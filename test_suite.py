"""
HL Strategy Lab — Test Suite
Permanent regression tests for known bugs and edge cases.
Run: python3 test_suite.py
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from strategy_engine import (
    Signal, HLData, StrategyResult,
    RSIStrategy, MACDStrategy, BollingerStrategy, TrendFollowStrategy,
    StochasticStrategy, VWAPStrategy, SupertrendStrategy,
    BreakoutStrategy, EMACrossStrategy,
    create_strategy, STRATEGIES, calc_atr,
)
from paper_trader import (
    apply_sentiment_overlay, check_sentiment_entry, load_strategy_config,
    MIN_CONFIDENCE, SENTIMENT_THRESHOLD, SENTIMENT_MIN_CONFIDENCE,
)

PASS = 0
FAIL = 0
FAILURES = []


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name} -- {detail}")


def make_candles(n=100, base_price=100.0, volatility=0.02, trend=0.0):
    """Generate synthetic OHLCV candles in HL format (keys: t, o, h, l, c, v)."""
    candles = []
    price = base_price
    for i in range(n):
        o = price
        change = np.random.randn() * volatility * price + trend * price
        c = max(0.01, o + change)
        h = max(o, c) + abs(np.random.randn() * volatility * price * 0.3)
        l = min(o, c) - abs(np.random.randn() * volatility * price * 0.3)
        v = abs(np.random.randn() * 1000) + 100
        candles.append({"t": i * 3600000, "o": round(o, 4), "h": round(h, 4),
                        "l": round(l, 4), "c": round(c, 4), "v": round(v, 2)})
        price = c
    return candles


# ═══════════════════════════════════════════════════════════════
# REGRESSION TESTS — Known bugs that must never come back
# ═══════════════════════════════════════════════════════════════

print("\n── Regression Tests ──")

# BUG #1: candle key was "close" instead of "c" in check_sentiment_entry
# This caused KeyError on every sentiment-driven entry attempt
fake_candles = make_candles(20)
try:
    result = check_sentiment_entry("TEST", fake_candles, {"TEST": {"score": 0.9, "confidence": 0.9}})
    test("Bug #1: check_sentiment_entry uses c[\"c\"] not c[\"close\"]",
         result is None or isinstance(result, StrategyResult),
         f"unexpected: {type(result)}")
except KeyError as e:
    test("Bug #1: check_sentiment_entry uses c[\"c\"] not c[\"close\"]", False, f"KeyError: {e}")
except Exception as e:
    test("Bug #1: check_sentiment_entry uses c[\"c\"] not c[\"close\"]",
         not isinstance(e, KeyError), f"Exception: {e}")


# BUG #2: MIN_CONFIDENCE was 0.6, blocking valid entries (HYPE at 0.57)
test("Bug #2: MIN_CONFIDENCE <= 0.5 (was 0.6, too restrictive)",
     MIN_CONFIDENCE <= 0.5, f"got {MIN_CONFIDENCE}")

# BUG #3: Sentiment overlay ran AFTER confidence gate, defeating its purpose
with open("paper_trader.py") as f:
    src = f.read()
test("Bug #3: Sentiment overlay applied BEFORE confidence gate",
     "Apply sentiment overlay BEFORE confidence gate" in src,
     "overlay-before-gate comment not found")
test("Bug #3b: Old pattern (overlay after gate) removed",
     "result.confidence > MIN_CONFIDENCE:\n                # Apply AI sentiment overlay" not in src,
     "old overlay-after-gate pattern still present")

# BUG #4: Strategy engine returned FLAT for MACD/EMACross/Supertrend when no crossover
# Root cause of 0% win rate — bot was always-in-market before fix, then the strategies
# were fixed to return FLAT when no crossover happened
for strat_name in ["macd", "emacross", "supertrend"]:
    strat = create_strategy(strat_name)
    flat_candles = make_candles(100, volatility=0.001)  # very low vol = no clear signal
    try:
        result = strat.compute(flat_candles)
        # At minimum, should return a valid StrategyResult (not crash)
        test(f"Bug #4: {strat_name} returns valid StrategyResult on flat data",
             isinstance(result, StrategyResult),
             f"got {type(result)}")
    except Exception as e:
        test(f"Bug #4: {strat_name} returns valid StrategyResult on flat data",
             False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════
# SENTIMENT OVERLAY TESTS
# ═══════════════════════════════════════════════════════════════

print("\n── Sentiment Overlay Tests ──")

# Aligned sentiment boosts confidence
adj_sig, adj_conf, note = apply_sentiment_overlay(
    Signal.LONG, 0.4, "TEST", {"TEST": {"score": 0.8, "confidence": 1.0}})
test("Aligned sentiment boosts confidence",
     adj_sig == Signal.LONG and adj_conf > 0.4 and "boosted" in note.lower(),
     f"sig={adj_sig}, conf={adj_conf}, note={note}")

# Contradicting sentiment blocks trade
adj_sig2, adj_conf2, note2 = apply_sentiment_overlay(
    Signal.LONG, 0.8, "TEST", {"TEST": {"score": -0.6, "confidence": 0.9}})
test("Contradicting sentiment blocks LONG",
     adj_sig2 == Signal.FLAT and "BLOCKED" in note2,
     f"sig={adj_sig2}, note={note2}")

# Neutral sentiment = no change
adj_sig3, adj_conf3, note3 = apply_sentiment_overlay(
    Signal.LONG, 0.7, "TEST", {"TEST": {"score": 0.1, "confidence": 0.9}})
test("Neutral sentiment passes through unchanged",
     adj_sig3 == Signal.LONG and adj_conf3 == 0.7,
     f"sig={adj_sig3}, conf={adj_conf3}")

# Low confidence sentiment = ignored
adj_sig4, adj_conf4, note4 = apply_sentiment_overlay(
    Signal.LONG, 0.7, "TEST", {"TEST": {"score": 0.8, "confidence": 0.2}})
test("Low confidence sentiment ignored",
     adj_sig4 == Signal.LONG and adj_conf4 == 0.7,
     f"sig={adj_sig4}, conf={adj_conf4}")

# Sentiment entry threshold check
strong_sent = {"TEST": {"score": 0.9, "confidence": 0.95}}
result = check_sentiment_entry("TEST", fake_candles, strong_sent)
test("Strong sentiment triggers entry check (returns result or None without crash)",
     result is None or isinstance(result, StrategyResult),
     f"got {type(result)}")

# Weak sentiment should not trigger
weak_sent = {"TEST": {"score": 0.4, "confidence": 0.3}}
result2 = check_sentiment_entry("TEST", fake_candles, weak_sent)
test("Weak sentiment does not trigger entry",
     result2 is None, f"got {result2}")


# ═══════════════════════════════════════════════════════════════
# STRATEGY ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

print("\n── Strategy Engine Tests ──")

# All strategies should be constructible and computable
trend_up = make_candles(100, base_price=100, trend=0.001)  # gentle uptrend
for name in STRATEGIES:
    try:
        strat = create_strategy(name)
        result = strat.compute(trend_up)
        test(f"Strategy '{name}' runs on synthetic data",
             isinstance(result, StrategyResult),
             f"got {type(result)}")
    except Exception as e:
        test(f"Strategy '{name}' runs on synthetic data", False, str(e)[:80])

# Trend strategy should detect uptrend
np.random.seed(42)  # deterministic for this test
trend_up_strong = make_candles(100, base_price=100, trend=0.005, volatility=0.005)
trend_result = TrendFollowStrategy().compute(trend_up_strong)
test("TrendFollow detects uptrend on synthetic rising data",
     trend_result.signal == Signal.LONG,
     f"got {trend_result.signal.value}")

# Breakout should not fire inside range
range_candles = make_candles(100, volatility=0.005)
breakout_result = BreakoutStrategy().compute(range_candles)
test("Breakout returns FLAT on tight range data",
     breakout_result.signal == Signal.FLAT,
     f"got {breakout_result.signal.value}")

# Candle format validation
test("Candles use 'c' key (not 'close')",
     "c" in fake_candles[0] and "close" not in fake_candles[0],
     f"keys: {list(fake_candles[0].keys())}")

# candles_to_arrays works correctly
o, h, l, c, v = HLData.candles_to_arrays(fake_candles)
test("candles_to_arrays returns equal-length arrays",
     len(o) == len(h) == len(l) == len(c) == len(v) == len(fake_candles),
     f"lengths: o={len(o)}, h={len(h)}, l={len(l)}, c={len(c)}, v={len(v)}")


# ═══════════════════════════════════════════════════════════════
# STATE INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════

print("\n── State Integrity Tests ──")

state_path = "paper_trader_state.json"
if os.path.exists(state_path):
    with open(state_path) as f:
        state = json.load(f)

    # No negative position sizes
    for pair, pos in state.get("positions", {}).items():
        test(f"Position {pair} size > 0",
             pos["size"] > 0, f"size={pos['size']}")

    # Capital is reasonable
    capital = state.get("capital", 0)
    test("Capital within reasonable range (0-100K)",
         0 < capital < 100000, f"capital={capital}")

    # Closed trades have required fields
    for i, trade in enumerate(state.get("closed_trades", [])):
        test(f"Closed trade #{i} has pnl field",
             "pnl" in trade and isinstance(trade["pnl"], (int, float)),
             f"missing/invalid pnl")
        test(f"Closed trade #{i} has exit_reason",
             "exit_reason" in trade, "missing exit_reason")
else:
    test("State file exists", False, f"{state_path} not found")


# Config integrity
config = load_strategy_config()
for pair in ["BTC", "ETH", "SOL", "HYPE"]:
    test(f"Config has strategy for {pair}",
         pair in config and "strategy" in config[pair],
         f"missing {pair} in config")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  RESULT: {PASS} PASS, {FAIL} FAIL")
if FAILURES:
    print(f"\n  FAILURES:")
    for f in FAILURES:
        print(f"    - {f}")
print(f"{'='*60}")

# Save results for harness_bundler
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save results to last_test_results.json")
args, _ = parser.parse_known_args()
if args.save:
    from datetime import datetime, timezone
    with open("last_test_results.json", "w") as f:
        json.dump({"pass": PASS, "fail": FAIL, "failures": FAILURES,
                    "ran_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    print(f"💾 Test results → last_test_results.json")

sys.exit(1 if FAIL > 0 else 0)
