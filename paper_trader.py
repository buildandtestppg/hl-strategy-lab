"""
HL Strategy Lab — Paper Trader
Live simulation engine. Fetches real HL prices, runs best strategy per pair,
tracks positions, records all trades to JSON for dashboard.
Designed to run via cron (every 1h) — zero LLM calls, pure deterministic.
"""
import json
import time
import os
import requests
import numpy as np
from datetime import datetime, timezone
from strategy_engine import (
    Signal, StrategyResult, HLData, RSIStrategy, MACDStrategy, BollingerStrategy,
    TrendFollowStrategy, StochasticStrategy, VWAPStrategy,
    SupertrendStrategy, BreakoutStrategy, EMACrossStrategy, calc_atr,
)
from audit_logger import log_decision, log_cycle_summary, rotate_log_if_needed

# ─── Config (loaded from strategy_config.json — AI can rewrite this) ───

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "strategy_config.json")

def load_strategy_config():
    """Load strategy assignments from JSON config. AI can modify this file."""
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    # Filter out _meta key
    return {k: v for k, v in cfg.items() if not k.startswith("_")}

def save_strategy_config(config, meta_note=""):
    """Save updated strategy config with metadata."""
    full_cfg = dict(config)
    full_cfg["_meta"] = {
        "last_optimized": datetime.now(timezone.utc).isoformat(),
        "optimization_count": 0,
        "history": []
    }
    # Load existing meta to preserve history
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            old = json.load(f)
        if "_meta" in old:
            full_cfg["_meta"]["optimization_count"] = old["_meta"].get("optimization_count", 0) + 1
            full_cfg["_meta"]["history"] = old["_meta"].get("history", [])[-9:] + [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": meta_note,
            }]
    with open(CONFIG_FILE, "w") as f:
        json.dump(full_cfg, f, indent=2)

# Load config at import time (also reloaded by get_strategy each cycle)
PAIR_STRATEGIES = load_strategy_config()

STRATEGY_MAP = {
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerStrategy,
    "trend": TrendFollowStrategy,
    "stochastic": StochasticStrategy,
    "vwap": VWAPStrategy,
    "supertrend": SupertrendStrategy,
    "breakout": BreakoutStrategy,
    "emacross": EMACrossStrategy,
}

def get_strategy(pair):
    """Create strategy instance from config."""
    cfg = load_strategy_config().get(pair)
    if not cfg:
        return None
    cls = STRATEGY_MAP.get(cfg["strategy"])
    if not cls:
        return None
    return cls(**cfg.get("params", {}))

INITIAL_CAPITAL = 5000.0
RISK_PER_TRADE = 0.01       # 1% of portfolio per position (was 2% — Kelly says less)
MAX_LEVERAGE = 1.5          # capped from 3 — 2.63x was reckless at 25% WR
STOP_LOSS_ATR = 3.5         # widened from 2.5 — stops were too tight, killing on noise
TAKE_PROFIT_ATR = 7.0       # widened from 5.0 — let winners run, fix inverted R:R
HL_FEE = 0.00045          # Real taker fee 0.045% (conservative — we'll use market orders)
SLIPPAGE = 0.0005         # 0.05% slippage on taker fills
MIN_CONFIDENCE = 0.65      # raised from 0.5 — too many bad entries at 0.5
MAX_POSITIONS = 4           # reduced from 6 — limit correlated exposure
CANDLE_INTERVAL = "1h"
CANDLE_DAYS = 30            # lookback for indicator calculation
MIN_HOLD_HOURS = 6          # increased from 3 — let trades breathe
REENTRY_COOLDOWN_HOURS = 8  # doubled from 4 — less overtrading
MAX_DRAWDOWN_PCT = 10.0     # halt all trading if drawdown exceeds 10%

# ─── Loop Engineering v2 (Jul 5 multi-model review) ───
# P1: Inverted exit hierarchy — signal_exit is now the DEFAULT.
#     regime_override is a secondary gate, not the primary exit.
REGIME_OVERRIDE_CONFIDENCE_GATE = 0.80  # must be this confident to override
REGIME_OVERRIDE_PROTECT_R = 2.0         # don't override trades already up >2R

# P3: Fee-aware confidence — penalize entries where fees eat the edge
FEE_IMPACT_ENABLED = True

# P5: Asset pruning — auto-disable pairs that consistently lose
ASSET_PRUNING_ENABLED = True
ASSET_PRUNING_MIN_TRADES = 7     # need this many trades before pruning
ASSET_PRUNING_MAX_WR = 35.0      # prune if WR below this after min_trades
ASSET_PRUNING_COOLDOWN_HOURS = 48  # re-enable after this long

# ─── AI Sentiment Overlay ───
SENTIMENT_FILE = os.path.join(os.path.dirname(__file__), "sentiment_scores.json")
SENTIMENT_THRESHOLD = 0.5   # raised from 0.3 — only strong sentiment overrides
SENTIMENT_MIN_CONFIDENCE = 0.6  # raised from 0.4 — require high confidence
SENTIMENT_MIN_SOURCES = 2   # NEW: require ≥2 sources before sentiment can override

# ─── Trading Lessons (feedback from self-review loop) ───
LESSONS_FILE = os.path.join(os.path.dirname(__file__), "trading_lessons.json")

# ─── Regime Override ───
# If sentiment flips hard against an open position AND position is losing,
# exit early to avoid bleeding against the trend.
REGIME_OVERRIDE_ENABLED = True
REGIME_OVERRIDE_THRESHOLD = 0.6   # sentiment must cross this to trigger override
REGIME_OVERRIDE_MIN_HOLD = 2      # don't override in first 2 hours (let trade breathe)

# ─── Exclusion List ───
# Pairs where buy-and-hold crushes all strategies — don't actively trade
EXCLUDE_PAIRS = []  # Populated dynamically by check_buy_hold()

STATE_FILE = os.path.join(os.path.dirname(__file__), "paper_trader_state.json")
TRADE_LOG_FILE = os.path.join(os.path.dirname(__file__), "paper_trades.json")
DASHBOARD_DATA = os.path.join(os.path.dirname(__file__), "dashboard_data.json")

# ─── State Management ───

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        # Sanitize: fix any corrupted sizes from older bugs
        for pair, pos in state.get("positions", {}).items():
            pos["size"] = abs(pos["size"])  # size must always be positive
        return state
    return {
        "capital": INITIAL_CAPITAL,
        "initial_capital": INITIAL_CAPITAL,
        "positions": {},      # {pair: {direction, entry_price, size, stop_loss, take_profit, entry_time, strategy}}
        "closed_trades": [],
        "equity_curve": [],   # [{timestamp, value, pnl_pct}]
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_update": None,
        "total_fees_paid": 0,
        "last_close_times": {},  # {pair: timestamp} for re-entry cooldown
    }

def save_state(state):
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    # Atomic write: write to temp then rename (prevents corruption from concurrent cron runs)
    tmp_file = STATE_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_file, STATE_FILE)

# ─── Helpers ───

def load_sentiment():
    """Load fleet sentiment scores. Returns {pair: {score, confidence, sources}}."""
    if not os.path.exists(SENTIMENT_FILE):
        return {}
    try:
        with open(SENTIMENT_FILE) as f:
            data = json.load(f)
        # Check freshness — staleness guard at 4 hours
        for pair, info in data.items():
            ts = info.get("timestamp", "")
            if ts:
                age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 3600
                if age_h > 4:
                    return {}  # stale — don't use
        return data
    except Exception:
        return {}


def load_lessons():
    """Load trading lessons from the self-review loop.
    Returns dict with pair performance and active lessons that should
    influence entry decisions. High-trust lessons can block or boost trades."""
    if not os.path.exists(LESSONS_FILE):
        return {"pairs": {}, "active_lessons": []}
    try:
        with open(LESSONS_FILE) as f:
            data = json.load(f)
        # Freshness guard — lessons older than 7 days are stale
        ts = data.get("generated_at", "")
        if ts:
            age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 3600
            if age_h > 168:  # 7 days
                return {"pairs": {}, "active_lessons": []}
        return data
    except Exception:
        return {"pairs": {}, "active_lessons": []}

# P5: Asset pruning — track disabled pairs in state
PRUNED_PAIRS_FILE = os.path.join(os.path.dirname(__file__), "pruned_pairs.json")

def load_pruned_pairs():
    """Load list of auto-pruned pairs with timestamps."""
    if not os.path.exists(PRUNED_PAIRS_FILE):
        return {}
    try:
        with open(PRUNED_PAIRS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_pruned_pairs(data):
    """Save pruned pairs data."""
    with open(PRUNED_PAIRS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def check_asset_pruning(closed_trades, pair):
    """P5: Check if a pair should be pruned based on performance.
    
    Returns (should_prune, reason) tuple.
    A pair is pruned if:
    - It has >= ASSET_PRUNING_MIN_TRADES closed trades
    - Win rate < ASSET_PRUNING_MAX_WR
    - Not already in cooldown period
    """
    if not ASSET_PRUNING_ENABLED:
        return False, ""
    
    pair_trades = [t for t in closed_trades if t.get("pair") == pair]
    if len(pair_trades) < ASSET_PRUNING_MIN_TRADES:
        return False, ""
    
    wins = sum(1 for t in pair_trades if t.get("pnl", 0) > 0)
    wr = wins / len(pair_trades) * 100
    total_pnl = sum(t.get("pnl", 0) for t in pair_trades)
    
    if wr < ASSET_PRUNING_MAX_WR:
        return True, f"{pair} pruned: {wr:.0f}% WR over {len(pair_trades)} trades (PnL ${total_pnl:.2f})"
    
    return False, ""


def check_lesson_warnings(pair, direction, lessons):
    """Check if accumulated trading lessons warn against this entry.
    Returns (should_warn, warning_text, confidence_adjustment).
    
    Uses high-trust patterns to identify known-bad entries:
    - If a pattern has <40% win rate and ≥3 occurrences, warn
    - If a pattern has >65% win rate, boost confidence
    """
    pair_data = lessons.get("pairs", {}).get(pair)
    if not pair_data:
        return False, "", 0.0
    
    patterns = pair_data.get("top_patterns", [])
    warning_parts = []
    adjustment = 0.0
    
    for p in patterns:
        pattern_name = p.get("pattern", "")
        wr = p.get("win_rate", 50)
        count = p.get("count", 0)
        
        # Only act on patterns with enough data
        if count < 3:
            continue
        
        # Known-bad pattern for this pair
        if wr <= 40:
            if "short_hold" in pattern_name and direction == Signal.LONG:
                warning_parts.append(f"⚠️ {pair} short holds: {wr:.0f}% WR ({count} trades)")
                adjustment -= 0.05
            elif "stopped_out" in pattern_name:
                warning_parts.append(f"⚠️ {pair} frequently stopped out ({wr:.0f}% WR)")
                adjustment -= 0.05
        
        # Known-good pattern
        elif wr >= 65 and count >= 3:
            adjustment += 0.03  # small boost
    
    if warning_parts:
        return True, " | ".join(warning_parts), max(adjustment, -0.15)
    return False, "", adjustment

def apply_sentiment_overlay(signal, confidence, pair, sentiment):
    """Apply AI sentiment overlay to a TA signal.
    
    Rules (enhanced with convergence detection):
    - If sentiment strongly contradicts TA (score > threshold, opposite direction): BLOCK the trade
    - If sentiment strongly agrees with TA AND convergence detected: BOOST confidence more
    - If sentiment is neutral or weak: no change
    - Convergence signals (≥2 sources agreeing) get extra weight
    
    Returns (adjusted_signal, adjusted_confidence, sentiment_note)
    """
    pair_data = sentiment.get(pair)
    if not pair_data or pair_data.get("confidence", 0) < SENTIMENT_MIN_CONFIDENCE:
        return signal, confidence, ""
    
    # Source diversity gate: require ≥SENTIMENT_MIN_SOURCES before sentiment can override
    sources = pair_data.get("sources", [])
    if isinstance(sources, list):
        active_sources = sources
    elif isinstance(sources, dict):
        active_sources = [s for s, v in sources.items() if v and abs(v.get("score", 0)) > 0.1]
    else:
        active_sources = []
    if len(active_sources) < SENTIMENT_MIN_SOURCES:
        return signal, confidence, f"sentiment skipped (only {len(active_sources)} sources, need {SENTIMENT_MIN_SOURCES})"
    
    score = pair_data["score"]
    is_convergence = pair_data.get("convergence", False)
    convergence_strength = pair_data.get("convergence_strength", 0)
    contradictions = pair_data.get("contradictions", 0)
    
    # Determine sentiment direction
    # Convergence lowers the bar: if ≥2 sources agree, smaller scores are actionable
    effective_threshold = SENTIMENT_THRESHOLD
    if is_convergence and convergence_strength >= 2:
        effective_threshold = SENTIMENT_THRESHOLD * 0.7  # 0.5 → 0.35

    if score > effective_threshold:
        sent_dir = Signal.LONG
    elif score < -effective_threshold:
        sent_dir = Signal.SHORT
    else:
        return signal, confidence, "sentiment neutral"
    
    # Check agreement
    if signal == sent_dir:
        # Aligned — boost confidence (extra boost for convergence)
        # Cap post-boost confidence at 0.85 per risk review — convergence can
        # rescue a marginal signal but shouldn't manufacture near-certainty
        boost = min(0.2, abs(score) * 0.15)
        if is_convergence:
            boost = min(0.3, boost + convergence_strength * 0.05)
        note = f"sentiment aligned ({score:+.2f}), conf boosted"
        if is_convergence:
            note += f" [convergence x{convergence_strength}]"
        return signal, min(0.85, confidence + boost), note
    elif signal in (Signal.LONG, Signal.SHORT):
        # Contradiction — block the trade
        note = f"BLOCKED by sentiment ({score:+.2f} vs {signal.value})"
        if is_convergence:
            note += f" [strong: {convergence_strength}-source convergence against]"
        return Signal.FLAT, 0, note
    else:
        return signal, confidence, ""

# Threshold for sentiment to trigger its own entry (higher than gate threshold)
SENTIMENT_ENTRY_THRESHOLD = 0.7   # min |score| to trigger entry without TA signal
SENTIMENT_ENTRY_MOMENTUM = 3      # price must be above/below MA this many candles

def check_sentiment_entry(pair, candles, sentiment):
    """Check if sentiment alone is strong enough to trigger an entry.
    
    Requires BOTH:
    1. Extreme sentiment score (|score| >= 0.7) with high confidence (>= 0.5)
    2. Price momentum confirmation (recent candles moving in sentiment direction)
    
    This lets the bot catch strong moves that TA hasn't signaled yet.
    Returns a StrategyResult if entry triggered, None otherwise.
    """
    pair_data = sentiment.get(pair)
    if not pair_data:
        return None
    
    score = pair_data.get("score", 0)
    confidence = pair_data.get("confidence", 0)
    
    # Need extreme sentiment AND decent confidence
    if abs(score) < SENTIMENT_ENTRY_THRESHOLD or confidence < 0.5:
        return None
    
    if len(candles) < SENTIMENT_ENTRY_MOMENTUM + 1:
        return None
    
    # Price momentum confirmation: recent candles should be trending in sentiment direction
    recent = candles[-SENTIMENT_ENTRY_MOMENTUM:]
    prices = [float(c["c"]) for c in recent]
    
    if score > SENTIMENT_ENTRY_THRESHOLD:
        # Bullish sentiment — check price is rising (at least 2 of last 3 candles up)
        up_candles = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
        if up_candles >= 2 and prices[-1] >= prices[0]:
            return StrategyResult(
                signal=Signal.LONG,
                confidence=min(0.75, 0.55 + abs(score) * 0.2),
                indicators={},
                reason=f"Sentiment-driven LONG (score {score:+.2f}, conf {confidence:.1f}, momentum confirmed)"
            )
    elif score < -SENTIMENT_ENTRY_THRESHOLD:
        # Bearish sentiment — check price is falling
        down_candles = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])
        if down_candles >= 2 and prices[-1] <= prices[0]:
            return StrategyResult(
                signal=Signal.SHORT,
                confidence=min(0.75, 0.55 + abs(score) * 0.2),
                indicators={},
                reason=f"Sentiment-driven SHORT (score {score:+.2f}, conf {confidence:.1f}, momentum confirmed)"
            )
    
    return None


def trade_cost(size, entry_price, exit_price):
    notional = size * entry_price + size * exit_price
    return notional * (HL_FEE + SLIPPAGE)

def calc_fee_impact(pair, price, candles):
    """P3: Calculate fee impact as a fraction of expected trade range.
    
    If round-trip fees eat >50% of the typical ATR move, the trade is 
    fee-doomed. Returns a multiplier (0.5–1.0) to penalize confidence.
    """
    if not FEE_IMPACT_ENABLED:
        return 1.0
    try:
        o, h, l, c, v = HLData.candles_to_arrays(candles)
        atr = calc_atr(h, l, c, 14)
        current_atr = atr[-1] if not np.isnan(atr[-1]) else price * 0.02
        # Expected round-trip cost as % of price
        round_trip_fee_pct = 2 * (HL_FEE + SLIPPAGE)  # entry + exit
        # Expected move (SL distance = 3.5 ATR — the risk we're taking)
        expected_move_pct = (current_atr * STOP_LOSS_ATR) / price
        if expected_move_pct <= 0:
            return 1.0
        # Fee impact ratio: how much of the expected move gets eaten by fees
        fee_ratio = round_trip_fee_pct / expected_move_pct
        # If fees eat 100%+ of expected move, kill the trade (0.5 floor)
        # If fees eat 10% or less, no penalty (1.0)
        impact = max(0.5, 1.0 - max(0, fee_ratio - 0.1) * 2)
        return round(impact, 3)
    except Exception:
        return 1.0

def get_current_prices():
    """Fetch all current prices from HL."""
    r = requests.post("https://api.hyperliquid.xyz/info", json={"type": "allMids"}, timeout=10)
    r.raise_for_status()
    return r.json()

def get_funding_rates():
    """Fetch live funding rates from HL. Returns {pair: rate_per_hour}.
    Positive = longs pay shorts, Negative = shorts pay longs."""
    r = requests.post("https://api.hyperliquid.xyz/info", json={
        "type": "metaAndAssetCtxs"
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    meta, ctxs = data[0], data[1]
    rates = {}
    for i, asset in enumerate(meta["universe"]):
        name = asset["name"]
        ctx = ctxs[i]
        funding = ctx.get("funding")
        if funding is not None:
            rates[name] = float(funding)
    return rates

def calc_funding_cost(pair, direction, notional, hold_hours, funding_rates):
    """Calculate funding cost for a position.
    Returns the cost (positive = we pay, negative = we earn)."""
    rate = funding_rates.get(pair)
    if rate is None:
        return 0.0
    # Funding charged per hour. Positive rate = longs pay shorts.
    # If we're LONG and rate is positive, we pay. If SHORT and rate negative, we pay.
    if direction == "LONG":
        return notional * rate * hold_hours
    else:  # SHORT — opposite sign
        return -notional * rate * hold_hours

# ─── Paper Trader Core ───

class PaperTrader:
    def __init__(self):
        self.state = load_state()

    def run_cycle(self):
        """Run one trading cycle: check exits, check entries, update dashboard data."""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions": [],
            "positions": {},
            "portfolio_value": 0,
            "prices": {},
        }

        # ─── Drawdown circuit breaker ───
        peak = self.state.get("peak_equity", self.state.get("initial_capital", INITIAL_CAPITAL))
        equity = self.state["capital"]
        self.state["peak_equity"] = max(peak, equity)
        dd_pct = ((self.state["peak_equity"] - equity) / self.state["peak_equity"]) * 100
        if dd_pct >= MAX_DRAWDOWN_PCT:
            results["actions"].append(
                f"🛑 CIRCUIT BREAKER: drawdown {dd_pct:.1f}% exceeds {MAX_DRAWDOWN_PCT}% — "
                f"closing all positions and halting new entries"
            )
            # Close all open positions
            prices = get_current_prices()
            for pair in list(self.state["positions"].keys()):
                pos = self.state["positions"][pair]
                current_price = float(prices.get(pair, 0))
                if current_price > 0:
                    self._close_position(pair, current_price, "circuit_breaker", results)
            self._save_dashboard(results, prices)
            return results

        prices = get_current_prices()
        candles_cache = {}

        # Load fleet sentiment for AI overlay
        sentiment = load_sentiment()
        
        # Load trading lessons from self-review loop
        lessons = load_lessons()

        # Fetch live funding rates for accurate cost tracking
        try:
            funding_rates = get_funding_rates()
        except Exception:
            funding_rates = {}  # Don't fail the cycle if funding API hiccups

        # Reload config fresh each cycle (AI may have changed it)
        active_pairs = load_strategy_config()
        
        # P5: Load pruned pairs and auto-re-enable after cooldown
        pruned = load_pruned_pairs()
        reenabled = []
        for p_pair in list(pruned.keys()):
            pruned_time = pruned[p_pair].get("timestamp", "")
            if pruned_time:
                try:
                    pruned_dt = datetime.fromisoformat(pruned_time.replace("Z", "+00:00"))
                    age_h = (datetime.now(timezone.utc) - pruned_dt).total_seconds() / 3600
                    if age_h >= ASSET_PRUNING_COOLDOWN_HOURS:
                        del pruned[p_pair]
                        reenabled.append(p_pair)
                except Exception:
                    pass
        if reenabled:
            save_pruned_pairs(pruned)
            results["actions"].append(f"♻️ Asset re-enabled after cooldown: {', '.join(reenabled)}")

        # Fetch candles for all pairs
        for pair in active_pairs:
            try:
                candles = HLData.get_candles(pair, interval=CANDLE_INTERVAL, days=CANDLE_DAYS)
                # CRITICAL: Drop the last (forming/unclosed) candle.
                # Without this, every 5-min cron run re-evaluates a shifting candle → whipsaw.
                if candles and len(candles) > 30:
                    candles = candles[:-1]
                candles_cache[pair] = candles
                price = float(prices.get(pair, candles[-1]["c"]))
                results["prices"][pair] = price
            except Exception as e:
                results["actions"].append(f"⚠️ {pair}: Failed to fetch candles: {e}")
                continue

        # 1. Check exits on open positions
        for pair in list(self.state["positions"].keys()):
            if pair not in candles_cache:
                continue
            pos = self.state["positions"][pair]
            current_price = results["prices"].get(pair)
            if not current_price:
                continue

            candles = candles_cache[pair]
            o, h, l, c, v = HLData.candles_to_arrays(candles)
            atr = calc_atr(h, l, c, 14)
            current_atr = atr[-1] if not np.isnan(atr[-1]) else current_price * 0.02

            exit_reason = None
            exit_price = None

            # Check stop loss / take profit
            if pos["direction"] == "LONG":
                if current_price <= pos["stop_loss"]:
                    exit_reason = "stop_loss"
                    exit_price = pos["stop_loss"]
                elif current_price >= pos["take_profit"]:
                    exit_reason = "take_profit"
                    exit_price = pos["take_profit"]
            else:  # SHORT
                if current_price >= pos["stop_loss"]:
                    exit_reason = "stop_loss"
                    exit_price = pos["stop_loss"]
                elif current_price <= pos["take_profit"]:
                    exit_reason = "take_profit"
                    exit_price = pos["take_profit"]

            # ─── EXIT HIERARCHY (v2 — INVERTED) ───
            # Priority order (check each, first match wins):
            #   1. Stop loss / take profit (hard limits — always checked first)
            #   2. Signal exit (DEFAULT — let the strategy drive exits)
            #   3. Regime override (GATED — only if confidence > 0.80 AND trade < 2R profit)
            #
            # Previously regime_override was checked before signal_exit, causing
            # 87% of exits to be sentiment-driven noise (+$0.82 PnL on 20 trades).
            # Signal exits generated 100% of profit (+$32.33 on 3 trades).
            # Fix: check signal_exit FIRST, then regime_override as a secondary gate.

            if not exit_reason:
                # ─── Signal Exit (PRIMARY) ───
                entry_strat_name = pos.get("strategy", "unknown")
                cfg = load_strategy_config().get(pair, {})
                current_strat_name = cfg.get("strategy", "unknown")
                hold_hours = (time.time() - pos["entry_ts"]) / 3600

                if hold_hours < MIN_HOLD_HOURS:
                    # Too soon — let SL/TP work, don't exit on signal
                    pass
                elif entry_strat_name == current_strat_name:
                    # Config hasn't changed and held long enough — check signal exit
                    strat = get_strategy(pair)
                    if strat:
                        result = strat.compute(candles)
                        if (pos["direction"] == "LONG" and result.signal == Signal.SHORT) or \
                           (pos["direction"] == "SHORT" and result.signal == Signal.LONG):
                            exit_reason = "signal_exit"
                            exit_price = current_price
                else:
                    # Strategy was swapped by optimizer while position is open.
                    # Only exit on SL/TP, not signal (avoids premature exit from new strategy).
                    pass

            if not exit_reason and REGIME_OVERRIDE_ENABLED:
                # ─── Regime Override (SECONDARY — GATED) ───
                # Only fires if ALL conditions met:
                #   1. Sentiment strongly flipped against position (> threshold)
                #   2. Sentiment confidence >= 0.80 (was 0.6 — too permissive)
                #   3. Trade is NOT already in >2R profit (let winners run)
                #   4. Held >= minimum override hold time
                pair_data = sentiment.get(pair) if sentiment else None
                if pair_data:
                    sent_score = pair_data.get("score", 0)
                    sent_conf = pair_data.get("confidence", 0)
                    hold_hours = (time.time() - pos["entry_ts"]) / 3600

                    # Check if sentiment has flipped against us significantly
                    override_triggered = False
                    if pos["direction"] == "LONG" and sent_score < -REGIME_OVERRIDE_THRESHOLD:
                        override_triggered = True
                    elif pos["direction"] == "SHORT" and sent_score > REGIME_OVERRIDE_THRESHOLD:
                        override_triggered = True

                    # P1 fix: Calculate R-multiple to protect winning trades
                    if override_triggered:
                        risk_per_unit = abs(pos["entry_price"] - pos["stop_loss"])
                        if risk_per_unit > 0:
                            if pos["direction"] == "LONG":
                                current_r = (current_price - pos["entry_price"]) / risk_per_unit
                            else:
                                current_r = (pos["entry_price"] - current_price) / risk_per_unit
                        else:
                            current_r = 0

                        # Gate 1: confidence must be very high
                        # Gate 2: trade must not be >2R in profit (protect winners)
                        # Gate 3: must have held minimum time
                        if sent_conf < REGIME_OVERRIDE_CONFIDENCE_GATE:
                            override_triggered = False  # not confident enough
                        elif current_r >= REGIME_OVERRIDE_PROTECT_R:
                            override_triggered = False  # trade is winning big — let it run
                        elif hold_hours < REGIME_OVERRIDE_MIN_HOLD:
                            override_triggered = False  # too early

                    if override_triggered:
                        exit_reason = "regime_override"
                        exit_price = current_price

            if exit_reason:
                self._close_position(pair, exit_price, exit_reason, results, funding_rates)

        # 2. Check for new entries
        # Track pairs closed THIS CYCLE to prevent same-cycle re-entry (whipsaw fix)
        _now = time.time()
        pairs_closed_this_cycle = set()
        for p, close_ts in self.state.get("last_close_times", {}).items():
            if _now - close_ts < 120:  # closed in last 2 minutes = this cycle
                pairs_closed_this_cycle.add(p)

        open_count = len(self.state["positions"])
        cycle_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        for pair in active_pairs:
            if pair in self.state["positions"]:
                log_decision(pair, self.state["positions"][pair]["direction"], 0,
                             "Already in position", "BLOCKED_ALREADY_OPEN",
                             action="held", price=results["prices"].get(pair),
                             cycle_id=cycle_id)
                continue  # already in position
            # P5: Skip pruned pairs
            if pair in pruned:
                log_decision(pair, "FLAT", 0, f"Pruned: {pruned[pair].get('reason','')}",
                             "BLOCKED_PRUNED", action="evaluated",
                             price=results["prices"].get(pair), cycle_id=cycle_id)
                continue
            if pair in pairs_closed_this_cycle:
                continue  # closed this cycle — no re-entry
            if open_count >= MAX_POSITIONS:
                log_decision(pair, "FLAT", 0, "Max positions reached",
                             "BLOCKED_MAX_POSITIONS", action="evaluated",
                             price=results["prices"].get(pair), cycle_id=cycle_id)
                break
            if pair not in candles_cache:
                continue

            # Re-entry cooldown: skip if we closed this pair recently
            last_close_ts = self.state.get("last_close_times", {}).get(pair)
            if last_close_ts and (time.time() - last_close_ts) / 3600 < REENTRY_COOLDOWN_HOURS:
                remaining = REENTRY_COOLDOWN_HOURS - (time.time() - last_close_ts) / 3600
                log_decision(pair, "FLAT", 0, f"Cooldown ({remaining:.1f}h left)",
                             "BLOCKED_COOLDOWN", action="evaluated",
                             price=results["prices"].get(pair), cycle_id=cycle_id)
                continue

            strat = get_strategy(pair)
            if not strat:
                continue

            candles = candles_cache[pair]
            result = strat.compute(candles)

            entered = False

            # Apply sentiment overlay BEFORE confidence gate —
            # fleet intelligence should boost weak-but-valid signals, not just
            # rubber-stamp ones that already passed.
            if result.signal in (Signal.LONG, Signal.SHORT):
                adj_signal, adj_conf, sent_note = apply_sentiment_overlay(
                    result.signal, result.confidence, pair, sentiment
                )
                if sent_note:
                    result.reason = f"{result.reason} | sentiment: {sent_note}"
            else:
                adj_signal = result.signal
                adj_conf = result.confidence

            if adj_signal in (Signal.LONG, Signal.SHORT) and adj_conf > MIN_CONFIDENCE:
                # P3: Apply fee-aware confidence penalty
                fee_impact = calc_fee_impact(pair, results["prices"][pair], candles)
                if fee_impact < 1.0:
                    pre_fee_conf = adj_conf
                    adj_conf *= fee_impact
                    result.reason += f" | fee_impact: {fee_impact:.2f} (conf {pre_fee_conf:.2f}→{adj_conf:.2f})"

                # Check trading lessons — warn or boost based on historical patterns
                lesson_warn, lesson_text, lesson_adj = check_lesson_warnings(pair, adj_signal, lessons)
                if lesson_adj:
                    adj_conf = max(0.0, min(0.95, adj_conf + lesson_adj))
                    result.reason += f" | lesson: {lesson_text}" if lesson_text else f" | lesson_adj: {lesson_adj:+.2f}"
                
                if adj_conf > MIN_CONFIDENCE:
                    result.signal = adj_signal
                    result.confidence = adj_conf
                    self._open_position(pair, result, results["prices"][pair], candles, results)
                    open_count += 1
                    entered = True
                    log_decision(pair, result.signal.value, result.confidence, result.reason,
                                 "PASSED", sentiment=sentiment.get(pair, {}) if sentiment else None,
                                 action="opened", price=results["prices"][pair], cycle_id=cycle_id)
                else:
                    # Lesson warning pushed confidence below threshold
                    results["actions"].append(
                        f"📚 {pair}: Entry blocked by lesson — {lesson_text}"
                    )
                    log_decision(pair, adj_signal.value, adj_conf, result.reason,
                                 "BLOCKED_LESSON", action="evaluated",
                                 price=results["prices"][pair], cycle_id=cycle_id)
            elif result.signal in (Signal.LONG, Signal.SHORT) and sent_note and "BLOCKED" in sent_note:
                results["actions"].append(
                    f"🧠 {pair}: TA signal {result.signal.value} blocked by sentiment overlay"
                )
                log_decision(pair, result.signal.value, result.confidence, result.reason,
                             "BLOCKED_SENTIMENT", sentiment=sentiment.get(pair, {}) if sentiment else None,
                             action="evaluated", price=results["prices"][pair], cycle_id=cycle_id)
            elif result.signal in (Signal.LONG, Signal.SHORT):
                log_decision(pair, result.signal.value, adj_conf, result.reason,
                             "BLOCKED_CONFIDENCE", action="evaluated",
                             price=results["prices"][pair], cycle_id=cycle_id)
            else:
                log_decision(pair, "FLAT", 0, result.reason,
                             "SKIPPED_FLAT", action="evaluated",
                             price=results["prices"][pair], cycle_id=cycle_id)

            # Sentiment-driven entry: extremely strong sentiment + price momentum confirmation
            if not entered:
                sent_entry = check_sentiment_entry(pair, candles, sentiment)
                if sent_entry:
                    self._open_position(pair, sent_entry, results["prices"][pair], candles, results)
                    open_count += 1
                    log_decision(pair, sent_entry.signal.value, sent_entry.confidence,
                                 sent_entry.reason, "PASSED_SENTIMENT",
                                 sentiment=sentiment.get(pair, {}) if sentiment else None,
                                 action="opened", price=results["prices"][pair], cycle_id=cycle_id)

        # 2b. P5: Auto-prune pairs with persistent poor performance
        for pair in list(active_pairs.keys()):
            if pair in pruned:
                continue  # already pruned
            should_prune, prune_reason = check_asset_pruning(self.state.get("closed_trades", []), pair)
            if should_prune:
                pruned[pair] = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": prune_reason,
                }
                save_pruned_pairs(pruned)
                results["actions"].append(f"✂️ PRUNED {prune_reason}")

        # 3. Calculate portfolio value (with unrealized funding cost estimate)
        portfolio_value = self.state["capital"]
        for pair, pos in self.state["positions"].items():
            current_price = results["prices"].get(pair, pos["entry_price"])
            if pos["direction"] == "LONG":
                unrealized = (current_price - pos["entry_price"]) * pos["size"]
            else:
                unrealized = (pos["entry_price"] - current_price) * pos["size"]
            # Subtract unrealized funding cost for honest portfolio valuation
            hold_hours = (time.time() - pos["entry_ts"]) / 3600
            current_notional = pos["size"] * current_price
            unrealized_funding = calc_funding_cost(pair, pos["direction"], current_notional, hold_hours, funding_rates)
            unrealized -= unrealized_funding
            portfolio_value += unrealized
            results["positions"][pair] = {
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "current_price": current_price,
                "size": pos["size"],
                "unrealized_pnl": round(unrealized, 2),
                "unrealized_pnl_pct": round(unrealized / (pos["entry_price"] * pos["size"]) * 100, 2),
                "stop_loss": pos["stop_loss"],
                "take_profit": pos["take_profit"],
                "strategy": pos["strategy"],
                "entry_time": pos["entry_time"],
                "duration_hours": round(hold_hours, 1),
                "funding_rate": round(funding_rates.get(pair, 0) * 100, 6),
                "unrealized_funding": round(unrealized_funding, 4),
            }

        results["portfolio_value"] = round(portfolio_value, 2)
        results["capital"] = round(self.state["capital"], 2)
        results["pnl"] = round(portfolio_value - INITIAL_CAPITAL, 2)
        results["pnl_pct"] = round((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)
        results["initial_capital"] = INITIAL_CAPITAL
        results["open_positions"] = len(self.state["positions"])
        results["total_closed_trades"] = len(self.state["closed_trades"])
        results["total_fees_paid"] = round(self.state["total_fees_paid"], 2)

        # Win/loss stats from closed trades
        closed = self.state["closed_trades"]
        if closed:
            wins = [t for t in closed if t["pnl"] > 0]
            losses = [t for t in closed if t["pnl"] <= 0]
            results["stats"] = {
                "total_trades": len(closed),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / len(closed) * 100, 1),
                "avg_win": round(np.mean([t["pnl"] for t in wins]), 2) if wins else 0,
                "avg_loss": round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
                "best_trade": round(max(t["pnl"] for t in closed), 2),
                "worst_trade": round(min(t["pnl"] for t in closed), 2),
                "total_realized_pnl": round(sum(t["pnl"] for t in closed), 2),
            }
        else:
            results["stats"] = {
                "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "best_trade": 0, "worst_trade": 0,
                "total_realized_pnl": 0,
            }

        # 4. Record equity curve point
        if "equity_curve" not in self.state:
            self.state["equity_curve"] = []
        self.state["equity_curve"].append({
            "t": results["timestamp"],
            "v": results["portfolio_value"],
            "p": results["pnl_pct"],
        })
        # Keep last 500 points (~42 hours at 5min intervals)
        if len(self.state["equity_curve"]) > 500:
            self.state["equity_curve"] = self.state["equity_curve"][-500:]

        # 5. Save state + write dashboard data
        save_state(self.state)

        # Compute per-strategy and per-pair stats from all closed trades
        all_trades = self.state["closed_trades"]
        strategy_stats = {}
        pair_stats = {}
        for t in all_trades:
            # Per strategy
            s_name = t.get("strategy", "unknown")
            if s_name not in strategy_stats:
                strategy_stats[s_name] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            strategy_stats[s_name]["trades"] += 1
            if t["pnl"] > 0:
                strategy_stats[s_name]["wins"] += 1
            else:
                strategy_stats[s_name]["losses"] += 1
            strategy_stats[s_name]["pnl"] += t["pnl"]

            # Per pair
            p_name = t.get("pair", "?")
            if p_name not in pair_stats:
                pair_stats[p_name] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            pair_stats[p_name]["trades"] += 1
            if t["pnl"] > 0:
                pair_stats[p_name]["wins"] += 1
            else:
                pair_stats[p_name]["losses"] += 1
            pair_stats[p_name]["pnl"] += t["pnl"]

        # Add win rates
        for s in strategy_stats.values():
            s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 0
            s["pnl"] = round(s["pnl"], 2)
        for p in pair_stats.values():
            p["win_rate"] = round(p["wins"] / p["trades"] * 100, 1) if p["trades"] > 0 else 0
            p["pnl"] = round(p["pnl"], 2)

        # Loop Engineering: exit-reason breakdown (P1/P2 diagnostics)
        exit_breakdown = {}
        for t in all_trades:
            reason = t.get("exit_reason", "unknown")
            if reason not in exit_breakdown:
                exit_breakdown[reason] = {"count": 0, "pnl": 0.0, "fees": 0.0, "wins": 0}
            exit_breakdown[reason]["count"] += 1
            exit_breakdown[reason]["pnl"] += t.get("pnl", 0)
            exit_breakdown[reason]["fees"] += t.get("fees_paid", 0)
            if t.get("pnl", 0) > 0:
                exit_breakdown[reason]["wins"] += 1
        for reason_data in exit_breakdown.values():
            reason_data["pnl"] = round(reason_data["pnl"], 2)
            reason_data["fees"] = round(reason_data["fees"], 2)
            reason_data["win_rate"] = round(reason_data["wins"] / reason_data["count"] * 100, 1) if reason_data["count"] > 0 else 0

        # Load AI optimizer history
        ai_history = []
        config_path = os.path.join(os.path.dirname(__file__), "strategy_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg_data = json.load(f)
            if "_meta" in cfg_data:
                ai_history = cfg_data["_meta"].get("history", [])

        # Write dashboard data (combines results + state)
        dashboard = {
            "last_update": results["timestamp"],
            "portfolio": {
                "value": results["portfolio_value"],
                "capital": results["capital"],
                "initial": INITIAL_CAPITAL,
                "pnl": results["pnl"],
                "pnl_pct": results["pnl_pct"],
                "open_positions": results["open_positions"],
            },
            "prices": results["prices"],
            "positions": results["positions"],
            "stats": results["stats"],
            "actions": results["actions"],
            "closed_trades": self.state["closed_trades"][-50:],  # last 50
            "pair_strategies": {p: active_pairs[p]["strategy"] for p in active_pairs},
            "strategy_stats": strategy_stats,
            "pair_stats": pair_stats,
            "equity_curve": self.state["equity_curve"][-200:],
            "ai_history": ai_history,
            "sentiment": sentiment,
            "config": {
                "risk_per_trade": f"{RISK_PER_TRADE*100}%",
                "max_leverage": f"{MAX_LEVERAGE}x",
                "stop_loss": f"{STOP_LOSS_ATR}x ATR",
                "take_profit": f"{TAKE_PROFIT_ATR}x ATR",
                "min_confidence": MIN_CONFIDENCE,
                "regime_override_gate": REGIME_OVERRIDE_CONFIDENCE_GATE,
                "regime_override_protect_r": REGIME_OVERRIDE_PROTECT_R,
                "fee_aware": FEE_IMPACT_ENABLED,
                "asset_pruning": ASSET_PRUNING_ENABLED,
            },
            # Loop Engineering v2 data
            "exit_breakdown": exit_breakdown,
            "pruned_pairs": load_pruned_pairs(),
            "loop_v2": {
                "version": "2.0",
                "implemented": "2026-07-05",
                "changes": [
                    "P1: Exit hierarchy inverted (signal_exit first, regime_override gated to conf>0.80 + protect >2R winners)",
                    "P3: Fee-aware confidence scoring (penalize fee-doomed entries)",
                    "P5: Asset pruning (auto-disable pairs with <35% WR after 7 trades)",
                ],
            },
        }
        with open(DASHBOARD_DATA, "w") as f:
            json.dump(dashboard, f, indent=2)

        return results

    def _open_position(self, pair, result, price, candles, results):
        """Open a new simulated position."""
        # CRITICAL: Use actual per-pair prices for portfolio valuation
        portfolio_value = self._portfolio_value(results["prices"])
        
        # Guard: stop trading if portfolio is in the red
        if portfolio_value <= 0:
            results["actions"].append(f"⚠️ {pair}: Skipping — portfolio value negative (${portfolio_value:.2f})")
            return

        o, h, l, c, v = HLData.candles_to_arrays(candles)
        atr = calc_atr(h, l, c, 14)
        current_atr = atr[-1] if not np.isnan(atr[-1]) else price * 0.02

        risk_amount = portfolio_value * RISK_PER_TRADE
        stop_distance = current_atr * STOP_LOSS_ATR
        if stop_distance <= 0:
            return

        size = risk_amount / stop_distance
        
        # CRITICAL: Cap notional at INITIAL_CAPITAL * leverage (not inflated portfolio value)
        max_notional = INITIAL_CAPITAL * MAX_LEVERAGE
        if size * price > max_notional:
            size = max_notional / price

        # CRITICAL: size must always be positive (direction is tracked separately)
        size = abs(size)

        direction = "LONG" if result.signal == Signal.LONG else "SHORT"
        if direction == "LONG":
            stop_loss = price - current_atr * STOP_LOSS_ATR
            take_profit = price + current_atr * TAKE_PROFIT_ATR
        else:
            stop_loss = price + current_atr * STOP_LOSS_ATR
            take_profit = price - current_atr * TAKE_PROFIT_ATR

        # Deduct entry fees from capital (entry fee only)
        entry_cost = size * price * (HL_FEE + SLIPPAGE)
        self.state["capital"] -= entry_cost
        self.state["total_fees_paid"] += entry_cost

        self.state["positions"][pair] = {
            "direction": direction,
            "entry_price": price,
            "size": size,
            "stop_loss": round(stop_loss, 6),
            "take_profit": round(take_profit, 6),
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_ts": time.time(),
            "strategy": load_strategy_config().get(pair, {}).get("strategy", "unknown"),
            "confidence": result.confidence,
            "reason": result.reason,
        }

        results["actions"].append(
            f"🟢 OPENED {direction} {pair} @ ${price:.4f} | "
            f"Size: {size:.4f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f} | "
            f"Strategy: {load_strategy_config().get(pair, {}).get('strategy', 'unknown')} | {result.reason}"
        )

    def _close_position(self, pair, exit_price, reason, results, funding_rates=None):
        """Close a position and record the trade."""
        if funding_rates is None:
            funding_rates = {}
        pos = self.state["positions"][pair]
        entry = pos["entry_price"]
        size = pos["size"]
        direction = pos["direction"]

        if direction == "LONG":
            raw_pnl = (exit_price - entry) * size
        else:
            raw_pnl = (entry - exit_price) * size

        exit_cost = size * exit_price * (HL_FEE + SLIPPAGE)  # exit fee only (entry already paid)

        # Funding cost: based on hold time, direction, and live funding rate
        hold_hours = (time.time() - pos["entry_ts"]) / 3600
        exit_notional = size * exit_price
        funding_cost = calc_funding_cost(pair, direction, exit_notional, hold_hours, funding_rates)

        net_pnl = raw_pnl - exit_cost - funding_cost
        self.state["capital"] += net_pnl
        self.state["total_fees_paid"] += exit_cost + max(0, funding_cost)  # only count positive costs

        trade_record = {
            "pair": pair,
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_price,
            "size": size,
            "pnl": round(net_pnl, 4),
            "pnl_pct": round(net_pnl / (entry * size) * 100, 2),
            "entry_time": pos["entry_time"],
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "duration_hours": round(hold_hours, 1),
            "exit_reason": reason,
            "strategy": pos["strategy"],
            "fees_paid": round(exit_cost, 4),
            "funding_cost": round(funding_cost, 4),
            "total_cost": round(exit_cost + funding_cost, 4),
        }

        self.state["closed_trades"].append(trade_record)
        del self.state["positions"][pair]
        
        # Record close time for re-entry cooldown
        if "last_close_times" not in self.state:
            self.state["last_close_times"] = {}
        self.state["last_close_times"][pair] = time.time()

        emoji = "✅" if net_pnl > 0 else "❌"
        results["actions"].append(
            f"{emoji} CLOSED {direction} {pair} @ ${exit_price:.4f} | "
            f"P&L: ${net_pnl:+.2f} ({net_pnl/(entry*size)*100:+.1f}%) | "
            f"Reason: {reason}"
        )

    def _portfolio_value(self, prices=None):
        """Calculate portfolio value using per-pair prices.
        
        Args:
            prices: dict of {pair: price} for accurate valuation,
                    or None to use entry prices (conservative).
        """
        val = self.state["capital"]
        for pair, pos in self.state["positions"].items():
            if prices and pair in prices:
                cp = prices[pair]
            else:
                cp = pos["entry_price"]
            if pos["direction"] == "LONG":
                val += (cp - pos["entry_price"]) * pos["size"]
            else:
                val += (pos["entry_price"] - cp) * pos["size"]
        return val


if __name__ == "__main__":
    trader = PaperTrader()
    print(f"=== HL Strategy Lab — Paper Trader ===")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Pairs: {', '.join(PAIR_STRATEGIES.keys())}")
    print(f"Strategies: {', '.join(v['strategy'] for v in PAIR_STRATEGIES.values())}")
    print(f"Risk: {RISK_PER_TRADE*100}% per trade | Max Leverage: {MAX_LEVERAGE}x")
    print(f"SL: {STOP_LOSS_ATR}x ATR | TP: {TAKE_PROFIT_ATR}x ATR")
    print()

    results = trader.run_cycle()

    print(f"--- Trading Cycle Complete ---")
    print(f"Portfolio Value: ${results['portfolio_value']:,.2f}")
    print(f"Cash: ${results['capital']:,.2f}")
    print(f"P&L: ${results['pnl']:+,.2f} ({results['pnl_pct']:+.2f}%)")
    print(f"Open Positions: {results['open_positions']}")
    print()

    if results["actions"]:
        print("--- Actions ---")
        for a in results["actions"]:
            print(f"  {a}")
        print()

    if results["positions"]:
        print("--- Open Positions ---")
        for pair, p in results["positions"].items():
            print(f"  {pair}: {p['direction']} ${p['entry_price']:.4f} → ${p['current_price']:.4f} "
                  f"| P&L: ${p['unrealized_pnl']:+.2f} ({p['unrealized_pnl_pct']:+.1f}%) "
                  f"| {p['strategy']}")
    else:
        print("--- No open positions ---")

    print()
    if results["stats"]["total_trades"] > 0:
        s = results["stats"]
        print(f"--- Trade Stats ---")
        print(f"  Total: {s['total_trades']} | W: {s['wins']} L: {s['losses']} | Win Rate: {s['win_rate']}%")
        print(f"  Realized P&L: ${s['total_realized_pnl']:+,.2f}")
        print(f"  Best: ${s['best_trade']:+.2f} | Worst: ${s['worst_trade']:+.2f}")

    print(f"\n📊 Dashboard data → {DASHBOARD_DATA}")
    print(f"💾 State → {STATE_FILE}")

    # Log cycle summary for audit trail
    log_cycle_summary(
        cycle_id=results.get("timestamp", ""),
        actions=results.get("actions", []),
        positions_count=results.get("open_positions", 0),
        portfolio_value=results.get("portfolio_value", 0),
        pnl=results.get("pnl", 0),
    )
    rotate_log_if_needed()
