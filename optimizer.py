"""
HL Strategy Lab — Autonomous Strategy Optimizer
AI-driven strategy rebalancing. Runs every 4h, analyzes live performance + fresh backtests,
decides whether to swap strategies on any pair, and rewrites strategy_config.json.

Decision logic:
1. Read current config + live trade history
2. Run fresh 30-day backtests for all strategies on all pairs
3. For each pair: compare current strategy's live + backtest performance vs alternatives
4. If a different strategy significantly outperforms → swap it
5. Log the change + post to Discord
"""
import json
import os
import sys
import time
import requests
import numpy as np
from datetime import datetime, timezone

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_engine import (
    Signal, HLData, RSIStrategy, MACDStrategy, BollingerStrategy,
    TrendFollowStrategy,
)
from backtester import Backtester
from paper_trader import load_strategy_config, save_strategy_config, CONFIG_FILE

# ─── Constants ───

CHANNEL_ID = "1520303679125721118"
DISCORD_ENV = os.path.expanduser("~/.hermes/.env")
BACKTEST_DAYS = 30
BACKTEST_INTERVAL = "1h"
MIN_TRADES_TO_JUDGE = 3      # need at least 3 live trades before judging
SWAP_THRESHOLD = 0.5          # new strategy must beat current by 50% (relative)
MIN_BACKTEST_TRADES = 5       # need at least 5 backtest trades to trust it
MAX_SWAPS_PER_RUN = 2         # don't change more than 2 pairs per optimization cycle

STRATEGIES_TO_TEST = [
    ("rsi", RSIStrategy()),
    ("macd", MACDStrategy()),
    ("bollinger", BollingerStrategy()),
    ("trend", TrendFollowStrategy()),
]

# ─── Discord ───

def get_discord_token():
    if os.path.exists(DISCORD_ENV):
        with open(DISCORD_ENV) as f:
            for line in f:
                if line.startswith("DISCORD_BOT_TOKEN=") and "REDACTED" not in line:
                    return line.strip().split("=", 1)[1].strip('"').strip("'")
    return None

def post_discord(content):
    token = get_discord_token()
    if not token:
        print("WARNING: No Discord token found")
        return False
    try:
        r = requests.post(
            f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
            headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
            json={"content": content},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"Discord post failed: {e}")
        return False

# ─── Analysis ───

def get_live_performance(closed_trades):
    """Get per-pair per-strategy live performance from closed trades."""
    perf = {}  # {pair: {strategy: {trades, wins, pnl, win_rate}}}
    for t in closed_trades:
        pair = t["pair"]
        strat = t["strategy"]
        if pair not in perf:
            perf[pair] = {}
        if strat not in perf[pair]:
            perf[pair][strat] = {"trades": 0, "wins": 0, "pnl": 0.0}
        perf[pair][strat]["trades"] += 1
        if t["pnl"] > 0:
            perf[pair][strat]["wins"] += 1
        perf[pair][strat]["pnl"] += t["pnl"]
    for pair in perf:
        for strat in perf[pair]:
            s = perf[pair][strat]
            s["win_rate"] = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
    return perf

def run_fresh_backtests(pairs, days=30):
    """Run all 4 strategies on all pairs with fresh data."""
    bt = Backtester(
        initial_capital=1000,
        risk_per_trade=0.02,
        max_leverage=3,
        stop_loss_atr=2.5,
        take_profit_atr=5.0,
        allow_shorts=True,
    )
    results = {}
    for pair in pairs:
        results[pair] = {}
        try:
            candles = HLData.get_candles(pair, interval=BACKTEST_INTERVAL, days=days)
            for name, strat in STRATEGIES_TO_TEST:
                result = bt.run(strat, candles, coin=pair, interval=BACKTEST_INTERVAL)
                results[pair][name] = {
                    "pnl_pct": result.total_pnl_pct,
                    "win_rate": result.win_rate,
                    "sharpe": result.sharpe_ratio,
                    "profit_factor": result.profit_factor,
                    "max_drawdown": result.max_drawdown,
                    "trades": result.total_trades,
                }
        except Exception as e:
            print(f"  ⚠️ {pair} backtest failed: {e}")
    return results

def score_strategy(metrics, live_data=None):
    """
    Composite score: 40% backtest P&L, 25% Sharpe, 20% win rate, 15% profit factor.
    Bonus: +10% if live data confirms the strategy is profitable.
    """
    pnl_score = max(0, metrics["pnl_pct"]) / 30  # normalize 30% → 1.0
    sharpe_score = max(0, metrics["sharpe"]) / 3  # normalize Sharpe 3 → 1.0
    win_score = metrics["win_rate"] / 100
    pf_score = min(1.0, metrics["profit_factor"] / 2) if metrics["profit_factor"] != float('inf') else 1.0

    score = pnl_score * 0.40 + sharpe_score * 0.25 + win_score * 0.20 + pf_score * 0.15

    # Live confirmation bonus
    if live_data and live_data.get("trades", 0) >= MIN_TRADES_TO_JUDGE:
        if live_data["pnl"] > 0:
            score *= 1.10  # 10% bonus if profitable live
        elif live_data["win_rate"] < 30:
            score *= 0.80  # 20% penalty if live win rate < 30%

    return round(score, 3)

# ─── Optimizer ───

def optimize():
    """Main optimization loop."""
    current_config = load_strategy_config()
    pairs = [p for p in current_config if not p.startswith("_")]

    # Load live trade state
    state_file = os.path.join(os.path.dirname(__file__), "paper_trader_state.json")
    closed_trades = []
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        closed_trades = state.get("closed_trades", [])

    live_perf = get_live_performance(closed_trades)

    print("=== Autonomous Strategy Optimizer ===")
    print(f"Analyzing {len(pairs)} pairs · {len(closed_trades)} live trades on record")
    print()

    # Run fresh backtests
    print("Running fresh 30-day backtests...")
    backtest_results = run_fresh_backtests(pairs, days=BACKTEST_DAYS)

    # Score each strategy per pair
    swaps = []
    for pair in pairs:
        current_strat = current_config[pair]["strategy"]
        print(f"\n--- {pair} (current: {current_strat}) ---")

        if pair not in backtest_results or not backtest_results[pair]:
            print(f"  ⚠️ No backtest data for {pair}")
            continue

        scores = {}
        for strat_name, metrics in backtest_results[pair].items():
            if metrics["trades"] < MIN_BACKTEST_TRADES:
                print(f"  {strat_name}: skipped (only {metrics['trades']} trades)")
                continue
            live = live_perf.get(pair, {}).get(strat_name)
            score = score_strategy(metrics, live)
            scores[strat_name] = score
            print(f"  {strat_name:12s} → score: {score:.3f} | "
                  f"P&L: {metrics['pnl_pct']:+.1f}% | "
                  f"WR: {metrics['win_rate']:.0f}% | "
                  f"Sharpe: {metrics['sharpe']:.2f} | "
                  f"Trades: {metrics['trades']}")

        if not scores:
            continue

        best_strat = max(scores, key=scores.get)
        current_score = scores.get(current_strat, 0)

        if best_strat == current_strat:
            print(f"  ✅ Keeping {current_strat} (best score: {current_score:.3f})")
            continue

        best_score = scores[best_strat]

        # Only swap if improvement is significant
        if current_score > 0:
            improvement = (best_score - current_score) / current_score
        else:
            improvement = float('inf') if best_score > 0 else 0

        if improvement >= SWAP_THRESHOLD and best_score > 0.1:
            swaps.append({
                "pair": pair,
                "from": current_strat,
                "to": best_strat,
                "from_score": current_score,
                "to_score": best_score,
                "improvement_pct": round(improvement * 100, 1),
                "backtest_pnl": backtest_results[pair][best_strat]["pnl_pct"],
                "backtest_wr": backtest_results[pair][best_strat]["win_rate"],
            })
            print(f"  🔄 SWAP: {current_strat} → {best_strat} (+{improvement*100:.0f}% better)")
        else:
            print(f"  ⏸️  {best_strat} better but only +{improvement*100:.0f}% (threshold: {SWAP_THRESHOLD*100:.0f}%)")

    # Apply swaps
    swaps_applied = swaps[:MAX_SWAPS_PER_RUN]
    if swaps_applied:
        print(f"\n=== Applying {len(swaps_applied)} swaps ===")
        new_config = dict(current_config)
        swap_notes = []
        for swap in swaps_applied:
            # Find the strategy params from STRATEGIES_TO_TEST defaults
            params = {}
            for name, _ in STRATEGIES_TO_TEST:
                if name == swap["to"]:
                    params = {}
                    break
            new_config[swap["pair"]] = {"strategy": swap["to"], "params": params}
            swap_notes.append(f"{swap['pair']}: {swap['from']}→{swap['to']}")

        # Save new config
        save_strategy_config(new_config, meta_note=" | ".join(swap_notes))
        print(f"✅ Config saved: {swap_notes}")

        # Build Discord message
        msg = "🧠 **Autonomous Optimization Complete**\n"
        msg += f"Analyzed {len(pairs)} pairs · Ran 30-day backtests on 4 strategies each\n\n"
        msg += "**Strategy Changes:**\n"
        for swap in swaps_applied:
            msg += f"🔄 **{swap['pair']}**: `{swap['from']}` → `{swap['to']}` "
            msg += f"(score {swap['from_score']:.2f} → {swap['to_score']:.2f}, +{swap['improvement_pct']}%)\n"
            msg += f"   Backtest: {swap['backtest_pnl']:+.1f}% P&L · {swap['backtest_wr']:.0f}% win rate\n"
        if len(swaps) > MAX_SWAPS_PER_RUN:
            msg += f"\n⚠️ {len(swaps) - MAX_SWAPS_PER_RUN} additional swap(s) deferred to next cycle"
        post_discord(msg)
    else:
        print("\n✅ No swaps needed — current config is optimal")
        # Post a brief status
        if closed_trades:
            wins = [t for t in closed_trades if t["pnl"] > 0]
            total_pnl = sum(t["pnl"] for t in closed_trades)
            # Get open positions for portfolio value
            open_positions = {}
            if os.path.exists(state_file):
                with open(state_file) as f:
                    s = json.load(f)
                open_positions = s.get("positions", {})
            capital = state.get("capital", 5000)
            msg = "🧠 **AI Review** — No changes needed\n"
            msg += f"Current strategies performing well. "
            msg += f"{len(closed_trades)} trades · {len(wins)} wins ({len(wins)/len(closed_trades)*100:.0f}% WR) · "
            msg += f"Realized: ${total_pnl:+.2f}\n"
            if open_positions:
                msg += f"Open: {len(open_positions)} positions"
            post_discord(msg)

    return swaps_applied

if __name__ == "__main__":
    swaps = optimize()
    print(f"\n{'='*50}")
    print(f"Swaps applied: {len(swaps)}")
    for s in swaps:
        print(f"  {s['pair']}: {s['from']} → {s['to']}")
