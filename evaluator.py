"""
HL Strategy Lab — Evaluation Engine
Scores every strategy-pair combination via backtests, ranks them on a composite
metric, produces a leaderboard, and auto-assigns the best strategy per pair.

Runnable standalone:  python3 evaluator.py [--dry-run]
Importable:           from evaluator import run_evaluation
"""
import argparse
import json
from datetime import datetime, timezone
from dataclasses import asdict

import numpy as np

from strategy_engine import (
    HLData, RSIStrategy, MACDStrategy, BollingerStrategy, TrendFollowStrategy,
    StochasticStrategy, VWAPStrategy, SupertrendStrategy, BreakoutStrategy,
    EMACrossStrategy, STRATEGIES,
)
from backtester import run_backtest_matrix

PROJECT_DIR = "/Users/mojoai/Projects/hl-strategy-lab"
LEADERBOARD_PATH = f"{PROJECT_DIR}/eval_leaderboard.json"
CONFIG_PATH = f"{PROJECT_DIR}/strategy_config.json"
PAIRS = ["BTC", "ETH", "SOL", "HYPE"]
DAYS = 90

# backtester friendly name → STRATEGIES dict key (config lowercase key)
NAME_TO_KEY = {
    "RSI": "rsi", "MACD": "macd", "Bollinger": "bollinger", "Trend": "trend",
    "Stochastic": "stochastic", "VWAP": "vwap", "Supertrend": "supertrend",
    "Breakout": "breakout", "EMACross": "emacross",
}

# Weights summing to 1.0: sharpe 40 / pf 25 / winrate 20 / drawdown 15
W_SHARPE, W_PF, W_WIN, W_DD = 0.40, 0.25, 0.20, 0.15


# ─── Scoring ──────────────────────────────────────────────────────────────

# P2: Net PnL weight — penalize fee drag heavily
NET_PNL_PENALTY = 0.3  # if total PnL after fees < 0, multiply score by this

def composite_score(result) -> float:
    """Composite 0-100 score for a BacktestResult.
    
    P2 Fix: Now includes net-PnL penalty (fees included) as primary gate.
    Previously optimized on gross metrics, ignoring that fees exceeded PnL.
    """
    sharpe = result.sharpe_ratio
    pf = result.profit_factor if np.isfinite(result.profit_factor) else 5.0
    win = result.win_rate
    dd = result.max_drawdown

    # Normalize each component to 0-1
    sharpe_norm = _clamp((sharpe + 1) / 4.0, 0, 1)       # -1..3 → 0..1
    pf_norm = _clamp((pf - 0.5) / 2.0, 0, 1)              # 0.5..2.5 → 0..1
    win_norm = _clamp(win / 60.0, 0, 1)                   # 0..60% → 0..1
    dd_penalty = _clamp(dd / 40.0, 0, 1)                  # 0..40% DD → 0..1

    score = (sharpe_norm * W_SHARPE + pf_norm * W_PF +
             win_norm * W_WIN - dd_penalty * W_DD)
    score = score * 100  # 0-100 scale

    # P2: Loser penalty — strategies that lose money get crushed
    if result.total_pnl_pct < 0:
        score *= NET_PNL_PENALTY

    # Edge over buy-hold: if it doesn't beat just holding, discount it
    if result.total_pnl_pct < result.buy_hold_return:
        score *= 0.8

    return round(_clamp(score, 0, 100), 2)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ─── Leaderboard builder ──────────────────────────────────────────────────

def build_leaderboard(results) -> list[dict]:
    """Flatten backtest matrix into a list of scored entries sorted desc."""
    entries = []
    for pair, strats in results.items():
        for friendly, res in strats.items():
            key = NAME_TO_KEY.get(friendly, friendly.lower())
            entries.append({
                "pair": pair,
                "strategy": key,
                "strategy_label": friendly,
                "score": composite_score(res),
                "total_pnl_pct": res.total_pnl_pct,
                "buy_hold_pct": res.buy_hold_return,
                "beats_hold": res.total_pnl_pct > res.buy_hold_return,
                "sharpe": res.sharpe_ratio,
                "profit_factor": res.profit_factor if np.isfinite(res.profit_factor) else None,
                "win_rate": res.win_rate,
                "max_drawdown": res.max_drawdown,
                "total_trades": res.total_trades,
            })
    entries.sort(key=lambda e: e["score"], reverse=True)
    return entries


def best_per_pair(leaderboard: list[dict]) -> dict[str, dict]:
    """Pick the top-scoring strategy for each pair + a confidence rating."""
    best = {}
    for pair in PAIRS:
        ranked = [e for e in leaderboard if e["pair"] == pair]
        if not ranked:
            continue
        winner = ranked[0]
        field = ranked[1:]
        margin = winner["score"] - (field[0]["score"] if field else 0)
        # confidence: how dominant the winner is vs the rest
        if not field:
            confidence = "LOW"
        elif margin >= 25:
            confidence = "HIGH"
        elif margin >= 10:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        best[pair] = {**winner, "margin": round(margin, 2), "confidence": confidence}
    return best


# ─── Config writer ────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def assign_strategies(assignments: dict[str, dict], dry_run=False) -> list[str]:
    """Write best strategy per pair to strategy_config.json (preserves _meta).
    
    Hysteresis: the incumbent strategy is NOT replaced unless a challenger
    beats it by >= HYSTERESIS_MARGIN points. This prevents flip-flopping
    when two strategies score within a few points of each other.
    
    P4: Promotion Gate — additional conditions before any strategy swap:
    - Champion freeze: incumbent must have held for >= CHAMPION_FREEZE_DAYS
    - Minimum trade evidence: challenger must have >= MIN_PROMOTION_TRADES
    Returns list of human-readable change notes."""
    HYSTERESIS_MARGIN = 5.0  # challenger must beat incumbent by this many points
    # P4: Promotion gate constants
    CHAMPION_FREEZE_DAYS = 14    # incumbent is frozen for this many days
    MIN_PROMOTION_TRADES = 30    # challenger needs this many backtest trades
    
    config = load_config()
    meta = config.get("_meta", {})
    history = meta.get("history", [])
    changes = []
    new_config = {}

    for pair in PAIRS:
        winner = assignments.get(pair)
        if not winner:
            new_config[pair] = config.get(pair, {"strategy": "trend", "params": {}})
            continue
        old = config.get(pair, {})
        old_strat = old.get("strategy")
        new_strat = winner["strategy"]

        # ── Hysteresis logic ──
        if old_strat and old_strat != new_strat:
            # Incumbent is being challenged. Find incumbent's score in leaderboard.
            # assignments[pair] is the winner; we need the full ranked list.
            # winner["score"] is the challenger's score. But we need incumbent's score.
            # We stored margin = winner_score - runner_up_score.
            # If old_strat is the runner-up, margin tells us the gap.
            # But we don't have the full leaderboard here — we have best_per_pair.
            # Solution: use margin directly. If margin < HYSTERESIS_MARGIN, keep incumbent.
            margin = winner.get("margin", 0)
            if margin < HYSTERESIS_MARGIN:
                # Challenger wins by too little — keep incumbent
                new_config[pair] = {"strategy": old_strat, "params": {}}
                changes.append(
                    f"{pair}: keep {old_strat} (challenger {new_strat} "
                    f"only +{margin:.1f} ahead — hysteresis holds, need +{HYSTERESIS_MARGIN:.0f})"
                )
                continue

            # P4: Check champion freeze period
            last_change_ts = None
            for h_entry in reversed(history):
                if pair in h_entry.get("note", "") and "→" in h_entry.get("note", ""):
                    last_change_ts = h_entry.get("timestamp", "")
                    break
            if last_change_ts:
                try:
                    change_dt = datetime.fromisoformat(last_change_ts.replace("Z", "+00:00"))
                    days_held = (datetime.now(timezone.utc) - change_dt).days
                    if days_held < CHAMPION_FREEZE_DAYS:
                        new_config[pair] = {"strategy": old_strat, "params": {}}
                        changes.append(
                            f"{pair}: keep {old_strat} (champion frozen — {days_held}d of {CHAMPION_FREEZE_DAYS}d minimum)"
                        )
                        continue
                except Exception:
                    pass  # if timestamp parsing fails, don't block the swap

            # P4: Check minimum trade evidence
            if winner.get("total_trades", 999) < MIN_PROMOTION_TRADES:
                new_config[pair] = {"strategy": old_strat, "params": {}}
                changes.append(
                    f"{pair}: keep {old_strat} (challenger {new_strat} has only "
                    f"{winner.get('total_trades', 0)} trades — need {MIN_PROMOTION_TRADES})"
                )
                continue

        # Either no incumbent, incumbent won outright, or challenger beat margin
        # Check if we're keeping the same strategy
        effective_strat = new_config.get(pair, {}).get("strategy", new_strat)
        if pair not in new_config:
            new_config[pair] = {"strategy": new_strat, "params": {}}

        if old_strat and old_strat != effective_strat:
            changes.append(f"{pair}: {old_strat}→{effective_strat} "
                           f"(score {winner['score']:.1f}, {winner['confidence']})")
        elif old_strat == effective_strat:
            changes.append(f"{pair}: keep {effective_strat} (score {winner['score']:.1f})")
        elif not old_strat:
            changes.append(f"{pair}: →{effective_strat} (score {winner['score']:.1f}, new)")

    if dry_run:
        return changes

    if any("→" in c for c in changes):
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": " | ".join(changes),
        })
        meta["history"] = history
        meta["last_optimized"] = datetime.now(timezone.utc).isoformat()
        meta["optimization_count"] = meta.get("optimization_count", 0) + 1

    new_config["_meta"] = meta
    with open(CONFIG_PATH, "w") as f:
        json.dump(new_config, f, indent=2)
    return changes


def save_leaderboard(leaderboard: list[dict], assignments: dict):
    # Sanitize numpy types for JSON serialization
    def _sanitize(val):
        if hasattr(val, 'item'):  # numpy scalar
            return val.item()
        if isinstance(val, float) and not np.isfinite(val):
            return None
        return val

    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "weights": {"sharpe": W_SHARPE, "profit_factor": W_PF,
                    "win_rate": W_WIN, "drawdown": W_DD},
        "leaderboard": [{k: _sanitize(v) for k, v in e.items()} for e in leaderboard],
        "best_per_pair": {k: {kk: _sanitize(vv) for kk, vv in v.items()}
                          for k, v in assignments.items()},
    }
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(out, f, indent=2)


# ─── Formatting ───────────────────────────────────────────────────────────

def print_leaderboard(leaderboard: list[dict], assignments: dict[str, dict]):
    print("\n" + "=" * 92)
    print(f"{'RANK':<5}{'PAIR':<6}{'STRATEGY':<13}{'SCORE':>7}{'PnL%':>9}{'HOLD%':>9}"
          f"{'SHARPE':>8}{'PF':>7}{'WIN%':>7}{'MAXDD':>8}{'CONF':>6}")
    print("-" * 92)
    for i, e in enumerate(leaderboard, 1):
        is_best = e["pair"] in assignments and assignments[e["pair"]]["strategy"] == e["strategy"]
        marker = " ★" if is_best else "  "
        pf = e['profit_factor'] if e['profit_factor'] is not None else float('inf')
        print(f"{i:<5}{e['pair']:<6}{e['strategy']:<13}{e['score']:>7.1f}"
              f"{e['total_pnl_pct']:>+9.2f}{e['buy_hold_pct']:>+9.2f}"
              f"{e['sharpe']:>8.2f}{pf:>7.2f}{e['win_rate']:>7.1f}"
              f"{e['max_drawdown']:>8.2f}{marker}")
    print("=" * 92)
    print("  ★ = best for pair\n")


def print_assignments(assignments: dict[str, dict], changes: list[str], dry_run=False):
    mode = "DRY RUN — no changes written" if dry_run else "strategy_config.json updated"
    print(f"{'─' * 60}\n  AUTO-ASSIGNMENT ({mode})\n{'─' * 60}")
    for pair in PAIRS:
        a = assignments.get(pair)
        if not a:
            continue
        print(f"  {pair:<5} → {a['strategy']:<12} score {a['score']:>5.1f}  "
              f"margin +{a['margin']:>5.1f}  [{a['confidence']}]")
    print(f"{'─' * 60}")
    if not any("→" in c for c in changes):
        print("  No strategy changes this round (all assignments held).\n")
    else:
        print("  Changes:")
        for c in changes:
            if "→" in c:
                print(f"    • {c}")
        print()


# ─── Main entry ───────────────────────────────────────────────────────────

def run_evaluation(dry_run=False, verbose=True) -> dict:
    """Run full evaluation: backtest → score → assign. Returns summary dict."""
    if verbose:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  HL STRATEGY LAB — EVALUATION ENGINE v1.0               ║")
        print(f"║  4 Pairs × {len(STRATEGIES)} Strategies × {DAYS}d 1h data  {'(DRY RUN)' if dry_run else '(LIVE)':>14} ║")
        print("╚══════════════════════════════════════════════════════════╝\n")

    results = run_backtest_matrix(
        pairs=PAIRS, interval="1h", days=DAYS,
        initial_capital=1000, risk_per_trade=0.02, max_leverage=3,
        stop_loss_atr=2.5, take_profit_atr=5.0, allow_shorts=True,
    )

    leaderboard = build_leaderboard(results)
    assignments = best_per_pair(leaderboard)

    save_leaderboard(leaderboard, assignments)
    changes = assign_strategies(assignments, dry_run=dry_run)

    if verbose:
        print_leaderboard(leaderboard, assignments)
        print_assignments(assignments, changes, dry_run)

    return {"leaderboard": leaderboard, "assignments": assignments, "changes": changes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HL Strategy Lab — Evaluation Engine")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be assigned without writing changes")
    args = ap.parse_args()
    run_evaluation(dry_run=args.dry_run)
