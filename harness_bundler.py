"""
HL Strategy Lab — Harness Data Bundler
Combines leaderboard, audit stats, and test results into a single JSON
that the dashboard's Harness tab consumes.

Run after: evaluator.py, test_suite.py, or any paper_trader.py cycle.
"""
import json
import os
from datetime import datetime, timezone

BASE = os.path.dirname(__file__)


def bundle():
    data = {"exported_at": datetime.now(timezone.utc).isoformat()}

    # 1. Leaderboard from evaluator
    lb_path = os.path.join(BASE, "eval_leaderboard.json")
    if os.path.exists(lb_path):
        with open(lb_path) as f:
            lb_raw = json.load(f)
        # Format: {"leaderboard": [...], "best_per_pair": {...}}
        lb = lb_raw.get("leaderboard", [])
        leaderboard = []
        for r in lb:
            leaderboard.append({
                "pair": r.get("pair", ""),
                "strategy": r.get("strategy", ""),
                "score": r.get("score", 0),
                "win_rate": r.get("win_rate", 0),
                "sharpe": r.get("sharpe", 0),
                "pnl_pct": r.get("total_pnl_pct", 0),
                "buy_hold": r.get("buy_hold_pct", 0),
                "profit_factor": r.get("profit_factor", 0) or 0,
                "max_drawdown": r.get("max_drawdown", 0),
                "total_trades": r.get("total_trades", 0),
            })
        leaderboard.sort(key=lambda x: x.get("score", 0), reverse=True)
        data["leaderboard"] = leaderboard
    else:
        data["leaderboard"] = []

    # 2. Audit trail (last 50 decisions + stats)
    try:
        from audit_logger import get_recent_decisions, get_decision_stats
        data["audit"] = {
            "recent": get_recent_decisions(limit=50),
            "stats": get_decision_stats(lookback_hours=24),
        }
    except Exception:
        data["audit"] = {"recent": [], "stats": {}}

    # 3. Test results — parse from last test_suite run
    test_log = os.path.join(BASE, "last_test_results.json")
    if os.path.exists(test_log):
        with open(test_log) as f:
            data["tests"] = json.load(f)
    else:
        data["tests"] = {"pass": 0, "fail": 0, "note": "run test_suite.py --save"}

    # 4. Config meta (last evaluator run)
    config_path = os.path.join(BASE, "strategy_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        data["config"] = {k: v for k, v in cfg.items() if k.startswith("_")}
        data["assignments"] = {k: v.get("strategy", "?") for k, v in cfg.items()
                               if not k.startswith("_")}
    else:
        data["config"] = {}
        data["assignments"] = {}

    return data


if __name__ == "__main__":
    output = bundle()
    out_path = os.path.join(BASE, "harness_data.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"✅ Harness data → {out_path}")
    lb = output.get("leaderboard", [])
    if lb:
        print(f"   Leaderboard: {len(lb)} entries")
        top3 = lb[:3]
        for r in top3:
            print(f"   #{lb.index(r)+1} {r['pair']}/{r['strategy']} — score {r['score']:.1f}")
    audits = output.get("audit", {}).get("stats", {})
    print(f"   Audit entries (24h): {audits.get('total_evaluated', 0)}")
