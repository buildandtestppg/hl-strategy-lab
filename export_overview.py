#!/usr/bin/env python3
"""
Dashboard Data Exporter — combines all fleet data into a single JSON.
This is the single data source for the overview dashboard.

Combines:
  - Paper trading state (portfolio, positions, trades, equity curve)
  - Signal synthesis (convergence, narratives, source breakdown)
  - Strategy assignments + performance
  - Fleet scanner feed (latest X/whale/news signals)
  - Harness evaluation results
  - LLM advisor report
  - Eval leaderboard
"""
import json
import os
import glob
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.absolute()
SGT = timezone(timedelta(hours=8))

def load_json(name):
    try:
        with open(PROJECT_DIR / name) as f:
            return json.load(f)
    except Exception:
        return None

def get_cron_output(job_id, max_age_hours=6):
    """Get latest cron output for a job."""
    cron_dir = Path.home() / ".hermes" / "cron" / "output" / job_id
    if not cron_dir.is_dir():
        return None
    files = sorted(glob.glob(str(cron_dir / "*.md")), reverse=True)
    if not files:
        return None
    mtime = os.path.getmtime(files[0])
    age_h = (datetime.now().timestamp() - mtime) / 3600
    if age_h > max_age_hours:
        return None
    try:
        with open(files[0]) as f:
            return f.read()[:5000]  # cap for dashboard payload
    except Exception:
        return None

def export():
    state = load_json("paper_trader_state.json") or {}
    dashboard = load_json("dashboard_data.json") or {}
    sentiment = load_json("sentiment_scores.json") or {}
    fleet_feed = load_json("fleet_feed.json") or {}
    harness = load_json("harness_data.json") or {}
    leaderboard = load_json("eval_leaderboard.json") or {}
    advisor = load_json("llm_advisor_report.json") or {}
    config = load_json("strategy_config.json") or {}

    # Cron scanner freshness
    SCANNER_JOBS = {
        "x_scanner": ("43ba1d393c5a", 2),
        "whale_monitor": ("66a98cf277aa", 1),
        "news_agent": ("8bd6a73fc440", 3),
        "polymarket": ("3876dd862ca1", 6),
    }
    scanner_status = {}
    for name, (job_id, max_age) in SCANNER_JOBS.items():
        cron_dir = Path.home() / ".hermes" / "cron" / "output" / job_id
        files = sorted(glob.glob(str(cron_dir / "*.md")), reverse=True) if cron_dir.is_dir() else []
        if files:
            mtime = os.path.getmtime(files[0])
            age_min = (datetime.now().timestamp() - mtime) / 60
            scanner_status[name] = {
                "last_run": datetime.fromtimestamp(mtime, SGT).strftime("%H:%M"),
                "age_minutes": round(age_min),
                "status": "fresh" if age_min < max_age * 60 else "stale",
            }
        else:
            scanner_status[name] = {"last_run": "never", "age_minutes": 9999, "status": "dead"}

    # Synthesizer cron status
    synth_cron_dir = Path.home() / ".hermes" / "cron" / "output" / "ee9ebdc2a789"
    synth_files = sorted(glob.glob(str(synth_cron_dir / "*.md")), reverse=True) if synth_cron_dir.is_dir() else []
    if synth_files:
        synth_mtime = os.path.getmtime(synth_files[0])
        synth_age = (datetime.now().timestamp() - synth_mtime) / 60
        scanner_status["synthesizer"] = {
            "last_run": datetime.fromtimestamp(synth_mtime, SGT).strftime("%H:%M"),
            "age_minutes": round(synth_age),
            "status": "fresh" if synth_age < 20 else "stale",
        }
    else:
        scanner_status["synthesizer"] = {"last_run": "pending", "age_minutes": 9999, "status": "waiting"}

    # Build unified export
    export_data = {
        "exported_at": datetime.now(SGT).isoformat(),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),

        # Trading
        "portfolio": {
            "value": state.get("capital", 0),
            "initial": state.get("initial_capital", 5000),
            "pnl": state.get("capital", 0) - state.get("initial_capital", 5000),
            "pnl_pct": ((state.get("capital", 0) - state.get("initial_capital", 5000)) / state.get("initial_capital", 5000)) * 100,
            "peak": state.get("peak_equity", state.get("capital", 0)),
            "open_positions": len(state.get("positions", {})),
            "max_positions": 4,
        },
        "positions": state.get("positions", {}),
        "closed_trades": state.get("closed_trades", [])[-20:],
        "equity_curve": state.get("equity_curve", [])[-200:],
        "reset_count": state.get("reset_count", 0),

        # Signal Synthesis
        "sentiment": sentiment,

        # Strategy
        "pair_strategies": {k: v for k, v in config.items() if not k.startswith("_")},
        "strategy_meta": config.get("_meta", {}),

        # Fleet
        "fleet_feed": fleet_feed,
        "scanner_status": scanner_status,

        # Harness
        "harness": {
            "leaderboard": harness.get("leaderboard", [])[:10] if harness else [],
            "tests": harness.get("tests", {}) if harness else {},
            "audit_recent": harness.get("audit", {}).get("recent", [])[:5] if harness else [],
            "advisor_regime": advisor.get("regime_analysis", {}) if advisor else {},
            "advisor_risk": advisor.get("risk_assessment", {}) if advisor else {},
        },

        # Eval
        "eval_leaderboard": leaderboard.get("leaderboard", [])[:10] if leaderboard else [],

        # Stats from dashboard_data
        "stats": dashboard.get("stats", {}),
        "pair_stats": dashboard.get("pair_stats", {}),
        "prices": dashboard.get("prices", {}),
        "actions": dashboard.get("actions", []),
        "ai_history": dashboard.get("ai_history", [])[-10:],

        # Config
        "config": dashboard.get("config", {}),

        # Trading Lessons (self-review loop)
        "lessons": load_json(PROJECT_DIR / "trading_lessons.json") or {"pairs": {}, "active_lessons": []},
        "trade_review_output": load_json(PROJECT_DIR / "trade_review_output.json") or {},
    }

    # Write
    out_path = PROJECT_DIR / "overview_data.json"
    with open(out_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"✅ Exported to {out_path} ({size_kb:.1f} KB)")
    print(f"   Portfolio: ${export_data['portfolio']['value']:.0f} ({export_data['portfolio']['pnl_pct']:+.1f}%)")
    print(f"   Positions: {export_data['portfolio']['open_positions']}/{export_data['portfolio']['max_positions']}")
    print(f"   Sentiment pairs: {len(sentiment)}")
    print(f"   Scanner status: {sum(1 for s in scanner_status.values() if s['status']=='fresh')}/{len(scanner_status)} fresh")

    return export_data

if __name__ == "__main__":
    export()
    # Auto-deploy to GitHub Pages
    import subprocess
    try:
        subprocess.run(["git", "add", "overview_data.json"], cwd=str(PROJECT_DIR),
                       capture_output=True, timeout=10)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"],
                               cwd=str(PROJECT_DIR), capture_output=True, timeout=10)
        if result.returncode != 0:
            subprocess.run(["git", "commit", "-m",
                           f"auto: overview dashboard {datetime.now().strftime('%H:%M UTC')}"],
                           cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=10)
            subprocess.run(["git", "push", "origin", "gh-pages"],
                           cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=30)
            print("✅ Pushed to GitHub Pages")
    except Exception as e:
        print(f"⚠️ Git push failed: {e}")
