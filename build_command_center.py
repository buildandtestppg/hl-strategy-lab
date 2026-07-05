#!/usr/bin/env python3
"""
Fleet Command Center — Data Builder
Combines ALL system data into one overview for the command center dashboard.
This is the CEO view: system health, trading, fleet, learning loop.
"""
import json, os, glob, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT = Path("/Users/mojoai/Projects/hl-strategy-lab")
HOME = Path.home()
SGT = timezone(timedelta(hours=8))

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}

def get_cron_health():
    """Get health status of all cron jobs."""
    try:
        result = subprocess.run(
            ["hermes", "cron", "list"], capture_output=True, text=True, timeout=15
        )
        output = result.stdout
        crons = []
        current = {}
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('Name:'):
                if current:
                    crons.append(current)
                current = {'name': line.split('Name:',1)[1].strip()}
            elif line.startswith('Schedule:') and current:
                current['schedule'] = line.split('Schedule:',1)[1].strip()
            elif line.startswith('Last run:') and current:
                last_raw = line.split('Last run:',1)[1].strip()
                current['last_run'] = last_raw
                current['status'] = 'ok' if ' ok' in last_raw else 'error'
            elif line.startswith('Deliver:') and current:
                current['deliver'] = line.split('Deliver:',1)[1].strip()
            elif line.startswith('Script:') and current:
                current['script'] = line.split('Script:',1)[1].strip()
        if current:
            crons.append(current)
        
        # Compute health summary
        total = len(crons)
        ok = sum(1 for c in crons if c.get('status') == 'ok')
        return {'jobs': crons[-40:], 'total': total, 'healthy': ok, 'issues': total - ok}
    except Exception as e:
        return {'jobs': [], 'total': 0, 'healthy': 0, 'issues': 0, 'error': str(e)}

def get_disk_health():
    """Get disk usage of key directories."""
    dirs = {
        'scripts': HOME / '.hermes' / 'scripts',
        'cron_output': HOME / '.hermes' / 'cron' / 'output',
        'hl_lab': PROJECT,
    }
    result = {}
    for name, path in dirs.items():
        if path.exists():
            total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            result[name] = round(total / (1024*1024), 1)
    return result

def build_command_data():
    """Build the unified command center data."""
    state = load_json(PROJECT / "paper_trader_state.json")
    dashboard = load_json(PROJECT / "dashboard_data.json")
    overview = load_json(PROJECT / "overview_data.json")
    sentiment = load_json(PROJECT / "sentiment_scores.json")
    fleet_feed = load_json(PROJECT / "fleet_feed.json")
    harness = load_json(PROJECT / "harness_data.json")
    leaderboard = load_json(PROJECT / "eval_leaderboard.json")
    lessons = load_json(PROJECT / "trading_lessons.json")
    pruned = load_json(PROJECT / "pruned_pairs.json")

    # Portfolio
    capital = state.get("capital", 0)
    initial = state.get("initial_capital", 5000)
    pnl = capital - initial
    pnl_pct = (pnl / initial * 100) if initial else 0
    peak = state.get("peak_equity", capital)
    drawdown = ((peak - capital) / peak * 100) if peak else 0

    closed_trades = state.get("closed_trades", [])
    wins = [t for t in closed_trades if t.get("pnl", 0) > 0]
    losses = [t for t in closed_trades if t.get("pnl", 0) <= 0]
    total_fees = sum(t.get("fees_paid", 0) + t.get("funding_cost", 0) for t in closed_trades)
    gross_pnl = sum(t.get("pnl", 0) for t in closed_trades)
    net_pnl = gross_pnl - total_fees

    # Exit breakdown
    exit_breakdown = dashboard.get("exit_breakdown", {})
    
    # Cron health
    cron_health = get_cron_health()
    disk = get_disk_health()
    
    # Scanner status from overview
    scanner_status = overview.get("scanner_status", {})
    
    # Fleet feed signals count
    fleet_signals = 0
    if fleet_feed:
        for key in ['x_scanner', 'whale_monitor', 'news_agent']:
            signals = fleet_feed.get(key, [])
            fleet_signals += len(signals) if isinstance(signals, list) else 0

    # Positions
    positions = state.get("positions", {})
    
    # Sentiment summary
    sent_scores = {k: v.get("score", 0) for k, v in sentiment.items()} if sentiment else {}
    avg_sentiment = sum(sent_scores.values()) / len(sent_scores) if sent_scores else 0

    # Build the command data
    data = {
        "generated_at": datetime.now(SGT).strftime("%Y-%m-%d %H:%M:%S SGT"),
        "generated_ts": datetime.now(SGT).isoformat(),
        
        # === COMMAND OVERVIEW ===
        "command": {
            "portfolio_value": round(capital, 2),
            "portfolio_initial": initial,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "peak_equity": round(peak, 2),
            "drawdown_pct": round(drawdown, 2),
            "open_positions": len(positions),
            "max_positions": 4,
            "total_trades": len(closed_trades),
            "win_rate": round(len(wins) / max(len(closed_trades), 1) * 100, 1),
            "wins": len(wins),
            "losses": len(losses),
            "gross_pnl": round(gross_pnl, 2),
            "total_fees": round(total_fees, 2),
            "net_pnl": round(net_pnl, 2),
            "avg_sentiment": round(avg_sentiment, 2),
            "fleet_signals": fleet_signals,
            "scanners_healthy": sum(1 for s in scanner_status.values() if s.get("status") == "fresh"),
            "scanners_total": len(scanner_status),
            "crons_healthy": cron_health.get("healthy", 0),
            "crons_total": cron_health.get("total", 0),
            "scripts_count": len(list((HOME / ".hermes" / "scripts").glob("*.py"))),
            "disk_mb": disk,
            "pruned_pairs": list(pruned.keys()),
            "optimization_count": (load_json(PROJECT / "strategy_config.json").get("_meta", {})).get("optimization_count", 0),
        },
        
        # === TRADING DETAIL ===
        "trading": {
            "positions": positions,
            "prices": dashboard.get("prices", {}),
            "pair_strategies": {k: v for k, v in load_json(PROJECT / "strategy_config.json").items() if not k.startswith("_")},
            "closed_trades": closed_trades[-20:],
            "equity_curve": (state.get("equity_curve") or [])[-200:],
            "pair_stats": dashboard.get("pair_stats", {}),
            "strategy_stats": dashboard.get("strategy_stats", {}),
            "stats": dashboard.get("stats", {}),
            "exit_breakdown": exit_breakdown,
            "sentiment": sentiment,
            "config": dashboard.get("config", {}),
            "ai_history": (load_json(PROJECT / "strategy_config.json").get("_meta", {})).get("history", [])[-10:],
        },
        
        # === FLEET DETAIL ===
        "fleet": {
            "scanner_status": scanner_status,
            "cron_jobs": cron_health.get("jobs", []),
            "cron_summary": {"total": cron_health.get("total", 0), "healthy": cron_health.get("healthy", 0), "issues": cron_health.get("issues", 0)},
            "fleet_feed": fleet_feed,
            "signals_count": fleet_signals,
        },
        
        # === SYSTEM HEALTH ===
        "system": {
            "disk": disk,
            "scripts_count": len(list((HOME / ".hermes" / "scripts").glob("*.py"))),
            "pruned_pairs": pruned,
            "lessons": lessons,
            "loop_v2": dashboard.get("loop_v2", {}),
            "eval_leaderboard": leaderboard.get("leaderboard", [])[:10] if leaderboard else [],
            "harness": {
                "leaderboard": harness.get("leaderboard", [])[:10] if harness else [],
                "tests": harness.get("tests", {}) if harness else {},
                "audit_recent": harness.get("audit", {}).get("recent", [])[:8] if harness else [],
            },
        },
    }
    
    return data

if __name__ == "__main__":
    data = build_command_data()
    out_path = PROJECT / "command_center.json"
    tmp = PROJECT / "command_center.json.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(out_path)
    
    c = data["command"]
    print(f"✅ Command Center data built: {out_path.name} ({os.path.getsize(out_path)/1024:.1f}KB)")
    print(f"   Portfolio: ${c['portfolio_value']:.0f} ({c['pnl_pct']:+.1f}%)")
    print(f"   Trading: {c['total_trades']} trades, {c['win_rate']}% WR, ${c['net_pnl']:+.2f} net")
    print(f"   Fleet: {c['scanners_healthy']}/{c['scanners_total']} scanners, {c['crons_healthy']}/{c['crons_total']} crons")
    print(f"   System: {c['scripts_count']} scripts, {sum(c['disk_mb'].values()):.1f}MB total")
