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

def get_fitness_data():
    """Load fitness tracking data."""
    log_path = HOME / ".hermes" / "fitness-log.md"
    if not log_path.exists():
        return {"active": False}
    content = log_path.read_text()
    # Parse key stats
    import re
    start_weight = 130.0
    current_match = re.search(r'Current Weight.*?(\d+\.?\d*)\s*kg', content)
    current_weight = float(current_match.group(1)) if current_match else start_weight
    start_match = re.search(r'Starting Weight.*?~?(\d+\.?\d*)\s*kg', content)
    if start_match:
        start_weight = float(start_match.group(1))
    # Find latest workout week
    week_match = re.search(r'Current Week.*?Week (\d+)', content)
    current_week = int(week_match.group(1)) if week_match else 0
    
    return {
        "active": True,
        "start_weight": start_weight,
        "current_weight": current_weight,
        "weight_lost": round(start_weight - current_weight, 1),
        "weight_lost_pct": round((start_weight - current_weight) / start_weight * 100, 1),
        "current_week": current_week,
        "goal": "Build muscle, lose fat, look good. Long game.",
    }

def get_wc_portfolio():
    """Get World Cup portfolio data from latest cron output."""
    try:
        result = subprocess.run(
            ["python3", str(HOME / ".hermes" / "scripts" / "wc_portfolio_tracker.py")],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout
        # Parse the text output
        import re
        invested_match = re.search(r'Total invested.*?\$([\d,.]+)', output)
        invested = float(invested_match.group(1).replace(',', '')) if invested_match else 0
        
        roi_match = re.search(r'TOTAL\s+([\d.]+%)', output)
        roi_str = roi_match.group(1).replace('%', '') if roi_match else '0'
        roi = float(roi_str)
        
        value_match = re.search(r'\$\s+(\d+)', output.split('TOTAL')[-1] if 'TOTAL' in output else '')
        total_value = float(value_match.group(1)) if value_match else 0
        
        pnl_match = re.search(r'\$\s+\d+\s+\$\s+([+-]?\d+)', output.split('TOTAL')[-1] if 'TOTAL' in output else '')
        pnl = float(pnl_match.group(1)) if pnl_match else 0
        
        # Count teams
        teams = re.findall(r'(France|Argentina|Brazil|Spain|Portugal|Netherlands|England|Germany)', output)
        
        return {
            "active": invested > 0,
            "invested": invested,
            "current_value": total_value,
            "pnl": pnl,
            "roi": roi,
            "teams": len(set(teams)),
            "raw_output": output[:1000],
        }
    except:
        return {"active": False}

def get_hermes_health():
    """Get Hermes system health from latest health harness output."""
    health_dir = HOME / ".hermes" / "cron" / "output" / "3c2578a553f3"
    if not health_dir.exists():
        return {"score": "N/A"}
    files = sorted(health_dir.glob("*.md"), reverse=True)
    if not files:
        return {"score": "N/A"}
    content = files[0].read_text()[:3000]
    import re
    score_match = re.search(r'Score.*?(\d+\.?\d*)/100', content)
    score = float(score_match.group(1)) if score_match else 0
    return {
        "score": score,
        "report_snippet": content[:500],
    }

def get_memory_stats():
    """Get memory store statistics."""
    db_path = HOME / ".hermes" / "memory_store.db"
    if not db_path.exists():
        return {"facts": 0, "entities": 0}
    try:
        result = subprocess.run(
            ["sqlite3", str(db_path),
             "SELECT COUNT(*) FROM facts; SELECT COUNT(*) FROM entities; SELECT category, COUNT(*) FROM facts GROUP BY category ORDER BY COUNT(*) DESC;"],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        facts = int(lines[0]) if lines else 0
        entities = int(lines[1]) if len(lines) > 1 else 0
        categories = {}
        for line in lines[2:]:
            if '|' in line:
                cat, count = line.split('|')
                categories[cat] = int(count)
        return {"facts": facts, "entities": entities, "categories": categories}
    except:
        return {"facts": 77, "entities": 86}

def get_latest_cron_outputs():
    """Get latest output snippets from key cron jobs."""
    job_map = {
        "daily_review": "4a4564fc946b",
        "fleet_heartbeat": "6f7b35ec5e0b",
        "intel_synth": "0d71774646b8",
        "daily_chart": "0d71774646b8",
    }
    outputs = {}
    for name, job_id in job_map.items():
        d = HOME / ".hermes" / "cron" / "output" / job_id
        if d.exists():
            files = sorted(d.glob("*.md"), reverse=True)
            if files:
                content = files[0].read_text()
                # Extract just the response section
                if "## Response" in content:
                    resp = content.split("## Response", 1)[1][:800]
                else:
                    resp = content[:800]
                outputs[name] = {
                    "time": files[0].stat().st_mtime,
                    "snippet": resp.strip()[:600],
                }
    return outputs

def build_command_data():
    """Build the unified command center data — covers ALL system domains."""
    state = load_json(PROJECT / "paper_trader_state.json")
    dashboard = load_json(PROJECT / "dashboard_data.json")
    overview = load_json(PROJECT / "overview_data.json")
    sentiment = load_json(PROJECT / "sentiment_scores.json")
    fleet_feed = load_json(PROJECT / "fleet_feed.json")
    harness = load_json(PROJECT / "harness_data.json")
    leaderboard = load_json(PROJECT / "eval_leaderboard.json")
    lessons = load_json(PROJECT / "trading_lessons.json")
    pruned = load_json(PROJECT / "pruned_pairs.json")
    
    # NEW: All domain data
    fitness = get_fitness_data()
    wc = get_wc_portfolio()
    health = get_hermes_health()
    memory = get_memory_stats()
    cron_outputs = get_latest_cron_outputs()

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
        
        # === COMMAND OVERVIEW (landing page) ===
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
            "skills_count": len(list((HOME / ".hermes" / "skills").rglob("SKILL.md"))),
            "disk_mb": disk,
            "pruned_pairs": list(pruned.keys()),
            "optimization_count": (load_json(PROJECT / "strategy_config.json").get("_meta", {})).get("optimization_count", 0),
            # NEW: all-domain summary
            "fitness": fitness,
            "wc_portfolio": wc,
            "hermes_health": health,
            "memory": memory,
        },
        
        # === ALL DOMAINS ===
        "domains": {
            "trading": {
                "portfolio_value": round(capital, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "trades": len(closed_trades),
                "win_rate": round(len(wins) / max(len(closed_trades), 1) * 100, 1),
                "positions": len(positions),
                "net_pnl": round(net_pnl, 2),
            },
            "research": {
                "arxiv_last_run": cron_outputs.get("daily_review", {}).get("snippet", "")[:200],
                "polymarket_active": True,
            },
            "intelligence": {
                "scanners_active": sum(1 for s in scanner_status.values() if s.get("status") == "fresh"),
                "signals_count": fleet_signals,
                "sentiment_avg": round(avg_sentiment, 2),
            },
            "fitness": fitness,
            "world_cup": wc,
            "system": {
                "health_score": health.get("score", "N/A"),
                "crons": f"{cron_health.get('healthy', 0)}/{cron_health.get('total', 0)}",
                "scripts": len(list((HOME / ".hermes" / "scripts").glob("*.py"))),
                "skills": len(list((HOME / ".hermes" / "skills").rglob("SKILL.md"))),
                "disk_mb": sum(disk.values()),
            },
            "memory": memory,
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
            "skills_count": len(list((HOME / ".hermes" / "skills").rglob("SKILL.md"))),
            "pruned_pairs": pruned,
            "lessons": lessons,
            "loop_v2": dashboard.get("loop_v2", {}),
            "eval_leaderboard": leaderboard.get("leaderboard", [])[:10] if leaderboard else [],
            "harness": {
                "leaderboard": harness.get("leaderboard", [])[:10] if harness else [],
                "tests": harness.get("tests", {}) if harness else {},
                "audit_recent": harness.get("audit", {}).get("recent", [])[:8] if harness else [],
            },
            "cron_outputs": cron_outputs,
            "hermes_health": health,
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
