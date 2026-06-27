#!/usr/bin/env python3
"""Full system health check for HL Strategy Lab."""
import json, os, subprocess, sys, time, requests
from datetime import datetime, timezone

PROJECT_DIR = "/Users/mojoai/Projects/hl-strategy-lab"
STATE_FILE = os.path.join(PROJECT_DIR, "paper_trader_state.json")
CONFIG_FILE = os.path.join(PROJECT_DIR, "strategy_config.json")
DATA_FILE = os.path.join(PROJECT_DIR, "dashboard_data.json")
SCRIPTS_DIR = os.path.mojoaihermes_scripts = os.path.expanduser("~/.hermes/scripts")
CHANNEL_ID = "1520303679125721118"

errors = []
warns = []
passes = []

def ok(name, detail=""):
    passes.append(name)
    print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))

def fail(name, detail=""):
    errors.append(f"{name}: {detail}")
    print(f"  FAIL  {name} — {detail}")

def warn(name, detail=""):
    warns.append(f"{name}: {detail}")
    print(f"  WARN  {name} — {detail}")

print("=" * 60)
print("  HL STRATEGY LAB — FULL SYSTEM HEALTH CHECK")
print("=" * 60)

# ─── 1. STATE FILE INTEGRITY ───
print("\n[1] State File Integrity")
if not os.path.exists(STATE_FILE):
    fail("State file exists", "paper_trader_state.json missing")
else:
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        ok("State file parses")
        
        # Check required keys
        required = ["capital", "initial_capital", "positions", "closed_trades", "equity_curve", "last_update"]
        for key in required:
            if key not in s:
                fail(f"State has key '{key}'", "missing")
            else:
                pass
        
        # Check values are sane
        cap = s.get("capital", 0)
        init = s.get("initial_capital", 0)
        if cap <= 0:
            fail("Capital positive", f"capital=${cap}")
        elif cap > init * 3:
            warn("Capital seems too high", f"${cap} vs initial ${init}")
        else:
            ok("Capital sane", f"${cap:.2f}")
        
        if init != 5000:
            warn("Initial capital", f"expected $5000, got ${init}")
        else:
            ok("Initial capital correct", f"${init}")
        
        # Check positions structure
        positions = s.get("positions", {})
        for pair, p in positions.items():
            for field in ["direction", "entry_price", "size", "stop_loss", "take_profit", "strategy"]:
                if field not in p:
                    fail(f"Position {pair} has '{field}'", "missing")
            if p.get("direction") not in ("LONG", "SHORT"):
                fail(f"Position {pair} direction valid", p.get("direction"))
            if p.get("entry_price", 0) <= 0:
                fail(f"Position {pair} entry price positive", p.get("entry_price"))
            if p.get("size", 0) <= 0:
                fail(f"Position {pair} size positive", p.get("size"))
        
        ok("Positions valid", f"{len(positions)} open")
        
        # Check equity curve
        eq = s.get("equity_curve", [])
        if len(eq) == 0:
            warn("Equity curve", "empty — will fill over time")
        else:
            ok("Equity curve", f"{len(eq)} data points")
            # Verify last point matches current state
            last_eq = eq[-1]
            if abs(last_eq["v"] - (cap + sum(
                (float(requests.post("https://api.hyperliquid.xyz/info", json={"type":"allMids"}, timeout=10).json().get(pair, p["entry_price"])) - p["entry_price"]) * p["size"] if p["direction"]=="LONG" else (p["entry_price"] - float(requests.post("https://api.hyperliquid.xyz/info", json={"type":"allMids"}, timeout=10).json().get(pair, p["entry_price"]))) * p["size"]
                for pair, p in positions.items()
            ))) > 500:
                warn("Equity curve last point", "doesn't match current portfolio (prices moved)")
        
        # Check last_update freshness
        last_upd = s.get("last_update")
        if last_upd:
            try:
                last_dt = datetime.fromisoformat(last_upd)
                age_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
                if age_min > 30:
                    fail("State freshness", f"{age_min:.0f} min old (>30min threshold)")
                elif age_min > 10:
                    warn("State freshness", f"{age_min:.0f} min old")
                else:
                    ok("State fresh", f"{age_min:.1f} min old")
            except Exception as e:
                warn("State timestamp parse", str(e))
        
    except json.JSONDecodeError as e:
        fail("State file parses", str(e))

# ─── 2. CONFIG FILE INTEGRITY ───
print("\n[2] Strategy Config Integrity")
try:
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    ok("Config parses")
    
    valid_strats = ["rsi", "macd", "bollinger", "trend"]
    pairs_found = 0
    for key, val in cfg.items():
        if key.startswith("_"):
            continue
        pairs_found += 1
        if not isinstance(val, dict) or "strategy" not in val:
            fail(f"Config[{key}] structure", str(val))
        elif val["strategy"] not in valid_strats:
            fail(f"Config[{key}] strategy valid", val["strategy"])
    
    if pairs_found != 6:
        fail("Config pair count", f"expected 6, got {pairs_found}")
    else:
        ok("Config pairs", f"{pairs_found} pairs configured")
    
    # Check meta
    meta = cfg.get("_meta", {})
    opt_count = meta.get("optimization_count", 0)
    ok("Optimizer history", f"{opt_count} optimizations run")
    
except Exception as e:
    fail("Config parses", str(e))

# ─── 3. DASHBOARD DATA FRESHNESS ───
print("\n[3] Dashboard Data")
try:
    with open(DATA_FILE) as f:
        dd = json.load(f)
    ok("Dashboard data parses")
    
    p = dd.get("portfolio", {})
    if not p.get("value"):
        fail("Dashboard portfolio value", "missing")
    else:
        ok("Dashboard portfolio", f"${p['value']:.2f}")
    
    # Check it has the new enriched fields
    for field in ["strategy_stats", "pair_stats", "equity_curve", "ai_history"]:
        if field not in dd:
            warn(f"Dashboard has '{field}'", "missing — may need one more cycle")
    
    # Compare with raw GitHub
    r = requests.get(
        "https://raw.githubusercontent.com/buildandtestppg/hl-strategy-lab/gh-pages/dashboard_data.json",
        timeout=10,
    )
    if r.status_code == 200:
        remote = r.json()
        local_ts = dd.get("last_update", "")
        remote_ts = remote.get("last_update", "")
        if local_ts == remote_ts:
            ok("GitHub Pages in sync", "local == remote")
        else:
            warn("GitHub Pages sync", f"local={local_ts[:19]} remote={remote_ts[:19]}")
    else:
        warn("GitHub Pages accessible", f"HTTP {r.status_code}")
        
except Exception as e:
    fail("Dashboard data", str(e))

# ─── 4. PAPER TRADER RUNS CLEAN ───
print("\n[4] Paper Trader Execution")
try:
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_DIR, "paper_trader.py")],
        cwd=PROJECT_DIR,
        capture_output=True, text=True, timeout=90,
    )
    output = result.stdout + result.stderr
    
    if result.returncode != 0:
        fail("Paper trader exit", f"exit {result.returncode}")
        print(f"  Output: {output[-500:]}")
    else:
        ok("Paper trader exit 0")
        
        # Check output has expected sections
        for section in ["Portfolio Value:", "P&L:", "Open Positions:"]:
            if section not in output:
                fail(f"Output has '{section}'", "missing")
            else:
                pass
        
        # Check for hidden errors
        error_patterns = ["Traceback", "Error", "Exception", "KeyError", "ValueError"]
        found_errors = [p for p in error_patterns if p in output]
        if found_errors:
            fail("No runtime errors in output", f"found: {found_errors}")
        else:
            ok("No runtime errors")
        
except subprocess.TimeoutExpired:
    fail("Paper trader", "timed out 90s")
except Exception as e:
    fail("Paper trader", str(e))

# ─── 5. GIT PUSH WORKS ───
print("\n[5] Git Push (GitHub Pages)")
try:
    # Check we're on gh-pages branch
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=PROJECT_DIR, capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    
    if branch != "gh-pages":
        fail("On gh-pages branch", f"on '{branch}'")
    else:
        ok("On gh-pages branch")
    
    # Check remote is set
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=PROJECT_DIR, capture_output=True, text=True, timeout=5,
    )
    if remote.returncode != 0:
        fail("Git remote origin", "not set")
    else:
        ok("Git remote set", remote.stdout.strip()[:50] + "...")
    
    # Check no uncommitted changes blocking
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_DIR, capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    if len(status.split("\n")) > 3:
        warn("Uncommitted files", f"{len(status.split(chr(10)))} files dirty")
    
except Exception as e:
    fail("Git check", str(e))

# ─── 6. WRAPPER SCRIPTS RUN CLEAN ───
print("\n[6] Wrapper Scripts")
# Paper trader wrapper
try:
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "hl_paper_trader.py")],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        fail("hl_paper_trader.py exit", f"exit {result.returncode}")
    else:
        ok("hl_paper_trader.py exit 0")
except Exception as e:
    fail("hl_paper_trader.py", str(e))

# Optimizer wrapper (just check it starts and the import works)
try:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, '" + SCRIPTS_DIR + "'); import hl_optimizer; print('OK')"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0 and "OK" in result.stdout:
        ok("hl_optimizer.py imports clean")
    else:
        fail("hl_optimizer.py imports", result.stderr[-200:])
except Exception as e:
    fail("hl_optimizer.py", str(e))

# ─── 7. HYPERLIQUID API LIVE ───
print("\n[7] Hyperliquid API")
try:
    r = requests.post("https://api.hyperliquid.xyz/info", json={"type": "allMids"}, timeout=10)
    prices = r.json()
    
    required_pairs = ["BTC", "ETH", "SOL", "HYPE", "AVAX", "DOGE"]
    for pair in required_pairs:
        if pair not in prices:
            fail(f"HL API has {pair}", "missing")
        elif float(prices[pair]) <= 0:
            fail(f"HL API {pair} price", prices[pair])
    
    if all(p in prices and float(prices[p]) > 0 for p in required_pairs):
        ok("All 6 pairs live on HL API", ", ".join(f"{p}=${float(prices[p]):.2f}" for p in required_pairs))
        
except Exception as e:
    fail("HL API", str(e))

# ─── 8. DISCORD NOTIFICATIONS ───
print("\n[8] Discord Channel")
try:
    token = None
    with open(os.path.expanduser("~/.hermes/.env")) as f:
        for line in f:
            parts = line.strip().split("=", 1)
            if parts[0] == "DISCORD_BOT_TOKEN" and len(parts) > 1:
                token = parts[1].strip('"').strip("'")
                break
    
    if not token:
        fail("Discord token", "not found in .env")
    else:
        r = requests.get(
            f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages?limit=3",
            headers={"Authorization": f"Bot {token}"},
            timeout=10,
        )
        if r.status_code == 200:
            ok("Discord channel accessible")
            msgs = r.json()
            # Check for recent errors
            recent_errors = [m for m in msgs if "ERROR" in m.get("content", "")]
            if recent_errors:
                warn("Recent Discord errors", f"{len(recent_errors)} error posts in last 3 msgs")
        else:
            fail("Discord channel", f"HTTP {r.status_code}")
except Exception as e:
    fail("Discord", str(e))

# ─── SUMMARY ───
print("\n" + "=" * 60)
total = len(passes) + len(warns) + len(errors)
print(f"  RESULTS: {len(passes)} PASS · {len(warns)} WARN · {len(errors)} FAIL")
print(f"  Total checks: {total}")
print("=" * 60)

if errors:
    print("\n  CRITICAL FAILURES:")
    for e in errors:
        print(f"    FAIL {e}")
    sys.exit(1)
elif warns:
    print("\n  WARNINGS (non-blocking):")
    for w in warns:
        print(f"    WARN {w}")
    print("\n  System is operational with minor warnings.")
else:
    print("\n  All systems operational.")

sys.exit(0 if not errors else 1)
