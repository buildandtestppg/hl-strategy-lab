#!/usr/bin/env python3
"""
HL Strategy Lab — Fleet Feed Builder
Merges latest outputs from X Scanner, Whale Monitor, and News Agent
into a single fleet_feed.json for the dashboard.
Output: fleet_feed.json
"""

import json, os, glob, re
from datetime import datetime

CRON_DIR = os.path.expanduser("~/.hermes/cron/output")
X_SCANNER_JOB = "43ba1d393c5a"
WHALE_JOB = "66a98cf277aa"
NEWS_JOB = "8bd6a73fc440"

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet_feed.json")


def parse_lines(text, max_items=20):
    """Extract individual signal lines from markdown/text output."""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        # Skip section headers
        if line.startswith("##") or line.startswith("==="):
            continue
        lines.append(line)
    return lines[:max_items]


def score_line(text):
    """Quick sentiment score for a single line."""
    text = text.lower()
    bullish = ["bullish", "pump", "breakout", "accumulate", "strong bid", 
               "flip long", "bid-heavy", "institutional", "upgrade", "rally",
               "building", "positive", "upside", "opportunity", "growth"]
    bearish = ["bearish", "dump", "crash", "distribute", "strong ask", 
               "flip short", "ask-heavy", "institutional sell", "downgrade", 
               "drop", "negative", "downside", "risk", "hack", "exploit", "bear"]
    
    score = 0
    for word in bullish:
        if word in text:
            score += 0.3
    for word in bearish:
        if word in text:
            score -= 0.3
    return max(-1, min(1, score))


def extract_timestamp(text):
    """Try to find a timestamp in the text."""
    # Look for patterns like "2026-06-29" or "HH:MM"
    m = re.search(r'(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2})', text)
    if m:
        return m.group(1)
    return None


def read_latest(job_id, prefix_filter=None):
    """Read the latest output file from a cron job."""
    job_dir = os.path.join(CRON_DIR, job_id)
    if not os.path.isdir(job_dir):
        return []
    
    files = sorted(glob.glob(os.path.join(job_dir, "*.md")), key=os.path.getmtime, reverse=True)
    if not files:
        files = sorted(glob.glob(os.path.join(job_dir, "*.txt")), key=os.path.getmtime, reverse=True)
    if not files:
        return []
    
    latest = files[0]
    try:
        with open(latest) as f:
            text = f.read()
    except:
        return []
    
    lines = parse_lines(text)
    signals = []
    for line in lines:
        ts = extract_timestamp(line) or os.path.getmtime(latest)
        s = score_line(line)
        signals.append({
            "text": line[:200],  # truncate long lines
            "sentiment": round(s, 2),
            "time": str(ts)
        })
    return signals


def main():
    data = {
        "x_scanner": read_latest(X_SCANNER_JOB),
        "whale_monitor": read_latest(WHALE_JOB),
        "news_agent": read_latest(NEWS_JOB),
        "generated": datetime.utcnow().isoformat() + "Z"
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    total = len(data["x_scanner"]) + len(data["whale_monitor"]) + len(data["news_agent"])
    print(f"Fleet feed: {total} signals across 3 sources → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
