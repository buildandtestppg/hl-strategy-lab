"""
HL Strategy Lab — Decision Audit Logger
Records every signal, gate decision, and trade action per cycle.
Creates a permanent audit trail so every trade can be traced back to WHY it happened.
"""
import json
import os
import time
from datetime import datetime, timezone

AUDIT_FILE = os.path.join(os.path.dirname(__file__), "decision_audit.jsonl")
MAX_AUDIT_ENTRIES = 10000  # rolling log — keep last 10k decisions


def log_decision(pair, signal, confidence, reason, gate_result, sentiment=None,
                 action="evaluated", price=None, cycle_id=None):
    """Log a single decision point in the trading pipeline.

    Args:
        pair: Trading pair (BTC, ETH, etc.)
        signal: Signal value (LONG, SHORT, FLAT)
        confidence: Confidence score 0-1
        reason: Human-readable reason from strategy
        gate_result: "PASSED", "BLOCKED_CONFIDENCE", "BLOCKED_SENTIMENT",
                     "BLOCKED_COOLDOWN", "BLOCKED_ALREADY_OPEN", "SKIPPED_FLAT"
        sentiment: Optional dict with sentiment overlay details
        action: "evaluated", "opened", "closed", "held"
        price: Current price at time of decision
        cycle_id: Optional cycle identifier for grouping
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "pair": pair,
        "signal": signal,
        "confidence": round(confidence, 4) if confidence else 0,
        "reason": reason[:200] if reason else "",
        "gate": gate_result,
        "action": action,
    }
    if sentiment:
        entry["sentiment"] = {
            "score": round(sentiment.get("score", 0), 3),
            "confidence": round(sentiment.get("confidence", 0), 3),
            "adjusted_conf": round(sentiment.get("adjusted_conf", 0), 3),
            "note": sentiment.get("note", ""),
        }
    if price is not None:
        entry["price"] = round(price, 4)
    if cycle_id:
        entry["cycle_id"] = cycle_id

    # Append to JSONL file
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_cycle_summary(cycle_id, actions, positions_count, portfolio_value, pnl):
    """Log a summary of the entire cycle."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "type": "cycle_summary",
        "cycle_id": cycle_id,
        "actions": actions,
        "positions_count": positions_count,
        "portfolio_value": round(portfolio_value, 2),
        "pnl": round(pnl, 2),
    }
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def rotate_log_if_needed():
    """Keep the log from growing forever — trim to MAX_AUDIT_ENTRIES."""
    if not os.path.exists(AUDIT_FILE):
        return
    with open(AUDIT_FILE) as f:
        lines = f.readlines()
    if len(lines) > MAX_AUDIT_ENTRIES:
        with open(AUDIT_FILE, "w") as f:
            f.writelines(lines[-MAX_AUDIT_ENTRIES:])


def get_recent_decisions(pair=None, limit=50, action_filter=None):
    """Query recent decisions from the audit log.

    Args:
        pair: Filter by pair (optional)
        limit: Max results
        action_filter: Filter by action type (optional)

    Returns list of decision dicts, newest first.
    """
    if not os.path.exists(AUDIT_FILE):
        return []

    with open(AUDIT_FILE) as f:
        lines = f.readlines()

    results = []
    for line in reversed(lines):
        try:
            entry = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if pair and entry.get("pair") != pair:
            continue
        if action_filter and entry.get("action") != action_filter:
            continue
        results.append(entry)
        if len(results) >= limit:
            break

    return results


def get_decision_stats(pair=None, lookback_hours=24):
    """Aggregate stats from the audit log.

    Returns dict with counts of each gate result, action type, etc.
    """
    if not os.path.exists(AUDIT_FILE):
        return {}

    cutoff = time.time() - (lookback_hours * 3600)
    stats = {
        "total_evaluated": 0,
        "gate_counts": {},
        "action_counts": {},
        "signal_counts": {},
        "avg_confidence": [],
    }

    with open(AUDIT_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if entry.get("epoch", 0) < cutoff:
                continue
            if pair and entry.get("pair") != pair:
                continue
            if entry.get("type") == "cycle_summary":
                continue

            stats["total_evaluated"] += 1
            gate = entry.get("gate", "UNKNOWN")
            stats["gate_counts"][gate] = stats["gate_counts"].get(gate, 0) + 1

            action = entry.get("action", "unknown")
            stats["action_counts"][action] = stats["action_counts"].get(action, 0) + 1

            signal = entry.get("signal", "FLAT")
            stats["signal_counts"][signal] = stats["signal_counts"].get(signal, 0) + 1

            conf = entry.get("confidence", 0)
            if conf > 0:
                stats["avg_confidence"].append(conf)

    if stats["avg_confidence"]:
        import numpy as np
        stats["avg_confidence"] = round(float(np.mean(stats["avg_confidence"])), 3)
    else:
        stats["avg_confidence"] = 0

    return stats


def export_for_dashboard(limit=100):
    """Export recent decisions in a format the dashboard can consume."""
    decisions = get_recent_decisions(limit=limit)
    stats = get_decision_stats(lookback_hours=48)

    return {
        "recent_decisions": decisions[:limit],
        "stats": stats,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
