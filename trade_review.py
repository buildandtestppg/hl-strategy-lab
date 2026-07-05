#!/usr/bin/env python3
"""
Trade Review Engine — Autonomous post-trade analysis and lesson extraction.

When a trade closes, this engine:
1. Analyzes WHY it won or lost (TA signal, sentiment overlay, market conditions)
2. Extracts a machine-readable lesson
3. Writes it to Obsidian vault + lessons.json
4. Updates trust scores on existing lessons (if this trade confirms/contradicts)
5. Outputs a summary for the review cron

This is the core of the self-improving loop. No human in the loop.
"""
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter

SGT = timezone(timedelta(hours=8))

# Paths
PROJECT_DIR = Path(__file__).parent.absolute()
VAULT = Path.home() / "Documents" / "Obsidian Vault" / "Fleet Intelligence"
LESSONS_DIR = VAULT / "Lessons"
LESSONS_JSON = LESSONS_DIR / "lessons.json"
REVIEWS_DIR = VAULT / "Reviews"
STATE_FILE = PROJECT_DIR / "paper_trader_state.json"
SENTIMENT_FILE = PROJECT_DIR / "sentiment_scores.json"
DASHBOARD_FILE = PROJECT_DIR / "dashboard_data.json"

# Trading constants (must match paper_trader.py)
INITIAL_CAPITAL = 5000.0
TRADED_PAIRS = ["BTC", "ETH", "SOL", "HYPE"]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, str(path))


def load_lessons():
    """Load the lessons index."""
    data = load_json(LESSONS_JSON)
    if not data:
        return {"lessons": [], "stats": {"total_lessons": 0, "verified_correct": 0,
                "verified_wrong": 0, "pending_review": 0}, "last_updated": None}
    return data


def save_lessons(data):
    data["last_updated"] = now_iso()
    save_json(LESSONS_JSON, data)


def analyze_trade(trade, sentiment_at_close=None):
    """Analyze a single closed trade and extract a lesson.
    
    Returns a lesson dict or None if nothing to learn.
    """
    pair = trade.get("pair", "?")
    direction = (trade.get("direction") or "").upper()
    pnl = float(trade.get("pnl") or 0)
    pnl_pct = float(trade.get("pnl_pct") or 0)
    entry_price = float(trade.get("entry_price") or 0)
    exit_price = float(trade.get("exit_price") or 0)
    duration_h = float(trade.get("duration_hours") or 0)
    exit_reason = trade.get("exit_reason") or "unknown"
    strategy = trade.get("strategy") or "unknown"
    
    won = pnl > 0
    
    # Build lesson
    lesson = {
        "id": f"{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": now_iso(),
        "pair": pair,
        "direction": direction,
        "strategy": strategy,
        "outcome": "win" if won else "loss",
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "duration_hours": round(duration_h, 1),
        "exit_reason": exit_reason,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "trust": 0.5,  # starts neutral
        "verified": False,
        "pattern_tags": [],
        "lesson_text": "",
    }
    
    # Analyze patterns
    tags = []
    
    # Duration pattern
    if duration_h < 4:
        tags.append("short_hold")
    elif duration_h > 12:
        tags.append("long_hold")
    else:
        tags.append("medium_hold")
    
    # Exit reason pattern
    if "stop" in exit_reason.lower():
        tags.append("stopped_out")
    elif "take_profit" in exit_reason.lower() or "tp" in exit_reason.lower():
        tags.append("tp_hit")
    elif "sentiment" in exit_reason.lower() or "regime" in exit_reason.lower():
        tags.append("sentiment_override")
    elif "signal" in exit_reason.lower():
        tags.append("signal_exit")
    
    # Sentiment context
    sent_data = sentiment_at_close or {}
    pair_sent = sent_data.get(pair, {})
    sent_score = pair_sent.get("score", 0)
    sent_conv = pair_sent.get("convergence", False)
    
    if sent_conv and abs(sent_score) > 0.3:
        if sent_score > 0 and direction == "LONG":
            tags.append("convergence_aligned")
        elif sent_score < 0 and direction == "SHORT":
            tags.append("convergence_aligned")
        else:
            tags.append("convergence_ignored")
    
    lesson["pattern_tags"] = tags
    
    # Build lesson text (human-readable insight)
    insights = []
    if won:
        if "convergence_aligned" in tags:
            insights.append(f"{pair}: Trade won WITH sentiment convergence ({sent_score:+.2f}). Convergence alignment profitable.")
        if "tp_hit" in tags:
            insights.append(f"{pair}: Take profit hit in {duration_h:.0f}h. {strategy} strategy caught the move.")
        if "long_hold" in tags and pnl_pct > 2:
            insights.append(f"{pair}: Long hold ({duration_h:.0f}h) was profitable. Patience rewarded.")
    else:
        if "stopped_out" in tags:
            insights.append(f"{pair}: Stopped out after {duration_h:.0f}h. {strategy} signal was wrong.")
        if "convergence_ignored" in tags:
            insights.append(f"{pair}: LOST despite sentiment disagreement ({sent_score:+.2f}). Trader ignored convergence signal.")
        if "short_hold" in tags:
            insights.append(f"{pair}: Lost on short hold ({duration_h:.0f}h). Entry timing was poor.")
        if exit_reason == "regime_override":
            insights.append(f"{pair}: Sentiment regime override triggered exit. Saved or lost money depending on P&L.")
    
    if not insights:
        if won:
            insights.append(f"{pair}: {direction} won +${pnl:.0f} ({pnl_pct:+.1f}%) via {strategy}. Duration {duration_h:.0f}h. Exit: {exit_reason}.")
        else:
            insights.append(f"{pair}: {direction} lost -${abs(pnl):.0f} ({pnl_pct:+.1f}%) via {strategy}. Duration {duration_h:.0f}h. Exit: {exit_reason}.")
    
    lesson["lesson_text"] = " ".join(insights)
    return lesson


def write_lesson_to_obsidian(lesson):
    """Write a lesson as a markdown note in the Obsidian vault."""
    pair = lesson["pair"]
    date_str = datetime.now(SGT).strftime("%Y-%m-%d_%H%M")
    filename = f"{date_str}_{pair}_{lesson['outcome']}.md"
    filepath = LESSONS_DIR / filename
    
    won = lesson["outcome"] == "win"
    emoji = "✅" if won else "❌"
    tags_str = " ".join(f"#{t}" for t in lesson["pattern_tags"])
    
    md = f"""# {emoji} {pair} {lesson['direction']} — {lesson['outcome'].upper()} ({lesson['pnl_pct']:+.1f}%)

**P&L:** ${lesson['pnl']:+.2f}
**Strategy:** {lesson['strategy']}
**Duration:** {lesson['duration_hours']:.1f}h
**Exit Reason:** {lesson['exit_reason']}
**Entry:** ${lesson['entry_price']:.2f} → **Exit:** ${lesson['exit_price']:.2f}
**Trust:** {lesson['trust']:.2f}
**Timestamp:** {lesson['timestamp']}

## Lesson
{lesson['lesson_text']}

## Tags
{tags_str}

---
[[Fleet Intelligence]]
"""
    with open(filepath, "w") as f:
        f.write(md)
    
    return filepath


def update_trust_scores(new_lesson, lessons_data):
    """When a new lesson is added, check if it confirms or contradicts existing lessons.
    Update trust scores accordingly."""
    new_pair = new_lesson["pair"]
    new_tags = set(new_lesson["pattern_tags"])
    new_outcome = new_lesson["outcome"]
    
    for existing in lessons_data["lessons"]:
        # Only compare same pair + overlapping tags
        if existing["pair"] != new_pair:
            continue
        existing_tags = set(existing.get("pattern_tags", []))
        overlap = new_tags & existing_tags
        if not overlap:
            continue
        
        # Same tags, same outcome → confirms
        if existing["outcome"] == new_outcome:
            existing["trust"] = min(1.0, existing.get("trust", 0.5) + 0.05)
            existing.setdefault("confirmations", 0)
            existing["confirmations"] += 1
        # Same tags, opposite outcome → contradicts
        else:
            existing["trust"] = max(0.0, existing.get("trust", 0.5) - 0.1)
            existing.setdefault("contradictions", 0)
            existing["contradictions"] += 1


def get_pair_performance(lessons_data, pair):
    """Get accumulated performance stats for a pair from lessons."""
    pair_lessons = [l for l in lessons_data["lessons"] if l["pair"] == pair]
    if not pair_lessons:
        return None
    
    wins = sum(1 for l in pair_lessons if l["outcome"] == "win")
    losses = sum(1 for l in pair_lessons if l["outcome"] == "loss")
    total = len(pair_lessons)
    total_pnl = sum(l["pnl"] for l in pair_lessons)
    
    # Pattern frequency
    tag_counter = Counter()
    for l in pair_lessons:
        for tag in l.get("pattern_tags", []):
            tag_counter[tag] += 1
    
    # High-trust patterns (confirmed multiple times)
    reliable_patterns = []
    for tag, count in tag_counter.most_common(5):
        tag_lessons = [l for l in pair_lessons if tag in l.get("pattern_tags", [])]
        tag_wins = sum(1 for l in tag_lessons if l["outcome"] == "win")
        tag_wr = tag_wins / len(tag_lessons) * 100 if tag_lessons else 0
        if count >= 2:
            reliable_patterns.append({
                "pattern": tag,
                "count": count,
                "win_rate": round(tag_wr, 1),
                "verdict": "profitable" if tag_wr >= 60 else "unprofitable" if tag_wr <= 40 else "mixed"
            })
    
    return {
        "pair": pair,
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "total_pnl": round(total_pnl, 2),
        "top_patterns": reliable_patterns,
        "avg_trust": round(sum(l.get("trust", 0.5) for l in pair_lessons) / total, 2),
    }


def generate_lessons_feed(lessons_data):
    """Generate a machine-readable lessons summary that the trader can consume.
    This is the FEEDBACK part of the loop.
    
    P6: Now includes exit-reason performance breakdown so the loop can
    detect if specific exit types are systematically losing money.
    """
    feed = {"generated_at": now_iso(), "pairs": {}}
    
    for pair in TRADED_PAIRS:
        perf = get_pair_performance(lessons_data, pair)
        if perf:
            feed["pairs"][pair] = perf
    
    # High-trust lessons that should influence future trades
    feed["active_lessons"] = []
    for lesson in lessons_data["lessons"]:
        if lesson.get("trust", 0.5) >= 0.65:
            feed["active_lessons"].append({
                "pair": lesson["pair"],
                "pattern": lesson["pattern_tags"],
                "lesson": lesson["lesson_text"],
                "trust": lesson["trust"],
            })
    
    # P6: Exit-reason performance breakdown
    exit_stats = {}
    for lesson in lessons_data["lessons"]:
        reason = lesson.get("exit_reason", "unknown")
        if reason not in exit_stats:
            exit_stats[reason] = {"count": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        exit_stats[reason]["count"] += 1
        if lesson["outcome"] == "win":
            exit_stats[reason]["wins"] += 1
        else:
            exit_stats[reason]["losses"] += 1
        exit_stats[reason]["pnl"] += lesson.get("pnl", 0)
    
    for reason_data in exit_stats.values():
        reason_data["pnl"] = round(reason_data["pnl"], 2)
        reason_data["win_rate"] = round(reason_data["wins"] / reason_data["count"] * 100, 1) if reason_data["count"] > 0 else 0
        reason_data["verdict"] = (
            "profitable" if reason_data["win_rate"] >= 60 and reason_data["pnl"] > 0
            else "unprofitable" if reason_data["win_rate"] <= 40 or reason_data["pnl"] < 0
            else "mixed"
        )
    
    feed["exit_breakdown"] = exit_stats
    
    return feed


def main():
    """Main entry — review newly closed trades and extract lessons."""
    print(f"=== Trade Review Engine — {datetime.now(SGT).strftime('%Y-%m-%d %H:%M SGT')} ===")
    
    # Load state
    state = load_json(STATE_FILE) or {}
    sentiment = load_json(SENTIMENT_FILE) or {}
    dashboard = load_json(DASHBOARD_FILE) or {}
    lessons_data = load_lessons()
    
    # Get all closed trades
    closed_trades = state.get("closed_trades", [])
    if not closed_trades:
        print("No closed trades to review.")
        return
    
    # Track which trades we've already reviewed
    reviewed_ids = set()
    for lesson in lessons_data["lessons"]:
        # Use pair + timestamp + pnl as unique key
        key = f"{lesson['pair']}_{lesson['timestamp']}"
        reviewed_ids.add(key)
    
    # Find new trades (not yet reviewed)
    new_lessons = []
    for trade in closed_trades:
        # Create a rough unique ID for this trade
        pair = trade.get("pair", "?")
        exit_time = trade.get("exit_time", "")
        pnl = trade.get("pnl", 0)
        trade_key = f"{pair}_{exit_time}_{pnl}"
        
        # Skip if already reviewed
        already = False
        for lesson in lessons_data["lessons"]:
            if lesson["pair"] == pair and abs(lesson["pnl"] - pnl) < 0.01:
                already = True
                break
        if already:
            continue
        
        # Analyze and extract lesson
        lesson = analyze_trade(trade, sentiment)
        if lesson:
            new_lessons.append(lesson)
    
    if not new_lessons:
        print(f"All {len(closed_trades)} closed trades already reviewed. No new lessons.")
        
        # Still regenerate the feed
        feed = generate_lessons_feed(lessons_data)
        feed_path = PROJECT_DIR / "trading_lessons.json"
        save_json(feed_path, feed)
        print(f"Lessons feed refreshed: {feed_path}")
        print(f"  Active lessons: {len(feed['active_lessons'])}")
        return
    
    # Process new lessons
    print(f"\nFound {len(new_lessons)} new trade(s) to review:")
    for lesson in new_lessons:
        print(f"  {'✅' if lesson['outcome']=='win' else '❌'} {lesson['pair']} {lesson['direction']}: "
              f"{lesson['pnl_pct']:+.1f}% ({lesson['pnl']:+.2f}) — {lesson['strategy']}")
        print(f"     Tags: {', '.join(lesson['pattern_tags'])}")
        print(f"     Lesson: {lesson['lesson_text'][:100]}")
        
        # Write to Obsidian
        md_path = write_lesson_to_obsidian(lesson)
        print(f"     → Obsidian: {md_path.name}")
        
        # Update trust scores on existing lessons
        update_trust_scores(lesson, lessons_data)
        
        # Add to index
        lessons_data["lessons"].append(lesson)
    
    # Update stats
    lessons_data["stats"]["total_lessons"] = len(lessons_data["lessons"])
    lessons_data["stats"]["pending_review"] = sum(1 for l in lessons_data["lessons"] if not l.get("verified"))
    
    # Save
    save_lessons(lessons_data)
    
    # Generate feedback feed for trader
    feed = generate_lessons_feed(lessons_data)
    feed_path = PROJECT_DIR / "trading_lessons.json"
    save_json(feed_path, feed)
    
    print(f"\n{'='*60}")
    print(f"Lessons stored: {len(lessons_data['lessons'])} total")
    print(f"  Wins: {sum(1 for l in lessons_data['lessons'] if l['outcome']=='win')}")
    print(f"  Losses: {sum(1 for l in lessons_data['lessons'] if l['outcome']=='loss')}")
    print(f"  Active (high-trust): {len(feed['active_lessons'])}")
    print(f"  Feed: {feed_path}")
    
    # Output for cron consumption
    output = {
        "new_reviews": len(new_lessons),
        "total_lessons": len(lessons_data["lessons"]),
        "active_lessons": len(feed["active_lessons"]),
        "pair_summaries": feed["pairs"],
        "new_lesson_texts": [l["lesson_text"] for l in new_lessons],
    }
    
    output_path = PROJECT_DIR / "trade_review_output.json"
    save_json(output_path, output)
    
    return output


if __name__ == "__main__":
    main()
