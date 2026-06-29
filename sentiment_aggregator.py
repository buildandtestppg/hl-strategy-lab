"""
HL Strategy Lab — Sentiment Aggregator
Reads latest outputs from fleet intelligence (X Scanner, Whale Monitor, News Agent)
and produces a structured sentiment score per pair.

Output: sentiment_scores.json
  {
    "BTC": {"score": -0.6, "confidence": 0.7, "sources": [...], "timestamp": "..."},
    "ETH": {"score": -0.3, "confidence": 0.5, "sources": [...], "timestamp": "..."},
    ...
  }

Score range: -1.0 (extremely bearish) to +1.0 (extremely bullish)
Confidence: 0.0 to 1.0 (how much data backs this score)

Zero LLM calls — pure deterministic parsing of fleet outputs.
"""
import json
import os
import re
import glob
from datetime import datetime, timezone, timedelta

# ─── Paths ───
CRON_OUTPUT = os.path.expanduser("~/.hermes/cron/output")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_scores.json")

# Job IDs from cron config
X_SCANNER_JOB = "43ba1d393c5a"
WHALE_MONITOR_JOB = "66a98cf277aa"
NEWS_AGENT_JOB = "8bd6a73fc440"

# Pairs we trade
TRADED_PAIRS = ["BTC", "ETH", "SOL", "HYPE", "AVAX", "DOGE"]

# ─── Sentiment keywords from X scanner emoji codes ───
BULLISH_EMOJIS = ["🏗", "🚀", "💎", "🐂", "📈", "🔥", "⚡"]
BEARISH_EMOJIS = ["📉", "🐻", "🔴", "⛔", "⚠️"]
NEUTRAL_EMOJIS = ["🧵", "🗺", "🌊", "📊"]


def get_latest_output(job_id, max_age_hours=6):
    """Get the latest output file from a cron job, within max_age."""
    job_dir = os.path.join(CRON_OUTPUT, job_id)
    if not os.path.isdir(job_dir):
        return None
    files = sorted(glob.glob(os.path.join(job_dir, "*.md")), reverse=True)
    if not files:
        return None
    # Check freshness
    mtime = os.path.getmtime(files[0])
    age_hours = (datetime.now().timestamp() - mtime) / 3600
    if age_hours > max_age_hours:
        return None
    with open(files[0]) as f:
        return f.read()


def parse_x_scanner(text):
    """Parse X scanner output for per-coin sentiment signals.
    Returns {pair: [(score, weight), ...]}"""
    if not text:
        return {}
    scores = {}
    # Find the "Response" section (actual scanner output)
    if "## Response" in text:
        text = text.split("## Response")[-1]
    # Parse signal lines: • [emoji $TICKER] @user (engagement) description [score: X.Y]
    # Pattern: captures emoji sentiment, ticker, and quality score
    pattern = r"•\s*\[([🏗🚀💎🐂📈🔥⚡📉🐻🔴⛔⚠️🧵🗺🌊📊]+)\s+\$([A-Z]+)\]"
    matches = re.findall(pattern, text)
    score_pattern = r"\[score:\s*([\d.]+)\]"

    lines = text.split("\n")
    for line in lines:
        match = re.search(r"\$([A-Z]+)", line)
        if not match:
            continue
        pair = match.group(1)
        if pair not in TRADED_PAIRS:
            continue

        # Determine bullish/bearish from emoji
        is_bullish = any(e in line for e in BULLISH_EMOJIS)
        is_bearish = any(e in line for e in BEARISH_EMOJIS)

        # Extract quality score
        qs_match = re.search(score_pattern, line)
        qs = float(qs_match.group(1)) if qs_match else 1.0

        # Also check for keyword sentiment
        lower_line = line.lower()
        bearish_words = ["short", "sell", "dump", "crash", "bear", "resistance", "reject", "liquidat"]
        bullish_words = ["long", "buy", "pump", "breakout", "bull", "support", "accumulate", "deposited"]

        keyword_bullish = any(w in lower_line for w in bullish_words)
        keyword_bearish = any(w in lower_line for w in bearish_words)

        # Combine: emoji + keyword
        signal = 0
        if is_bullish or keyword_bullish:
            signal += 1
        if is_bearish or keyword_bearish:
            signal -= 1

        if signal != 0:
            if pair not in scores:
                scores[pair] = []
            scores[pair].append((signal, qs))

    return scores


def parse_whale_monitor(text):
    """Parse whale monitor output for orderbook imbalance and whale flows.
    Returns {pair: [(score, weight), ...]}"""
    if not text:
        return {}
    scores = {}
    if "## Response" in text:
        text = text.split("## Response")[-1]

    # Parse the coin table for orderbook imbalance
    # Pattern: | BTC | $59,266 | $1.98B | -0.79% | 📉 81% ask | 0.0002% |
    table_pattern = r"\|\s*([A-Z]+)\s*\|.*?\|\s*([\d.]+)%\s*\|.*?(bid|ask|⚖️)"
    for match in re.finditer(r"\|\s*([A-Z]+)\s*\|.*?(\d{2,3})%\s*(bid|ask)", text):
        pair = match.group(1)
        pct = int(match.group(2))
        side = match.group(3)
        if pair not in TRADED_PAIRS:
            continue
        if pair not in scores:
            scores[pair] = []
        # bid-heavy = bullish, ask-heavy = bearish
        if side == "bid":
            imbalance_score = (pct - 50) / 50.0  # 82% bid → +0.64
        else:
            imbalance_score = (50 - pct) / 50.0  # 81% ask → -0.62
        scores[pair].append((imbalance_score, 2.0))  # whale data weighted higher

    # Parse whale position changes
    # Pattern: 🐋 0xabc... flipped BTC SHORT→LONG
    for line in text.split("\n"):
        if "🐋" not in line:
            continue
        for pair in TRADED_PAIRS:
            if pair not in line:
                continue
            if pair not in scores:
                scores[pair] = []
            lower = line.lower()
            if "short→long" in lower or "increased" in lower and "long" in lower:
                scores[pair].append((0.5, 1.5))
            elif "long→short" in lower or "increased" in lower and "short" in lower:
                scores[pair].append((-0.5, 1.5))
            elif "reduced" in lower or "closed" in lower:
                scores[pair].append((-0.2, 0.5))  # uncertainty

    return scores


def parse_news_agent(text):
    """Parse news agent output for breaking headlines affecting specific coins.
    Returns {pair: [(score, weight), ...]}"""
    if not text:
        return {}
    scores = {}
    if "## Response" in text:
        text = text.split("## Response")[-1]

    # Check for [SILENT] — no news
    if "[SILENT]" in text:
        return {}

    for pair in TRADED_PAIRS:
        # Look for pair mentions with context
        pattern = rf"\$?{pair}\b"
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get surrounding context (the line)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].lower()

            # Classify sentiment from context
            if any(w in context for w in ["hack", "exploit", "drain", "depeg", "crash", "ban", "sec ", "lawsuit"]):
                scores.setdefault(pair, []).append((-0.8, 3.0))  # Very bearish, high weight
            elif any(w in context for w in ["etf", "approved", "institutional", "adoption", "partnership", "launch"]):
                scores.setdefault(pair, []).append((0.6, 2.0))
            elif any(w in context for w in ["record", "all-time", "breakout", "surge", " milestone"]):
                scores.setdefault(pair, []).append((0.5, 2.0))

    return scores


def aggregate_scores(x_scores, whale_scores, news_scores):
    """Combine all sentiment sources into final per-pair scores."""
    all_pairs = set(list(x_scores.keys()) + list(whale_scores.keys()) + list(news_scores.keys()))
    result = {}

    for pair in TRADED_PAIRS:
        signals = []
        sources = []

        if pair in x_scores:
            for score, weight in x_scores[pair]:
                signals.append((score, weight))
                sources.append("x_scanner")

        if pair in whale_scores:
            for score, weight in whale_scores[pair]:
                signals.append((score, weight))
                sources.append("whale_monitor")

        if pair in news_scores:
            for score, weight in news_scores[pair]:
                signals.append((score, weight))
                sources.append("news_agent")

        if not signals:
            result[pair] = {
                "score": 0.0,
                "confidence": 0.0,
                "sources": [],
                "signal_count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            continue

        # Weighted average
        total_weight = sum(w for _, w in signals)
        weighted_sum = sum(s * w for s, w in signals)
        score = max(-1.0, min(1.0, weighted_sum / total_weight))

        # Confidence based on signal count and source diversity
        unique_sources = list(set(sources))
        confidence = min(1.0, len(signals) * 0.15 + len(unique_sources) * 0.2)

        result[pair] = {
            "score": round(score, 2),
            "confidence": round(confidence, 2),
            "sources": unique_sources,
            "signal_count": len(signals),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return result


def main():
    """Main entry point — aggregate all fleet sentiment."""
    now = datetime.now(timezone.utc)
    print(f"=== Sentiment Aggregator — {now.isoformat()} ===\n")

    # Fetch latest outputs
    x_text = get_latest_output(X_SCANNER_JOB, max_age_hours=2)
    whale_text = get_latest_output(WHALE_MONITOR_JOB, max_age_hours=1)
    news_text = get_latest_output(NEWS_AGENT_JOB, max_age_hours=3)

    print(f"X Scanner: {'✅ ' + str(len(x_text)) + ' chars' if x_text else '❌ stale/missing'}")
    print(f"Whale Monitor: {'✅ ' + str(len(whale_text)) + ' chars' if whale_text else '❌ stale/missing'}")
    print(f"News Agent: {'✅ ' + str(len(news_text)) + ' chars' if news_text else '❌ stale/missing'}")

    # Parse each source
    x_scores = parse_x_scanner(x_text)
    whale_scores = parse_whale_monitor(whale_text)
    news_scores = parse_news_agent(news_text)

    print(f"\nParsed signals:")
    print(f"  X Scanner: {sum(len(v) for v in x_scores.values())} signals across {len(x_scores)} pairs")
    print(f"  Whale Monitor: {sum(len(v) for v in whale_scores.values())} signals across {len(whale_scores)} pairs")
    print(f"  News Agent: {sum(len(v) for v in news_scores.values())} signals across {len(news_scores)} pairs")

    # Aggregate
    final = aggregate_scores(x_scores, whale_scores, news_scores)

    # Print summary
    print(f"\n{'='*50}")
    print(f"{'Pair':6s} {'Score':>7s} {'Conf':>6s} {'Signals':>8s} Sources")
    print(f"{'-'*50}")
    for pair in TRADED_PAIRS:
        s = final[pair]
        emoji = "🟢" if s["score"] > 0.2 else "🔴" if s["score"] < -0.2 else "⚪"
        print(f"{pair:6s} {s['score']:+7.2f} {s['confidence']:6.2f} {s['signal_count']:8d} {','.join(s['sources'])} {emoji}")

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n✅ Written to {OUTPUT_FILE}")

    return final


if __name__ == "__main__":
    main()
