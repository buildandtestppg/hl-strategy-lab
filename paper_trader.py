"""
HL Strategy Lab — Paper Trader
Live simulation engine. Fetches real HL prices, runs best strategy per pair,
tracks positions, records all trades to JSON for dashboard.
Designed to run via cron (every 1h) — zero LLM calls, pure deterministic.
"""
import json
import time
import os
import requests
import numpy as np
from datetime import datetime, timezone
from strategy_engine import (
    Signal, HLData, RSIStrategy, MACDStrategy, BollingerStrategy,
    TrendFollowStrategy, StochasticStrategy, VWAPStrategy,
    SupertrendStrategy, BreakoutStrategy, EMACrossStrategy, calc_atr,
)

# ─── Config (loaded from strategy_config.json — AI can rewrite this) ───

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "strategy_config.json")

def load_strategy_config():
    """Load strategy assignments from JSON config. AI can modify this file."""
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    # Filter out _meta key
    return {k: v for k, v in cfg.items() if not k.startswith("_")}

def save_strategy_config(config, meta_note=""):
    """Save updated strategy config with metadata."""
    full_cfg = dict(config)
    full_cfg["_meta"] = {
        "last_optimized": datetime.now(timezone.utc).isoformat(),
        "optimization_count": 0,
        "history": []
    }
    # Load existing meta to preserve history
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            old = json.load(f)
        if "_meta" in old:
            full_cfg["_meta"]["optimization_count"] = old["_meta"].get("optimization_count", 0) + 1
            full_cfg["_meta"]["history"] = old["_meta"].get("history", [])[-9:] + [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": meta_note,
            }]
    with open(CONFIG_FILE, "w") as f:
        json.dump(full_cfg, f, indent=2)

# Load config at import time (also reloaded by get_strategy each cycle)
PAIR_STRATEGIES = load_strategy_config()

STRATEGY_MAP = {
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerStrategy,
    "trend": TrendFollowStrategy,
    "stochastic": StochasticStrategy,
    "vwap": VWAPStrategy,
    "supertrend": SupertrendStrategy,
    "breakout": BreakoutStrategy,
    "emacross": EMACrossStrategy,
}

def get_strategy(pair):
    """Create strategy instance from config."""
    cfg = load_strategy_config().get(pair)
    if not cfg:
        return None
    cls = STRATEGY_MAP.get(cfg["strategy"])
    if not cls:
        return None
    return cls(**cfg.get("params", {}))

INITIAL_CAPITAL = 5000.0
RISK_PER_TRADE = 0.02       # 2% of portfolio per position
MAX_LEVERAGE = 3
STOP_LOSS_ATR = 2.5
TAKE_PROFIT_ATR = 5.0
HL_FEE = 0.00045          # Real taker fee 0.045% (conservative — we'll use market orders)
SLIPPAGE = 0.0005         # 0.05% slippage on taker fills
MIN_CONFIDENCE = 0.6       # Raised from 0.3 — only strong signals enter
MAX_POSITIONS = 6           # one per pair max
CANDLE_INTERVAL = "1h"
CANDLE_DAYS = 30            # lookback for indicator calculation
MIN_HOLD_HOURS = 3          # minimum hold time before signal exit allowed
REENTRY_COOLDOWN_HOURS = 4  # cooldown after closing before re-entering same pair

# ─── AI Sentiment Overlay ───
SENTIMENT_FILE = os.path.join(os.path.dirname(__file__), "sentiment_scores.json")
SENTIMENT_THRESHOLD = 0.3   # min |score| to override TA signal
SENTIMENT_MIN_CONFIDENCE = 0.4  # min aggregator confidence to trust sentiment

STATE_FILE = os.path.join(os.path.dirname(__file__), "paper_trader_state.json")
TRADE_LOG_FILE = os.path.join(os.path.dirname(__file__), "paper_trades.json")
DASHBOARD_DATA = os.path.join(os.path.dirname(__file__), "dashboard_data.json")

# ─── State Management ───

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        # Sanitize: fix any corrupted sizes from older bugs
        for pair, pos in state.get("positions", {}).items():
            pos["size"] = abs(pos["size"])  # size must always be positive
        return state
    return {
        "capital": INITIAL_CAPITAL,
        "initial_capital": INITIAL_CAPITAL,
        "positions": {},      # {pair: {direction, entry_price, size, stop_loss, take_profit, entry_time, strategy}}
        "closed_trades": [],
        "equity_curve": [],   # [{timestamp, value, pnl_pct}]
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_update": None,
        "total_fees_paid": 0,
        "last_close_times": {},  # {pair: timestamp} for re-entry cooldown
    }

def save_state(state):
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    # Atomic write: write to temp then rename (prevents corruption from concurrent cron runs)
    tmp_file = STATE_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_file, STATE_FILE)

# ─── Helpers ───

def load_sentiment():
    """Load fleet sentiment scores. Returns {pair: {score, confidence, sources}}."""
    if not os.path.exists(SENTIMENT_FILE):
        return {}
    try:
        with open(SENTIMENT_FILE) as f:
            data = json.load(f)
        # Check freshness — staleness guard at 4 hours
        for pair, info in data.items():
            ts = info.get("timestamp", "")
            if ts:
                age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 3600
                if age_h > 4:
                    return {}  # stale — don't use
        return data
    except Exception:
        return {}

def apply_sentiment_overlay(signal, confidence, pair, sentiment):
    """Apply AI sentiment overlay to a TA signal.
    
    Rules:
    - If sentiment strongly contradicts TA (score > threshold, opposite direction): BLOCK the trade
    - If sentiment strongly agrees with TA: BOOST confidence (trade bigger)
    - If sentiment is neutral or weak: no change
    
    Returns (adjusted_signal, adjusted_confidence, sentiment_note)
    """
    pair_data = sentiment.get(pair)
    if not pair_data or pair_data.get("confidence", 0) < SENTIMENT_MIN_CONFIDENCE:
        return signal, confidence, ""
    
    score = pair_data["score"]
    
    # Determine sentiment direction
    if score > SENTIMENT_THRESHOLD:
        sent_dir = Signal.LONG
    elif score < -SENTIMENT_THRESHOLD:
        sent_dir = Signal.SHORT
    else:
        return signal, confidence, "sentiment neutral"
    
    # Check agreement
    if signal == sent_dir:
        # Aligned — boost confidence
        boost = min(0.2, abs(score) * 0.15)
        return signal, min(1.0, confidence + boost), f"sentiment aligned ({score:+.2f}), conf boosted"
    elif signal in (Signal.LONG, Signal.SHORT):
        # Contradiction — block the trade
        return Signal.FLAT, 0, f"BLOCKED by sentiment ({score:+.2f} vs {signal.value})"
    else:
        return signal, confidence, ""


def trade_cost(size, entry_price, exit_price):
    notional = size * entry_price + size * exit_price
    return notional * (HL_FEE + SLIPPAGE)

def get_current_prices():
    """Fetch all current prices from HL."""
    r = requests.post("https://api.hyperliquid.xyz/info", json={"type": "allMids"}, timeout=10)
    r.raise_for_status()
    return r.json()

def get_funding_rates():
    """Fetch live funding rates from HL. Returns {pair: rate_per_hour}.
    Positive = longs pay shorts, Negative = shorts pay longs."""
    r = requests.post("https://api.hyperliquid.xyz/info", json={
        "type": "metaAndAssetCtxs"
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    meta, ctxs = data[0], data[1]
    rates = {}
    for i, asset in enumerate(meta["universe"]):
        name = asset["name"]
        ctx = ctxs[i]
        funding = ctx.get("funding")
        if funding is not None:
            rates[name] = float(funding)
    return rates

def calc_funding_cost(pair, direction, notional, hold_hours, funding_rates):
    """Calculate funding cost for a position.
    Returns the cost (positive = we pay, negative = we earn)."""
    rate = funding_rates.get(pair)
    if rate is None:
        return 0.0
    # Funding charged per hour. Positive rate = longs pay shorts.
    # If we're LONG and rate is positive, we pay. If SHORT and rate negative, we pay.
    if direction == "LONG":
        return notional * rate * hold_hours
    else:  # SHORT — opposite sign
        return -notional * rate * hold_hours

# ─── Paper Trader Core ───

class PaperTrader:
    def __init__(self):
        self.state = load_state()

    def run_cycle(self):
        """Run one trading cycle: check exits, check entries, update dashboard data."""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions": [],
            "positions": {},
            "portfolio_value": 0,
            "prices": {},
        }

        prices = get_current_prices()
        candles_cache = {}

        # Load fleet sentiment for AI overlay
        sentiment = load_sentiment()

        # Fetch live funding rates for accurate cost tracking
        try:
            funding_rates = get_funding_rates()
        except Exception:
            funding_rates = {}  # Don't fail the cycle if funding API hiccups

        # Reload config fresh each cycle (AI may have changed it)
        active_pairs = load_strategy_config()

        # Fetch candles for all pairs
        for pair in active_pairs:
            try:
                candles = HLData.get_candles(pair, interval=CANDLE_INTERVAL, days=CANDLE_DAYS)
                # CRITICAL: Drop the last (forming/unclosed) candle.
                # Without this, every 5-min cron run re-evaluates a shifting candle → whipsaw.
                if candles and len(candles) > 30:
                    candles = candles[:-1]
                candles_cache[pair] = candles
                price = float(prices.get(pair, candles[-1]["c"]))
                results["prices"][pair] = price
            except Exception as e:
                results["actions"].append(f"⚠️ {pair}: Failed to fetch candles: {e}")
                continue

        # 1. Check exits on open positions
        for pair in list(self.state["positions"].keys()):
            if pair not in candles_cache:
                continue
            pos = self.state["positions"][pair]
            current_price = results["prices"].get(pair)
            if not current_price:
                continue

            candles = candles_cache[pair]
            o, h, l, c, v = HLData.candles_to_arrays(candles)
            atr = calc_atr(h, l, c, 14)
            current_atr = atr[-1] if not np.isnan(atr[-1]) else current_price * 0.02

            exit_reason = None
            exit_price = None

            # Check stop loss / take profit
            if pos["direction"] == "LONG":
                if current_price <= pos["stop_loss"]:
                    exit_reason = "stop_loss"
                    exit_price = pos["stop_loss"]
                elif current_price >= pos["take_profit"]:
                    exit_reason = "take_profit"
                    exit_price = pos["take_profit"]
            else:  # SHORT
                if current_price >= pos["stop_loss"]:
                    exit_reason = "stop_loss"
                    exit_price = pos["stop_loss"]
                elif current_price <= pos["take_profit"]:
                    exit_reason = "take_profit"
                    exit_price = pos["take_profit"]

            # Check signal exit (if strategy says flip or go flat)
            # CRITICAL: Use the strategy that opened the position, not current config
            # (optimizer may have changed it mid-position)
            # CRITICAL: Enforce minimum hold time to prevent whipsaw churning
            if not exit_reason:
                entry_strat_name = pos.get("strategy", "unknown")
                cfg = load_strategy_config().get(pair, {})
                current_strat_name = cfg.get("strategy", "unknown")
                hold_hours = (time.time() - pos["entry_ts"]) / 3600
                
                if hold_hours < MIN_HOLD_HOURS:
                    # Too soon — let SL/TP work, don't exit on signal
                    pass
                elif entry_strat_name == current_strat_name:
                    # Config hasn't changed and held long enough — safe to check signal exit
                    strat = get_strategy(pair)
                    if strat:
                        result = strat.compute(candles)
                        if (pos["direction"] == "LONG" and result.signal == Signal.SHORT) or \
                           (pos["direction"] == "SHORT" and result.signal == Signal.LONG):
                            exit_reason = "signal_exit"
                            exit_price = current_price
                else:
                    # Strategy was swapped by optimizer while position is open.
                    # Only exit on SL/TP, not signal (avoids premature exit from new strategy).
                    pass

            if exit_reason:
                self._close_position(pair, exit_price, exit_reason, results, funding_rates)

        # 2. Check for new entries
        # Track pairs closed THIS CYCLE to prevent same-cycle re-entry (whipsaw fix)
        _now = time.time()
        pairs_closed_this_cycle = set()
        for p, close_ts in self.state.get("last_close_times", {}).items():
            if _now - close_ts < 120:  # closed in last 2 minutes = this cycle
                pairs_closed_this_cycle.add(p)

        open_count = len(self.state["positions"])
        for pair in active_pairs:
            if pair in self.state["positions"]:
                continue  # already in position
            if pair in pairs_closed_this_cycle:
                continue  # closed this cycle — no re-entry
            if open_count >= MAX_POSITIONS:
                break
            if pair not in candles_cache:
                continue

            # Re-entry cooldown: skip if we closed this pair recently
            last_close_ts = self.state.get("last_close_times", {}).get(pair)
            if last_close_ts and (time.time() - last_close_ts) / 3600 < REENTRY_COOLDOWN_HOURS:
                continue

            strat = get_strategy(pair)
            if not strat:
                continue

            candles = candles_cache[pair]
            result = strat.compute(candles)

            if result.signal in (Signal.LONG, Signal.SHORT) and result.confidence > MIN_CONFIDENCE:
                # Apply AI sentiment overlay — fleet intelligence gate
                adj_signal, adj_conf, sent_note = apply_sentiment_overlay(
                    result.signal, result.confidence, pair, sentiment
                )
                if sent_note:
                    result.reason = f"{result.reason} | sentiment: {sent_note}"
                
                if adj_signal in (Signal.LONG, Signal.SHORT) and adj_conf > MIN_CONFIDENCE:
                    # Update confidence (may be boosted by aligned sentiment)
                    result.confidence = adj_conf
                    self._open_position(pair, result, results["prices"][pair], candles, results)
                    open_count += 1
                elif result.signal in (Signal.LONG, Signal.SHORT):
                    results["actions"].append(
                        f"🧠 {pair}: TA signal {result.signal.value} blocked by sentiment overlay"
                    )

        # 3. Calculate portfolio value (with unrealized funding cost estimate)
        portfolio_value = self.state["capital"]
        for pair, pos in self.state["positions"].items():
            current_price = results["prices"].get(pair, pos["entry_price"])
            if pos["direction"] == "LONG":
                unrealized = (current_price - pos["entry_price"]) * pos["size"]
            else:
                unrealized = (pos["entry_price"] - current_price) * pos["size"]
            # Subtract unrealized funding cost for honest portfolio valuation
            hold_hours = (time.time() - pos["entry_ts"]) / 3600
            current_notional = pos["size"] * current_price
            unrealized_funding = calc_funding_cost(pair, pos["direction"], current_notional, hold_hours, funding_rates)
            unrealized -= unrealized_funding
            portfolio_value += unrealized
            results["positions"][pair] = {
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "current_price": current_price,
                "size": pos["size"],
                "unrealized_pnl": round(unrealized, 2),
                "unrealized_pnl_pct": round(unrealized / (pos["entry_price"] * pos["size"]) * 100, 2),
                "stop_loss": pos["stop_loss"],
                "take_profit": pos["take_profit"],
                "strategy": pos["strategy"],
                "entry_time": pos["entry_time"],
                "duration_hours": round(hold_hours, 1),
                "funding_rate": round(funding_rates.get(pair, 0) * 100, 6),
                "unrealized_funding": round(unrealized_funding, 4),
            }

        results["portfolio_value"] = round(portfolio_value, 2)
        results["capital"] = round(self.state["capital"], 2)
        results["pnl"] = round(portfolio_value - INITIAL_CAPITAL, 2)
        results["pnl_pct"] = round((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)
        results["initial_capital"] = INITIAL_CAPITAL
        results["open_positions"] = len(self.state["positions"])
        results["total_closed_trades"] = len(self.state["closed_trades"])
        results["total_fees_paid"] = round(self.state["total_fees_paid"], 2)

        # Win/loss stats from closed trades
        closed = self.state["closed_trades"]
        if closed:
            wins = [t for t in closed if t["pnl"] > 0]
            losses = [t for t in closed if t["pnl"] <= 0]
            results["stats"] = {
                "total_trades": len(closed),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / len(closed) * 100, 1),
                "avg_win": round(np.mean([t["pnl"] for t in wins]), 2) if wins else 0,
                "avg_loss": round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
                "best_trade": round(max(t["pnl"] for t in closed), 2),
                "worst_trade": round(min(t["pnl"] for t in closed), 2),
                "total_realized_pnl": round(sum(t["pnl"] for t in closed), 2),
            }
        else:
            results["stats"] = {
                "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "best_trade": 0, "worst_trade": 0,
                "total_realized_pnl": 0,
            }

        # 4. Record equity curve point
        if "equity_curve" not in self.state:
            self.state["equity_curve"] = []
        self.state["equity_curve"].append({
            "t": results["timestamp"],
            "v": results["portfolio_value"],
            "p": results["pnl_pct"],
        })
        # Keep last 500 points (~42 hours at 5min intervals)
        if len(self.state["equity_curve"]) > 500:
            self.state["equity_curve"] = self.state["equity_curve"][-500:]

        # 5. Save state + write dashboard data
        save_state(self.state)

        # Compute per-strategy and per-pair stats from all closed trades
        all_trades = self.state["closed_trades"]
        strategy_stats = {}
        pair_stats = {}
        for t in all_trades:
            # Per strategy
            s_name = t.get("strategy", "unknown")
            if s_name not in strategy_stats:
                strategy_stats[s_name] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            strategy_stats[s_name]["trades"] += 1
            if t["pnl"] > 0:
                strategy_stats[s_name]["wins"] += 1
            else:
                strategy_stats[s_name]["losses"] += 1
            strategy_stats[s_name]["pnl"] += t["pnl"]

            # Per pair
            p_name = t.get("pair", "?")
            if p_name not in pair_stats:
                pair_stats[p_name] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            pair_stats[p_name]["trades"] += 1
            if t["pnl"] > 0:
                pair_stats[p_name]["wins"] += 1
            else:
                pair_stats[p_name]["losses"] += 1
            pair_stats[p_name]["pnl"] += t["pnl"]

        # Add win rates
        for s in strategy_stats.values():
            s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 0
            s["pnl"] = round(s["pnl"], 2)
        for p in pair_stats.values():
            p["win_rate"] = round(p["wins"] / p["trades"] * 100, 1) if p["trades"] > 0 else 0
            p["pnl"] = round(p["pnl"], 2)

        # Load AI optimizer history
        ai_history = []
        config_path = os.path.join(os.path.dirname(__file__), "strategy_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg_data = json.load(f)
            if "_meta" in cfg_data:
                ai_history = cfg_data["_meta"].get("history", [])

        # Write dashboard data (combines results + state)
        dashboard = {
            "last_update": results["timestamp"],
            "portfolio": {
                "value": results["portfolio_value"],
                "capital": results["capital"],
                "initial": INITIAL_CAPITAL,
                "pnl": results["pnl"],
                "pnl_pct": results["pnl_pct"],
                "open_positions": results["open_positions"],
            },
            "prices": results["prices"],
            "positions": results["positions"],
            "stats": results["stats"],
            "actions": results["actions"],
            "closed_trades": self.state["closed_trades"][-50:],  # last 50
            "pair_strategies": {p: active_pairs[p]["strategy"] for p in active_pairs},
            "strategy_stats": strategy_stats,
            "pair_stats": pair_stats,
            "equity_curve": self.state["equity_curve"][-200:],
            "ai_history": ai_history,
            "sentiment": sentiment,
            "config": {
                "risk_per_trade": f"{RISK_PER_TRADE*100}%",
                "max_leverage": f"{MAX_LEVERAGE}x",
                "stop_loss": f"{STOP_LOSS_ATR}x ATR",
                "take_profit": f"{TAKE_PROFIT_ATR}x ATR",
            },
        }
        with open(DASHBOARD_DATA, "w") as f:
            json.dump(dashboard, f, indent=2)

        return results

    def _open_position(self, pair, result, price, candles, results):
        """Open a new simulated position."""
        # CRITICAL: Use actual per-pair prices for portfolio valuation
        portfolio_value = self._portfolio_value(results["prices"])
        
        # Guard: stop trading if portfolio is in the red
        if portfolio_value <= 0:
            results["actions"].append(f"⚠️ {pair}: Skipping — portfolio value negative (${portfolio_value:.2f})")
            return

        o, h, l, c, v = HLData.candles_to_arrays(candles)
        atr = calc_atr(h, l, c, 14)
        current_atr = atr[-1] if not np.isnan(atr[-1]) else price * 0.02

        risk_amount = portfolio_value * RISK_PER_TRADE
        stop_distance = current_atr * STOP_LOSS_ATR
        if stop_distance <= 0:
            return

        size = risk_amount / stop_distance
        
        # CRITICAL: Cap notional at INITIAL_CAPITAL * leverage (not inflated portfolio value)
        max_notional = INITIAL_CAPITAL * MAX_LEVERAGE
        if size * price > max_notional:
            size = max_notional / price

        # CRITICAL: size must always be positive (direction is tracked separately)
        size = abs(size)

        direction = "LONG" if result.signal == Signal.LONG else "SHORT"
        if direction == "LONG":
            stop_loss = price - current_atr * STOP_LOSS_ATR
            take_profit = price + current_atr * TAKE_PROFIT_ATR
        else:
            stop_loss = price + current_atr * STOP_LOSS_ATR
            take_profit = price - current_atr * TAKE_PROFIT_ATR

        # Deduct entry fees from capital (entry fee only)
        entry_cost = size * price * (HL_FEE + SLIPPAGE)
        self.state["capital"] -= entry_cost
        self.state["total_fees_paid"] += entry_cost

        self.state["positions"][pair] = {
            "direction": direction,
            "entry_price": price,
            "size": size,
            "stop_loss": round(stop_loss, 6),
            "take_profit": round(take_profit, 6),
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_ts": time.time(),
            "strategy": load_strategy_config().get(pair, {}).get("strategy", "unknown"),
            "confidence": result.confidence,
            "reason": result.reason,
        }

        results["actions"].append(
            f"🟢 OPENED {direction} {pair} @ ${price:.4f} | "
            f"Size: {size:.4f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f} | "
            f"Strategy: {load_strategy_config().get(pair, {}).get('strategy', 'unknown')} | {result.reason}"
        )

    def _close_position(self, pair, exit_price, reason, results, funding_rates=None):
        """Close a position and record the trade."""
        if funding_rates is None:
            funding_rates = {}
        pos = self.state["positions"][pair]
        entry = pos["entry_price"]
        size = pos["size"]
        direction = pos["direction"]

        if direction == "LONG":
            raw_pnl = (exit_price - entry) * size
        else:
            raw_pnl = (entry - exit_price) * size

        exit_cost = size * exit_price * (HL_FEE + SLIPPAGE)  # exit fee only (entry already paid)

        # Funding cost: based on hold time, direction, and live funding rate
        hold_hours = (time.time() - pos["entry_ts"]) / 3600
        exit_notional = size * exit_price
        funding_cost = calc_funding_cost(pair, direction, exit_notional, hold_hours, funding_rates)

        net_pnl = raw_pnl - exit_cost - funding_cost
        self.state["capital"] += net_pnl
        self.state["total_fees_paid"] += exit_cost + max(0, funding_cost)  # only count positive costs

        trade_record = {
            "pair": pair,
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_price,
            "size": size,
            "pnl": round(net_pnl, 4),
            "pnl_pct": round(net_pnl / (entry * size) * 100, 2),
            "entry_time": pos["entry_time"],
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "duration_hours": round(hold_hours, 1),
            "exit_reason": reason,
            "strategy": pos["strategy"],
            "fees_paid": round(exit_cost, 4),
            "funding_cost": round(funding_cost, 4),
            "total_cost": round(exit_cost + funding_cost, 4),
        }

        self.state["closed_trades"].append(trade_record)
        del self.state["positions"][pair]
        
        # Record close time for re-entry cooldown
        if "last_close_times" not in self.state:
            self.state["last_close_times"] = {}
        self.state["last_close_times"][pair] = time.time()

        emoji = "✅" if net_pnl > 0 else "❌"
        results["actions"].append(
            f"{emoji} CLOSED {direction} {pair} @ ${exit_price:.4f} | "
            f"P&L: ${net_pnl:+.2f} ({net_pnl/(entry*size)*100:+.1f}%) | "
            f"Reason: {reason}"
        )

    def _portfolio_value(self, prices=None):
        """Calculate portfolio value using per-pair prices.
        
        Args:
            prices: dict of {pair: price} for accurate valuation,
                    or None to use entry prices (conservative).
        """
        val = self.state["capital"]
        for pair, pos in self.state["positions"].items():
            if prices and pair in prices:
                cp = prices[pair]
            else:
                cp = pos["entry_price"]
            if pos["direction"] == "LONG":
                val += (cp - pos["entry_price"]) * pos["size"]
            else:
                val += (pos["entry_price"] - cp) * pos["size"]
        return val


if __name__ == "__main__":
    trader = PaperTrader()
    print(f"=== HL Strategy Lab — Paper Trader ===")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Pairs: {', '.join(PAIR_STRATEGIES.keys())}")
    print(f"Strategies: {', '.join(v['strategy'] for v in PAIR_STRATEGIES.values())}")
    print(f"Risk: {RISK_PER_TRADE*100}% per trade | Max Leverage: {MAX_LEVERAGE}x")
    print(f"SL: {STOP_LOSS_ATR}x ATR | TP: {TAKE_PROFIT_ATR}x ATR")
    print()

    results = trader.run_cycle()

    print(f"--- Trading Cycle Complete ---")
    print(f"Portfolio Value: ${results['portfolio_value']:,.2f}")
    print(f"Cash: ${results['capital']:,.2f}")
    print(f"P&L: ${results['pnl']:+,.2f} ({results['pnl_pct']:+.2f}%)")
    print(f"Open Positions: {results['open_positions']}")
    print()

    if results["actions"]:
        print("--- Actions ---")
        for a in results["actions"]:
            print(f"  {a}")
        print()

    if results["positions"]:
        print("--- Open Positions ---")
        for pair, p in results["positions"].items():
            print(f"  {pair}: {p['direction']} ${p['entry_price']:.4f} → ${p['current_price']:.4f} "
                  f"| P&L: ${p['unrealized_pnl']:+.2f} ({p['unrealized_pnl_pct']:+.1f}%) "
                  f"| {p['strategy']}")
    else:
        print("--- No open positions ---")

    print()
    if results["stats"]["total_trades"] > 0:
        s = results["stats"]
        print(f"--- Trade Stats ---")
        print(f"  Total: {s['total_trades']} | W: {s['wins']} L: {s['losses']} | Win Rate: {s['win_rate']}%")
        print(f"  Realized P&L: ${s['total_realized_pnl']:+,.2f}")
        print(f"  Best: ${s['best_trade']:+.2f} | Worst: ${s['worst_trade']:+.2f}")

    print(f"\n📊 Dashboard data → {DASHBOARD_DATA}")
    print(f"💾 State → {STATE_FILE}")
