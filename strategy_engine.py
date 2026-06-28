"""
HL Strategy Lab — Strategy Engine
Generates trading signals from technical indicators for multiple pairs.
"""
import time
import requests
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Signal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

@dataclass
class Indicator:
    name: str
    value: float

@dataclass
class StrategyResult:
    signal: Signal
    confidence: float  # 0-1
    indicators: dict
    reason: str

# ─── Indicators ───

def sma(data, period):
    result = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    return result

def ema(data, period):
    result = np.full(len(data), np.nan)
    k = 2 / (period + 1)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1 - k)
    return result

def calc_rsi(data, period=14):
    result = np.full(len(data), np.nan)
    if len(data) < period + 1:
        return result
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    result[period] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))
    for i in range(period + 1, len(data)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        result[i] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))
    return result

def calc_macd(data, fast=12, slow=26, signal=9):
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    # Compute signal line: EMA of the MACD line
    # Find first non-NaN index in macd_line
    first_valid = slow - 1  # EMA(slow) needs at least `slow` data points
    if len(data) - first_valid < signal:
        # Not enough data for signal line
        full_signal = np.full(len(data), np.nan)
        histogram = macd_line - full_signal
        return macd_line, full_signal, histogram
    # Compute EMA of macd_line starting from first_valid
    valid_macd = macd_line[first_valid:]
    signal_line = ema(valid_macd, signal)
    # Pad signal to match full length
    full_signal = np.full(len(data), np.nan)
    offset = first_valid  # signal_line[i] corresponds to macd_line[first_valid + i]
    valid_signal_start = offset + (signal - 1)  # first valid signal
    full_signal[valid_signal_start:] = signal_line[signal - 1:]
    histogram = macd_line - full_signal
    return macd_line, full_signal, histogram

def calc_bollinger(data, period=20, std_mult=2):
    mid = sma(data, period)
    rolling_std = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        rolling_std[i] = np.std(data[i - period + 1:i + 1])
    upper = mid + std_mult * rolling_std
    lower = mid - std_mult * rolling_std
    return upper, mid, lower

def calc_atr(highs, lows, closes, period=14):
    result = np.full(len(highs), np.nan)
    if len(highs) < period + 1:
        return result
    trs = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    )
    for i in range(period, len(highs)):
        result[i] = np.mean(trs[i - period:i])
    return result

# ─── Data Fetcher ───

class HLData:
    BASE = "https://api.hyperliquid.xyz"

    @staticmethod
    def get_candles(coin, interval="1h", days=90):
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - days * 86400 * 1000
        r = requests.post(f"{HLData.BASE}/info", json={
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval,
                    "startTime": start_ms, "endTime": now_ms}
        }, timeout=15)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def get_prices():
        r = requests.post(f"{HLData.BASE}/info", json={"type": "allMids"}, timeout=10)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def candles_to_arrays(candles):
        """Convert raw candle list to OHLCV numpy arrays."""
        o = np.array([float(c["o"]) for c in candles])
        h = np.array([float(c["h"]) for c in candles])
        l = np.array([float(c["l"]) for c in candles])
        c = np.array([float(c["c"]) for c in candles])
        v = np.array([float(c["v"]) for c in candles])
        return o, h, l, c, v

# ─── Strategies ───

class RSIStrategy:
    """RSI mean reversion — buy oversold, sell overbought."""
    name = "RSI Mean Reversion"

    def __init__(self, period=14, oversold=30, overbought=70, exit_mid=50):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_mid = exit_mid

    def compute(self, candles):
        o, h, l, c, v = HLData.candles_to_arrays(candles)
        rsi = calc_rsi(c, self.period)
        atr = calc_atr(h, l, c, 14)

        idx = len(c) - 1
        if np.isnan(rsi[idx]):
            return StrategyResult(Signal.FLAT, 0, {}, "Not enough data")

        current_rsi = rsi[idx]
        indicators = {"rsi": round(current_rsi, 2), "atr": round(atr[idx], 2) if not np.isnan(atr[idx]) else 0}

        # Track RSI trend (last 3 bars)
        rsi_trend = rsi[idx] - rsi[idx - 3] if idx >= 3 and not np.isnan(rsi[idx - 3]) else 0

        if current_rsi < self.oversold:
            conf = min(1.0, (self.oversold - current_rsi) / self.oversold)
            return StrategyResult(Signal.LONG, conf, indicators,
                f"RSI {current_rsi:.1f} < {self.oversold} (oversold)")
        elif current_rsi > self.overbought:
            conf = min(1.0, (current_rsi - self.overbought) / (100 - self.overbought))
            return StrategyResult(Signal.SHORT, conf, indicators,
                f"RSI {current_rsi:.1f} > {self.overbought} (overbought)")
        else:
            return StrategyResult(Signal.FLAT, 0, indicators,
                f"RSI {current_rsi:.1f} — neutral zone")

class MACDStrategy:
    """MACD crossover — buy on bullish cross, sell on bearish."""
    name = "MACD Crossover"

    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, candles):
        o, h, l, c, v = HLData.candles_to_arrays(candles)
        macd_line, signal_line, histogram = calc_macd(c, self.fast, self.slow, self.signal)

        idx = len(c) - 1
        if np.isnan(histogram[idx]) or np.isnan(histogram[idx - 1]):
            return StrategyResult(Signal.FLAT, 0, {}, "Not enough data")

        current_hist = histogram[idx]
        prev_hist = histogram[idx - 1]
        indicators = {
            "macd": round(macd_line[idx], 4),
            "signal": round(signal_line[idx], 4),
            "histogram": round(current_hist, 4),
        }

        # Bullish cross: histogram crosses from negative to positive
        if prev_hist < 0 and current_hist >= 0:
            return StrategyResult(Signal.LONG, 0.8, indicators,
                f"MACD bullish cross (hist {prev_hist:.4f} → {current_hist:.4f})")
        # Bearish cross
        elif prev_hist > 0 and current_hist <= 0:
            return StrategyResult(Signal.SHORT, 0.8, indicators,
                f"MACD bearish cross (hist {prev_hist:.4f} → {current_hist:.4f})")
        # Trend continuation
        elif current_hist > 0:
            return StrategyResult(Signal.LONG, 0.4, indicators,
                f"MACD bullish (hist {current_hist:.4f})")
        elif current_hist < 0:
            return StrategyResult(Signal.SHORT, 0.4, indicators,
                f"MACD bearish (hist {current_hist:.4f})")
        else:
            return StrategyResult(Signal.FLAT, 0, indicators, "MACD flat")

class BollingerStrategy:
    """Bollinger Band bounce — buy at lower band, sell at upper."""
    name = "Bollinger Bands"

    def __init__(self, period=20, std_mult=2.0):
        self.period = period
        self.std_mult = std_mult

    def compute(self, candles):
        o, h, l, c, v = HLData.candles_to_arrays(candles)
        upper, mid, lower = calc_bollinger(c, self.period, self.std_mult)

        idx = len(c) - 1
        if np.isnan(upper[idx]):
            return StrategyResult(Signal.FLAT, 0, {}, "Not enough data")

        price = c[idx]
        indicators = {
            "price": round(price, 2),
            "upper": round(upper[idx], 2),
            "mid": round(mid[idx], 2),
            "lower": round(lower[idx], 2),
        }

        band_width = upper[idx] - lower[idx]

        if price <= lower[idx]:
            # %B = (price - lower) / (upper - lower)
            pct_b = (price - lower[idx]) / band_width if band_width > 0 else 0
            return StrategyResult(Signal.LONG, min(1.0, 1 - pct_b), indicators,
                f"Price {price:.2f} at/below lower BB {lower[idx]:.2f}")
        elif price >= upper[idx]:
            band_width = upper[idx] - lower[idx]
            pct_b = (price - lower[idx]) / band_width if band_width > 0 else 1
            return StrategyResult(Signal.SHORT, min(1.0, pct_b), indicators,
                f"Price {price:.2f} at/above upper BB {upper[idx]:.2f}")
        else:
            return StrategyResult(Signal.FLAT, 0, indicators,
                f"Price {price:.2f} inside bands ({lower[idx]:.2f} - {upper[idx]:.2f})")

class TrendFollowStrategy:
    """Multi-SMA trend following — buy when fast > slow > baseline."""
    name = "Trend Following (SMA)"

    def __init__(self, fast=9, slow=21, baseline=50):
        self.fast = fast
        self.slow = slow
        self.baseline = baseline

    def compute(self, candles):
        o, h, l, c, v = HLData.candles_to_arrays(candles)
        sma_fast = sma(c, self.fast)
        sma_slow = sma(c, self.slow)
        sma_base = sma(c, self.baseline)

        idx = len(c) - 1
        if np.isnan(sma_base[idx]):
            return StrategyResult(Signal.FLAT, 0, {}, "Not enough data")

        indicators = {
            "sma_fast": round(sma_fast[idx], 2),
            "sma_slow": round(sma_slow[idx], 2),
            "sma_base": round(sma_base[idx], 2),
            "price": round(c[idx], 2),
        }

        bullish = sma_fast[idx] > sma_slow[idx] > sma_base[idx]
        bearish = sma_fast[idx] < sma_slow[idx] < sma_base[idx]

        if bullish:
            spread = (sma_fast[idx] - sma_slow[idx]) / sma_slow[idx]
            return StrategyResult(Signal.LONG, min(1.0, spread * 20), indicators,
                f"Uptrend: SMA{self.fast} > SMA{self.slow} > SMA{self.baseline}")
        elif bearish:
            spread = (sma_slow[idx] - sma_fast[idx]) / sma_slow[idx]
            return StrategyResult(Signal.SHORT, min(1.0, spread * 20), indicators,
                f"Downtrend: SMA{self.fast} < SMA{self.slow} < SMA{self.baseline}")
        else:
            return StrategyResult(Signal.FLAT, 0, indicators,
                f"Mixed signals — no clear trend")

# ─── Strategy Registry ───

STRATEGIES = {
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerStrategy,
    "trend": TrendFollowStrategy,
}

def create_strategy(name, **params):
    """Create a strategy instance by name with optional params."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**params)

# ─── Multi-Strategy Voting ───

class MultiStrategyVoter:
    """Run multiple strategies and vote on direction."""

    def __init__(self, strategy_configs):
        """strategy_configs: [{name: "rsi", params: {...}}, ...]"""
        self.strategies = []
        for cfg in strategy_configs:
            s = create_strategy(cfg["name"], **cfg.get("params", {}))
            self.strategies.append(s)

    def compute(self, candles):
        results = []
        votes = {Signal.LONG: 0, Signal.SHORT: 0, Signal.FLAT: 0}
        weighted = {Signal.LONG: 0.0, Signal.SHORT: 0.0, Signal.FLAT: 0.0}

        for s in self.strategies:
            result = s.compute(candles)
            results.append({"strategy": s.name, "result": result})
            votes[result.signal] += 1
            weighted[result.signal] += result.confidence

        # Winner = highest weighted votes
        total_weight = sum(weighted.values())
        if total_weight == 0:
            return StrategyResult(Signal.FLAT, 0, {"votes": votes},
                "All strategies flat"), results

        best_signal = max(weighted, key=weighted.get)
        best_conf = weighted[best_signal] / total_weight if total_weight > 0 else 0

        reasons = [r["result"].reason for r in results if r["result"].signal == best_signal]
        reason = f"{' + '.join(reasons[:2])}" if reasons else "No signal"

        indicators = {"votes": votes, "weighted": {k: round(v, 2) for k, v in weighted.items()}}
        return StrategyResult(best_signal, best_conf, indicators, reason), results


if __name__ == "__main__":
    # Quick test
    print("=== Testing HL Strategy Engine ===\n")

    pairs = ["BTC", "ETH", "SOL", "HYPE", "AVAX", "DOGE"]

    for pair in pairs:
        candles = HLData.get_candles(pair, interval="1h", days=30)
        print(f"\n--- {pair} ({len(candles)} candles) ---")

        rsi_s = RSIStrategy()
        macd_s = MACDStrategy()
        bb_s = BollingerStrategy()
        trend_s = TrendFollowStrategy()

        for s in [rsi_s, macd_s, bb_s, trend_s]:
            result = s.compute(candles)
            print(f"  {s.name:30s} → {result.signal.value:5s} (conf: {result.confidence:.2f}) — {result.reason}")
