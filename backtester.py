"""
HL Strategy Lab — Backtester
Runs strategies against historical HL data with realistic fees, slippage, and risk management.
Outputs detailed performance metrics.
"""
import time
import requests
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from strategy_engine import (
    Signal, HLData, RSIStrategy, MACDStrategy, BollingerStrategy, TrendFollowStrategy,
    MultiStrategyVoter, create_strategy, STRATEGIES,
    sma, ema, calc_rsi, calc_macd, calc_bollinger, calc_atr
)

# ─── Config ───

HL_FEE = 0.00035       # 0.035% taker fee on Hyperliquid
SLIPPAGE = 0.0005      # 0.05% assumed slippage
FUNDING_COST = 0.0001  # 0.01% per 8h (conservative avg for shorts)

# ─── Trade & Position ───

@dataclass
class Trade:
    entry_time: float
    entry_price: float
    exit_time: float
    exit_price: float
    direction: str  # "LONG" or "SHORT"
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # "signal", "stop_loss", "take_profit", "end_of_data"
    indicators: dict = field(default_factory=dict)

@dataclass
class BacktestResult:
    strategy_name: str
    coin: str
    interval: str
    start_price: float
    end_price: float
    buy_hold_return: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    equity_curve: list
    trades: list
    duration_hours: float

    def summary(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════╗
║  {self.strategy_name} — {self.coin}
╠══════════════════════════════════════════════════════════╣
║  Period: {self.duration_hours:.0f}h ({len(self.equity_curve)} bars)
║  Buy & Hold: {self.buy_hold_return:+.2f}%
║  ────────────────────────────────────────────────────────
║  Total P&L:    ${self.total_pnl:+,.2f} ({self.total_pnl_pct:+.2f}%)
║  Trades:       {self.total_trades} ({self.wins}W / {self.losses}L)
║  Win Rate:     {self.win_rate:.1f}%
║  Avg Win:      +${self.avg_win:,.2f}
║  Avg Loss:     -${abs(self.avg_loss):,.2f}
║  Profit Factor: {self.profit_factor:.2f}
║  Max Drawdown: {self.max_drawdown:.2f}%
║  Sharpe:       {self.sharpe_ratio:.2f}
╚══════════════════════════════════════════════════════════╝"""


# ─── Backtester ───

class Backtester:
    def __init__(
        self,
        initial_capital=1000.0,
        risk_per_trade=0.02,      # 2% of equity per trade
        max_leverage=3,
        stop_loss_atr=2.5,        # SL = 2.5x ATR
        take_profit_atr=5.0,      # TP = 5x ATR
        allow_shorts=True,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.allow_shorts = allow_shorts

    def run(self, strategy, candles, coin="?", interval="1h"):
        """
        Run a backtest on historical candles.
        Strategy must have a .compute(candles) method that returns (StrategyResult, [details]) or StrategyResult.
        """
        if len(candles) < 60:
            raise ValueError(f"Need at least 60 candles, got {len(candles)}")

        o, h, l, c, v = HLData.candles_to_arrays(candles)
        atr = calc_atr(h, l, c, 14)

        equity = self.initial_capital
        equity_curve = []
        trades = []

        position = None  # {"direction", "entry_price", "size", "stop_loss", "take_profit", "entry_idx"}

        for i in range(60, len(candles)):
            # Check stop loss / take profit FIRST
            if position:
                pos_candles = candles[:i+1]
                result = self._check_exit(position, h[i], l[i], c[i], i, atr[i])
                if result:
                    direction, entry, exit_price, reason = result
                    cost = self._trade_cost(position["size"], entry, exit_price)
                    if direction == "LONG":
                        raw_pnl = (exit_price - entry) * position["size"]
                    else:
                        raw_pnl = (entry - exit_price) * position["size"]
                    pnl = raw_pnl - cost
                    equity += pnl
                    trade = Trade(
                        entry_time=candles[position["entry_idx"]]["t"],
                        entry_price=entry,
                        exit_time=candles[i]["t"],
                        exit_price=exit_price,
                        direction=direction,
                        size=position["size"],
                        pnl=round(pnl, 4),
                        pnl_pct=round(pnl / (entry * position["size"]) * 100, 2),
                        exit_reason=reason,
                    )
                    trades.append(asdict(trade))
                    position = None

            # Check for new entry
            if position is None:
                slice_candles = candles[:i+1]
                ret = strategy.compute(slice_candles)

                # Handle both (result, details) and just result
                if isinstance(ret, tuple):
                    result = ret[0]
                else:
                    result = ret

                if result.signal in (Signal.LONG, Signal.SHORT) and result.confidence > 0.3:
                    if result.signal == Signal.SHORT and not self.allow_shorts:
                        continue

                    price = c[i]
                    current_atr = atr[i] if not np.isnan(atr[i]) else price * 0.02

                    # Position sizing: risk-based
                    risk_amount = equity * self.risk_per_trade
                    stop_distance = current_atr * self.stop_loss_atr
                    if stop_distance <= 0:
                        continue
                    position_size = risk_amount / stop_distance

                    # Cap position notional at equity * leverage
                    max_notional = equity * self.max_leverage
                    if position_size * price > max_notional:
                        position_size = max_notional / price

                    direction = "LONG" if result.signal == Signal.LONG else "SHORT"

                    if direction == "LONG":
                        stop_loss = price - current_atr * self.stop_loss_atr
                        take_profit = price + current_atr * self.take_profit_atr
                    else:
                        stop_loss = price + current_atr * self.stop_loss_atr
                        take_profit = price - current_atr * self.take_profit_atr

                    position = {
                        "direction": direction,
                        "entry_price": price,
                        "size": position_size,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "entry_idx": i,
                    }

            equity_curve.append(round(equity, 2))

        # Close any open position at last price
        if position:
            last_price = c[-1]
            entry = position["entry_price"]
            if position["direction"] == "LONG":
                raw_pnl = (last_price - entry) * position["size"]
            else:
                raw_pnl = (entry - last_price) * position["size"]
            cost = self._trade_cost(position["size"], entry, last_price)
            pnl = raw_pnl - cost
            equity += pnl
            equity_curve.append(round(equity, 2))
            trades.append(asdict(Trade(
                entry_time=candles[position["entry_idx"]]["t"],
                entry_price=entry,
                exit_time=candles[-1]["t"],
                exit_price=last_price,
                direction=position["direction"],
                size=position["size"],
                pnl=round(pnl, 4),
                pnl_pct=round(pnl / (entry * position["size"]) * 100, 2),
                exit_reason="end_of_data",
            )))

        # ─── Metrics ───
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        buy_hold = ((c[-1] - c[60]) / c[60]) * 100

        # Max drawdown
        eq = np.array(equity_curve)
        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq - running_max) / running_max * 100
        max_dd = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

        # Sharpe (simplified — per-bar returns)
        if len(eq) > 1:
            returns = np.diff(eq) / eq[:-1]
            if np.std(returns) > 0:
                bars_per_year = {"1m": 525600, "5m": 105120, "15m": 35040,
                                 "1h": 8760, "4h": 2190, "1d": 365}
                bpy = bars_per_year.get(interval, 8760)
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(bpy)
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        duration_h = (candles[-1]["t"] - candles[60]["t"]) / 3600000

        return BacktestResult(
            strategy_name=strategy.name if hasattr(strategy, 'name') else type(strategy).__name__,
            coin=coin,
            interval=interval,
            start_price=c[60],
            end_price=c[-1],
            buy_hold_return=round(buy_hold, 2),
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=round(len(wins) / len(trades) * 100, 1) if trades else 0,
            avg_win=round(np.mean([t["pnl"] for t in wins]), 2) if wins else 0,
            avg_loss=round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl / self.initial_capital * 100, 2),
            max_drawdown=round(max_dd, 2),
            sharpe_ratio=round(float(sharpe), 2),
            profit_factor=round(float(profit_factor), 2),
            equity_curve=equity_curve,
            trades=trades,
            duration_hours=duration_h,
        )

    def _check_exit(self, position, current_high, current_low, current_close, idx, current_atr):
        """Check if position should be exited. Returns (direction, entry, exit_price, reason) or None."""
        direction = position["direction"]
        entry = position["entry_price"]
        sl = position["stop_loss"]
        tp = position["take_profit"]

        if direction == "LONG":
            if current_low <= sl:
                return direction, entry, sl, "stop_loss"
            if current_high >= tp:
                return direction, entry, tp, "take_profit"
        else:  # SHORT
            if current_high >= sl:
                return direction, entry, sl, "stop_loss"
            if current_low <= tp:
                return direction, entry, tp, "take_profit"
        return None

    def _trade_cost(self, size, entry_price, exit_price):
        """Estimate total cost: fees + slippage for entry + exit."""
        notional_entry = size * entry_price
        notional_exit = size * exit_price
        fees = (notional_entry + notional_exit) * HL_FEE
        slip = (notional_entry + notional_exit) * SLIPPAGE
        return fees + slip


# ─── Multi-asset Backtest Runner ───

def run_backtest_matrix(
    pairs=["BTC", "ETH", "SOL", "HYPE", "AVAX", "DOGE"],
    strategies=None,
    interval="1h",
    days=90,
    **backtest_kwargs
):
    """Run all strategies across all pairs. Returns dict of results."""
    if strategies is None:
        strategies = [
            ("RSI", RSIStrategy()),
            ("MACD", MACDStrategy()),
            ("Bollinger", BollingerStrategy()),
            ("Trend", TrendFollowStrategy()),
        ]

    bt = Backtester(**backtest_kwargs)
    results = {}

    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"  Fetching {pair} ({days}d, {interval})...")
        candles = HLData.get_candles(pair, interval=interval, days=days)
        results[pair] = {}

        for name, strat in strategies:
            print(f"  → Backtesting {name}...")
            result = bt.run(strat, candles, coin=pair, interval=interval)
            results[pair][name] = result
            print(result.summary())

    return results


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  HL Strategy Lab — BACKTEST ENGINE v1.0                     ║")
    print("║  6 Pairs × 4 Strategies × 90 Days Historical Data           ║")
    print("╚══════════════════════════════HL STRATEGY LAB — BACKTEST ENGINE v1.0 ===")
    print("6 Pairs × 4 Strategies × 90 Days Historical Data\n")

    # Run backtests
    results = run_backtest_matrix(
        pairs=["BTC", "ETH", "SOL", "HYPE", "AVAX", "DOGE"],
        interval="1h",
        days=90,
        initial_capital=1000,
        risk_per_trade=0.02,
        max_leverage=3,
        stop_loss_atr=2.5,
        take_profit_atr=5.0,
        allow_shorts=True,
    )

    # Save results to JSON
    output = {}
    for pair, strats in results.items():
        output[pair] = {}
        for sname, result in strats.items():
            r = asdict(result)
            output[pair][sname] = {
                "strategy": r["strategy_name"],
                "coin": r["coin"],
                "total_pnl": r["total_pnl"],
                "total_pnl_pct": r["total_pnl_pct"],
                "buy_hold": r["buy_hold_return"],
                "trades": r["total_trades"],
                "win_rate": r["win_rate"],
                "max_drawdown": r["max_drawdown"],
                "sharpe": r["sharpe_ratio"],
                "profit_factor": r["profit_factor"],
                "equity_curve": r["equity_curve"],
            }

    out_path = "/Users/mojoai/Projects/hl-strategy-lab/backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")
