# Paper Trading Lessons Learned — Multi-Model Review (Jul 1 2026)

## Context
3-model review (Quant + Risk + Signal Quality) after 4 trades lost -$277.82 (5.6% of $5K).
Kelly criterion = -31%. System had zero edge.

## Root Causes

1. **Inverted R:R** — Winners exited on signal reversal at +$2.23 while losers ran to stop at -$93 avg. Intended 2:1 R:R was actually 0.024:1.

2. **Competing optimizers** — `optimizer.py` (30d window, no hysteresis) fought `evaluator.py` (90d window, 5pt hysteresis). Flipped ETH 4 times in 2 hours on Jun 30.

3. **Sentiment overlay with bad data** — Single-source fleet sentiment (known bugs in scorer) overrode TA signals. 3/4 trades were sentiment-driven, all losers.

4. **Stops too tight for leverage** — BTC short stop at 1.27% with 3x leverage = noise trigger. Crypto hourly ATR routinely exceeds 1.27%.

5. **HYPE: trading destroys value** — Best strategy returned +12.3% vs buy-and-hold +80.8%. Every active trade = opportunity cost.

6. **Fees eat marginal wins** — Only winning trade ($2.23) had $3.29 in fees = net loss of -$1.06.

## Fixes Applied

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| Risk per trade | 2% | 1% | Kelly says less |
| Max leverage | 3x | 1.5x | Survive low WR |
| Stop loss ATR | 2.5 | 3.5 | Wider stops |
| Take profit ATR | 5.0 | 7.0 | R:R now 2:1 |
| Min confidence | 0.5 | 0.65 | Fewer bad entries |
| Max positions | 6 | 4 | Less correlation |
| Min hold hours | 3 | 6 | Let trades breathe |
| Reentry cooldown | 4h | 8h | Less overtrading |
| Sentiment threshold | 0.3 | 0.5 | Only strong signals |
| Sentiment min confidence | 0.4 | 0.6 | Higher bar |
| Sentiment min sources | 1 (any) | 2 | Diverse sources |
| Max drawdown | None | 10% | Circuit breaker |
| Optimizer cron | Active | Paused | Evaluator is sole authority |

## Rules Going Forward

1. **HYPE stays in portfolio** — we need to LEARN how to trade it profitably, not avoid it
2. **Evaluator is sole strategy authority** — optimizer.py stays paused until it proves it can respect hysteresis
3. **Multi-model review** — any system losing money gets 3+ model review before changes
4. **Kelly criterion gates** — don't risk capital when Kelly is negative
5. **Cross-model scoring** — eval/harness crons use different model than the one that built the system
