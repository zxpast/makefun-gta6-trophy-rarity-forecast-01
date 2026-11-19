# gta6-trophy-rarity-forecast-01 — recalibrated forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zxpast/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zxpast/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PS5 trophy's Exophase earn rate at
2026-11-26T00:00:00Z as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95),
scored by mean pinball loss on the logit scale.

## What changed vs the submitted round (loss 0.22, rank 1136/4349)

The old program forecast the last-read rate with a 0.76-logit spread.
That spread alone cost ~0.15 of pinball loss: rank 1's 0.00598 implies a
spread of ~0.03 logit with a near-exact median, and our median was already
good (typical miss 0.04 pp, ~rank 103 in that column). This rebuild fixes
the median law and earns the spread from data instead of setting it from
caution.

## Method

All work happens in integer earners / players count space; the rate is
only ever a ratio at the end, and is never trended.

- **Snapshots.** The history collapses to one row per distinct snapshot
  (the page is frozen between Exophase refreshes). Earners are recovered
  as rate * N / 100 and reconciled to the published total by largest
  remainder.
- **Few snapshots (practice):** the median is the logit-mean of six
  total-consistent allocations of the next refresh's new earners --
  uniform / proportional-to-rate / gamma-power, each concentrated on the
  top 25 / 28 / 30 trophies by current rate (new players can only hold
  early trophies; the rarer half gets ~0). Spread is a fixed normal-z
  s = 0.20 logit.
- **Many snapshots (the final):** each refresh is inverted per trophy
  (new earners ~ intercept + slope * new players; slope shrunk 60/40
  toward a proportional prior, intercept taken from the last 3 refreshes
  because old-cohort earns accelerate). Future players and totals come
  from saturating exponential fits, never log-log extrapolation. The
  median is aggregate-constrained to the forecast total. The spread is
  conformal-style: leave-one-refresh-out residuals in logit space, pooled,
  centred on their median, scaled by sqrt(future refreshes).

## Validation (measured first-hand, 2026-11-20)

| Gate | Result |
|---|---|
| A — practice proxy-truth sweep (22 laws, p00/p01) | mean 0.0959, worst 0.1679, beats submitted 22/22 (submitted 0.2077 / 0.2317) — PASS |
| B — synthetic final-like cases (4 seeds) | mean 0.8419 vs submitted 0.9178 — beats submitted; 90% coverage 0.25 (spread calibration on synthetic finals remains unproven, as in the playbook) |
| C — rolling-origin backtest (29 origins) | mean loss 0.2751, coverage 0.84; loss does not rise with origin index (no flip mis-specification) |

Determinism: two runs on the kit practice input are byte-identical.
Runtime: ~0.13 s per case (limit 8 s). Python 3.13 stdlib only, no network.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in`, writes one `*.out` per case:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=6hmpy1rq&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
