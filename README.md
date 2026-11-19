# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zxpast/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zxpast/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's Exophase earn
rate at the case's target, as five quantiles per trophy, scored by pinball
loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` in the input directory and writes `<id>.out` per case. Two
regimes, decided by what the history shows:

- **Trend regime** — the history contains real movement. Exophase
  refreshes a game's counts in jumps, so the page is a step function
  frozen between refreshes; each distinct read is one refresh point.
  Per trophy, a weighted least-squares line of logit(rate) against
  ln(1 + hours since release) is fitted over the distinct reads, the
  slope is shrunk toward a pooled level-dependent slope, the move is
  damped by 0.7 (rates saturate — a log-time line overstates the move
  to a target far beyond the history), and the line is read at the
  target. Spread combines a random-walk term in sqrt(horizon) with the
  fit's residual.
- **Flat regime** — every read is identical (no refresh seen yet). The
  next refresh re-bases rates on a much larger player count, modelled
  as a new cohort joining: players_tracked grows by the first-refresh
  factor the house recorded on launch night (638 → 3,868, ~6.06x), and
  a new player holds trophy *i* with probability p_i = r_i^1.6. That
  exponent is pinned by the same refresh's totals — it reproduces the
  recorded post-refresh trophies-earned count (4,276) to within 0.2%.
  New players hold almost only the near-universal intro trophy;
  everything else dilutes roughly four- to five-fold.

Quantiles are the median's logit plus/minus the standard-normal
quantiles times the spread. Deterministic: no randomness, fixed file
order, two runs are byte-identical.

## Practice results

On the practice cases (p00/p01, flat launch-night history, answer is the
first refresh), the implied total trophies earned at the target is
4,263 vs the recorded 4,276 (0.3% off). Against proxy answers spanning
the dilution exponent γ ∈ [1.3, 2.5], pinball loss is 0.18–0.30 with
90% coverage 0.98, vs the house persistence baseline's 0.90–1.69 with
coverage 0.00. On two synthetic final-like cases (two days of reads
with refreshes, 4.5-day horizon), loss is 0.23–0.24 vs the baseline's
0.45–0.52.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runtime is ~0.1 s per case, far under the 8 s cap.

## License

MIT — see `LICENSE`.
