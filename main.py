"""GTA VI trophy-rarity forecaster (Python 3.13, standard library only).

Forecasts every trophy's Exophase earn rate at a case's target as five
quantiles, from the house's reads of the game's trophy page before the
case's checkpoint. Two regimes, decided by what the history shows:

1. TREND REGIME -- the history contains real movement (distinct reads).
   Exophase refreshes a game's counts in jumps, so the page is a step
   function frozen between refreshes. Each distinct read is one refresh
   point. Per trophy, fit a weighted least-squares line of logit(rate)
   against ln(1 + hours since release) over the distinct reads, shrink
   the slope toward a pooled slope that depends on the trophy's current
   level (rare trophies drift differently from common ones), and read
   the line at the target. The median is that line's value; the spread
   combines a random-walk term in sqrt(horizon) with the fit's residual.

2. FLAT REGIME -- every read in the history is identical (no refresh
   yet seen). A flat page does not mean the rates will stay flat: the
   next refresh re-bases them on a much larger player count. Model the
   refresh as a new cohort joining: players_tracked grows by the
   first-refresh factor the house itself recorded on launch night
   (638 -> 3,868, about 6.06x; see the kit's competition notes), and a
   new player holds trophy i with probability p_i = r_i ** 1.6, where
   r_i is the trophy's current rate as a fraction. That exponent is
   pinned by the same refresh's totals: it reproduces the recorded
   post-refresh trophies-earned count (4,276) to within 0.2%, where a
   uniform dilution (exponent 1) overshoots it by 12%. New players
   hold almost only the near-universal intro trophy; everything else
   dilutes roughly four- to five-fold.

Spread is honest on purpose: the practice cases show a single refresh
moving every rate outside a narrow band, so the flat regime carries a
wide refresh term. Quantiles are the median's logit plus/minus the
standard-normal quantiles times the spread, rounded to 4 decimals.

Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
reads every <id>.in in the input directory and writes <id>.out per
case. Deterministic: no randomness anywhere, fixed file order.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Random-walk spread, logit units per square-root day (the house baseline's).
RW_SIGMA = 0.5
# Flat-regime refresh uncertainty, logit units.
FLAT_SPREAD = 0.75
# New-cohort dilution exponent, calibrated on the launch-night refresh.
DILUTION_GAMMA = 1.6
# First-refresh growth factor for players_tracked, from the kit's notes.
REFRESH_GROWTH = 3868.0 / 638.0
# How far a fitted trend may move a rate, in logit units, over the horizon.
MAX_LOGIT_MOVE = 3.0
# Damping on the extrapolated trend: rates saturate (a trophy cannot
# pass 100%, and early trophies approach their plateau), so a line
# fitted in log-time overstates the move to a target far beyond the
# history. 0.7 keeps seventy percent of the fitted move.
TREND_DAMP = 0.7
# Recency half-life for the trend fit, in hours.
HALF_LIFE_HOURS = 48.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since(release, moment):
    h = (moment - release).total_seconds() / 3600.0
    return max(h, -0.99)


def distinct_reads(history):
    """One entry per distinct rate vector, in order, always keeping the last read.

    The page is frozen between Exophase refreshes, so repeated identical
    reads carry no new information and would overweight the plateau.
    """
    out = []
    seen = set()
    for read in history:
        key = tuple(read["rates"])
        if key not in seen:
            seen.add(key)
            out.append(read)
    if history and out[-1] is not history[-1]:
        out.append(history[-1])
    return out


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_sd, n)."""
    n = len(xs)
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, n
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        rss = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        resid = math.sqrt(rss / sw * n / (n - 2))
    else:
        resid = 0.0
    return a, b, resid, n


def pooled_slope(fits, logits_last):
    """Median slope, plus a level-dependent slope from regressing slopes
    on current logits. Returns a function level -> pooled slope."""
    slopes = [f[1] for f in fits if f is not None]
    if not slopes:
        return lambda _level: 0.0
    med = sorted(slopes)[len(slopes) // 2]
    if len(fits) < 5:
        return lambda _level: med
    xs = [lg for lg, f in zip(logits_last, fits) if f is not None]
    ys = [f[1] for f in fits if f is not None]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return lambda _level: med
    coef = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    base = my - coef * mx
    lo, hi = min(slopes), max(slopes)
    return lambda level: min(hi, max(lo, base + coef * level))


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    history = case["history"]
    if not history:
        raise ValueError("no history")
    trophies = case["trophies"]
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(hours_since(release, target))
    x_last = math.log1p(hours_since(release, checkpoint))

    reads = distinct_reads(history)
    last = history[-1]
    n_last = last["players_tracked"]
    flat = len(reads) <= 2

    # Per-trophy series over the distinct reads.
    series = []
    for i in range(len(trophies)):
        pts = []
        for read in reads:
            rate = read["rates"][i]
            if rate is None:
                continue
            t = hours_since(release, parse_time(read["at"]))
            pts.append((math.log1p(t), logit(rate), t))
        series.append(pts)

    # Per-trophy trend fits, where identifiable.
    fits = []
    logits_last = []
    for i, pts in enumerate(series):
        last_rate = last["rates"][i]
        if last_rate is None:
            prev = None
            for r in reversed(history):
                if r["rates"][i] is not None:
                    prev = r
                    break
            last_rate = prev["rates"][i] if prev else 1.0
        logits_last.append(logit(last_rate))
        if len(pts) >= 3 and max(p for _, p, _ in pts) - min(p for _, p, _ in pts) > 1e-9:
            t_last = pts[-1][2]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ws = [0.5 ** ((t_last - p[2]) / HALF_LIFE_HOURS) for p in pts]
            fits.append(weighted_line(xs, ys, ws))
        else:
            fits.append(None)
    pool = pooled_slope(fits, logits_last)

    rw = RW_SIGMA * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 1.0
        med_logit, spread = median_and_spread(
            last_rate, series[i], fits[i], pool, logits_last[i],
            x_target, x_last, rw, flat, n_last,
        )
        qs = []
        for z in Z:
            q = round4(clamp_pct(inv_logit(med_logit + z * spread)))
            qs.append(q)
        # Rounding can only equalise neighbours, never invert them.
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def median_and_spread(last_rate, pts, fit, pool, level_logit,
                      x_target, x_last, rw, flat, n_last):
    """Median logit and spread (logit units) for one trophy."""
    if last_rate >= 99.995:
        # A trophy every tracked player holds: it stays near-universal.
        return logit(99.9), 0.8

    if flat:
        # No refresh seen yet: the next refresh re-bases on a larger
        # player count. New-cohort dilution, calibrated on launch night.
        n_target = n_last * REFRESH_GROWTH
        earned = last_rate / 100.0 * n_last
        p_new = (last_rate / 100.0) ** DILUTION_GAMMA
        earned_new = earned + (n_target - n_last) * p_new
        median = earned_new / n_target * 100.0
        return logit(median), math.sqrt(rw * rw + FLAT_SPREAD ** 2)

    if fit is not None:
        a, b, resid, _n = fit
        pooled = pool(level_logit)
        b_shrunk = 0.65 * b + 0.35 * pooled
        # Anchor the shrunk line at the last read so a noisy intercept
        # cannot drag the median away from what the page actually shows.
        med = level_logit + TREND_DAMP * b_shrunk * (x_target - x_last)
        delta = med - level_logit
        if delta > MAX_LOGIT_MOVE:
            med = level_logit + MAX_LOGIT_MOVE
        elif delta < -MAX_LOGIT_MOVE:
            med = level_logit - MAX_LOGIT_MOVE
        resid = min(max(resid, 0.15), 0.8)
        spread = math.sqrt(rw * rw + (1.5 * resid) ** 2 + 0.01)
        return med, spread

    # Trophy itself never moved, but the game did: follow the pooled
    # level-dependent slope from its last rate.
    b_pool = pool(level_logit)
    med = level_logit + TREND_DAMP * b_pool * (x_target - x_last)
    delta = med - level_logit
    if delta > MAX_LOGIT_MOVE:
        med = level_logit + MAX_LOGIT_MOVE
    elif delta < -MAX_LOGIT_MOVE:
        med = level_logit - MAX_LOGIT_MOVE
    return med, math.sqrt(rw * rw + 0.25 + 0.01)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
