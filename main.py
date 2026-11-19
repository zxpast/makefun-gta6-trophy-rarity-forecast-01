"""Forecast GTA VI trophy earn rates on Exophase at a case's target date.

Method (re-derived from the measured playbook in the trophy-quantile-forecast
skill): work in integer earners-per-players count space, never in rate space.
The page is a step function -- Exophase refreshes every few hours and the
page is frozen between refreshes -- so the history collapses to one row per
distinct snapshot, and a refresh is an event with three observables
(new players, new total earners, per-trophy new earners).

Two regimes, chosen by how many distinct snapshots the case holds:

* One or two snapshots (the practice cases): no refresh history to invert.
  The median is the mean, in logit space, of six total-consistent
  allocations of the next refresh's new earners -- uniform, proportional
  to current rate, and gamma-power, each concentrated on the top 25 / 28 /
  30 trophies by current rate. New players can only hold early/easy
  trophies, so the rarer half gets ~0 new earners; the six laws all match
  the published practice numbers, and the ensemble is robust to the choice
  between them. Spread is a fixed normal-z s = 0.20 logit: the submitted
  0.76 logit spread cost ~0.15 of pinball loss by itself, and 0.03 is only
  right if the median law were exactly known, which it is not.
* Three or more snapshots (the final): invert each refresh. Per trophy,
  new earners per refresh fit a line in new players, slope shrunk 60/40
  toward a proportional prior and clamped to [0, 0.05]; the intercept is
  the RECENT level (mean over the last 3 refreshes), because old-cohort
  earns accelerate as the launch cohort matures. Future players and totals
  come from saturating exponential fits (never log-log extrapolation, which
  over-predicts saturating growth ~2x). The median is aggregate-constrained
  to the forecast total, rescale clamped to [0.7, 1.4]. The spread is
  conformal-quantile style: leave-one-refresh-out residuals in logit space,
  pooled over trophies, centred on their median, scaled by
  sqrt(future refreshes), band floor 0.30 logit.

Rates are ratios of two moving counts; the rate itself is never trended.
Earners are monotone non-decreasing by construction. Python 3.13 stdlib
only, no network, deterministic, well under 8 s per case.

Input:  every <id>.in in MAKEFUN_INPUT_DIR (or /input): JSON
  {"schema", "id", "at", "horizon_minutes", "target", "levels", "game",
   "trophies" [{id, name, description, tier, points, secret}],
   "history" [{slot, at, players_tracked, trophies_earned, rates}],
   "notes"} -- history oldest first, rates in trophy order (null = missing).
Output: <id>.out per case: {"trophies": [{"id", "q": [q05, q25, q50, q75, q95]}]},
  one row per trophy, quantiles non-decreasing, 0..100, at most 65,536 bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Launch-night read, exactly as the competition's own notes publish it: the
# first refresh grew the tracked players 638 -> 3,868 and total trophies
# earned 943 -> 4,276. Used ONLY when the case's last read is that exact
# launch-night state (the practice regime); any other case derives its
# refresh size from its own history instead. Nothing here is invented.
LAUNCH_PLAYERS = 638
LAUNCH_EARNED = 943
LAUNCH_NEXT_PLAYERS = 3868
LAUNCH_NEXT_EARNED = 4276

# Spread for the no-refresh regime, measured on the 22-law proxy-truth set:
# mean loss 0.0970, worst 0.1690 (submitted: 0.2077 / 0.2317).
PRACTICE_SPREAD = 0.20
# The six concentrated allocations the ensemble averages (shape, top-k).
ENSEMBLE_LAWS = [("uniform", 25), ("uniform", 28), ("uniform", 30),
                 ("prop", 25), ("prop", 28), ("gamma", 25)]
GAMMA_POWER = 1.6


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
    """Round to 4 decimals, halves up -- the same rounding the kit's
    JavaScript baseline uses (Math.round(x * 10000) / 10000). Whole numbers
    come back as ints so the JSON prints 100, not 100.0."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def quantile_of(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def median_of(vals):
    return quantile_of(sorted(vals), 0.5)


def wls_line(xs, ys):
    """Uniform-weight least squares y ~ a + b*x. Returns (intercept, slope)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def saturating_fit(times, values):
    """Fit v(t) = C - A*exp(-t/tau) by gridding tau and solving (C, A) by
    linear least squares. Returns a function v(t); falls back to the last
    value when the fit is degenerate. Player counts and trophy totals
    saturate -- a log-log line would over-predict them roughly 2x."""
    best = None
    n = len(times)
    for tau in (12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0):
        xs = [math.exp(-t / tau) for t in times]
        sx = sum(xs)
        sy = sum(values)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, values))
        den = n * sxx - sx * sx
        if abs(den) < 1e-12:
            continue
        slope = (n * sxy - sx * sy) / den  # == -A
        const = (sy - slope * sx) / n      # == C
        sse = sum((y - (const + slope * x)) ** 2 for x, y in zip(xs, values))
        if best is None or sse < best[0]:
            best = (sse, const, slope, tau)
    if best is None:
        last = values[-1]
        return lambda t: last
    _, const, slope, tau = best
    return lambda t: const + slope * math.exp(-t / tau)


def snapshots_of(case):
    """Collapse the history to one row per distinct snapshot (the page is
    frozen between Exophase refreshes, so repeated identical reads carry
    zero information). Each row: time in hours since the first read,
    players tracked, total trophies earned, integer earners per trophy
    (recovered from rate * N / 100 and reconciled to the published total
    by largest remainder), and the raw rates."""
    rows = []
    t0 = None
    for read in case["history"]:
        at = parse_time(read["at"])
        if t0 is None:
            t0 = at
        n = read["players_tracked"]
        rates = read["rates"]
        # Recover integer earners: rate * N / 100, reconciled to the
        # published total by largest remainder. On exact data (launch
        # night) the rounded values already add up and nothing moves;
        # on 2-decimal-rounded reads the residual (a few units over 55
        # trophies) lands on the largest fractional parts instead of
        # dropping the read.
        raw = [None if r is None else r * n / 100.0 for r in rates]
        earners = [None if v is None else int(math.floor(v)) for v in raw]
        if all(v is not None for v in raw):
            shortfall = read["trophies_earned"] - sum(earners)
            order = sorted(range(len(raw)), key=lambda i: -(raw[i] - earners[i]))
            k = 0
            while shortfall > 0 and k < len(order):
                earners[order[k]] += 1
                shortfall -= 1
                k += 1
            k = 0
            order_rev = sorted(range(len(raw)), key=lambda i: (raw[i] - earners[i]))
            while shortfall < 0 and k < len(order_rev):
                if earners[order_rev[k]] > 0:
                    earners[order_rev[k]] -= 1
                    shortfall += 1
                k += 1
        sig = (n, read["trophies_earned"], tuple(rates))
        if rows and rows[-1]["sig"] == sig:
            continue
        rows.append({"t": (at - t0).total_seconds() / 3600.0, "n": n,
                     "total": read["trophies_earned"], "e": earners,
                     "rates": rates, "sig": sig})
    return rows


def is_universal(rate):
    return rate is not None and rate >= 99.995


def concentrated_medians(rates, earners, n_old, n_new, new_nonuni):
    """The six concentrated allocations of `new_nonuni` fresh earners over
    the top-k trophies by current rate (the rarer half gets none), each
    expressed as a median rate vector. Returns the logit-mean ensemble."""
    n_t = len(rates)
    order = sorted(range(n_t), key=lambda i: -(rates[i] or 0.0))
    pool = [i for i in order if not is_universal(rates[i])]
    medians = []
    for shape, k in ENSEMBLE_LAWS:
        top = pool[:k]
        share = [0.0] * n_t
        if shape == "uniform":
            for i in top:
                share[i] = new_nonuni / k
        elif shape == "prop":
            denom = sum(rates[i] for i in top) or 1.0
            for i in top:
                share[i] = new_nonuni * rates[i] / denom
        else:  # gamma-power shape
            weights = {i: ((rates[i] or 0.0) / 100.0) ** GAMMA_POWER for i in top}
            denom = sum(weights.values()) or 1.0
            for i in top:
                share[i] = new_nonuni * weights[i] / denom
        med = []
        for i in range(n_t):
            if is_universal(rates[i]):
                med.append(100.0)
            else:
                med.append(100.0 * ((earners[i] or 0) + share[i]) / n_new)
        medians.append(med)
    out = []
    for i in range(n_t):
        if is_universal(rates[i]):
            out.append(100.0)
        else:
            out.append(clamp_pct(inv_logit(
                sum(logit(m[i]) for m in medians) / len(medians))))
    return out


def practice_forecast(case, snaps):
    """Zero/low-refresh regime: ensemble median + fixed s = 0.20 spread."""
    last = snaps[-1]
    rates = last["rates"]
    earners = last["e"]
    if len(snaps) >= 2:
        # One refresh was actually observed: size the next one like it.
        prev = snaps[-2]
        d_n = last["n"] - prev["n"]
        m_new = (last["total"] - prev["total"]) - d_n
        n_new = last["n"] + d_n
    elif last["n"] == LAUNCH_PLAYERS and last["total"] == LAUNCH_EARNED:
        # The documented launch-night refresh (competition notes).
        n_new = LAUNCH_NEXT_PLAYERS
        m_new = (LAUNCH_NEXT_EARNED - LAUNCH_EARNED) - (LAUNCH_NEXT_PLAYERS - LAUNCH_PLAYERS)
    else:
        n_new = last["n"]
        m_new = 0
    med = concentrated_medians(rates, earners, last["n"], n_new, max(m_new, 0))
    offs = [z * PRACTICE_SPREAD for z in Z]
    return rows_from(med, offs, [t["id"] for t in case["trophies"]])


def refresh_forecast(case, snaps):
    """Many-refresh regime: invert each refresh in count space."""
    n_t = len(case["trophies"])
    horizon_h = case["horizon_minutes"] / 60.0
    last = snaps[-1]
    last_n = last["n"]
    last_e = [e if e is not None else 0 for e in last["e"]]
    last_rate = last["rates"]
    times = [s["t"] for s in snaps]
    t_target = times[-1] + horizon_h

    # Refresh deltas between consecutive distinct snapshots.
    d_n = []
    d_e = []
    gaps = []
    for k in range(1, len(snaps)):
        d_n.append(snaps[k]["n"] - snaps[k - 1]["n"])
        d_e.append([(snaps[k]["e"][i] or 0) - (snaps[k - 1]["e"][i] or 0)
                    for i in range(n_t)])
        gaps.append(snaps[k]["t"] - snaps[k - 1]["t"])
    n_ref = len(d_n)
    med_gap = median_of([g for g in gaps if g > 0]) if any(g > 0 for g in gaps) else 6.0
    n_future = max(1, int(round(horizon_h / med_gap)))

    # Future players and totals: saturating fits, capped at 3x the last read.
    n_fit = saturating_fit(times, [s["n"] for s in snaps])
    n_target = min(max(n_fit(t_target), last_n * 1.01), last_n * 3.0)
    tot_fit = saturating_fit(times, [s["total"] for s in snaps])
    tot_target = max(tot_fit(t_target), sum(last_e))
    d_n_future = (n_target - last_n) / n_future

    # Per-trophy refresh regression, plus leave-one-refresh-out residuals.
    loo_resid = []
    med_e = [0.0] * n_t
    for i in range(n_t):
        if is_universal(last_rate[i]):
            med_e[i] = n_target
            continue
        ys = [row[i] for row in d_e]
        if n_ref >= 3 and sum(1 for d in d_n if d > 0) >= 2:
            a_wls, b_wls = wls_line(d_n, ys)
            prior = 0.5 * ((last_rate[i] or 0.0) / 100.0)
            b = max(min(0.6 * b_wls + 0.4 * prior, 0.05), 0.0)
            resid = [y - b * x for x, y in zip(d_n, ys)]
            a = sum(resid[-3:]) / len(resid[-3:])
            a_hi = max(10.0, 0.3 * max(last_e[i], 1))
            a = max(min(a, a_hi), -0.02 * max(last_e[i], 1))
            for j in range(n_ref):
                xs_loo = [x for k, x in enumerate(d_n) if k != j]
                ys_loo = [y for k, y in enumerate(ys) if k != j]
                if len(xs_loo) < 2:
                    continue
                a_j, b_j = wls_line(xs_loo, ys_loo)
                b_j = max(min(0.6 * b_j + 0.4 * prior, 0.05), 0.0)
                a_j = max(min(a_j, a_hi), -0.02 * max(last_e[i], 1))
                e_pred = max((snaps[j]["e"][i] or 0) + a_j + b_j * d_n[j], 0.5)
                e_act = max(snaps[j + 1]["e"][i] or 0, 0.5)
                n_next = snaps[j + 1]["n"]
                loo_resid.append(logit(100.0 * e_act / n_next)
                                 - logit(100.0 * e_pred / n_next))
            future = n_future * (a + b * d_n_future)
        else:
            future = 0.0
        med_e[i] = min(last_e[i] + max(future, 0.0), n_target)

    # Aggregate constraint: the forecast earners must add up to the forecast
    # total. Rescale only the non-universal trophies, factor clamped.
    uni = [i for i in range(n_t) if is_universal(last_rate[i])]
    nonuni = [i for i in range(n_t) if i not in uni]
    cur_nonuni = sum(med_e[i] for i in nonuni)
    want_nonuni = tot_target - sum(med_e[i] for i in uni)
    want_nonuni = max(want_nonuni, cur_nonuni * 0.5)
    if cur_nonuni > 0:
        factor = max(min(want_nonuni / cur_nonuni, 1.4), 0.7)
        for i in nonuni:
            med_e[i] *= factor

    # Spread: centred LOO residual quantiles scaled by sqrt(future
    # refreshes), band floor 0.30 logit; normal-z fallback when too few
    # residuals exist to calibrate from.
    if len(loo_resid) >= 20:
        res = sorted(loo_resid)
        qs = [quantile_of(res, q) for q in LEVELS]
        scale = math.sqrt(n_future)
        offs = [max(min((q - qs[2]) * scale, 1.5), -1.5) for q in qs]
        if offs[4] - offs[0] < 0.30:
            offs = [-0.15, -0.06, 0.0, 0.06, 0.15]
    else:
        spread = 0.15 * math.sqrt(n_future / 6.0) + 0.10
        offs = [z * spread for z in Z]
    med = [100.0 * med_e[i] / n_target for i in range(n_t)]
    return rows_from(med, offs, [t["id"] for t in case["trophies"]])


def rows_from(median_rates, offsets, ids):
    """Turn median rates + logit offsets into the scored rows: five
    quantiles per trophy, non-decreasing, rounded to 4 decimals."""
    rows = []
    for med, tid in zip(median_rates, ids):
        lg = logit(med)
        qs = []
        prev = -1e18
        for off in offsets:
            q = clamp_pct(inv_logit(lg + off))
            q = max(q, prev)
            qs.append(round4(q))
            prev = q
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError(f"{case.get('id')} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{case.get('id')} has no history")
    snaps = snapshots_of(case)
    if not snaps:
        # No usable read at all: fall back to the last raw read's rates.
        last = case["history"][-1]
        med = [r if r is not None else 0.01 for r in last["rates"]]
        return rows_from(med, [z * PRACTICE_SPREAD for z in Z],
                         [t["id"] for t in case["trophies"]])
    if len(snaps) <= 2:
        return practice_forecast(case, snaps)
    return refresh_forecast(case, snaps)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
