#!/usr/bin/env python3
"""gta6-trophy-rarity-forecast-01: forecast Exophase trophy earn rates at target.

Approach (my own implementation; standard library only, deterministic).

What the data looks like: Exophase re-reads its trophy page on a fixed clock,
but the page itself only refreshes every few hours, so long runs of reads are
byte-identical plateaus. A trophy's earn rate is pulled in two directions at
once -- existing players keep earning it (pushes the rate up) while fresh
players keep joining with almost no trophies (dilutes the rate down). The
net drift is what we have to forecast, a week out, from at most two days of
history.

Model, one trophy at a time, using only the case file:
  * Collapse the history to distinct snapshots: a snapshot is new only when
    (players_tracked, rates) changes. Plateaus carry no information.
  * Work on the logit scale, against log time since release:
        y = logit(rate),  x = ln(1 + hours since 2026-11-19T00:00Z).
    Fit an ordinary least-squares line per trophy.
  * Shrink each slope toward the cross-trophy median slope with an
    empirical-Bayes weight: a trophy whose own slope is precisely estimated
    keeps it, one whose slope is noise gets pulled to the median. The
    median is the shared dilution factor -- most trophies are diluted by
    the same flood of new players, so the median slope is the best prior
    for any single trophy.
  * Refit the intercept so the shrunken line still passes through the
    trophy's own data centre, then evaluate it at the target time.
  * Spread: the regression's prediction standard error (residual scatter
    times leverage, so it grows when the target is far outside the
    history's time range) combined in quadrature with the house baseline's
    random-walk term, 0.5 * sqrt(horizon in days). Floored at 0.35 and
    capped at 2.5 logit, so the interval stays honest -- wide when the fit
    is shaky or the horizon is long, narrow when the trophy's curve is
    clean.
  * Flat history (a single distinct snapshot, as on launch night): the
    fitted trend is zero, but a refresh is overdue. The next Exophase
    refresh adds a flood of new players who hold almost no trophies. The
    launch-night refresh jumped tracked players 638 -> 3,868 and rates
    fell ~4.5x across the board, i.e. about -1.5 in logit terms for the
    small rates involved. So the median is shifted down by 1.5 logit and
    the spread is floored at 0.9 (a refresh is a big, lumpy event).
  * A trophy already at ~100% stays ~100%: every new player earns it the
    moment they start the game, so dilution cannot touch it.

Output: {"trophies": [{"id": ..., "q": [q05, q25, q50, q75, q95]}, ...]},
quantiles rounded to 4 decimals half-up (as the house's JavaScript rounds),
non-decreasing, each in [0, 100].

Run by the grader as `python3 -P -s /bot/main.py` in python:3.13-slim.
Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for
/input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Random-walk spread of the house baseline, logit per sqrt(day).
BASE_SIGMA = 0.5
# Spread floor / cap, in logit.
SPREAD_FLOOR = 0.35
SPREAD_CAP = 2.5
# Flat-history dilution shift (logit) and spread floor, see module docstring.
FLAT_SHIFT = -1.5
FLAT_SPREAD_FLOOR = 0.9


def clamp_rate(pct):
    """Clamp a percent into (0, 100) so the logit stays finite."""
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def pct_of(logit_value):
    """Inverse logit, as a percent in [0, 100]."""
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round_half_up_4(x):
    """Round to 4 decimals, half up, matching JS Math.round(x*10000)/10000."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_hours(moment):
    """ln(1 + hours since release) -- the time axis of the trend."""
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return math.log(1.0 + max(hours, 0.0))


def distinct_reads(history):
    """Collapse consecutive identical reads to distinct snapshots."""
    snapshots = []
    previous_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read["rates"]))
        if key != previous_key:
            snapshots.append(read)
            previous_key = key
    return snapshots


def fit_line(points):
    """OLS line y = a + b*x for [(x, y), ...].

    Returns (slope, intercept, resid_sd, slope_se, x_mean, sxx), or None
    when the points cannot define a line.
    """
    n = len(points)
    if n < 2:
        return None
    x_mean = sum(p[0] for p in points) / n
    y_mean = sum(p[1] for p in points) / n
    sxx = sum((p[0] - x_mean) ** 2 for p in points)
    if sxx <= 1e-12:
        return None
    sxy = sum((p[0] - x_mean) * (p[1] - y_mean) for p in points)
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    if n > 2:
        dof = n - 2
        resid_sd = math.sqrt(
            sum((p[1] - (intercept + slope * p[0])) ** 2 for p in points) / dof
        )
    else:
        resid_sd = 0.0
    slope_se = resid_sd / math.sqrt(sxx) if resid_sd > 0 else 0.0
    return (slope, intercept, resid_sd, slope_se, x_mean, sxx)


def median(values):
    ordered = sorted(values)
    m = len(ordered)
    if m == 0:
        return 0.0
    if m % 2:
        return ordered[m // 2]
    return (ordered[m // 2 - 1] + ordered[m // 2]) / 2.0


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels mismatch")
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("no history")

    horizon_days = case["horizon_minutes"] / 1440.0
    target_x = log_hours(parse_time(case["target"]))
    snapshots = distinct_reads(history)
    last_rates = history[-1]["rates"]

    # Flat history: a single distinct snapshot (or no rate moved at all).
    flat = len(snapshots) <= 1
    if not flat:
        first_rates = snapshots[0]["rates"]
        flat = all(
            a == b or (a is None and b is None)
            for a, b in zip(first_rates, last_rates)
        )

    snap_x = [log_hours(parse_time(s["at"])) for s in snapshots]

    # Per-trophy line fits on the distinct snapshots.
    fits = []
    for i in range(len(trophies)):
        points = []
        for j, snap in enumerate(snapshots):
            rate = snap["rates"][i]
            if rate is None:
                continue
            points.append((snap_x[j], logit_of(rate)))
        fits.append(fit_line(points))

    # Empirical-Bayes shrinkage of the slopes toward their median.
    real_fits = [f for f in fits if f is not None]
    median_slope = median([f[0] for f in real_fits]) if real_fits else 0.0
    # Cross-trophy variance of slopes, net of estimation noise.
    if len(real_fits) >= 2:
        mean_slope = sum(f[0] for f in real_fits) / len(real_fits)
        var_slopes = sum((f[0] - mean_slope) ** 2 for f in real_fits) / (
            len(real_fits) - 1
        )
        mean_noise = sum(f[3] ** 2 for f in real_fits) / len(real_fits)
        tau2 = max(var_slopes - mean_noise, 0.0)
    else:
        tau2 = 0.0

    base_spread = BASE_SIGMA * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = 0.0

        # Already ~100%: every new player earns it immediately.
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.9, 99.95, 100, 100, 100]})
            continue

        fit = fits[i]
        if fit is None:
            # No usable trend for this trophy: persist the last rate.
            centre = logit_of(last_rate)
            spread = base_spread
        else:
            slope, intercept, resid_sd, slope_se, x_mean, sxx = fit
            # Shrink the slope toward the median, by its precision.
            weight = tau2 / (tau2 + slope_se ** 2) if tau2 > 0 else 0.0
            shrunk_slope = median_slope + (slope - median_slope) * weight
            # Refit the intercept through the data centre with the new slope.
            centre_y = intercept + slope * x_mean
            centre = centre_y + shrunk_slope * (target_x - x_mean)
            # Prediction standard error; grows with extrapolation distance.
            if sxx > 1e-12 and len(snapshots) > 2:
                se = resid_sd * math.sqrt(
                    1.0 + 1.0 / len(snapshots) + (target_x - x_mean) ** 2 / sxx
                )
            else:
                se = resid_sd
            spread = math.sqrt(base_spread ** 2 + se ** 2)

        if flat:
            centre += FLAT_SHIFT
            spread = max(spread, FLAT_SPREAD_FLOOR)

        spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))
        quantiles = [round_half_up_4(pct_of(centre + z * spread)) for z in Z_SCORES]
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
