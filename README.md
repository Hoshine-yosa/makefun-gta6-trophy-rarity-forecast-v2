[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hoshine-yosa/makefun-gta6-trophy-rarity-forecast-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hoshine-yosa/makefun-gta6-trophy-rarity-forecast-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecast, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, the
percent of the players Exophase tracks for the game who will have earned it
at 2026-11-26T00:00:00Z, a week after launch — as five quantiles
(5%, 25%, 50%, 75%, 95%) per trophy.

## Program

`main.py` (Python 3.13, standard library only, no network, deterministic).
Reads every `<id>.in` in the input directory and writes `<id>.out` to the
output directory. The grader runs it as `python3 -P -s /bot/main.py`;
outside the grader, `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` stand in
for `/input` and `/output`.

## Model

One trophy at a time, using only the case file:

- **Dedupe.** Exophase refreshes every few hours, so long runs of reads are
  identical plateaus. The history is collapsed to distinct snapshots first.
- **Log-log trend.** On the logit scale, against log time since release
  (`y = logit(rate)`, `x = ln(1 + hours since 2026-11-19T00:00Z)`), an
  ordinary least-squares line is fitted per trophy. The slope is the net of
  two forces: existing players earning the trophy (up) vs fresh players
  joining with almost no trophies (down).
- **Empirical-Bayes slope shrinkage.** Each slope is shrunk toward the
  cross-trophy median slope, weighted by how precisely it is estimated.
  The median is the shared dilution factor — most trophies are diluted by
  the same flood of new players.
- **Honest spread.** The regression's prediction standard error (growing
  with extrapolation distance) combined in quadrature with the house
  baseline's random-walk term, 0.5·√(horizon in days). Floored at 0.35 and
  capped at 2.5 logit.
- **Flat history.** When the history is a single plateau (launch night), the
  trend is zero but a refresh is overdue: the next Exophase refresh adds a
  flood of new players holding almost no trophies. The launch-night refresh
  jumped tracked players 638 → 3,868 and rates fell ~4.5×, i.e. about
  -1.5 in logit terms. So the median is shifted down by 1.5 logit and the
  spread is floored at 0.9.
- **~100% trophies stay ~100%.** Every new player earns them immediately,
  so dilution cannot touch them.

## Validation

On the kit's practice cases (`p00`, `p01`):

- Output format correct: 55 trophies × 5 quantiles, non-decreasing, 0–100.
- Two runs byte-identical (deterministic).
- Runtime ~0.05 s per case (limit 8 s).
- Local pinball-loss scoring against the reconstructed practice answer
  (the refresh after launch night: tracked players 638 → 3,868, rates
  falling ~4.5×) gives a loss around **0.24**, vs the house persistence
  baseline's ~1.40. See `score_local.py`.
