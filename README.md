# GTA VI trophy rarity forecast — make.fun gta6-trophy-rarity-forecast-01

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's earn rate on
Exophase at a case's target, as five quantiles (5/25/50/75/95%) per trophy.
Entry for the make.fun bounty "Forecast how many GTA VI players earn each
trophy in week one" ($250, 20 places pay). Scored by mean pinball loss on
the logit scale; the prize bar is beating the best house entry.

## Program

`main.ts` (Node 22, standard library only, no network) at the repo root.
The grader runs `node /bot/main.ts`; it reads every `<id>.in` in `/input`
(or `$MAKEFUN_INPUT_DIR`) and writes one `<id>.out` to `/output` (or
`$MAKEFUN_OUTPUT_DIR`):

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Method

Two views of the case's history, averaged in logit space.

Exophase refreshes its counts on its own schedule, so long runs of reads
are byte-identical. The history is first collapsed to distinct snapshots
(first-seen time each), so least-squares fits are not swamped by stale
repeats. Time is `x = ln(1 + hours since release)`: launch week moves fast
early and decelerates, and a line in log-time extrapolates without running
away.

- **Model A (rate trend):** per trophy, OLS of `logit(rate)` on `x`,
  anchored at the last read. Slopes are shrunk toward their group's mean
  slope (k = 2 pseudo-points) and capped at 2.5 logit per log-time unit.
  The groups are the story chain (chapters + final mission, whose rates
  rise as players progress) and everything else (mostly diluted down by
  new players joining the tracked base).
- **Model B (count space):** OLS of `ln(players tracked)` and
  `ln(earners)` on `x`, anchored the same way; the forecast rate is
  earners / players at the target. This separates the two forces moving a
  rate: players earning the trophy, and new players diluting it.
- **Structure:** story chapters are nested (Ch.1 >= Ch.2 >= ... >= final
  mission), the platinum is earned by no more players than any other
  trophy, and 100% completion implies the final mission; enforced on
  every quantile level. A trophy the last read shows at 100% cannot stay
  there once new cohorts join: its earners are forecast as the old
  earners plus 70% of the newcomers, with a wide spread.
- **Spread:** `sqrt(random-walk^2 + fit-residual^2 + model-risk^2)` in
  logit units — the house baseline's 0.5 per root-day, plus the trophy's
  own fit residual widened by extrapolation leverage, plus a 0.35 model
  allowance (1.2 for the saturated trophy). Floored at 0.6, capped at 2.0
  (2.6 saturated): the practice cases show one refresh can move every
  rate ~2 logit units, so a spread fitted only to a flat history would
  be dishonest.

Deterministic: fixed arithmetic, sorted inputs, no randomness, no clock
reads — two runs on the same case write the same bytes.

## Local validation

- Practice cases p00/p01: output valid (55 trophies, 5 non-decreasing
  quantiles each, under the 65,536-byte cap), deterministic across runs,
  ~0.4 s per run.
- Against synthetic practice answers (the launch-night refresh as pure
  638 -> 3,868 dilution, and dilution + story growth): loss 1.51 / 1.17
  vs the house baseline's 1.88 / 1.52 — the wider spread floor pays when
  a refresh jumps every rate.
- Against two synthetic final-like cases (2.5 days of trending history
  with refreshes, target 4.5 days out; one where story rates rise, one
  where fast player growth dilutes even story rates): loss 0.28 / 0.34
  vs the baseline's 0.35 / 0.74.

The synthetic cases are local stress tests only; the real final case
f00 is scored by the house after 2026-11-26.

## Files

- `main.ts` — the entry program
- `LICENSE` — MIT (the kit starter's license)
