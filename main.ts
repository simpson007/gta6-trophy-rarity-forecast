// Forecast each GTA VI PS5 trophy's earn rate on Exophase at a case's target.
// make.fun gta6-trophy-rarity-forecast-01. Node 22, standard library only,
// no network. Reads every <id>.in in the input directory and writes one
// <id>.out per case: {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
// The grader runs `node /bot/main.ts` with /input and /output.
//
// Method: two views of the same history, averaged in logit space.
//
// The case's history is the house's reads of Exophase's trophy page.
// Exophase refreshes its counts on its own schedule, so long runs of reads
// are byte-identical: the first thing the program does is collapse the
// history to distinct snapshots (first-seen time each), so a least-squares
// fit is not swamped by stale repeats.
//
// Time is x = ln(1 + hours since release, 2026-11-19T00:00:00Z). Log-time
// is the right clock for a launch: the first day moves rates far more than
// the fifth, and a line in log-time decelerates instead of running away.
//
//   Model A (rate trend): per trophy, OLS of logit(rate) on x over the
//     distinct snapshots, anchored at the last read: the forecast is the
//     last rate moved by the fitted slope over (x_target - x_last). Slopes
//     are shrunk toward the mean slope of their group (k = 2 pseudo-points)
//     and capped, so a trophy with a noisy or short history borrows its
//     group's drift instead of extrapolating its own noise. The groups are
//     the story chain (chapters and the final mission, whose rates rise as
//     players progress) and everything else (whose rates are mostly dragged
//     down by new players diluting the tracked base): shrinking a chapter
//     toward the average of grindy trophies, or vice versa, would bias it
//     in the wrong direction, so each group shrinks toward its own mean.
//   Model B (count space): OLS of ln(players tracked) and ln(earners)
//     on x, anchored the same way; the forecast rate is earners / players
//     at the target. This is the same trend seen from the other side, but
//     it separates the two forces that move a rate: players earning the
//     trophy, and new players diluting it. When the tracked base grows
//     many-fold over the horizon, dilution alone can halve a slow
//     trophy's rate while its earner count still grows.
//
// The median is the mean of A and B in logit space. Two structural facts
// are then enforced on every quantile level: story chapters are nested
// (Chapter 1 >= Chapter 2 >= ... >= final mission), the platinum is earned
// by no more players than any other trophy, and 100% completion implies
// the final mission. A trophy the last read shows at 100% is a special
// case: it cannot stay at 100% once new cohorts join, so its earners are
// forecast as the old earners plus 70% of the newcomers (a documented
// prior for an opening activity every early player did), with a wide
// spread, instead of the logit of a clamped 99.99%.
//
// Spread: s = sqrt(s_rw^2 + s_fit^2 + s_model^2), where s_rw is the house
// baseline's random walk (0.5 logit per root-day), s_fit the trophy's own
// fit residual widened by the extrapolation leverage, and s_model a 0.35
// logit allowance for refresh staleness and functional-form risk (1.2 for
// the saturated trophy). Floored at 0.6, capped at 2.0 (2.6 saturated):
// the practice cases show a single refresh can move every rate ~2 logit
// units, so a spread fitted only to a flat history would be dishonest.
//
// Deterministic: fixed arithmetic, sorted inputs, no randomness, no clock
// reads, so two runs on the same case write the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The house baseline's random-walk spread, logit units per root-day. */
const SIGMA_RW = 0.5;
/** Pseudo-points of shrinkage of each trophy's slope toward the mean slope. */
const SHRINK_K = 2;
/** Largest |slope| allowed, logit units per unit of log-time. */
const SLOPE_CAP = 2.5;
/** Model-risk allowance in the spread, logit units (saturated trophy: 1.2). */
const S_MODEL = 0.35;
const S_MODEL_SATURATED = 1.2;
/** Spread floor / cap, logit units. */
const S_FLOOR = 0.6;
const S_CAP = 2.0;
const S_CAP_SATURATED = 2.6;
/** A rate at or above this in the last read counts as saturated (100%). */
const SATURATED_AT = 99.5;
/** Share of newcomers assumed to earn a saturated opening activity. */
const SATURATED_NEW_SHARE = 0.7;
/** Story chain, trophy ids in completion order (nested: each implies the last). */
const STORY_CHAIN = [3, 4, 5, 6, 7];
/** Platinum id and 100%-completion id. */
const PLATINUM_ID = 1;
const COMPLETION_ID = 14;

interface Trophy {
  id: number;
  name: string;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
}

interface Snap {
  /** Hours since release at first sight of this snapshot. */
  h: number;
  /** log-time coordinate. */
  x: number;
  players: number;
  rates: Array<number | null>;
}

interface Fit {
  slope: number;
  residSd: number;
  leverage: number;
  n: number;
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Ordinary least squares of y on x. Returns null when x does not vary. */
function ols(xs: number[], ys: number[]): Fit | null {
  const n = xs.length;
  if (n < 3) return null;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx <= 1e-12) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    ss += r * r;
  }
  // Residual SD with n - 2 degrees of freedom; 0 when the fit is exact.
  const residSd = n > 2 ? Math.sqrt(ss / (n - 2)) : 0;
  return { slope, residSd, leverage: 0, n };
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error('case has no history');

  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const targetMs = Date.parse(c.target);
  const hTarget = Math.max(0, (targetMs - releaseMs) / 3_600_000);
  const xTarget = Math.log(1 + hTarget);
  const horizonDays = c.horizon_minutes / 1440;

  // Collapse the history to distinct snapshots: consecutive reads that
  // Exophase had not refreshed are one observation, at first-seen time.
  const snaps: Snap[] = [];
  let prevKey = '';
  for (const read of c.history) {
    const key = `${read.players_tracked}|${read.rates.join(',')}`;
    if (key === prevKey) continue;
    prevKey = key;
    const h = Math.max(-0.9, (Date.parse(read.at) - releaseMs) / 3_600_000);
    snaps.push({ h, x: Math.log(1 + h), players: read.players_tracked, rates: read.rates });
  }
  const last = snaps[snaps.length - 1];
  const xLast = last.x;

  const nT = c.trophies.length;

  // --- Model A: per-trophy logit-rate trend, anchored at the last read. ---
  const fitsA: Array<Fit | null> = [];
  const lastLogit: number[] = [];
  for (let i = 0; i < nT; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const s of snaps) {
      const r = s.rates[i];
      if (r !== null && r !== undefined) {
        xs.push(s.x);
        ys.push(logit(r));
      }
    }
    fitsA.push(ols(xs, ys));
    // Last non-null rate, walking back over reads that lacked the trophy.
    let rLast: number | null = null;
    for (let k = snaps.length - 1; k >= 0; k--) {
      const r = snaps[k].rates[i];
      if (r !== null && r !== undefined) {
        rLast = r;
        break;
      }
    }
    lastLogit.push(logit(rLast ?? 1));
  }
  const validSlopesA = fitsA.filter((f): f is Fit => f !== null).map((f) => f.slope);
  const meanSlopeA =
    validSlopesA.length > 0 ? validSlopesA.reduce((a, b) => a + b, 0) / validSlopesA.length : 0;
  // Group means for shrinkage: the story chain shares a progression dynamic
  // that the other trophies do not, so each group shrinks toward its own mean.
  const storySet = new Set(STORY_CHAIN);
  const storySlopes = fitsA
    .map((f, i) => (storySet.has(c.trophies[i].id) ? f : null))
    .filter((f): f is Fit => f !== null)
    .map((f) => f.slope);
  const otherSlopes = fitsA
    .map((f, i) => (storySet.has(c.trophies[i].id) ? null : f))
    .filter((f): f is Fit => f !== null)
    .map((f) => f.slope);
  const meanSlope = (slopes: number[]): number =>
    slopes.length > 0 ? slopes.reduce((a, b) => a + b, 0) / slopes.length : meanSlopeA;
  const meanSlopeStory = meanSlope(storySlopes);
  const meanSlopeOther = meanSlope(otherSlopes);
  const medianResid =
    fitsA.filter((f): f is Fit => f !== null).map((f) => f.residSd).sort((a, b) => a - b)[
      Math.floor(fitsA.filter((f) => f !== null).length / 2)
    ] ?? 0.5;

  // --- Model B: count space. ln(players) trend, per-trophy ln(earners) trend. ---
  const xsN = snaps.map((s) => s.x);
  const ysN = snaps.map((s) => Math.log(Math.max(1, s.players)));
  const fitN = ols(xsN, ysN);
  const slopeN = fitN ? fitN.slope : 0;
  const lnNLast = Math.log(Math.max(1, last.players));
  const lnNTarget = lnNLast + slopeN * (xTarget - xLast);
  const nTarget = Math.exp(lnNTarget);

  // Implied logit slope of B for each trophy, shrunk toward its group's mean slope.
  const shrink = (slope: number, n: number, id: number): number => {
    const target = storySet.has(id) ? meanSlopeStory : meanSlopeOther;
    const s = (n * slope + SHRINK_K * target) / (n + SHRINK_K);
    return Math.max(-SLOPE_CAP, Math.min(SLOPE_CAP, s));
  };

  const medians: number[] = [];
  const spreads: number[] = [];
  for (let i = 0; i < nT; i++) {
    const fit = fitsA[i];
    const slopeA = shrink(fit ? fit.slope : meanSlopeA, fit ? fit.n : 0, c.trophies[i].id);
    const logitA = lastLogit[i] + slopeA * (xTarget - xLast);

    // Model B: earners trend. Earners = rate * players / 100.
    const xsE: number[] = [];
    const ysE: number[] = [];
    for (const s of snaps) {
      const r = s.rates[i];
      if (r !== null && r !== undefined) {
        xsE.push(s.x);
        ysE.push(Math.log(Math.max(0.5, (r / 100) * s.players)));
      }
    }
    const fitE = ols(xsE, ysE);
    const slopeB = shrink(
      fitE ? fitE.slope - slopeN : meanSlopeA,
      fitE ? fitE.n : 0,
      c.trophies[i].id,
    );
    const rLastPct = 100 / (1 + Math.exp(-lastLogit[i]));
    const lnELast = Math.log(Math.max(0.5, (rLastPct / 100) * last.players));
    const lnETarget = lnELast + (slopeB + slopeN) * (xTarget - xLast);
    const rateB = clampRate((100 * Math.exp(lnETarget)) / nTarget);
    const logitB = logit(rateB);

    const saturated = rLastPct >= SATURATED_AT;
    let medianLogit: number;
    let sModel = S_MODEL;
    let sCap = S_CAP;
    if (saturated) {
      // A saturated trophy cannot stay saturated as new cohorts join:
      // its earners grow by the assumed share of newcomers only.
      const eTarget = last.players + SATURATED_NEW_SHARE * Math.max(0, nTarget - last.players);
      medianLogit = logit(clampRate((100 * eTarget) / Math.max(1, nTarget)));
      sModel = S_MODEL_SATURATED;
      sCap = S_CAP_SATURATED;
    } else {
      medianLogit = (logitA + logitB) / 2;
    }

    // Spread: random walk + own fit residual (leverage-widened) + model risk.
    const sRw = SIGMA_RW * Math.sqrt(horizonDays);
    let sFit = medianResid;
    if (fit) {
      const xbar = xsE.length > 0 ? xsE.reduce((a, b) => a + b, 0) / xsE.length : xLast;
      let sxx = 0;
      for (const x of xsE) sxx += (x - xbar) * (x - xbar);
      const lev = sxx > 1e-12 ? Math.sqrt(1 + ((xTarget - xbar) * (xTarget - xbar)) / sxx) : 2;
      sFit = fit.residSd * Math.min(2, lev);
    }
    const s = Math.max(S_FLOOR, Math.min(sCap, Math.sqrt(sRw * sRw + sFit * sFit + sModel * sModel)));
    medians.push(medianLogit);
    spreads.push(s);
  }

  // --- Structural constraints, applied per quantile level. ---
  const rows = c.trophies.map((t, i) => {
    const q = Z.map((z) => round4(sigmoidPct(medians[i] + z * spreads[i])));
    // Non-decreasing after rounding.
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
  const byId = new Map(rows.map((r) => [r.id, r]));
  const enforceChain = (chain: number[]): void => {
    for (let j = 0; j < LEVELS.length; j++) {
      for (let k = 1; k < chain.length; k++) {
        const prev = byId.get(chain[k - 1]);
        const cur = byId.get(chain[k]);
        if (prev && cur) cur.q[j] = Math.min(cur.q[j], prev.q[j]);
      }
    }
  };
  enforceChain(STORY_CHAIN);
  // Platinum earns no more than any trophy; 100% completion no more than the final mission.
  for (let j = 0; j < LEVELS.length; j++) {
    const plat = byId.get(PLATINUM_ID);
    const comp = byId.get(COMPLETION_ID);
    const fin = byId.get(STORY_CHAIN[STORY_CHAIN.length - 1]);
    if (plat && comp) plat.q[j] = Math.min(plat.q[j], comp.q[j]);
    if (comp && fin) comp.q[j] = Math.min(comp.q[j], fin.q[j]);
    if (plat) {
      let m = plat.q[j];
      for (const r of rows) m = Math.min(m, r.q[j]);
      plat.q[j] = m;
    }
  }
  // Re-enforce monotonicity inside each row after the constraints.
  for (const r of rows) {
    for (let j = 1; j < r.q.length; j++) r.q[j] = Math.max(r.q[j], r.q[j - 1]);
  }

  return { trophies: rows };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const out = forecastCase(c);
  const text = `${JSON.stringify({ trophies: out.trophies })}\n`;
  if (Buffer.byteLength(text, 'utf8') > 65_536) throw new Error(`${id}.out would exceed 65536 bytes`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), text);
}
