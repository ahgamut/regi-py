/* Node smoke test for the Embind featurizer kernels (core/featurize.{cc,h}).
 * Build first:  emcmake cmake -G Ninja -B build && cmake --build build
 * Then run:     node smoke_features.mjs   (or all suites via `npm run smoke`)
 *
 * These are the SAME kernels the pybind ext exposes to training; bit-for-bit
 * parity vs the old numpy is covered by tests/test_features_parity.py in a Python
 * env. This test checks the BROWSER surface: the Embind wrappers return correctly
 * shaped Float32Arrays, fuse_card_tokens is internally consistent with the three
 * component kernels (it copies their outputs frame-major), and featurization is
 * deterministic under a fixed seed.
 *
 * Exports runSmoke() so smoke_all.mjs can call it as a function (returns the failure
 * count); running this file directly runs just this suite and exits non-zero on fail. */
import { pathToFileURL } from 'node:url';
import RegiModule from './dist/regicore.mjs';

const MAX_CARDS = 56;
const LC = 9, UC = 22, CAP = 2, FW = LC + UC + CAP; // 33
const WINDOW = 8;
const eqf = (a, b) => a === b || (Number.isNaN(a) && Number.isNaN(b));

export async function runSmoke() {
  const M = await RegiModule();
  let failures = 0;
  const check = (name, cond) => { console.log(`${cond ? 'ok  ' : 'FAIL'}  ${name}`); if (!cond) failures++; };

  /* Capture featurization at the first attack decision that offers combos: the
   * combos arrive as the callback argument, and game.export_phaseinfo() is the
   * current phase. Returns the captured Float32Arrays (or null). */
  const captureFeatures = (seedVal) => {
    M.seed(seedVal);
    const log = new M.NoOpLog();
    const g = new M.GameState(log);
    let cap = null;
    const Capture = M.Strategy.extend('Strategy', {
      setup(player, game) { return 0; },
      getAttackIndex(combos, player, yieldAllowed, game) {
        if (cap === null && combos.size() >= 1) {
          const phase = game.export_phaseinfo();
          const persp = game.active_player;
          const window = new M.VectorPhaseInfo();
          for (let i = 0; i < WINDOW; i++) window.push_back(phase);
          cap = {
            k: combos.size(),
            persp,
            loc: M.features_location_array(phase, persp),
            usp: M.features_used_pile_array(phase),
            caps: M.features_card_capabilities(phase),
            cand: M.features_candidate_semantics(phase, combos),
            fused: M.features_fuse_card_tokens(window, persp),
          };
          window.delete();
          phase.delete();
        }
        return 0;
      },
      getDefenseIndex(combos, player, damage, game) { return 0; },
      getRedirectIndex(player, game) { return 0; },
    });
    const s0 = new Capture(), s1 = new Capture();
    g.add_player(s0);
    g.add_player(s1);
    g.initialize();
    let steps = 0;
    while (g.is_runnable() && steps < 100000) { g.step(); steps++; }
    [s0, s1, g, log].forEach((x) => x.delete());
    return cap;
  };

  const f = captureFeatures(2024);
  check('captured a decision with offered combos', f !== null && f.k >= 1);

  if (f) {
    /* 1) shapes */
    check('location_array length == 56*9', f.loc.length === MAX_CARDS * LC);
    check('used_pile_array length == 56*22', f.usp.length === MAX_CARDS * UC);
    check('card_capabilities length == 56*2', f.caps.length === MAX_CARDS * CAP);
    check('fuse_card_tokens length == 56*(8*33)', f.fused.length === MAX_CARDS * WINDOW * FW);
    check('candidate_semantics length == K*9', f.cand.length === f.k * 9);

    /* 2) capabilities are finite and in a sane scaled range (|.| <= ~1.5 = 60/40) */
    let capsOk = true;
    for (const v of f.caps) if (!Number.isFinite(v) || Math.abs(v) > 2.0) capsOk = false;
    check('capabilities finite and within scaled range', capsOk);

    /* 3) candidate semantics: finite; parts col (5) in [0,1]; is_yield col (6) is 0/1 */
    let candOk = true;
    for (let i = 0; i < f.k; i++) {
      const row = f.cand.subarray(i * 9, i * 9 + 9);
      for (const v of row) if (!Number.isFinite(v)) candOk = false;
      if (row[5] < 0 || row[5] > 1) candOk = false;
      if (row[6] !== 0 && row[6] !== 1) candOk = false;
    }
    check('candidate_semantics finite, parts in [0,1], is_yield binary', candOk);

    /* 4) fuse_card_tokens is bit-consistent with the component kernels: for a window
     *    of one repeated phase, every frame block equals [loc|usp|cap] for that card
     *    (fuseCardTokens copies those same kernel outputs frame-major). */
    const rowlen = WINDOW * FW;
    let fuseOk = true, checkedCells = 0;
    for (let card = 0; card < MAX_CARDS && fuseOk; card++) {
      for (let fr = 0; fr < WINDOW && fuseOk; fr++) {
        const base = card * rowlen + fr * FW;
        for (let k = 0; k < LC; k++) { if (!eqf(f.fused[base + k], f.loc[card * LC + k])) fuseOk = false; }
        for (let k = 0; k < UC; k++) { if (!eqf(f.fused[base + LC + k], f.usp[card * UC + k])) fuseOk = false; }
        for (let k = 0; k < CAP; k++) { if (!eqf(f.fused[base + LC + UC + k], f.caps[card * CAP + k])) fuseOk = false; }
        checkedCells += FW;
      }
    }
    check(`fuse_card_tokens consistent with loc/usp/cap kernels (${checkedCells} cells)`, fuseOk);

    /* 5) determinism: same seed reproduces identical featurization byte-for-byte */
    const g2 = captureFeatures(2024);
    let same = g2 !== null && g2.fused.length === f.fused.length && g2.cand.length === f.cand.length;
    if (same) {
      for (let i = 0; i < f.fused.length; i++) if (!eqf(f.fused[i], g2.fused[i])) { same = false; break; }
      for (let i = 0; same && i < f.cand.length; i++) if (!eqf(f.cand[i], g2.cand[i])) { same = false; break; }
    }
    check('same seed -> identical featurization', same);
  }

  console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
  return failures;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exit((await runSmoke()) ? 1 : 0);
}
