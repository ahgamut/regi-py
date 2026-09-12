/* Golden cross-check: the JS ADZ Direct-net bot picks the SAME combo index as the
 * Python ADZDirectStrategy, over the fixture from gen_golden.py. This closes the
 * Phase-4 MVP loop (featurizer + ONNX + argmax parity), the piece the in-env node
 * smoke tests can't cover because they lack onnxruntime.
 *
 * Prereqs: build the WASM module, export the net (trainers/export_onnx.py) into
 * <dist>, generate the fixture (gen_golden.py), and have an onnxruntime package
 * installed (onnxruntime-web by default; onnxruntime-node also works).
 *
 *   node check_golden.mjs --net adzpool --dist ./dist \
 *       --golden ./golden/adzpool.json [--ort onnxruntime-node]
 *
 * Each case is featurized with {reshuffle:false} so the phase matches Python's
 * byte-for-byte (no RNG). A combo-order check runs first, so an index mismatch is
 * never masked by the two sides enumerating the offered combos differently. */
import { readFileSync } from 'node:fs';
import RegiModule from './dist/regicore.mjs';
import { NetBot } from './adz_bot.mjs';
import { AZBot } from './az_bot.mjs';

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

const NET = arg('net', 'adzpool');
const DIST = arg('dist', './dist');
const GOLDEN = arg('golden', `./golden/${NET}.json`);
const ORT_PKG = arg('ort', 'onnxruntime-web');

const ortMod = await import(ORT_PKG);
const ort = ortMod.Tensor ? ortMod : ortMod.default;
const M = await RegiModule();

const contract = JSON.parse(readFileSync(`${DIST}/${NET}.io.json`, 'utf8'));
const golden = JSON.parse(readFileSync(GOLDEN, 'utf8'));
if (golden.net !== NET) throw new Error(`fixture net ${golden.net} != --net ${NET}`);
const session = await ort.InferenceSession.create(`${DIST}/${NET}.onnx`);
const bot = contract.paradigm === 'az'
  ? new AZBot(M, ort, session, contract, M.combo_map())
  : new NetBot(M, ort, session, contract);

const sameArr = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

/* Drive one loaded phase to its decision and build the feeds there (sync, while the
 * offered VectorCombo is valid). Returns {built, combosLocs} or null. */
function buildForCase(c) {
  const log = new M.NoOpLog();
  const g = new M.GameState(log);
  let captured = null;
  const Capture = M.Strategy.extend('Strategy', {
    setup() { return 0; },
    capture(combos, game) {
      if (captured !== null) return;
      const phase = game.export_phaseinfo();
      const combosLocs = [];
      for (let i = 0; i < combos.size(); i++) {
        const cb = combos.get(i), parts = cb.parts, locs = [];
        for (let j = 0; j < parts.size(); j++) { const cd = parts.get(j); locs.push(cd.location); cd.delete(); }
        parts.delete(); cb.delete();
        combosLocs.push(locs.sort((a, b) => a - b));
      }
      const built = bot.buildFeeds(phase, combos, [], { reshuffle: false });
      phase.delete();
      captured = { built, combosLocs };
    },
    getAttackIndex(combos, player, yieldAllowed, game) { this.capture(combos, game); return 0; },
    getDefenseIndex(combos, player, damage, game) { this.capture(combos, game); return 0; },
    getRedirectIndex() { return 0; },
  });
  const strats = [];
  for (let i = 0; i < c.num_players; i++) { const s = new Capture(); strats.push(s); g.add_player(s); }
  g.init_string(c.phase);
  let steps = 0;
  while (captured === null && g.is_runnable() && steps < 10000) { g.step(); steps++; }
  strats.forEach((s) => s.delete());
  g.delete(); log.delete();
  return captured;
}

// A mismatch whose score gap (js pick vs py pick) is below this is a benign fp
// near-tie -- two combos the net scores equally, resolved differently by torch vs ORT
// op-ordering (e.g. two equally-discardable cards, keepy ~0). Not a real divergence.
const TIE_TOL = 1e-6;
let checked = 0, comboMismatch = 0, indexMismatch = 0, ties = 0;
const examples = [];
for (let ci = 0; ci < golden.cases.length; ci++) {
  const c = golden.cases[ci];
  const built = buildForCase(c);
  if (!built) { examples.push(`case ${ci}: no decision captured`); comboMismatch++; continue; }

  // self-diagnosing: the two sides must enumerate the offered combos identically,
  // else an index is meaningless. Compare member-location lists order-for-order.
  const orderOk = built.combosLocs.length === c.combos.length &&
    built.combosLocs.every((locs, i) => sameArr(locs, c.combos[i]));
  if (!orderOk) {
    comboMismatch++;
    if (examples.length < 8) examples.push(`case ${ci}: combo order differs (k js=${built.combosLocs.length} py=${c.combos.length})`);
    continue;
  }

  // Each bot scores the offered combos its own way (ADZ: cand_logits; AZ: combomap
  // grid for attack, keepy for defense) -- the same code the live app argmaxes.
  const scores = await bot.scoreCombos(built.built);
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
  if (best !== c.index) {
    const gap = scores[best] - scores[c.index]; // >= 0
    if (gap <= TIE_TOL) {
      ties++; // benign fp near-tie: both picks score equally
    } else {
      indexMismatch++;
      if (examples.length < 12) examples.push(`case ${ci}: js=${best} py=${c.index} k=${built.built.K} ${c.attacking ? 'atk' : 'def'} gap=${gap.toExponential(2)}`);
    }
  }
  checked++;
}

console.log(`\n${NET} (${contract.paradigm}): ${checked} cases with matching combo order, ${indexMismatch} index mismatch, ${ties} benign fp ties, ${comboMismatch} combo-order/capture failures`);
if (examples.length) console.log('examples:\n  ' + examples.join('\n  '));
const ok = indexMismatch === 0 && comboMismatch === 0 && checked > 0;
console.log(ok ? 'GOLDEN CHECK PASSED' : 'GOLDEN CHECK FAILED');
process.exit(ok ? 0 : 1);
