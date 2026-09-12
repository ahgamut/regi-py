/* Golden cross-check: the JS Direct-net bots pick the SAME combo index as their
 * Python reference strategy (ADZDirectStrategy / NetDirectStrategy), over the
 * fixtures from gen_golden.py / az_gen_golden.py. This closes the featurizer + ONNX
 * + argmax parity loop that the in-env node smoke tests can't cover (no onnxruntime).
 *
 * Checks EVERY net for which both a golden fixture (<golden>/<net>.json) and an export
 * (<dist>/<net>.onnx + <net>.io.json) exist -- pass --net to restrict to one.
 *
 * Prereqs: build the WASM module, export the nets (trainers/export_onnx.py) into
 * <dist>, generate the fixtures, and have an onnxruntime package installed
 * (onnxruntime-web by default; onnxruntime-node also works).
 *
 *   node check_golden.mjs [--net adzpool] [--dist ./dist] [--golden ./golden] \
 *       [--ort onnxruntime-node]
 *
 * Each case is featurized with {reshuffle:false} so the phase matches Python's
 * byte-for-byte (no RNG). A combo-order check runs first, so an index mismatch is
 * never masked by the two sides enumerating the offered combos differently. */
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import RegiModule from './dist/regicore.mjs';
import { NetBot } from './adz_bot.mjs';
import { AZBot } from './az_bot.mjs';

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

const ONLY_NET = arg('net', null);
const DIST = arg('dist', './dist');
const GOLDEN_DIR = arg('golden', './golden');
const ORT_PKG = arg('ort', 'onnxruntime-web');

const ortMod = await import(ORT_PKG);
const ort = ortMod.Tensor ? ortMod : ortMod.default;
const M = await RegiModule();
const comboMap = M.combo_map(); // shared by every AZ net

// A mismatch whose score gap (js pick vs py pick) is below this is a benign fp
// near-tie -- two combos the net scores equally, resolved differently by torch vs ORT
// op-ordering (e.g. two equally-discardable cards, keepy ~0). Not a real divergence.
const TIE_TOL = 1e-6;
const sameArr = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

/* The nets to check: those with BOTH a golden fixture and an export. --net restricts
 * to one (and errors if its artifacts are missing). */
function selectNets() {
  if (ONLY_NET) {
    const miss = [`${GOLDEN_DIR}/${ONLY_NET}.json`, `${DIST}/${ONLY_NET}.onnx`, `${DIST}/${ONLY_NET}.io.json`]
      .filter((p) => !existsSync(p));
    if (miss.length) throw new Error(`--net ${ONLY_NET}: missing ${miss.join(', ')}`);
    return [ONLY_NET];
  }
  const golden = readdirSync(GOLDEN_DIR).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5));
  const nets = golden.filter((n) => existsSync(`${DIST}/${n}.onnx`) && existsSync(`${DIST}/${n}.io.json`)).sort();
  for (const n of golden) {
    if (!nets.includes(n)) console.log(`skip ${n}: golden present but no export in ${DIST}`);
  }
  return nets;
}

/* Drive one loaded phase to its decision and build the feeds there (sync, while the
 * offered VectorCombo is valid). Returns {built, combosLocs} or null. */
function buildForCase(bot, c) {
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

/* Check one net against its fixture; returns { ok, checked, indexMismatch, ... }. */
async function checkNet(net) {
  const contract = JSON.parse(readFileSync(`${DIST}/${net}.io.json`, 'utf8'));
  const golden = JSON.parse(readFileSync(`${GOLDEN_DIR}/${net}.json`, 'utf8'));
  if (golden.net !== net) throw new Error(`fixture net ${golden.net} != ${net}`);
  const session = await ort.InferenceSession.create(`${DIST}/${net}.onnx`);
  const bot = contract.paradigm === 'az'
    ? new AZBot(M, ort, session, contract, comboMap)
    : new NetBot(M, ort, session, contract);

  let checked = 0, comboMismatch = 0, indexMismatch = 0, ties = 0;
  const examples = [];
  for (let ci = 0; ci < golden.cases.length; ci++) {
    const c = golden.cases[ci];
    const built = buildForCase(bot, c);
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

  const ok = indexMismatch === 0 && comboMismatch === 0 && checked > 0;
  console.log(`${ok ? 'ok  ' : 'FAIL'}  ${net} (${contract.paradigm}): ${checked} cases, ${indexMismatch} index mismatch, ${ties} benign fp ties, ${comboMismatch} combo-order/capture failures`);
  if (examples.length) console.log('        ' + examples.join('\n        '));
  return { net, ok };
}

const nets = selectNets();
if (!nets.length) { console.error(`no nets to check (looked in ${GOLDEN_DIR} + ${DIST})`); process.exit(1); }

const results = [];
for (const net of nets) results.push(await checkNet(net));

const failed = results.filter((r) => !r.ok).map((r) => r.net);
console.log(`\n${results.length} net(s): ${results.length - failed.length} passed, ${failed.length} failed`);
if (failed.length) { console.log('GOLDEN CHECK FAILED: ' + failed.join(', ')); process.exit(1); }
console.log('GOLDEN CHECK PASSED');
