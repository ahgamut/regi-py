/* Node smoke test for the ADZ Direct-net bot's feed assembly (adz_bot.mjs).
 * Build first:  emcmake cmake -G Ninja -B build && cmake --build build
 * Then run:     node smoke_bot.mjs
 *
 * onnxruntime-web is NOT needed here: a mock `ort.Tensor` captures the tensors the
 * bot WOULD feed the model, so we can validate the drift-prone assembly (shapes,
 * dtypes, cand_mask, and the adzmulti/adzpool membership encodings) against the
 * real offered combos. The actual forward pass + argmax parity vs the Python
 * ADZDirectStrategy is a torch/ORT-env check (trainers/export_onnx --verify covers
 * the graph; a JS-vs-Python golden check comes with onnxruntime-web wired up). */
import RegiModule from './dist/regicore.mjs';
import { NetBot } from './adz_bot.mjs';

const M = await RegiModule();
let failures = 0;
function check(name, cond) {
  console.log(`${cond ? 'ok  ' : 'FAIL'}  ${name}`);
  if (!cond) failures++;
}

const MAX_CARDS = 56, MAX_CANDIDATES = 128, MAX_PARTS = 7, CFD = 9;

/* Mock ort namespace: Tensor just records what it was handed. */
const ort = { Tensor: class { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } } };

const CONTRACTS = {
  adzmulti: {
    net: 'adzmulti', paradigm: 'adz', max_history: 8, outputs: ['value', 'cand_logits', 'keepy'],
    inputs: [{ name: 'cand_feats' }, { name: 'cand_mask' }, { name: 'cand_members' }, { name: 'tokens' }],
  },
  adzpool: {
    net: 'adzpool', paradigm: 'adz', max_history: 8, outputs: ['value', 'cand_logits', 'keepy'],
    inputs: [{ name: 'cand_feats' }, { name: 'cand_mask' }, { name: 'cand_idx' }, { name: 'cand_partmask' }, { name: 'tokens' }],
  },
};

/* Capture the first attack decision offering >=2 combos, plus the ground-truth
 * member locations of each offered combo (from Combo.parts). */
function captureDecision(seedVal) {
  M.seed(seedVal);
  const log = new M.NoOpLog();
  const g = new M.GameState(log);
  let cap = null;
  const Capture = M.Strategy.extend('Strategy', {
    setup() { return 0; },
    getAttackIndex(combos, player, yieldAllowed, game) {
      if (cap === null && combos.size() >= 2) {
        const phase = game.export_phaseinfo();
        const truthLocs = [];
        for (let i = 0; i < combos.size(); i++) {
          const c = combos.get(i), parts = c.parts, locs = [];
          for (let j = 0; j < parts.size(); j++) { const cd = parts.get(j); locs.push(cd.location); cd.delete(); }
          parts.delete(); c.delete();
          truthLocs.push(locs.sort((a, b) => a - b));
        }
        // run the bot's feed builder for each contract on this exact decision
        const feeds = {};
        for (const name of Object.keys(CONTRACTS)) {
          const bot = new NetBot(M, ort, /*session*/ null, CONTRACTS[name]);
          feeds[name] = bot.buildFeeds(phase, combos, []);
        }
        cap = { K: combos.size(), truthLocs, feeds, phaseStr: phase.to_string(), numPlayers: phase.num_players };
        phase.delete();
      }
      return 0;
    },
    getDefenseIndex() { return 0; },
    getRedirectIndex() { return 0; },
  });
  const s0 = new Capture(), s1 = new Capture();
  g.add_player(s0); g.add_player(s1);
  g.initialize();
  let steps = 0;
  while (g.is_runnable() && steps < 100000) { g.step(); steps++; }
  [s0, s1, g, log].forEach((x) => x.delete());
  return cap;
}

const d = captureDecision(31337);
check('captured an attack decision with >=2 combos', d !== null && d.K >= 2);

if (d) {
  const K = d.K;
  const sameSet = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

  for (const net of ['adzmulti', 'adzpool']) {
    const { feeds } = d.feeds[net];
    const tag = (s) => `[${net}] ${s}`;

    // shared inputs
    check(tag('tokens dims [1,56,264] f32'),
      feeds.tokens && feeds.tokens.type === 'float32' &&
      sameSet(feeds.tokens.dims, [1, 56, 264]) && feeds.tokens.data.length === 56 * 264);
    check(tag('cand_feats dims [1,128,9] f32'),
      feeds.cand_feats.type === 'float32' && sameSet(feeds.cand_feats.dims, [1, MAX_CANDIDATES, CFD]) &&
      feeds.cand_feats.data.length === MAX_CANDIDATES * CFD);
    // cand_mask: first K ones, rest zero
    const mask = feeds.cand_mask.data;
    let maskOk = feeds.cand_mask.data.length === MAX_CANDIDATES;
    for (let i = 0; i < MAX_CANDIDATES; i++) if (mask[i] !== (i < K ? 1 : 0)) maskOk = false;
    check(tag('cand_mask is K ones then zeros'), maskOk);
    // padded candidate feature rows (>= K) are zero
    let padOk = true;
    for (let i = K * CFD; i < feeds.cand_feats.data.length; i++) if (feeds.cand_feats.data[i] !== 0) padOk = false;
    check(tag('cand_feats padded rows are zero'), padOk);

    // membership matches the offered combos' parts locations
    if (net === 'adzmulti') {
      const mem = feeds.cand_members;
      check(tag('cand_members dims [1,128,56] f32'),
        mem.type === 'float32' && sameSet(mem.dims, [1, MAX_CANDIDATES, MAX_CARDS]));
      let memOk = true;
      for (let i = 0; i < MAX_CANDIDATES; i++) {
        const locs = [];
        for (let loc = 0; loc < MAX_CARDS; loc++) if (mem.data[i * MAX_CARDS + loc] === 1) locs.push(loc);
        const expect = i < K ? d.truthLocs[i] : [];
        if (!sameSet(locs, expect)) memOk = false;
      }
      check(tag('cand_members multi-hot matches combo parts'), memOk);
    } else {
      const idx = feeds.cand_idx, pm = feeds.cand_partmask;
      check(tag('cand_idx dims [1,128,7] int64'),
        idx.type === 'int64' && idx.data instanceof BigInt64Array && sameSet(idx.dims, [1, MAX_CANDIDATES, MAX_PARTS]));
      check(tag('cand_partmask dims [1,128,7] f32'),
        pm.type === 'float32' && sameSet(pm.dims, [1, MAX_CANDIDATES, MAX_PARTS]));
      let idxOk = true;
      for (let i = 0; i < MAX_CANDIDATES; i++) {
        const locs = [];
        for (let j = 0; j < MAX_PARTS; j++) {
          if (pm.data[i * MAX_PARTS + j] === 1) locs.push(Number(idx.data[i * MAX_PARTS + j]));
          else if (idx.data[i * MAX_PARTS + j] !== 0n) idxOk = false; // masked slots must be 0
        }
        locs.sort((a, b) => a - b);
        const expect = i < K ? d.truthLocs[i] : [];
        if (!sameSet(locs, expect)) idxOk = false;
      }
      check(tag('cand_idx+partmask index encoding matches combo parts'), idxOk);
    }
  }
}

/* 6) check_golden plumbing: reload the captured phase via init_string, drive to the
 *    decision, and build feeds with {reshuffle:false} -- the exact path check_golden.mjs
 *    uses. The reloaded offer must reproduce the same combos (order-for-order), and the
 *    no-reshuffle feeds must carry the same membership. Validates everything except the
 *    onnxruntime forward pass. */
if (d) {
  const contract = CONTRACTS.adzpool;
  const bot = new NetBot(M, ort, null, contract);
  const log = new M.NoOpLog();
  const g = new M.GameState(log);
  let reloaded = null;
  const Capture = M.Strategy.extend('Strategy', {
    setup() { return 0; },
    grab(combos, game) {
      if (reloaded !== null) return;
      const phase = game.export_phaseinfo();
      const locs = [];
      for (let i = 0; i < combos.size(); i++) {
        const c = combos.get(i), parts = c.parts, l = [];
        for (let j = 0; j < parts.size(); j++) { const cd = parts.get(j); l.push(cd.location); cd.delete(); }
        parts.delete(); c.delete();
        locs.push(l.sort((a, b) => a - b));
      }
      const { feeds, K } = bot.buildFeeds(phase, combos, [], { reshuffle: false });
      phase.delete();
      reloaded = { locs, feeds, K };
    },
    getAttackIndex(combos, player, y, game) { this.grab(combos, game); return 0; },
    getDefenseIndex(combos, player, dmg, game) { this.grab(combos, game); return 0; },
    getRedirectIndex() { return 0; },
  });
  const strats = [];
  for (let i = 0; i < d.numPlayers; i++) { const s = new Capture(); strats.push(s); g.add_player(s); }
  g.init_string(d.phaseStr);
  let steps = 0;
  while (reloaded === null && g.is_runnable() && steps < 10000) { g.step(); steps++; }
  strats.forEach((s) => s.delete());
  g.delete(); log.delete();

  const sameSet = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  check('reloaded phase captures a decision', reloaded !== null);
  if (reloaded) {
    check('reloaded offer reproduces the same combos (order-for-order)',
      reloaded.locs.length === d.truthLocs.length && reloaded.locs.every((l, i) => sameSet(l, d.truthLocs[i])));
    // reshuffle:false membership must still match the combos' parts
    let memOk = reloaded.K === d.K;
    const idx = reloaded.feeds.cand_idx.data, pm = reloaded.feeds.cand_partmask.data;
    for (let i = 0; i < d.K; i++) {
      const l = [];
      for (let j = 0; j < MAX_PARTS; j++) if (pm[i * MAX_PARTS + j] === 1) l.push(Number(idx[i * MAX_PARTS + j]));
      l.sort((a, b) => a - b);
      if (!sameSet(l, d.truthLocs[i])) memOk = false;
    }
    check('reshuffle:false feeds membership matches combos', memOk);
  }
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
