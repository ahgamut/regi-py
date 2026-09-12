/* Node smoke test for the WASM engine build.
 * Build first:  emcmake cmake -G Ninja -B build && cmake --build build
 * Then run:     node smoke.mjs
 * Drives full games to a terminal state through the Embind API (native and a
 * JS-subclassed Strategy), and checks determinism + the export_string round-trip. */
import RegiModule from './dist/regicore.mjs';

const M = await RegiModule();
let failures = 0;
function check(name, cond) {
  console.log(`${cond ? 'ok  ' : 'FAIL'}  ${name}`);
  if (!cond) failures++;
}

/* Run a game to completion with `nplayers` seats built by makeStrat(). */
function runGame(makeStrat, nplayers, seedVal) {
  M.seed(seedVal);
  const log = new M.NoOpLog();
  const g = new M.GameState(log);
  const strats = [];
  for (let i = 0; i < nplayers; i++) {
    const s = makeStrat();
    strats.push(s);
    g.add_player(s);
  }
  g.initialize();
  let steps = 0;
  while (g.is_runnable() && steps < 100000) { g.step(); steps++; }
  const out = { ended: g.status.value === M.GameStatus.ENDED.value, export: g.export_string(), steps };
  strats.forEach((s) => s.delete());
  g.delete();
  log.delete();
  return out;
}

/* 1) native DamageStrategy drives a 2p game to ENDED. */
const nat = runGame(() => new M.DamageStrategy(), 2, 12345);
check('native strategy game reaches ENDED', nat.ended);

/* 2) a JS-subclassed Strategy (offered combos arrive as an argument) drives a
 *    3p game to ENDED -- the browser-driver model, no Asyncify. */
const PickFirst = M.Strategy.extend('Strategy', {
  setup(player, game) { return 0; },
  getAttackIndex(combos, player, yieldAllowed, game) { return 0; },
  getDefenseIndex(combos, player, damage, game) { return 0; },
  getRedirectIndex(player, game) { return 0; },
});
const js = runGame(() => new PickFirst(), 3, 777);
check('JS-subclassed strategy game reaches ENDED', js.ended);

/* 3) determinism: the same seed reproduces the same game byte-for-byte. */
const a = runGame(() => new M.DamageStrategy(), 2, 42);
const b = runGame(() => new M.DamageStrategy(), 2, 42);
check('same seed -> identical export_string', a.export === b.export);

/* 4) export_string round-trips: re-seating from the string reproduces it. */
M.seed(99);
const log = new M.NoOpLog();
const g = new M.GameState(log);
const s0 = new M.RandomStrategy();
const s1 = new M.RandomStrategy();
g.add_player(s0);
g.add_player(s1);
g.initialize();
const es = g.export_string();
const g2 = new M.GameState(log);
g2.add_player(s0);
g2.add_player(s1);
g2.init_string(es);
check('init_string(export_string()) round-trips', g2.export_string() === es);
[s0, s1, g, g2, log].forEach((x) => x.delete());

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
