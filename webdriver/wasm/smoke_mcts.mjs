/* Headless integration test for the MCTS Explorer (mcts.mjs) with the REAL
 * onnxruntime-web nets. Build the WASM module + export the nets first, then:
 *   node smoke_mcts.mjs [--adz adzpool] [--az basic] [--dist ./dist] [--ort onnxruntime-web]
 *                       (or all suites via `npm run smoke`)
 *
 * Checks, per paradigm: (1) buildBot(iters>0) yields an Explorer wrapping the Direct
 * bot; (2) an Explorer seat drives a full game to a terminal state via the app's
 * isExplorer branch, indices always in range; (3) each Explorer move runs ~iters+1
 * net forward passes (one per node), never more; (4) search is deterministic under
 * the fixed seed; (5) ADZ Explorer exposes no chooseRedirect (random handoff), AZ
 * Explorer delegates one (value-argmax).
 *
 * Exports runSmoke() so smoke_all.mjs can call it as a function (returns the failure
 * count); running this file directly runs just this suite and exits non-zero on fail. */
import { pathToFileURL } from 'node:url';
import { readFileSync } from 'node:fs';
import RegiModule from './dist/regicore.mjs';
import { buildBot } from './load_bot.mjs';
import { ExplorerBot } from './mcts.mjs';
import { GameDriver } from './game_driver.mjs';

function arg(name, def) { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : def; }
const ADZ = arg('adz', 'adzpool');
const AZ = arg('az', 'basic');
const DIST = arg('dist', './dist');
const ORT_PKG = arg('ort', 'onnxruntime-web');

export async function runSmoke() {
  const ortMod = await import(ORT_PKG);
  const ort = ortMod.Tensor ? ortMod : ortMod.default;
  const M = await RegiModule();
  const comboMap = M.combo_map();
  let failures = 0;
  const check = (name, cond) => { console.log(`${cond ? 'ok  ' : 'FAIL'}  ${name}`); if (!cond) failures++; };

  /* Build a bot at a given search depth, wrapping session.run with a call counter. */
  const makeBot = async (net, iters) => {
    const contract = JSON.parse(readFileSync(`${DIST}/${net}.io.json`, 'utf8'));
    const session = await ort.InferenceSession.create(`${DIST}/${net}.onnx`);
    const counter = { runs: 0 };
    const origRun = session.run.bind(session);
    session.run = async (...a) => { counter.runs++; return origRun(...a); };
    const bot = buildBot(M, ort, session, contract, { iters, comboMap });
    return { bot, counter, session, contract };
  };

  /* Drive a full game, mirroring app.mjs's loop (Direct vs Explorer branch). Returns
   * { ended, moves, badIndex, maxRunsOverBudget } where the last is the worst excess of
   * a single Explorer move's forward passes over its iters+1 budget (should be <= 0). */
  const playGame = async (driver, iters, counter) => {
    driver.newGame();
    let moves = 0, badIndex = 0, maxOver = -1, ended = false, endValue = 0;
    for (let steps = 0; steps < 40000; steps++) {
      const dec = driver.prepare();
      if (dec.kind === 'ended') { ended = true; endValue = dec.endValue; break; }
      if (dec.kind === 'auto') { driver.commit(-1); continue; }
      const bot = driver.seatBots[dec.activeSeat];
      let index;
      if (bot && bot.isExplorer) {
        const before = counter.runs;
        const hist = driver.history.map((p) => p.to_string());
        index = await bot.search(dec.decisionPhaseString, hist, dec.comboData);
        maxOver = Math.max(maxOver, (counter.runs - before) - (iters + 1)); // <= 0 expected
      } else if (bot) {
        index = await bot.runFeeds(dec.built);
      } else {
        index = 0; // scripted human: first offered combo
      }
      if (!(index >= 0 && index < dec.comboData.length)) { badIndex++; index = 0; }
      driver.commit(index);
      moves++;
    }
    return { ended, moves, badIndex, maxOver, endValue };
  };

  // ---- ADZ Explorer ----
  {
    const iters = 16;
    const { bot, counter } = await makeBot(ADZ, iters);
    check(`buildBot(${ADZ}, iters=${iters}) is an Explorer`, bot instanceof ExplorerBot && bot.isExplorer === true);
    check(`ADZ Explorer exposes no chooseRedirect (random handoff)`, typeof bot.chooseRedirect !== 'function');
    const driver = new GameDriver(M, { numPlayers: 2, seatBots: [bot, null], maxHistory: bot.maxHistory, seed: 4242 });
    const r = await playGame(driver, iters, counter);
    check(`ADZ Explorer game reaches a terminal state (moves=${r.moves}, endValue=${r.endValue})`, r.ended && r.moves > 0);
    check('ADZ Explorer never returned an out-of-range index', r.badIndex === 0);
    check(`ADZ Explorer runs <= iters+1 forward passes/move (worst over-budget=${r.maxOver})`, r.maxOver <= 0);
    driver.dispose();
  }

  // ---- AZ Explorer ----
  {
    const iters = 16;
    const { bot, counter } = await makeBot(AZ, iters);
    check(`buildBot(${AZ}, iters=${iters}) is an Explorer`, bot instanceof ExplorerBot && bot.isExplorer === true);
    check(`AZ Explorer delegates chooseRedirect (value-argmax)`, typeof bot.chooseRedirect === 'function');
    const driver = new GameDriver(M, { numPlayers: 3, seatBots: [bot, bot, null], maxHistory: bot.maxHistory, seed: 77 });
    const r = await playGame(driver, iters, counter);
    check(`AZ Explorer game reaches a terminal state (moves=${r.moves}, endValue=${r.endValue})`, r.ended && r.moves > 0);
    check('AZ Explorer never returned an out-of-range index', r.badIndex === 0);
    check(`AZ Explorer runs <= iters+1 forward passes/move (worst over-budget=${r.maxOver})`, r.maxOver <= 0);
    driver.dispose();
  }

  // ---- determinism: same root + same seed -> same chosen index ----
  {
    const { bot: a } = await makeBot(ADZ, 32);
    const { bot: b } = await makeBot(ADZ, 32);
    const driver = new GameDriver(M, { numPlayers: 2, seatBots: [null, null], seed: 5 });
    driver.newGame();
    let checked = 0, mismatch = 0;
    for (let steps = 0; steps < 60 && checked < 4; steps++) {
      const dec = driver.prepare();
      if (dec.kind === 'ended') { driver.newGame(); continue; }
      if (dec.kind === 'auto') { driver.commit(-1); continue; }
      const hist = driver.history.map((p) => p.to_string());
      const ia = await a.search(dec.decisionPhaseString, hist, dec.comboData);
      const ib = await b.search(dec.decisionPhaseString, hist, dec.comboData);
      if (ia !== ib) mismatch++;
      checked++;
      driver.commit(0);
    }
    check(`Explorer search is deterministic under a fixed seed (checked=${checked})`, checked > 0 && mismatch === 0);
    driver.dispose();
  }

  console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
  return failures;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exit((await runSmoke()) ? 1 : 0);
}
