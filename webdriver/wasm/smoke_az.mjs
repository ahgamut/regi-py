/* In-env smoke for the AZ Direct-net bot (real onnxruntime-web under node). Drives
 * full all-bot games with each AZ net through GameDriver and checks: indices in range,
 * snapshots well-formed, every offered ATTACK combo has a combomap cell, net outputs
 * are finite, and chooseRedirect returns a valid other seat. This is the in-env
 * structural check; exact index parity vs Python is the golden step (check_golden.mjs).
 *
 *   node smoke_az.mjs            (needs onnxruntime-web installed + dist/<net>.onnx;
 *                                 or run all suites via `npm run smoke`)
 *
 * Exports runSmoke() so smoke_all.mjs can call it as a function (returns the failure
 * count); running this file directly runs just this suite and exits non-zero on fail. */
import { pathToFileURL } from 'node:url';
import { readFileSync } from 'node:fs';
import RegiModule from './dist/regicore.mjs';
import { GameDriver } from './game_driver.mjs';
import { AZBot } from './az_bot.mjs';
import { bitwiseOfLocations } from './net_common.mjs';

const NETS = ['basic', 'percardmlp', 'cardtx', 'mixer'];
const snapshotOk = (s) => s && s.numPlayers >= 2 && s.hands.length === s.numPlayers &&
  Number.isFinite(s.progress) && s.enemiesLeft >= 0;

export async function runSmoke() {
  const ortMod = await import('onnxruntime-web');
  const ort = ortMod.Tensor ? ortMod : ortMod.default;
  ort.env.wasm.numThreads = 1;
  const M = await RegiModule();
  const comboMap = M.combo_map();
  let failures = 0;
  const ok = (cond, msg) => { console.log(`${cond ? 'ok  ' : 'FAIL'}  ${msg}`); if (!cond) failures++; };

  const loadBot = async (net) => {
    const contract = JSON.parse(readFileSync(`./dist/${net}.io.json`, 'utf8'));
    const session = await ort.InferenceSession.create(`./dist/${net}.onnx`);
    return new AZBot(M, ort, session, contract, comboMap);
  };

  for (const net of NETS) {
    const bot = await loadBot(net);
    let moves = 0, badIndex = 0, badSnap = 0, atkChecked = 0, comboMisses = 0, nonFinite = 0, reached = 0;
    for (const numPlayers of [2, 3, 4]) {
      const seatBots = new Array(numPlayers).fill(bot);
      const driver = new GameDriver(M, { numPlayers, seatBots, maxHistory: bot.maxHistory, seed: 100 + numPlayers });
      driver.newGame();
      let guard = 0;
      while (guard++ < 400) {
        const dec = driver.prepare();
        if (dec.kind === 'ended') { reached++; break; }
        if (!snapshotOk(driver.snapshot())) badSnap++;
        if (dec.kind === 'auto') { driver.commit(-1); continue; }
        // combomap coverage + finite outputs on attack decisions
        if (dec.attacking) {
          for (const locs of dec.built.comboLocs) {
            atkChecked++;
            if (!(bitwiseOfLocations(locs).toString() in comboMap)) comboMisses++;
          }
          const out = await bot.session.run(dec.built.feeds);
          for (const v of out.a.data) if (!Number.isFinite(v)) { nonFinite++; break; }
        }
        const index = await bot.runFeeds(dec.built);
        if (index < -1 || index >= dec.K) badIndex++;
        driver.commit(index);
        moves++;
      }
      driver.dispose();
    }
    ok(moves > 0 && reached === 3, `${net}: 3 all-bot games (2/3/4p) reached terminal (moves=${moves})`);
    ok(badIndex === 0, `${net}: bot indices in range (bad=${badIndex})`);
    ok(badSnap === 0, `${net}: snapshots well-formed (bad=${badSnap})`);
    ok(comboMisses === 0, `${net}: every offered attack combo has a combomap cell (checked=${atkChecked}, misses=${comboMisses})`);
    ok(nonFinite === 0, `${net}: net 'a' outputs finite (nonfinite decisions=${nonFinite})`);

    // redirect: a valid other seat in a 3p game
    const g = new GameDriver(M, { numPlayers: 3, seatBots: [bot, bot, bot], maxHistory: bot.maxHistory, seed: 7 });
    g.newGame();
    const rawPhase = M.PhaseInfo.from_string(g.phaseString);
    const active = rawPhase.active_player;
    const target = await bot.chooseRedirect(rawPhase, [], 3);
    rawPhase.delete(); g.dispose();
    ok(target >= 0 && target < 3 && target !== active, `${net}: chooseRedirect -> valid other seat (${target} != ${active})`);
  }

  console.log(failures === 0 ? '\nall checks passed' : `\n${failures} checks FAILED`);
  return failures;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exit((await runSmoke()) ? 1 : 0);
}
