/* Headless integration test for GameDriver (the browser game loop) using the REAL
 * onnxruntime-web bot -- the whole Phase-5 engine loop minus the DOM.
 * Build the WASM module + export at least dist/adzpool.onnx first, then:
 *   node smoke_driver.mjs [--net adzpool] [--dist ./dist] [--ort onnxruntime-web]
 *                         (or all suites via `npm run smoke`)
 *
 * Checks: (1) two NetBot seats play a full game to a terminal state via the
 * peek/advance re-seat loop; (2) every peeked combo index the bot picks is in
 * range; (3) a mixed human(scripted)/bot game also terminates; (4) the board
 * snapshot stays well-formed throughout.
 *
 * Exports runSmoke() so smoke_all.mjs can call it as a function (returns the failure
 * count); running this file directly runs just this suite and exits non-zero on fail. */
import { pathToFileURL } from 'node:url';
import { readFileSync } from 'node:fs';
import RegiModule from './dist/regicore.mjs';
import { buildBot } from './load_bot.mjs';
import { GameDriver } from './game_driver.mjs';

function arg(name, def) { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : def; }
const NET = arg('net', 'adzpool');       // ADZ net for the two-bot / mixed games
const AZ_NET = arg('az-net', 'basic');   // AZ net to exercise loadBot's paradigm dispatch
const DIST = arg('dist', './dist');
const ORT_PKG = arg('ort', 'onnxruntime-web');

function snapshotOk(s) {
  return s && Number.isInteger(s.numPlayers) && Array.isArray(s.hands) &&
    s.hands.length === s.numPlayers && s.handCounts.every((n) => n >= 0) &&
    (s.enemy === null || (typeof s.enemy.hp === 'number'));
}

export async function runSmoke() {
  const ortMod = await import(ORT_PKG);
  const ort = ortMod.Tensor ? ortMod : ortMod.default;
  const M = await RegiModule();
  let failures = 0;
  const check = (name, cond) => { console.log(`${cond ? 'ok  ' : 'FAIL'}  ${name}`); if (!cond) failures++; };

  // buildBot is the exact dispatch app.mjs's loadBot uses (adz -> NetBot, az -> AZBot
  // + combomap); here we read the files ourselves since node fetch can't take a path.
  const comboMap = M.combo_map();
  const makeBot = async (net = NET) => {
    const contract = JSON.parse(readFileSync(`${DIST}/${net}.io.json`, 'utf8'));
    const session = await ort.InferenceSession.create(`${DIST}/${net}.onnx`);
    return buildBot(M, ort, session, contract, { comboMap });
  };

  /* Play a full game; decideFor(kind, dec, seat) -> index for human seats. */
  const playGame = async (driver, decideFor) => {
    driver.newGame();
    let moves = 0, autos = 0, badIndex = 0, badSnap = 0, endValue = 0, steps = 0;
    for (; steps < 40000; steps++) {
      const dec = driver.prepare();
      if (dec.kind === 'ended') { endValue = dec.endValue; break; }
      if (dec.kind === 'auto') { driver.commit(-1); autos++; continue; }
      if (!snapshotOk(driver.snapshot())) badSnap++;
      let index;
      if (dec.isBot) index = await driver.seatBots[dec.activeSeat].runFeeds(dec.built);
      else index = decideFor(dec.attacking ? 'attack' : 'defense', dec, dec.activeSeat);
      if (!(index >= 0 && index < dec.comboData.length)) { badIndex++; index = 0; }
      driver.commit(index);
      moves++;
    }
    return { ended: driver.prepare().kind === 'ended', moves, autos, badIndex, badSnap, endValue, snap: driver.snapshot() };
  };

  // 1) two-bot game
  {
    const bot = await makeBot();
    const driver = new GameDriver(M, { numPlayers: 2, seatBots: [bot, bot], maxHistory: 8, seed: 12345 });
    const r = await playGame(driver, () => 0);
    check(`two-bot game reaches a terminal state (moves=${r.moves}, autos=${r.autos})`, r.ended && r.moves > 0);
    check('bot never returned an out-of-range index', r.badIndex === 0);
    check('board snapshot well-formed throughout (2-bot)', r.badSnap === 0);
    check(`terminal endValue is win/loss, not 0 (endValue=${r.endValue})`, r.endValue === 1 || r.endValue === -1);
    driver.dispose();
  }

  // 2) mixed human(scripted: always the last offered combo)/bot 3-player game
  {
    const bot = await makeBot();
    const driver = new GameDriver(M, { numPlayers: 3, seatBots: [null, bot, bot], maxHistory: 8, seed: 999 });
    const r = await playGame(driver, (kind, dec) => dec.comboData.length - 1);
    check(`mixed human/bot game terminates (moves=${r.moves})`, r.ended && r.moves > 0);
    check('board snapshot well-formed throughout (mixed)', r.badSnap === 0);
    driver.dispose();
  }

  // 3) AZ card-space bot drives through the SAME prepare/commit `built` path (loadBot
  //    must have dispatched to AZBot, which scores via the combomap grid / keepy).
  {
    const bot = await makeBot(AZ_NET);
    check(`loadBot('${AZ_NET}') dispatches to an AZ bot (has chooseRedirect)`, typeof bot.chooseRedirect === 'function');
    const driver = new GameDriver(M, { numPlayers: 4, seatBots: [bot, bot, bot, bot], maxHistory: 8, seed: 7 });
    const r = await playGame(driver, () => 0);
    check(`AZ-net 4-bot game reaches a terminal state (moves=${r.moves})`, r.ended && r.moves > 0);
    check('AZ bot never returned an out-of-range index', r.badIndex === 0);

    // 4) botRedirect path: on a real decision phase, chooseRedirect (AZ value-argmax)
    //    must hand off to a VALID other seat -- what app.mjs asks on a bot Joker attack.
    driver.newGame();
    let redirOk = true, redirChecked = 0;
    for (let steps = 0; steps < 200 && redirChecked < 5; steps++) {
      const dec = driver.prepare();
      if (dec.kind === 'ended') { driver.newGame(); continue; }
      if (dec.kind === 'auto') { driver.commit(-1); continue; }
      const phase = M.PhaseInfo.from_string(dec.decisionPhaseString);
      const t = await bot.chooseRedirect(phase, driver.history, driver.numPlayers);
      phase.delete();
      if (!(t >= 0 && t < driver.numPlayers && t !== dec.activeSeat)) redirOk = false;
      redirChecked++;
      driver.commit(await bot.runFeeds(dec.built));
    }
    check(`AZ chooseRedirect returns a valid other seat (checked=${redirChecked})`, redirOk && redirChecked > 0);
    driver.dispose();
  }

  // 5) preset opening: newGame(startPhase) replays a committed preset via init_string
  //    (what the menu's #cfg-preset does), captures start{Phase,Seed}, and plays out.
  {
    const numPlayers = 3;
    const preset = JSON.parse(readFileSync('./tables/presets_3p.json', 'utf8'));
    check(`presets_3p.json is a ${numPlayers}p set with phases`,
      preset.num_players === numPlayers && Array.isArray(preset.phases) && preset.phases.length > 0);
    const startPhase = preset.phases[0];
    const bot = await makeBot();
    const driver = new GameDriver(M, { numPlayers, seatBots: [bot, bot, bot], maxHistory: 8, seed: 4242 });
    const snap = driver.newGame(startPhase);
    check('newGame(startPhase) deals the requested player count', snap.numPlayers === numPlayers);
    check('newGame(startPhase) captures startPhase/startSeed', driver.startPhase === driver.phaseString && driver.startSeed === 4242);

    // Same preset + same seed => identical opening board (reproducible replay).
    const d2 = new GameDriver(M, { numPlayers, seatBots: [bot, bot, bot], maxHistory: 8, seed: 4242 });
    const s2 = d2.newGame(startPhase);
    check('preset replay is reproducible under a fixed seed',
      d2.startPhase === driver.startPhase && s2.drawPileSize === snap.drawPileSize && JSON.stringify(s2.handCounts) === JSON.stringify(snap.handCounts));
    d2.dispose();

    // And it still drives to a terminal state.
    let steps = 0, ended = false;
    for (; steps < 40000; steps++) {
      const dec = driver.prepare();
      if (dec.kind === 'ended') { ended = true; break; }
      if (dec.kind === 'auto') { driver.commit(-1); continue; }
      driver.commit(dec.isBot ? await driver.seatBots[dec.activeSeat].runFeeds(dec.built) : 0);
    }
    check('preset game reaches a terminal state', ended);
    driver.dispose();
  }

  console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
  return failures;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exit((await runSmoke()) ? 1 : 0);
}
