/* Headless integration test for GameDriver (the browser game loop) using the REAL
 * onnxruntime-web bot -- the whole Phase-5 engine loop minus the DOM.
 * Build the WASM module + export at least dist/adzpool.onnx first, then:
 *   node smoke_driver.mjs [--net adzpool] [--dist ./dist] [--ort onnxruntime-web]
 *
 * Checks: (1) two NetBot seats play a full game to a terminal state via the
 * peek/advance re-seat loop; (2) every peeked combo index the bot picks is in
 * range; (3) a mixed human(scripted)/bot game also terminates; (4) the board
 * snapshot stays well-formed throughout. */
import { readFileSync } from 'node:fs';
import RegiModule from './dist/regicore.mjs';
import { NetBot } from './adz_bot.mjs';
import { GameDriver } from './game_driver.mjs';

function arg(name, def) { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : def; }
const NET = arg('net', 'adzpool');
const DIST = arg('dist', './dist');
const ORT_PKG = arg('ort', 'onnxruntime-web');

const ortMod = await import(ORT_PKG);
const ort = ortMod.Tensor ? ortMod : ortMod.default;
const M = await RegiModule();
let failures = 0;
const check = (name, cond) => { console.log(`${cond ? 'ok  ' : 'FAIL'}  ${name}`); if (!cond) failures++; };

async function makeBot() {
  const contract = JSON.parse(readFileSync(`${DIST}/${NET}.io.json`, 'utf8'));
  const session = await ort.InferenceSession.create(`${DIST}/${NET}.onnx`);
  return new NetBot(M, ort, session, contract);
}

function snapshotOk(s) {
  return s && Number.isInteger(s.numPlayers) && Array.isArray(s.hands) &&
    s.hands.length === s.numPlayers && s.handCounts.every((n) => n >= 0) &&
    (s.enemy === null || (typeof s.enemy.hp === 'number'));
}

/* Play a full game; decideFor(kind, dec, seat) -> index for human seats. */
async function playGame(driver, decideFor) {
  driver.newGame();
  let moves = 0, autos = 0, badIndex = 0, badSnap = 0, endValue = 0, steps = 0;
  for (; steps < 40000; steps++) {
    const dec = driver.prepare();
    if (dec.kind === 'ended') { endValue = dec.endValue; break; }
    if (dec.kind === 'auto') { driver.commit(-1); autos++; continue; }
    if (!snapshotOk(driver.snapshot())) badSnap++;
    let index;
    if (dec.isBot) index = await driver.seatBots[dec.activeSeat].runFeeds(dec.feeds, dec.K);
    else index = decideFor(dec.attacking ? 'attack' : 'defense', dec, dec.activeSeat);
    if (!(index >= 0 && index < dec.comboData.length)) { badIndex++; index = 0; }
    driver.commit(index);
    moves++;
  }
  return { ended: driver.prepare().kind === 'ended', moves, autos, badIndex, badSnap, endValue, snap: driver.snapshot() };
}

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

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
