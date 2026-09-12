/* Client-side Regicide webapp: the C++ engine (WASM) + net bots (onnxruntime-web)
 * run entirely in the browser. No server game state, no per-move round-trip.
 * This module wires GameDriver's prepare()/commit() loop to the DOM: bot seats run
 * a NetBot forward pass, the human seat waits for a click on an offered move. */
import RegiModule from './dist/regicore.mjs';
import { GameDriver } from './game_driver.mjs';
import { loadNetBot } from './adz_bot.mjs';

// ---- onnxruntime-web source (edit these two lines to change where ORT loads) ----
// The ESM loads its wasm sidecar (~10MB) from ORT_WASM_PATHS, NOT inlined, so both
// must point at the same dist dir.
//   npm:            ./node_modules/onnxruntime-web/dist/         (after `npm install`)
//   no-npm, CDN:    https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/
//   no-npm, vendor: ./vendor/   (copy ort.wasm.bundle.min.mjs + ort-wasm-*.wasm there)
const ORT_DIST = './node_modules/onnxruntime-web/dist/';
const ort = await import(ORT_DIST + 'ort.wasm.bundle.min.mjs');
// single-threaded wasm backend: no SharedArrayBuffer, so no COOP/COEP headers
// needed (works from a plain static host / GitHub Pages).
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = ORT_DIST; // where the ort-wasm-*.wasm sidecar is fetched from

const BOT_MOVE_DELAY_MS = 550; // let the human watch bot moves
const $ = (id) => document.getElementById(id);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };

let M = null;
const botCache = new Map(); // net name -> NetBot (session reused across games)
let driver = null;
let humanSeat = 0;
let running = false; // a game loop is active (guards New-game re-entrancy)

RegiModule().then((mod) => { M = mod; $('boot').textContent = 'ready'; $('new-game').disabled = false; });
$('new-game').disabled = true;

/* ---- card rendering ---- */
function cardEl(label) {
  const e = el('span', 'pc', label);
  const suit = label[label.length - 1];
  if (suit === 'H' || suit === 'D') e.classList.add('suit-H');
  return e;
}
function renderCards(container, labels) {
  container.replaceChildren();
  for (const l of labels) container.appendChild(cardEl(l));
}

/* ---- board rendering ---- */
function render(snap) {
  const enemy = snap.enemy;
  $('enemy-name').textContent = enemy ? enemy.label : '—';
  $('enemy-strength').textContent = enemy ? enemy.strength : 0;
  const hp = enemy ? Math.max(0, enemy.hp) : 0;
  const hpMax = enemy ? Math.max(hp, enemy.strength, 10) : 10; // rough scale for the bar
  $('enemy-hp-txt').textContent = enemy ? `${enemy.hp}` : '—';
  $('enemy-hp-bar').style.width = `${enemy ? Math.min(100, (hp / hpMax) * 100) : 0}%`;
  $('prog-txt').textContent = `${snap.progress} / 360`;
  $('prog-bar').style.width = `${Math.max(0, Math.min(100, (snap.progress / 360) * 100))}%`;
  $('enemies-left').textContent = snap.enemiesLeft;
  $('draw-size').textContent = snap.drawPileSize;
  $('cur-block').textContent = snap.currentBlock;

  const seats = $('seats'); seats.replaceChildren();
  for (let i = 0; i < snap.numPlayers; i++) {
    const s = el('div', 'seat' + (i === snap.activeSeat ? ' active' : ''));
    s.appendChild(el('div', 'who', i === humanSeat ? `Seat ${i} — you` : `Seat ${i}`));
    s.appendChild(el('div', 'kind', `${snap.handCounts[i]} cards · ${i === humanSeat ? 'human' : 'bot'}`));
    seats.appendChild(s);
  }
  renderCards($('used-combos'), snap.usedCombos);
  renderCards($('your-hand'), (snap.hands[humanSeat] || []).map((c) => c.label));
}

/* ---- log ---- */
function log(html, cls) {
  const row = el('div', 'row' + (cls ? ' ' + cls : ''));
  row.innerHTML = html;
  $('log').prepend(row);
}
function moveText(dec, idx) {
  const c = dec.comboData[idx];
  const who = dec.isBot ? `Bot seat ${dec.activeSeat}` : `You`;
  const what = !c ? '?' : c.isYield ? 'yields' : `${dec.attacking ? 'attacks' : 'defends'} with ${c.labels.join(' ')}`;
  return `<b>${who}</b> ${what}`;
}

/* ---- human decision ---- */
let resolveHuman = null;
function presentHuman(dec) {
  $('turn-status').textContent = `Your turn — ${dec.attacking ? 'choose an attack' : 'choose a defense'}:`;
  $('turn-status').className = 'status you';
  const moves = $('moves'); moves.replaceChildren();
  for (const c of dec.comboData) {
    const b = el('button', 'move' + (c.isYield ? ' yield' : ''));
    if (c.isYield) b.appendChild(el('span', null, 'Yield'));
    else {
      const cc = el('div', 'combo-cards');
      for (const l of c.labels) cc.appendChild(cardEl(l));
      b.appendChild(cc);
    }
    b.appendChild(el('span', 'lab', c.isYield ? 'pass' : `move ${c.index}`));
    b.addEventListener('click', () => { if (resolveHuman) { const r = resolveHuman; resolveHuman = null; r(c.index); } });
    moves.appendChild(b);
  }
  return new Promise((res) => { resolveHuman = res; });
}
function clearMoves() { $('moves').replaceChildren(); }

/* ---- main loop ---- */
async function loop() {
  while (true) {
    const dec = driver.prepare();
    render(driver.snapshot());
    if (dec.kind === 'ended') { finish(dec.endValue); return; }
    if (dec.kind === 'auto') { driver.commit(-1); continue; }

    let index;
    if (dec.isBot) {
      $('turn-status').textContent = `Bot (seat ${dec.activeSeat}) is thinking…`;
      $('turn-status').className = 'status think';
      clearMoves();
      index = await driver.seatBots[dec.activeSeat].runFeeds(dec.feeds, dec.K);
      log(moveText(dec, index));
      driver.commit(index);
      await new Promise((r) => setTimeout(r, BOT_MOVE_DELAY_MS));
    } else {
      index = await presentHuman(dec);
      clearMoves();
      log(moveText(dec, index));
      driver.commit(index);
    }
  }
}

function finish(endValue) {
  running = false;
  clearMoves();
  const won = endValue === 1;
  $('turn-status').textContent = won ? 'Victory — all enemies defeated!' : 'Defeat — the castle falls.';
  $('turn-status').className = 'status';
  log(won ? 'The party <b>WINS</b> — all 12 royals defeated.' : 'The party <b>LOSES</b>.', won ? 'win' : 'loss');
  $('overlay-res').textContent = won ? 'Victory' : 'Defeat';
  $('overlay-res').className = 'res ' + (won ? 'win' : 'loss');
  $('overlay-sub').textContent = won ? 'All royals cleared.' : 'A player could not block, or ran out of moves.';
  $('overlay').classList.add('show');
}

/* ---- new game ---- */
async function newGame() {
  if (!M || running) return;
  $('overlay').classList.remove('show');
  const numPlayers = parseInt($('cfg-players').value, 10);
  const netName = $('cfg-net').value;
  const spectate = $('cfg-spectate').checked;
  const seedRaw = $('cfg-seed').value.trim();
  const seed = seedRaw === '' ? null : (parseInt(seedRaw, 10) >>> 0);
  humanSeat = spectate ? -1 : 0;

  $('new-game').disabled = true;
  $('boot').textContent = 'loading net…';
  let bot = botCache.get(netName);
  if (!bot) { bot = await loadNetBot(M, ort, './dist', netName); botCache.set(netName, bot); }
  $('boot').textContent = 'ready';
  $('new-game').disabled = false;

  const seatBots = [];
  for (let i = 0; i < numPlayers; i++) seatBots.push(i === humanSeat ? null : bot);
  if (driver) driver.dispose();
  driver = new GameDriver(M, { numPlayers, seatBots, maxHistory: bot.maxHistory, seed });
  driver.newGame();
  $('log').replaceChildren();
  $('you-seat-note').textContent = spectate ? '(spectating — all bots)' : '(seat 0)';
  $('you-panel').classList.toggle('hidden', false);
  log(`New ${numPlayers}-player game — ${spectate ? 'all bots' : 'you are seat 0'} · net ${netName}`);
  running = true;
  loop();
}

$('new-game').addEventListener('click', newGame);
$('overlay-again').addEventListener('click', () => { $('overlay').classList.remove('show'); newGame(); });
