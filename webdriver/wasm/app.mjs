/* Client-side Regicide (regi-py) webapp. The C++ engine (WASM) + net bots
 * (onnxruntime-web) run entirely in the browser -- no server game state, no
 * per-move round-trip. Four screens (intro / how-to / menu / game); on the game
 * screen the human picks CARDS from their hand (not a move list) and submits. */
import RegiModule from './dist/regicore.mjs';
import { GameDriver } from './game_driver.mjs';
import { loadNetBot } from './adz_bot.mjs';

// ---- onnxruntime-web source (edit these two lines to change where ORT loads) ----
//   npm:            ./node_modules/onnxruntime-web/dist/
//   no-npm, CDN:    https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/
//   no-npm, vendor: ./vendor/
const ORT_DIST = './node_modules/onnxruntime-web/dist/';
const ort = await import(ORT_DIST + 'ort.wasm.bundle.min.mjs');
ort.env.wasm.numThreads = 1; // single-thread wasm: no SharedArrayBuffer / COOP-COEP
// ort.env.wasm.wasmPaths = ORT_DIST; // where the ort-wasm-*.wasm sidecar is fetched from

const NETS = ['adzpool', 'adzmulti'];
const HUMAN_SEAT = 0;          // the human always takes seat 0
const BOT_MOVE_DELAY_MS = 550; // let the human watch bot moves

const $ = (id) => document.getElementById(id);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };

let M = null;
const botCache = new Map();      // net name -> NetBot (session reused across games)
let driver = null;
let playerName = 'Player';
let loopToken = 0;               // bumped to abandon an in-flight game loop

/* ================= screens ================= */
function showScreen(id) {
  for (const s of document.querySelectorAll('.screen')) s.classList.toggle('show', s.id === `screen-${id}`);
  if (id === 'menu') buildOpponents();
}
document.querySelectorAll('[data-nav]').forEach((b) => b.addEventListener('click', () => showScreen(b.dataset.nav)));
$('intro-play').addEventListener('click', () => showScreen('menu'));
$('intro-howto').addEventListener('click', () => showScreen('howto'));

/* how-to tabs */
document.querySelectorAll('#howto-tabs .tab').forEach((t) => t.addEventListener('click', () => {
  document.querySelectorAll('#howto-tabs .tab').forEach((x) => x.classList.toggle('active', x === t));
  document.querySelectorAll('.howto-pane').forEach((p) => { p.hidden = p.dataset.pane !== t.dataset.howto; });
}));

/* boot the engine, then unlock Play */
$('intro-play').disabled = true;
RegiModule().then((mod) => { M = mod; $('boot').textContent = 'engine ready'; $('intro-play').disabled = false; });

/* ================= menu ================= */
function buildOpponents() {
  const numPlayers = parseInt($('cfg-players').value, 10);
  const wrap = $('opponents');
  const prev = {};
  wrap.querySelectorAll('select').forEach((s) => { prev[s.dataset.seat] = s.value; });
  wrap.replaceChildren();
  for (let i = 1; i < numPlayers; i++) { // seat 0 is you
    const row = el('div', 'opp-row');
    row.appendChild(el('span', 'seat-name', `Seat ${i}`));
    const sel = el('select');
    sel.dataset.seat = i;
    for (const n of NETS) { const o = el('option', null, n); o.value = n; sel.appendChild(o); }
    sel.value = prev[i] || 'adzpool';
    row.appendChild(sel);
    row.appendChild(el('span', 'tag', 'bot'));
    wrap.appendChild(row);
  }
}
$('cfg-players').addEventListener('change', buildOpponents);
$('menu-start').addEventListener('click', startGame);

async function startGame() {
  if (!M) return;
  const numPlayers = parseInt($('cfg-players').value, 10);
  const seedRaw = $('cfg-seed').value.trim();
  const seed = seedRaw === '' ? null : (parseInt(seedRaw, 10) >>> 0);
  playerName = ($('cfg-name').value.trim() || 'Player').slice(0, 16);

  const netForSeat = {};
  $('opponents').querySelectorAll('select').forEach((s) => { netForSeat[+s.dataset.seat] = s.value; });

  $('menu-start').disabled = true;
  const seatBots = [];
  for (let i = 0; i < numPlayers; i++) {
    if (i === HUMAN_SEAT) { seatBots.push(null); continue; }
    const net = netForSeat[i] || 'adzpool';
    let bot = botCache.get(net);
    if (!bot) { bot = await loadNetBot(M, ort, './dist', net); botCache.set(net, bot); }
    seatBots.push(bot);
  }
  $('menu-start').disabled = false;

  const maxHistory = seatBots.find((b) => b)?.maxHistory ?? 8;
  if (driver) driver.dispose();
  driver = new GameDriver(M, { numPlayers, seatBots, maxHistory, seed });
  driver.newGame();

  $('log').replaceChildren();
  $('overlay').classList.remove('show');
  $('you-seat-note').textContent = `· you are “${playerName}”`;
  log(`New ${numPlayers}-player game — you are seat 0.`);
  showScreen('game');
  loopToken++;
  loop(loopToken);
}

$('game-menu').addEventListener('click', () => { loopToken++; showScreen('menu'); });
$('overlay-menu').addEventListener('click', () => { $('overlay').classList.remove('show'); showScreen('menu'); });
$('overlay-again').addEventListener('click', startGame);

/* ================= card rendering ================= */
const SUIT_SYM = { C: '♣', D: '♦', H: '♥', S: '♠' };
const RANK_NAME = { J: 'Jack', Q: 'Queen', K: 'King' };
function parseCard(label) {
  // labels are "<rank><suit>" with rank in A23456789TJQK and suit CDHS; joker = "X!"
  const rank = label[0], suitCh = label[1];
  const joker = rank === 'X';
  const disp = joker ? '★' : (rank === 'T' ? '10' : rank);
  const suit = joker ? '' : (SUIT_SYM[suitCh] || suitCh);
  const red = suitCh === 'H' || suitCh === 'D';
  return { disp, suit, red, joker };
}
function cardStrength(label) {
  // pip values (core Card::strength): A=1, 2-9, T=J=10, Q=15, K=20, joker=0
  const r = label[0];
  if (r === 'A') return 1; if (r === 'T' || r === 'J') return 10;
  if (r === 'Q') return 15; if (r === 'K') return 20; if (r === 'X') return 0;
  const n = parseInt(r, 10); return Number.isNaN(n) ? 0 : n;
}
function cardEl(card, opts = {}) {
  const p = parseCard(card.label);
  const cls = 'card' + (opts.mini ? ' mini' : '') + (opts.royal ? ' royal' : '') +
    (p.red ? ' red' : '') + (p.joker ? ' joker' : '');
  const e = el('div', cls);
  if (card.location != null) e.dataset.loc = card.location;
  e.appendChild(el('span', 'rank', p.disp));
  e.appendChild(el('span', 'suit', p.suit));
  if (!opts.mini && !p.joker) e.appendChild(el('span', 'rank br', p.disp));
  return e;
}
function royalName(label) {
  const p = parseCard(label);
  return `${RANK_NAME[label[0]] || label[0]} ${p.suit}`;
}
function renderPile(id, count) {
  const p = $(id); p.replaceChildren();
  p.appendChild(el('div', 'card back' + (count > 0 ? '' : ' empty')));
}

/* ================= board render ================= */
function render(snap) {
  const enemy = snap.enemy;
  // current royal as a face-up card + piles as card backs
  const rc = $('royal-card'); rc.replaceChildren();
  if (enemy) { rc.appendChild(cardEl({ label: enemy.label }, { royal: true })); $('royal-label').textContent = royalName(enemy.label); }
  else { rc.appendChild(el('div', 'card royal empty')); $('royal-label').textContent = 'cleared'; }
  renderPile('draw-pile', snap.drawPileSize);
  renderPile('discard-pile', snap.discardPileSize);
  $('draw-size').textContent = snap.drawPileSize;
  $('disc-size').textContent = snap.discardPileSize;

  $('enemy-strength').textContent = enemy ? enemy.strength : 0;
  $('cur-block').textContent = snap.currentBlock;
  const hp = enemy ? Math.max(0, enemy.hp) : 0;
  const hpMax = enemy ? Math.max(hp, enemy.strength, 10) : 10;
  $('enemy-hp-lbl').textContent = enemy ? `${enemy.hp}` : '—';
  $('enemy-hp-bar').style.width = `${enemy ? Math.min(100, (hp / hpMax) * 100) : 0}%`;

  const incoming = enemy ? Math.max(0, enemy.strength - snap.currentBlock) : 0;
  const inc = $('incoming');
  inc.textContent = enemy ? (incoming > 0 ? `Hits for ${incoming}` : 'Full-blocked') : '—';
  inc.classList.toggle('blocked', incoming === 0);

  $('prog-txt').textContent = `${snap.progress} / 360`;
  $('prog-bar').style.width = `${Math.max(0, Math.min(100, (snap.progress / 360) * 100))}%`;
  $('enemies-left').textContent = snap.enemiesLeft;

  const seats = $('seats'); seats.replaceChildren();
  for (let i = 0; i < snap.numPlayers; i++) {
    const you = i === HUMAN_SEAT;
    const s = el('div', 'seat' + (i === snap.activeSeat ? ' active' : '') + (you ? ' you-seat' : ''));
    s.appendChild(el('div', 'who', you ? playerName : `Seat ${i}`));
    s.appendChild(el('div', 'kind', `${snap.handCounts[i]} cards · ${you ? 'you' : 'bot'}`));
    seats.appendChild(s);
  }

  const used = $('used-combos'); used.replaceChildren();
  if (!snap.usedCombos.length) used.appendChild(el('span', 'note', 'nothing yet'));
  for (const combo of snap.usedCombos) {
    const g = el('div', 'combo-group');
    if (!combo.length) g.appendChild(el('span', 'note', 'yield'));
    for (const c of combo) g.appendChild(cardEl(c, { mini: true }));
    used.appendChild(g);
  }

  // your hand, read-only here (presentHuman makes it pickable on your turn)
  const hand = $('your-hand'); hand.replaceChildren();
  hand.classList.remove('pickable'); hand.classList.add('locked');
  for (const c of (snap.hands[HUMAN_SEAT] || [])) hand.appendChild(cardEl(c));
}

/* ================= log + status ================= */
function log(html, cls) { const row = el('div', 'row' + (cls ? ' ' + cls : '')); row.innerHTML = html; $('log').prepend(row); }
function moveText(dec, idx) {
  const c = dec.comboData[idx];
  const who = dec.isBot ? `Seat ${dec.activeSeat}` : playerName;
  const what = !c ? '?' : c.isYield ? 'yields' : `${dec.attacking ? 'attacks' : 'defends'} with ${c.labels.join(' ')}`;
  return `<b>${who}</b> ${what}`;
}
function setPill(text, cls) { const p = $('turn-status'); p.textContent = text; p.className = 'turn-pill' + (cls ? ' ' + cls : ''); }

/* ================= human decision (pick cards) ================= */
function setCombat(dec, snap) {
  const combat = $('combat'); combat.replaceChildren();
  if (dec.attacking) {
    combat.appendChild(el('span', null, `Attack ${snap.enemy ? snap.enemy.label : ''}.`));
  } else {
    combat.appendChild(el('span', 'dmg', `Defend ${dec.damage ?? 0} damage.`));
    combat.appendChild(el('span', 'need', `(block ${snap.currentBlock} already up)`));
  }
  return combat;
}

function presentHuman(dec, snap) {
  setPill(`Your turn — ${dec.attacking ? 'choose an attack' : 'defend'}`, 'you');
  const combat = setCombat(dec, snap);
  const selSum = el('span', 'sel-sum'); combat.appendChild(selSum);

  const hand = $('your-hand');
  hand.classList.add('pickable'); hand.classList.remove('locked');
  const selected = new Set();
  const yieldCombo = dec.comboData.find((c) => c.isYield);
  const sb = $('btn-submit');
  let hovering = false;

  const canYield = () => selected.size === 0 && !!yieldCombo;
  const submitLabel = () => (hovering && canYield()) ? 'Yield' : 'Submit';
  const matchIndex = () => {
    for (const c of dec.comboData) {
      if (c.locations.length === selected.size && c.locations.every((l) => selected.has(l))) return c.index;
    }
    return -1;
  };
  const refresh = () => {
    let sum = 0; for (const c of (snap.hands[HUMAN_SEAT] || [])) if (selected.has(c.location)) sum += cardStrength(c.label);
    selSum.textContent = selected.size ? `— selected ${selected.size} card${selected.size > 1 ? 's' : ''} (${sum})` : '';
    sb.disabled = !(matchIndex() >= 0 || canYield());
    sb.textContent = submitLabel();
    $('btn-clear').disabled = selected.size === 0;
  };

  for (const cel of hand.children) {
    cel.onclick = () => {
      const loc = +cel.dataset.loc;
      if (selected.has(loc)) { selected.delete(loc); cel.classList.remove('sel'); }
      else { selected.add(loc); cel.classList.add('sel'); }
      refresh();
    };
  }
  sb.onmouseenter = () => { hovering = true; sb.textContent = submitLabel(); };
  sb.onmouseleave = () => { hovering = false; sb.textContent = submitLabel(); };
  refresh();

  return new Promise((resolve) => {
    const done = (index) => {
      hand.classList.remove('pickable'); hand.classList.add('locked');
      for (const cel of hand.children) cel.onclick = null;
      sb.onmouseenter = sb.onmouseleave = sb.onclick = null;
      sb.disabled = $('btn-clear').disabled = true;
      sb.textContent = 'Submit';
      resolve(index);
    };
    sb.onclick = () => {
      if (canYield()) return done(yieldCombo.index);
      const idx = matchIndex();
      if (idx < 0) { flashInvalid(); return; }
      done(idx);
    };
    $('btn-clear').onclick = () => { selected.clear(); for (const cel of hand.children) cel.classList.remove('sel'); refresh(); };
  });
}
function flashInvalid() {
  const w = el('span', 'need', ' — not a legal combo, pick again');
  $('combat').appendChild(w);
  setTimeout(() => w.remove(), 1600);
}

/* ================= main loop ================= */
async function loop(token) {
  while (token === loopToken) {
    const dec = driver.prepare();
    const snap = driver.snapshot();
    render(snap);
    if (dec.kind === 'ended') { finish(dec.endValue); return; }
    if (dec.kind === 'auto') { driver.commit(-1); continue; }

    if (dec.isBot) {
      setPill(`Seat ${dec.activeSeat} is thinking…`, 'think');
      $('combat').replaceChildren(el('span', 'note', 'waiting for the other players'));
      const index = await driver.seatBots[dec.activeSeat].runFeeds(dec.feeds, dec.K);
      if (token !== loopToken) return;
      log(moveText(dec, index));
      driver.commit(index);
      render(driver.snapshot());
      await new Promise((r) => setTimeout(r, BOT_MOVE_DELAY_MS));
    } else {
      const index = await presentHuman(dec, snap);
      if (token !== loopToken) return;
      log(moveText(dec, index));
      driver.commit(index);
    }
  }
}

function finish(endValue) {
  const won = endValue === 1;
  setPill(won ? 'Victory' : 'Defeat', won ? 'win' : 'loss');
  $('combat').replaceChildren();
  $('your-hand').classList.add('locked');
  log(won ? 'The party <b>WINS</b> — all 12 royals defeated.' : 'The party <b>LOSES</b>.', won ? 'win' : 'loss');
  $('overlay-res').textContent = won ? 'Victory' : 'Defeat';
  $('overlay-res').className = 'res ' + (won ? 'win' : 'loss');
  $('overlay-sub').textContent = won ? 'All royals cleared.' : 'A player could not block, or ran out of moves.';
  $('overlay').classList.add('show');
}
