/* Client-side Regicide (regi-py) webapp. The C++ engine (WASM) + net bots
 * (onnxruntime-web) run entirely in the browser -- no server game state, no
 * per-move round-trip. Four screens (intro / how-to / menu / game); on the game
 * screen the human picks CARDS from their hand (not a move list) and submits. */
import RegiModule from './dist/regicore.mjs';
import { GameDriver } from './game_driver.mjs';
import { loadBot } from './load_bot.mjs';
import { ExplorerBot } from './mcts.mjs';

// ---- onnxruntime-web source (edit these two lines to change where ORT loads) ----
//   npm:            ./node_modules/onnxruntime-web/dist/
//   no-npm, CDN:    https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/
//   no-npm, vendor: ./vendor/
const ORT_DIST = './node_modules/onnxruntime-web/dist/';
const ort = await import(ORT_DIST + 'ort.wasm.bundle.min.mjs');
ort.env.wasm.numThreads = 1; // single-thread wasm: no SharedArrayBuffer / COOP-COEP
// ort.env.wasm.wasmPaths = ORT_DIST; // where the ort-wasm-*.wasm sidecar is fetched from

// Selectable bots. ADZ (candidate-scoring) + AZ (card-space) Direct-net nets;
// attntrunk is intentionally omitted (heaviest payload, redundant with basic).
// movetoken is an AZ card-token net (reasons in move space, same v/k/a contract).
const NETS = ['adzpool', 'adzmulti', 'basic', 'percardmlp', 'cardtx', 'mixer', 'movetoken'];
// Search depth per bot: 0 = Direct-net (one forward pass, argmax); > 0 = an N-iteration
// MCTS Explorer (~N+1 serial forward passes/move, so bigger = stronger but slower).
const ITERS = [0, 16, 32, 64, 128];
const itersLabel = (n) => (n === 0 ? 'Direct' : `MCTS ${n}`);
const EVENT_DELAY_MS = 450; // per-event reveal pace in the log during bot/auto turns

let humanSeat = 0; // which seat (turn-order position) the human took THIS game; shuffled at start
// Player labels are 1-indexed for display ("Player 1".."Player N"); the human's
// own seat shows their name. Seat ids stay 0-indexed internally.
const seatLabel = (seat) => (seat === humanSeat ? playerName : `Player ${seat + 1}`);

const $ = (id) => document.getElementById(id);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };

let M = null;
let comboMap = null;             // AZ nets' bitwise -> grid cell map (built once at boot)
const botCache = new Map();      // net name -> bot (session reused across games)
const presetCache = new Map();   // numPlayers -> preset opening phase strings (fetched once)
let driver = null;
let playerName = 'Player';
let loopToken = 0;               // bumped to abandon an in-flight game loop
let moveCount = 0;               // decisions committed this game (for the end summary)

/* ================= screens ================= */
function showScreen(id) {
  for (const s of document.querySelectorAll('.screen')) s.classList.toggle('show', s.id === `screen-${id}`);
  if (id === 'menu') { buildOpponents(); refreshPresets(); }
}
document.querySelectorAll('[data-nav]').forEach((b) => b.addEventListener('click', () => showScreen(b.dataset.nav)));
$('intro-play').addEventListener('click', () => showScreen('menu'));
$('intro-howto').addEventListener('click', () => showScreen('howto'));

/* how-to tabs */
document.querySelectorAll('#howto-tabs .tab').forEach((t) => t.addEventListener('click', () => {
  document.querySelectorAll('#howto-tabs .tab').forEach((x) => x.classList.toggle('active', x === t));
  document.querySelectorAll('.howto-pane').forEach((p) => { p.hidden = p.dataset.pane !== t.dataset.howto; });
}));

/* ================= boot / loading screen ================= */
// Everything the app needs is loaded up front on the loading screen: the WASM
// engine, the combomap (AZ policy grid), every player-count's opening presets, and
// an onnxruntime session per net. Nets whose artifacts aren't in ./dist just get
// dropped from the menu (marked "unavailable") rather than blocking play.
const availableNets = []; // NETS that loaded a session (also warmed into botCache)

function setLoadBar(done, total, status) {
  $('load-bar').style.width = `${total ? Math.round((done / total) * 100) : 0}%`;
  if (status != null) $('load-status').textContent = status;
}
function addLoadStep(text) { const li = el('li', 'load-step', text); $('load-steps').appendChild(li); return li; }
function markStep(li, ok, text) { if (text) li.textContent = text; li.classList.add(ok ? 'done' : 'fail'); }

async function boot() {
  const total = 2 + NETS.length; // engine, presets, then one step per net
  let done = 0;
  try {
    const s1 = addLoadStep('Game engine (WebAssembly)');
    M = await RegiModule();
    comboMap = M.combo_map();
    markStep(s1, true); setLoadBar(++done, total, 'engine ready');

    const s2 = addLoadStep('Opening deals (presets)');
    await Promise.all([2, 3, 4].map(loadPresets));
    markStep(s2, true); setLoadBar(++done, total, 'presets ready');

    // A Direct bot session per net; an Explorer just wraps it, so this warms every
    // selectable bot. loadBot fetches <net>.onnx + <net>.io.json from ./dist.
    for (const net of NETS) {
      const step = addLoadStep(`Bot: ${net}`);
      setLoadBar(done, total, `loading ${net}…`);
      try {
        botCache.set(net, await loadBot(M, ort, './dist', net, { comboMap }));
        availableNets.push(net);
        markStep(step, true);
      } catch (err) {
        console.warn(`net ${net} unavailable:`, err);
        markStep(step, false, `Bot: ${net} — unavailable`);
      }
      setLoadBar(++done, total);
    }

    if (!availableNets.length) throw new Error('no bot nets could be loaded from ./dist');
    setLoadBar(total, total, 'ready');
    showScreen('intro');
  } catch (err) {
    const box = $('load-error');
    box.hidden = false;
    box.textContent = `Could not start: ${err?.message || err}. Build the engine and export the ` +
      `nets into ./dist (see README), then reload.`;
    $('load-status').textContent = 'failed to load';
  }
}
boot();

/* ================= menu ================= */
function buildOpponents() {
  const numPlayers = parseInt($('cfg-players').value, 10);
  const wrap = $('opponents');
  const prevNet = {}, prevIters = {};
  wrap.querySelectorAll('select.opp-net').forEach((s) => { prevNet[s.dataset.bot] = s.value; });
  wrap.querySelectorAll('select.opp-iters').forEach((s) => { prevIters[s.dataset.bot] = s.value; });
  wrap.replaceChildren();
  // Configure the N-1 bot opponents; their turn-order seats are decided (shuffled)
  // at game start, so these are just numbered bots, not fixed seats. Each picks a net
  // AND a search depth (Direct vs an MCTS Explorer).
  for (let i = 1; i < numPlayers; i++) {
    const row = el('div', 'opp-row');
    row.appendChild(el('span', 'seat-name', `Bot ${i}`));
    const net = el('select', 'opp-net');
    net.dataset.bot = i;
    for (const n of availableNets) { const o = el('option', null, n); o.value = n; net.appendChild(o); }
    const dflt = availableNets.includes('adzpool') ? 'adzpool' : availableNets[0];
    net.value = availableNets.includes(prevNet[i]) ? prevNet[i] : dflt;
    row.appendChild(net);
    const iters = el('select', 'opp-iters');
    iters.dataset.bot = i;
    for (const n of ITERS) { const o = el('option', null, itersLabel(n)); o.value = n; iters.appendChild(o); }
    iters.value = prevIters[i] ?? '0';
    row.appendChild(iters);
    row.appendChild(el('span', 'tag', 'bot'));
    wrap.appendChild(row);
  }
}
$('cfg-players').addEventListener('change', () => { buildOpponents(); refreshPresets(); });
$('menu-start').addEventListener('click', startGame);

/* Fetch (once) the committed starter openings for a player count. A preset is an
 * opening phase string GameDriver.newGame(startPhase) replays via init_string; on any
 * fetch error we just fall back to "Random deal" (an empty list). */
async function loadPresets(numPlayers) {
  if (presetCache.has(numPlayers)) return presetCache.get(numPlayers);
  let phases = [];
  try {
    const data = await fetch(`./tables/presets_${numPlayers}p.json`).then((r) => r.json());
    if (Array.isArray(data.phases)) phases = data.phases;
  } catch { phases = []; }
  presetCache.set(numPlayers, phases);
  return phases;
}

/* Repopulate #cfg-preset for the current player count: "Random deal" + one entry per
 * committed preset. Keeps the prior pick if it's still in range, else Random. */
async function refreshPresets() {
  const numPlayers = parseInt($('cfg-players').value, 10);
  const sel = $('cfg-preset');
  const prev = sel.value;
  const phases = await loadPresets(numPlayers);
  sel.replaceChildren();
  const rand = el('option', null, 'Random deal'); rand.value = ''; sel.appendChild(rand);
  phases.forEach((_, i) => { const o = el('option', null, `Preset ${i + 1}`); o.value = String(i); sel.appendChild(o); });
  sel.value = [...sel.options].some((o) => o.value === prev) ? prev : '';
}

async function startGame() {
  if (!M) return;
  const numPlayers = parseInt($('cfg-players').value, 10);
  const seedRaw = $('cfg-seed').value.trim();
  const seed = seedRaw === '' ? null : (parseInt(seedRaw, 10) >>> 0);
  playerName = ($('cfg-name').value.trim() || 'Player').slice(0, 16);

  // The configured bot nets + search depths, in menu order (N-1 of them).
  const botNets = [], botIters = [];
  $('opponents').querySelectorAll('select.opp-net').forEach((s) => { botNets.push(s.value); });
  $('opponents').querySelectorAll('select.opp-iters').forEach((s) => { botIters.push(parseInt(s.value, 10) || 0); });

  // A chosen preset replays a fixed opening deal; "Random deal" ('') deals fresh.
  const presetVal = $('cfg-preset').value;
  let startPhase = null;
  if (presetVal !== '') { const phases = await loadPresets(numPlayers); startPhase = phases[parseInt(presetVal, 10)] || null; }

  // Shuffle the human into a random seat so turn order varies each game.
  humanSeat = Math.floor(Math.random() * numPlayers);

  const seatBots = [];
  let bi = 0;
  for (let i = 0; i < numPlayers; i++) {
    if (i === humanSeat) { seatBots.push(null); continue; }
    const net = botNets[bi] || availableNets[0];
    const iters = botIters[bi] || 0;
    bi++;
    // Every net was warmed into botCache during boot; an Explorer just wraps the
    // cached Direct bot, so switching a seat's search depth reloads nothing.
    const direct = botCache.get(net);
    seatBots.push(iters > 0 ? new ExplorerBot(M, direct, { iterations: iters }) : direct);
  }

  const maxHistory = seatBots.find((b) => b)?.maxHistory ?? 8;
  if (driver) driver.dispose();
  driver = new GameDriver(M, { numPlayers, seatBots, maxHistory, seed });
  driver.newGame(startPhase);

  moveCount = 0;
  $('log').replaceChildren();
  $('overlay').classList.remove('show');
  $('show-summary').hidden = true;
  setBotTurn(false);
  $('you-seat-note').textContent = `· you are “${playerName}”`;
  const opening = startPhase ? `preset ${parseInt(presetVal, 10) + 1}` : 'a random deal';
  log(`New ${numPlayers}-player game (${opening}) — you are <b>Player ${humanSeat + 1}</b>.`);
  showScreen('game');
  loopToken++;
  loop(loopToken);
}

$('game-menu').addEventListener('click', () => { loopToken++; setBotTurn(false); $('show-summary').hidden = true; showScreen('menu'); });
$('overlay-menu').addEventListener('click', () => { $('overlay').classList.remove('show'); $('show-summary').hidden = true; setBotTurn(false); showScreen('menu'); });
$('overlay-again').addEventListener('click', startGame);
// Dismiss the result overlay to look over the finished board; a floating button
// (bottom-right) brings the summary back. Both keep you on the game screen.
$('overlay-review').addEventListener('click', () => { $('overlay').classList.remove('show'); $('show-summary').hidden = false; });
$('show-summary').addEventListener('click', () => { $('overlay').classList.add('show'); $('show-summary').hidden = true; });

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
  // The engine sets each royal's max HP = 2 * its attack (Jack 20, Queen 30, King 40),
  // so the bar is scaled to THIS royal's max and shown as "current / max".
  const hp = enemy ? Math.max(0, enemy.hp) : 0;
  const hpMax = enemy ? 2 * enemy.strength : 0;
  $('enemy-hp-lbl').textContent = enemy ? `${hp} / ${hpMax}` : '—';
  $('enemy-hp-bar').style.width = `${enemy && hpMax > 0 ? Math.max(0, Math.min(100, (hp / hpMax) * 100)) : 0}%`;

  const incoming = enemy ? Math.max(0, enemy.strength - snap.currentBlock) : 0;
  const inc = $('incoming');
  inc.textContent = enemy ? (incoming > 0 ? `Hits for ${incoming}` : 'Full-blocked') : '—';
  inc.classList.toggle('blocked', incoming === 0);

  $('prog-txt').textContent = `${snap.progress} / 360`;
  $('prog-bar').style.width = `${Math.max(0, Math.min(100, (snap.progress / 360) * 100))}%`;
  $('enemies-left').textContent = snap.enemiesLeft;

  const seats = $('seats'); seats.replaceChildren();
  for (let i = 0; i < snap.numPlayers; i++) {
    const you = i === humanSeat;
    const s = el('div', 'seat' + (i === snap.activeSeat ? ' active' : '') + (you ? ' you-seat' : ''));
    s.appendChild(el('div', 'who', `Player ${i + 1}${you ? ` · ${playerName}` : ''}`));
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
  for (const c of (snap.hands[humanSeat] || [])) hand.appendChild(cardEl(c));
}

/* ================= log + status ================= */
function log(html, cls) { const row = el('div', 'row' + (cls ? ' ' + cls : '')); row.innerHTML = html; $('log').prepend(row); }
/* Prepend a turn's lines so they read top-to-bottom in chronological order while the
   newest turn still sits on top (the log is newest-first). */
function logLines(lines) {
  for (let i = lines.length - 1; i >= 0; i--) {
    const r = el('div', 'row' + (lines[i].cls ? ' ' + lines[i].cls : ''));
    r.innerHTML = lines[i].html;
    $('log').prepend(r);
  }
}
function prettyLabel(label) { const p = parseCard(label); return p.joker ? '★' : `${p.disp}${p.suit}`; }
function moveText(dec, idx) {
  const c = dec.comboData[idx];
  const who = seatLabel(dec.activeSeat);
  const what = !c ? '?' : c.isYield ? 'yields'
    : `${dec.attacking ? 'attacks' : 'defends'} with ${c.labels.map(prettyLabel).join(' ')}`;
  return `<b>${who}</b> ${what}`;
}
/* Turn a commit()'s captured engine events into log lines (chronological). An
   optional leading `actionHtml` (the chosen-move text) heads the block. */
function eventLines(events, actionHtml) {
  const lines = [];
  if (actionHtml) lines.push({ html: actionHtml });
  if (!events) return lines;
  let draws = new Map();
  const flushDraws = () => {
    if (!draws.size) return;
    const parts = [...draws.entries()].map(([p, k]) => `<b>${seatLabel(p)}</b> +${k}`);
    lines.push({ html: `<span class="ev">♦</span> draw — ${parts.join(', ')}` });
    draws = new Map();
  };
  for (const e of events) {
    if (e.type === 'draw') { draws.set(e.player, (draws.get(e.player) || 0) + 1); continue; }
    flushDraws();
    switch (e.type) {
      case 'replenish':
        if (e.amount > 0) lines.push({ html: `<span class="ev">♥</span> heal — ${e.amount} card${e.amount > 1 ? 's' : ''} back under the deck` });
        break;
      case 'redirect':
        lines.push({ html: `<span class="ev">★</span> <b>${seatLabel(e.player)}</b> plays a Joker — <b>${seatLabel(e.target)}</b> plays next`, cls: 'joker' });
        break;
      case 'enemyKill':
        lines.push({ html: `<span class="ev">☠</span> <b>${prettyLabel(e.enemy)}</b> defeated`, cls: 'kill' });
        break;
      case 'fullBlock':
        lines.push({ html: `<span class="ev">🛡</span> <b>${seatLabel(e.player)}</b> fully blocks (${e.block} block ≥ ${e.damage})` });
        break;
      case 'failBlock':
        lines.push({ html: `<span class="ev">✖</span> <b>${seatLabel(e.player)}</b> can't block ${e.damage} (only ${e.block})`, cls: 'bad' });
        break;
    }
  }
  flushDraws();
  return lines;
}
function setPill(text, cls) { const p = $('turn-status'); p.textContent = text; p.className = 'turn-pill' + (cls ? ' ' + cls : ''); }
/* C4: while a bot holds the turn, foreground the event log (CSS `.board.bot-turn`)
   and dim your (idle) hand; cleared the moment it's your turn again, or the game ends. */
function setBotTurn(on) { document.querySelector('.board')?.classList.toggle('bot-turn', on); }
/* C3: reveal a turn's log lines one at a time so the action is followable. The whole
   block is prepended first (keeping its final chronological order), then rows light up
   top-to-bottom, EVENT_DELAY_MS apart. An interrupt (loopToken bumped) reveals the rest
   at once and bails, so an abandoned game never leaves half-hidden rows. */
async function emitLines(lines, token) {
  if (!lines.length) return;
  const rows = [];
  for (let i = lines.length - 1; i >= 0; i--) {
    const r = el('div', 'row reveal' + (lines[i].cls ? ' ' + lines[i].cls : ''));
    r.innerHTML = lines[i].html;
    $('log').prepend(r);
    rows.unshift(r);
  }
  for (const r of rows) {
    if (token !== loopToken) { rows.forEach((x) => x.classList.add('shown')); return; }
    r.classList.add('shown');
    await new Promise((res) => setTimeout(res, EVENT_DELAY_MS));
  }
}

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
    let sum = 0; for (const c of (snap.hands[humanSeat] || [])) if (selected.has(c.location)) sum += cardStrength(c.label);
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

/* Jester: after you play a Joker, choose which OTHER player takes the next turn
   (the engine/reference bots never redirect to self). Resolves to that seat id. */
function presentRedirect(dec) {
  setPill('You played a Joker — choose who plays next', 'you');
  const combat = $('combat'); combat.replaceChildren();
  combat.appendChild(el('span', 'dmg', '★ Joker — who plays next?'));
  const actions = $('actions');
  const submit = $('btn-submit'), clear = $('btn-clear');
  submit.hidden = true; clear.hidden = true;
  const btns = [];
  return new Promise((resolve) => {
    const done = (seat) => {
      for (const b of btns) b.remove();
      submit.hidden = false; clear.hidden = false;
      resolve(seat);
    };
    for (let i = 0; i < driver.numPlayers; i++) {
      if (i === dec.activeSeat) continue;
      const b = el('button', 'btn primary', `Player ${i + 1}`);
      b.onclick = () => done(i);
      actions.appendChild(b);
      btns.push(b);
    }
  });
}

/* When a bot plays a Joker, decide who takes the next turn the way that bot's
   reference strategy does: AZ nets value-argmax the other seats (AZBot.chooseRedirect,
   N-1 forward passes); ADZ nets have no redirect head -- returning null lets the driver
   fall back to a random OTHER player (== rl/adz/explorer.py _random_redirect). */
async function botRedirect(bot, dec) {
  if (typeof bot.chooseRedirect !== 'function') return null;
  const phase = M.PhaseInfo.from_string(dec.decisionPhaseString);
  const target = await bot.chooseRedirect(phase, driver.history, driver.numPlayers);
  phase.delete();
  return target;
}

/* ================= main loop ================= */
async function loop(token) {
  while (token === loopToken) {
    const dec = driver.prepare();
    const snap = driver.snapshot();
    render(snap);
    if (dec.kind === 'ended') { setBotTurn(false); logLines(eventLines(dec.events, null)); finish(dec.endValue); return; }
    if (dec.kind === 'auto') { driver.commit(-1); render(driver.snapshot()); await emitLines(eventLines(driver.lastEvents, null), token); continue; }

    if (dec.isBot) {
      setBotTurn(true);
      setPill(`${seatLabel(dec.activeSeat)} is thinking…`, 'think');
      $('combat').replaceChildren(el('span', 'note', 'waiting for the other players'));
      const bot = driver.seatBots[dec.activeSeat];
      // A Direct bot argmaxes its pre-built feeds; an Explorer searches from the
      // captured decision phase (+ the real past decisions as its history window).
      const index = bot.isExplorer
        ? await bot.search(dec.decisionPhaseString, driver.history.map((p) => p.to_string()), dec.comboData)
        : await bot.runFeeds(dec.built);
      if (token !== loopToken) return;
      // A bot Joker attack redirects like its reference strategy (AZ value-argmax,
      // ADZ random-other); non-joker moves pass null (no redirect asked).
      const played = dec.comboData[index];
      let redirect = null;
      if (dec.attacking && played && played.isJoker && driver.numPlayers > 1) {
        redirect = await botRedirect(bot, dec);
        if (token !== loopToken) return;
      }
      driver.commit(index, redirect);
      moveCount++;
      render(driver.snapshot());
      await emitLines(eventLines(driver.lastEvents, moveText(dec, index)), token);
      if (token !== loopToken) return;
    } else {
      setBotTurn(false);
      const index = await presentHuman(dec, snap);
      if (token !== loopToken) return;
      const played = dec.comboData[index];
      let redirect = null;
      if (dec.attacking && played && played.isJoker && driver.numPlayers > 1) {
        redirect = await presentRedirect(dec);
        if (token !== loopToken) return;
      }
      driver.commit(index, redirect);
      moveCount++;
      logLines(eventLines(driver.lastEvents, moveText(dec, index)));
    }
  }
}

function statEl(num, lbl) {
  const s = el('div', 'stat'); s.appendChild(el('div', 'num', num)); s.appendChild(el('div', 'lbl', lbl)); return s;
}
function finish(endValue) {
  const won = endValue === 1;
  const snap = driver.snapshot();
  setPill(won ? 'Victory' : 'Defeat', won ? 'win' : 'loss');
  $('combat').replaceChildren();
  $('your-hand').classList.add('locked');
  log(won ? 'The party <b>WINS</b> — all 12 royals defeated.' : 'The party <b>LOSES</b>.', won ? 'win' : 'loss');
  $('overlay-res').textContent = won ? 'Victory' : 'Defeat';
  $('overlay-res').className = 'res ' + (won ? 'win' : 'loss');
  $('overlay-sub').textContent = won ? 'All royals cleared.' : 'A player could not block, or ran out of moves.';
  // Final progress: royals cleared out of 12, damage dealt out of 360, moves played.
  $('overlay-stats').replaceChildren(
    statEl(`${12 - snap.enemiesLeft} / 12`, 'Royals cleared'),
    statEl(`${snap.progress} / 360`, 'Damage dealt'),
    statEl(String(moveCount), moveCount === 1 ? 'Move' : 'Moves'),
  );
  $('show-summary').hidden = true;
  // View-start: show the seed + opening phase this game began from (so it can be
  // replayed via the menu's Seed field, or shared). Collapsed until the player opens it.
  $('start-seed').textContent = driver?.startSeed ?? '—';
  $('start-phase').textContent = driver?.startPhase ?? '—';
  $('overlay-start').open = false;
  $('start-copy').textContent = 'Copy opening phase';
  $('overlay').classList.add('show');
}

/* Copy the opening phase string to the clipboard (best-effort; clipboard API needs a
   secure context, so fall back to selecting the text for a manual copy). */
$('start-copy').addEventListener('click', async () => {
  const text = driver?.startPhase || '';
  if (!text) return;
  const btn = $('start-copy');
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = 'Copied ✓';
    setTimeout(() => { btn.textContent = 'Copy opening phase'; }, 1400);
  } catch {
    const range = document.createRange();
    range.selectNodeContents($('start-phase'));
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    btn.textContent = 'Select + ⌘/Ctrl-C';
    setTimeout(() => { btn.textContent = 'Copy opening phase'; }, 1800);
  }
});
