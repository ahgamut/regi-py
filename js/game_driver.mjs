/* GameDriver: owns a Regicide game entirely in the browser and advances it one
 * decision at a time, so async decisions (a bot's onnxruntime forward pass, or a
 * human's click) fit a SYNCHRONOUS engine with no Asyncify. The state is just the
 * serialized phase string plus a rolling history window; each decision (a) peeks
 * the current decision on a throwaway game seeded from the string (capturing the
 * offered combos, and -- for a bot seat -- the net feeds while the combos are
 * alive), then (b) advances by re-seating and replaying the chosen index for one
 * onePhase. This is the PhaseExpander re-seat pattern (strats/phase_utils.py),
 * proven by check_golden.mjs.
 *
 * DOM-free on purpose: app.mjs wires this to the UI, and smoke_driver.mjs drives it
 * headless in node. onePhase() == exactly one active-player decision (attack, or
 * that player's defense); redirect is a separate rare callback, default-handled. */

export class GameDriver {
  /* seatBots: array of length numPlayers; entry i is a NetBot for a bot seat or
   * null/undefined for a human seat. maxHistory = the nets' window. */
  constructor(M, { numPlayers = 2, seatBots = [], maxHistory = 8, seed = null } = {}) {
    this.M = M;
    this.numPlayers = numPlayers;
    this.seatBots = seatBots;
    this.maxHistory = maxHistory;
    this.phaseString = null;
    this.lastEvents = []; // engine side-effect events from the most recent commit()
    this.history = []; // past decision phases (PhaseInfo), newest last, <= maxHistory
    // Per-cycle RNG seed: prepare() and commit() each simulate ONE onePhase and MUST
    // seed identically so the phase they produce (and any card draw in it) matches,
    // else two independent re-seats diverge and the offered combos disagree. The seed
    // advances (LCG) after each committed onePhase so the game keeps varying.
    this.rng = (seed === null ? (Date.now() & 0x7fffffff) : (seed >>> 0)) || 1;
    // The opening this game started from, captured by newGame() so the UI can show /
    // share it (a preset phase string, or a fresh seeded deal): { startPhase, startSeed }.
    this.startPhase = null;
    this.startSeed = null;
  }

  _seedCycle() { this.M.seed(this.rng); }
  _bumpCycle() { this.rng = (Math.imul(this.rng, 1664525) + 1013904223) >>> 0 || 1; }

  /* Start a game and record its opening phase string. With no `startPhase`, deal a
   * fresh game (seeded from `this.rng` so the deal is reproducible from the game seed);
   * with a `startPhase` (a make_phases-style preset), replay that exact opening via
   * init_string instead of dealing. Either way the opening + seed are captured on
   * `startPhase`/`startSeed` for the view-start affordance. */
  newGame(startPhase = null) {
    const M = this.M;
    const log = new M.NoOpLog();
    const g = new M.GameState(log);
    const seats = [];
    for (let i = 0; i < this.numPlayers; i++) { const s = new M.RandomStrategy(); seats.push(s); g.add_player(s); }
    this.startSeed = this.rng;
    if (startPhase) {
      g.init_string(startPhase); // replay the preset opening (num_players must match)
    } else {
      M.seed(this.rng); // reproducible fresh deal from the game seed
      g.initialize();
    }
    this.phaseString = g.export_string();
    this.startPhase = this.phaseString;
    this._clearHistory();
    seats.forEach((s) => s.delete());
    g.delete(); log.delete();
    return this.snapshot();
  }

  _clearHistory() { this.history.forEach((p) => p.delete()); this.history = []; }

  _pushHistory(phase) {
    this.history.push(phase);
    while (this.history.length > this.maxHistory) this.history.shift().delete();
  }

  /* Seat `numPlayers` copies of one JS Strategy (built by makeMethods) on a game
   * seeded from the current phase string, and return it for the caller to step.
   * An optional `log` (e.g. an EventLog) captures the engine's side-effect events
   * for that onePhase; when omitted a throwaway NoOpLog is used and disposed. */
  _seat(makeMethods, log = null) {
    const M = this.M;
    const ownLog = log === null;
    if (ownLog) log = new M.NoOpLog();
    const g = new M.GameState(log);
    const Strat = M.Strategy.extend('Strategy', makeMethods);
    const seats = [];
    for (let i = 0; i < this.numPlayers; i++) { const s = new Strat(); seats.push(s); g.add_player(s); }
    g.init_string(this.phaseString);
    return { g, log, seats, done() { seats.forEach((s) => s.delete()); g.delete(); if (ownLog) log.delete(); } };
  }

  /* Board snapshot for rendering (mirrors serialize.game_to_dict's readable fields,
   * assembled from a PhaseInfo). Cheap; safe to call anytime. */
  snapshot() {
    const M = this.M;
    const ph = M.PhaseInfo.from_string(this.phaseString);
    const readCards = (vec) => {
      const out = [];
      for (let i = 0; i < vec.size(); i++) { const c = vec.get(i); out.push({ location: c.location, label: c.label }); c.delete(); }
      return out;
    };
    const pc = ph.player_cards;
    const hands = [];
    for (let i = 0; i < pc.size(); i++) { const h = pc.get(i); hands.push(readCards(h)); h.delete(); }
    pc.delete();

    const ep = ph.enemy_pile;
    let enemy = null;
    const enemyLabels = [];
    let hpLeft = 0;
    for (let i = 0; i < ep.size(); i++) {
      const e = ep.get(i);
      enemyLabels.push(e.label);
      if (e.hp > 0) hpLeft += e.hp;
      if (i === 0) enemy = { label: e.label, hp: e.hp, strength: e.strength };
      e.delete();
    }
    ep.delete();
    const progress = 360 - hpLeft; // 12 enemies * (10|15|20) hp = 360 total to clear

    const dp = ph.draw_pile; const drawPileSize = dp.size(); dp.delete();
    const disc = ph.discard_pile; const discardPileSize = disc.size(); disc.delete();
    const up = ph.used_combos;
    const usedCombos = []; // each entry is that combo's cards (for card-stack rendering)
    for (let i = 0; i < up.size(); i++) {
      const c = up.get(i); const parts = c.parts; const cards = [];
      for (let j = 0; j < parts.size(); j++) { const cd = parts.get(j); cards.push({ location: cd.location, label: cd.label }); cd.delete(); }
      parts.delete(); c.delete();
      usedCombos.push(cards);
    }
    up.delete();

    const snap = {
      numPlayers: ph.num_players,
      activeSeat: ph.active_player,
      attacking: ph.phase_attacking,
      endValue: ph.game_endvalue,
      ended: ph.game_endvalue !== 0,
      currentBlock: ph.current_block(),
      hands,
      handCounts: hands.map((h) => h.length),
      enemy,
      enemyLabels,
      enemiesLeft: enemyLabels.length,
      progress,
      drawPileSize,
      discardPileSize,
      usedCombos,
    };
    ph.delete();
    return snap;
  }

  /* Look at the NEXT onePhase (a throwaway re-seat, one step under the cycle seed)
   * without committing. Returns one of:
   *   { kind:'ended', endValue }                              game over
   *   { kind:'auto' }                                          a no-decision onePhase
   *                                                            (e.g. a full block)
   *   { kind:'decision', activeSeat, attacking, isBot,
   *     comboData:[{index,locations,labels,isYield}], feeds, K }
   * For a bot seat the net feeds are built here while the offered combos are alive
   * (returned as `built`); the caller runs them with bot.runFeeds(built), then calls
   * commit(index). `decisionPhaseString` is provided for Explorer bots that search. */
  prepare() {
    const M = this.M;
    const history = this.history;
    let captured = null;
    const grab = (combos, game, extra) => {
      const phase = game.export_phaseinfo();
      const activeSeat = phase.active_player;
      const attacking = phase.phase_attacking;
      const comboData = [];
      for (let i = 0; i < combos.size(); i++) {
        const c = combos.get(i), parts = c.parts, locs = [], labels = [];
        for (let j = 0; j < parts.size(); j++) { const cd = parts.get(j); locs.push(cd.location); labels.push(cd.label); cd.delete(); }
        const isYield = parts.size() === 0;
        // a joker card stringifies as "X!"; playing one triggers the jester redirect
        const isJoker = labels.some((lb) => lb[0] === 'X');
        parts.delete(); c.delete();
        comboData.push({ index: i, locations: locs, labels, isYield, isJoker });
      }
      const K = combos.size();
      const seatBot = this.seatBots[activeSeat] || null;
      // The decision phase string is the search root for an Explorer (MCTS) bot; a
      // Direct bot instead uses `built` (feeds assembled here while the combos live).
      const decisionPhaseString = phase.to_string();
      // A Direct bot consumes `built` (feeds assembled here while the combos live); an
      // Explorer bot searches from `decisionPhaseString` instead, so skip the wasted
      // featurization for it.
      const built = (seatBot && !seatBot.isExplorer) ? seatBot.buildFeeds(phase, combos, history, { reshuffle: true }) : null;
      phase.delete();
      // `damage` (defense: what must be blocked) and `yieldAllowed` (attack: an
      // empty combo is offered) drive the human panel's combat readout + yield gate.
      captured = { kind: 'decision', activeSeat, attacking, isBot: !!seatBot, comboData, built, K,
        decisionPhaseString, damage: extra.damage ?? null, yieldAllowed: !!extra.yieldAllowed };
      return 0; // throwaway; commit() replays this onePhase with the real index
    };
    // Capture events even on the peek so a game-ending onePhase (a loss's
    // failBlock, the final NO_ENEMIES) -- which is never committed -- can still be
    // reported. On a decision/auto peek these events are discarded; commit() re-runs
    // the same onePhase and re-captures them for real.
    const evlog = new M.EventLog();
    this._seedCycle();
    const ctx = this._seat({
      setup() { return 0; },
      getAttackIndex(combos, p, y, game) { return grab(combos, game, { yieldAllowed: y }); },
      getDefenseIndex(combos, p, d, game) { return grab(combos, game, { damage: d }); },
      getRedirectIndex() { return 0; },
    }, evlog);
    const running = ctx.g.is_runnable();
    if (running) ctx.g.step(); // exactly one onePhase
    let result;
    if (captured !== null) result = captured;
    else if (!ctx.g.is_runnable()) {
      const term = ctx.g.export_string();
      const tp = M.PhaseInfo.from_string(term); const endValue = tp.game_endvalue; tp.delete();
      this.phaseString = term; // freeze the terminal board for snapshot()
      result = { kind: 'ended', endValue, events: evlog.drain() };
    } else {
      result = { kind: 'auto' }; // a no-decision onePhase; commit(-1) skips it
    }
    ctx.done();
    evlog.delete();
    return result;
  }

  /* Commit the onePhase prepare() showed, replaying it under the SAME cycle seed and
   * applying `index` at the decision (ignored for an 'auto' phase; pass -1). Updates
   * the phase string, records the decided phase into history, advances the seed.
   *
   * If `index` plays a JOKER, the engine also asks who takes the next turn (the
   * jester). `redirectTarget` names that seat; when null (a bot, or a non-joker move)
   * it defaults to a random OTHER player -- matching the ADZ reference
   * `_random_redirect`. Never self. */
  commit(index, redirectTarget = null) {
    const M = this.M;
    const decided = M.PhaseInfo.from_string(this.phaseString);
    const isDecision = index >= 0;
    const n = this.numPlayers;
    const active = decided.active_player;
    const redir = (redirectTarget != null && redirectTarget >= 0)
      ? redirectTarget
      : (active + 1 + Math.floor(Math.random() * (n - 1))) % n;
    const evlog = new M.EventLog();
    this._seedCycle();
    const ctx = this._seat({
      setup() { return 0; },
      getAttackIndex() { return index; },
      getDefenseIndex() { return index; },
      getRedirectIndex() { return redir; },
    }, evlog);
    if (ctx.g.is_runnable()) ctx.g.step(); // the same onePhase, same seed as prepare()
    this.phaseString = ctx.g.export_string();
    this.lastEvents = evlog.drain(); // side-effect events for this committed onePhase
    ctx.done();
    evlog.delete();
    this._bumpCycle();
    if (isDecision) this._pushHistory(decided); else decided.delete();
    return this.snapshot();
  }

  dispose() { this._clearHistory(); }
}
