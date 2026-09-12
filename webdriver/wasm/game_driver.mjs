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
    this.history = []; // past decision phases (PhaseInfo), newest last, <= maxHistory
    // Per-cycle RNG seed: prepare() and commit() each simulate ONE onePhase and MUST
    // seed identically so the phase they produce (and any card draw in it) matches,
    // else two independent re-seats diverge and the offered combos disagree. The seed
    // advances (LCG) after each committed onePhase so the game keeps varying.
    this.rng = (seed === null ? (Date.now() & 0x7fffffff) : (seed >>> 0)) || 1;
  }

  _seedCycle() { this.M.seed(this.rng); }
  _bumpCycle() { this.rng = (Math.imul(this.rng, 1664525) + 1013904223) >>> 0 || 1; }

  /* Deal a fresh game and record its opening phase string. */
  newGame() {
    const M = this.M;
    const log = new M.NoOpLog();
    const g = new M.GameState(log);
    const seats = [];
    for (let i = 0; i < this.numPlayers; i++) { const s = new M.RandomStrategy(); seats.push(s); g.add_player(s); }
    g.initialize();
    this.phaseString = g.export_string();
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
   * seeded from the current phase string, and return it for the caller to step. */
  _seat(makeMethods) {
    const M = this.M;
    const log = new M.NoOpLog();
    const g = new M.GameState(log);
    const Strat = M.Strategy.extend('Strategy', makeMethods);
    const seats = [];
    for (let i = 0; i < this.numPlayers; i++) { const s = new Strat(); seats.push(s); g.add_player(s); }
    g.init_string(this.phaseString);
    return { g, log, seats, done() { seats.forEach((s) => s.delete()); g.delete(); log.delete(); } };
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
    const up = ph.used_combos;
    const usedCombos = [];
    for (let i = 0; i < up.size(); i++) { const c = up.get(i); usedCombos.push(c.label); c.delete(); }
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
   * For a bot seat the net feeds are built here while the offered combos are alive;
   * the caller runs them with bot.runFeeds(feeds, K), then calls commit(index). */
  prepare() {
    const M = this.M;
    const history = this.history;
    let captured = null;
    const grab = (combos, game) => {
      const phase = game.export_phaseinfo();
      const activeSeat = phase.active_player;
      const attacking = phase.phase_attacking;
      const comboData = [];
      for (let i = 0; i < combos.size(); i++) {
        const c = combos.get(i), parts = c.parts, locs = [], labels = [];
        for (let j = 0; j < parts.size(); j++) { const cd = parts.get(j); locs.push(cd.location); labels.push(cd.label); cd.delete(); }
        const isYield = parts.size() === 0;
        parts.delete(); c.delete();
        comboData.push({ index: i, locations: locs, labels, isYield });
      }
      let feeds = null, K = combos.size();
      const seatBot = this.seatBots[activeSeat] || null;
      if (seatBot) { const built = seatBot.buildFeeds(phase, combos, history, { reshuffle: true }); feeds = built.feeds; K = built.K; }
      phase.delete();
      captured = { kind: 'decision', activeSeat, attacking, isBot: !!seatBot, comboData, feeds, K };
      return 0; // throwaway; commit() replays this onePhase with the real index
    };
    this._seedCycle();
    const ctx = this._seat({
      setup() { return 0; },
      getAttackIndex(combos, p, y, game) { return grab(combos, game); },
      getDefenseIndex(combos, p, d, game) { return grab(combos, game); },
      getRedirectIndex() { return 0; },
    });
    const running = ctx.g.is_runnable();
    if (running) ctx.g.step(); // exactly one onePhase
    let result;
    if (captured !== null) result = captured;
    else if (!ctx.g.is_runnable()) {
      const term = ctx.g.export_string();
      const tp = M.PhaseInfo.from_string(term); const endValue = tp.game_endvalue; tp.delete();
      this.phaseString = term; // freeze the terminal board for snapshot()
      result = { kind: 'ended', endValue };
    } else {
      result = { kind: 'auto' }; // a no-decision onePhase; commit(-1) skips it
    }
    ctx.done();
    return result;
  }

  /* Commit the onePhase prepare() showed, replaying it under the SAME cycle seed and
   * applying `index` at the decision (ignored for an 'auto' phase; pass -1). Updates
   * the phase string, records the decided phase into history, advances the seed. */
  commit(index) {
    const M = this.M;
    const decided = M.PhaseInfo.from_string(this.phaseString);
    const isDecision = index >= 0;
    const redirectTarget = (decided.active_player + 1) % this.numPlayers; // valid default (rare)
    this._seedCycle();
    const ctx = this._seat({
      setup() { return 0; },
      getAttackIndex() { return index; },
      getDefenseIndex() { return index; },
      getRedirectIndex() { return redirectTarget; },
    });
    if (ctx.g.is_runnable()) ctx.g.step(); // the same onePhase, same seed as prepare()
    this.phaseString = ctx.g.export_string();
    ctx.done();
    this._bumpCycle();
    if (isDecision) this._pushHistory(decided); else decided.delete();
    return this.snapshot();
  }

  dispose() { this._clearHistory(); }
}
