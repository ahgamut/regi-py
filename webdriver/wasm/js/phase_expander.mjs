/* PhaseExpander: the JS analogue of strats/phase_utils.py PhaseExpander +
 * _ExpansionStrategy -- the engine primitive the MCTS Explorer (Part B) needs and
 * the one thing GameDriver does NOT provide. GameDriver.prepare() steps exactly one
 * onePhase (one active-player decision); the expander instead, for a chosen root
 * combo, LOOPS the engine forward past any auto-resolved onePhases (e.g. a full
 * block) until the NEXT decision node -- or the end of the game. That "phase at the
 * next decision node" is the child state an MCTS node stores.
 *
 * Lazy by design: offered() lists the root combos once; step(bitwise) materializes
 * one child on demand, so the search only builds the children it actually visits.
 * Each call re-seats a throwaway GameState from the root string (the PhaseExpander
 * re-seat pattern, proven by check_golden.mjs / GameDriver), forces the chosen
 * combo, and steps to the next decision. State is strings + bitwise BigInts only --
 * no live engine handles escape a call (they don't survive in WASM/Embind).
 *
 * Determinism: the root string is expected to be ALREADY determinized (the MCTS
 * perspectivizes once at the search root); a fixed `seed` is re-applied before every
 * re-seat so a draw-pile reshuffle mid-step is reproducible across calls (same
 * reason GameDriver seeds each prepare/commit cycle). Hidden-card reshuffling per
 * node would be an info leak, so the expander never perspectivizes -- callers step
 * children with the faithful string round-trip.
 *
 * `trim` (the Python offered(trim) yield/throwy-combo pruner) is intentionally NOT
 * ported: it's an RNG exploration reducer used in self-play (alongside Dirichlet),
 * and competitive play offers every legal combo. */

import { bitwiseOfLocations } from './net_common.mjs';

export class PhaseExpander {
  /* `rootPhaseString` is GameState.export_string() at a decision node. `seed`
   * (optional) fixes the engine RNG before each re-seat; null leaves it free. */
  constructor(M, rootPhaseString, { seed = null } = {}) {
    this.M = M;
    this.rootPhaseString = rootPhaseString;
    this.seed = seed;
    const ph = M.PhaseInfo.from_string(rootPhaseString);
    this.numPlayers = ph.num_players;
    this.endValue = ph.game_endvalue; // != 0 -> the root is already terminal
    ph.delete();
    this._offered = null;   // cached root-combo descriptors
    this.capture = null;    // payload from offered()'s onOffer callback (Part B leaf feeds)
  }

  isTerminal() { return this.endValue !== 0; }

  /* Seat `numPlayers` copies of one capturing JS Strategy (built by makeMethods) on a
   * throwaway game re-seated from the root string; returns it for _run to step. */
  _seat(makeMethods) {
    const M = this.M;
    const log = new M.NoOpLog();
    const g = new M.GameState(log);
    const Strat = M.Strategy.extend('Strategy', makeMethods);
    const seats = [];
    for (let i = 0; i < this.numPlayers; i++) { const s = new Strat(); seats.push(s); g.add_player(s); }
    g.init_string(this.rootPhaseString);
    return { g, seats, done() { seats.forEach((s) => s.delete()); g.delete(); log.delete(); } };
  }

  /* Re-seat, arm the forced combo (null = just discover the root offers), step the
   * root decision, then step forward to the next decision node (or game end).
   *   forceBitwise: BigInt|null   which root combo to play (by bitwise identity)
   *   onOffer:      fn|null       called ONCE at the root decision with the live
   *                               (phase, combos, descriptors); its return is stashed
   *                               as this.capture (used by Part B to build leaf feeds
   *                               while the combos are alive). phase is deleted after.
   * Returns { rootOffered, child } where child = { phaseString, endValue }. */
  _run(forceBitwise, onOffer) {
    const M = this.M;
    if (this.seed != null) M.seed(this.seed >>> 0);
    let atRoot = true;
    let rootOffered = null;
    let captured = null; // the child decision phase string, set at the 2nd decision

    const describe = (combos) => {
      const out = [];
      for (let i = 0; i < combos.size(); i++) {
        const c = combos.get(i), parts = c.parts, locs = [], labels = [];
        for (let j = 0; j < parts.size(); j++) { const cd = parts.get(j); locs.push(cd.location); labels.push(cd.label); cd.delete(); }
        parts.delete(); c.delete();
        out.push({ index: i, bitwise: bitwiseOfLocations(locs), locations: locs, labels,
          isYield: locs.length === 0, isJoker: labels.some((lb) => lb[0] === 'X') });
      }
      return out;
    };

    const choose = (combos, game) => {
      if (atRoot) {
        rootOffered = describe(combos);
        if (onOffer) {
          const phase = game.export_phaseinfo();
          this.capture = onOffer(phase, combos, rootOffered);
          phase.delete();
        }
        atRoot = false;
        if (forceBitwise != null) {
          for (const d of rootOffered) if (d.bitwise === forceBitwise) return d.index;
          return 0; // impossible miss: fall back to combo 0 (matches Python)
        }
        return 0;
      }
      // the first decision reached AFTER the forced root move == the child node
      if (captured === null) captured = game.export_string();
      return 0;
    };

    const n = this.numPlayers;
    const ctx = this._seat({
      setup() { return 0; },
      getAttackIndex(combos, p, y, game) { return choose(combos, game); },
      getDefenseIndex(combos, p, d, game) { return choose(combos, game); },
      // A joker stepped INSIDE the search hands off to a fixed other seat so step()
      // stays reproducible under the seed; the REAL redirect on a committed joker is
      // decided separately by botRedirect at the game level, not by this model.
      getRedirectIndex(p, game) { return (game.active_player + 1) % n; },
    });

    let child;
    if (!ctx.g.is_runnable()) {
      child = { phaseString: this.rootPhaseString, endValue: this.endValue };
    } else {
      ctx.g.step(); // root decision (plays the forced combo)
      while (ctx.g.is_runnable() && captured === null) ctx.g.step(); // to the next decision / end
      if (captured !== null) {
        child = { phaseString: captured, endValue: 0 }; // a decision child (not terminal)
      } else {
        const term = ctx.g.export_string();
        const tp = M.PhaseInfo.from_string(term); const ev = tp.game_endvalue; tp.delete();
        child = { phaseString: term, endValue: ev }; // the game ended along the way
      }
    }
    ctx.done();
    return { rootOffered: rootOffered || [], child };
  }

  /* The combos legally playable at the root, as descriptors
   * ({ index, bitwise, locations, labels, isYield, isJoker }). `onOffer(phase, combos,
   * descriptors)` (optional) runs while the engine's combos are alive so a caller can
   * assemble net feeds synchronously (Part B); its return is stored as this.capture.
   * Cached after the first call. */
  offered(onOffer = null) {
    if (this._offered === null) this._offered = this._run(null, onOffer).rootOffered;
    return this._offered;
  }

  /* The phase at the next decision node reached after playing the root combo whose
   * bitwise is `bitwise`, as { phaseString, endValue } (endValue != 0 => the game
   * ended along the way; that child is terminal). */
  step(bitwise) {
    return this._run(bitwise, null).child;
  }

  dispose() { /* no persistent engine handles are held between calls */ }
}
