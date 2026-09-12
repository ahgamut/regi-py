/* AZ (AlphaZero, card-space) Direct-net bot: the client-side analogue of
 * rl/az/explorer.py NetDirectStrategy (iters=0, no search). One onnxruntime forward
 * pass yields (v, k, a): value (tanh), keepyness k[56] (sigmoid), and a (56,22) action
 * grid (already masked-softmaxed). Scoring:
 *   ATTACK  argmax a[loc,pst], (loc,pst) = comboMap[combo.bitwise]  (== export_onnx
 *           _direct_index_az_attack). Combos with no cell score 0 (never for legal atk).
 *   DEFENSE argmax max(0, 1 - Σ k[card.location]) over the combo's parts (keepy
 *           fallback -- AZ has no defense policy head).
 *   REDIRECT per-opponent value argmax: re-perspectivize from each other seat, forward,
 *           read v, argmax.
 *
 * Two input paradigms (from the <net>.io.json contract):
 *   token nets (percardmlp/cardtx/mixer): tokens (1,56,264)   -- reuses fuse_card_tokens
 *   conv  nets (basic/attntrunk):         location (1,8,56,9), used_pile (1,8,56,22),
 *                                         capability (1,8,56,2) -- 8 frames stacked here
 * Same buildFeeds/runFeeds shape as adz_bot.NetBot so GameDriver drives it unchanged. */

import { MAX_CARDS, trimmedHistory, comboLocations, bitwiseOfLocations } from './net_common.mjs';

const LOC_DIM = 9;
const USP_DIM = 22;
const CAP_DIM = 2;
const PLAYED_STATUS = 22; // a grid is (56, MAX_PLAYED_STATUS=22)

export class AZBot {
  /* `contract` is the parsed <net>.io.json; `comboMap` the loaded tables/combomap.json
   * ({ "<bitwise>": [loc, pst] }). AZ nets only. */
  constructor(M, ort, session, contract, comboMap) {
    if (contract.paradigm !== 'az') {
      throw new Error(`AZBot supports AZ nets only; got ${contract.paradigm} (${contract.net})`);
    }
    this.M = M;
    this.ort = ort;
    this.session = session;
    this.contract = contract;
    this.comboMap = comboMap;
    this.maxHistory = contract.max_history;
    this.inputNames = contract.inputs.map((i) => i.name);
    this.isToken = this.inputNames.includes('tokens');
  }

  /* Featurize `rawPhase` from `viewSeat`'s perspective (defaults to the active player)
   * into the net's input feeds. `reshuffle` re-randomizes hidden cards (real play);
   * false is a faithful string round-trip (determinism / golden). Returns the feeds. */
  _featurize(rawPhase, priorHistory, viewSeat = null, reshuffle = true) {
    const { M, ort } = this;
    const persp = viewSeat == null ? rawPhase.active_player : viewSeat;
    const root = reshuffle
      ? M.PhaseInfo.randomize_from(rawPhase, persp)
      : M.PhaseInfo.from_string(rawPhase.to_string());
    const window = trimmedHistory(priorHistory, root, this.maxHistory);
    const feeds = {};
    if (this.isToken) {
      const wv = new M.VectorPhaseInfo();
      for (const ph of window) wv.push_back(ph);
      const tokens = M.features_fuse_card_tokens(wv, persp); // Float32Array(56*264)
      wv.delete();
      feeds.tokens = new ort.Tensor('float32', tokens, [1, MAX_CARDS, tokens.length / MAX_CARDS]);
    } else {
      // conv: stack the window frames as channels -> (1, F, 56, W), frame-major
      const F = window.length; // == maxHistory after trim/pad
      const loc = new Float32Array(F * MAX_CARDS * LOC_DIM);
      const usp = new Float32Array(F * MAX_CARDS * USP_DIM);
      const cap = new Float32Array(F * MAX_CARDS * CAP_DIM);
      for (let f = 0; f < F; f++) {
        loc.set(M.features_location_array(window[f], persp), f * MAX_CARDS * LOC_DIM);
        usp.set(M.features_used_pile_array(window[f]), f * MAX_CARDS * USP_DIM);
        cap.set(M.features_card_capabilities(window[f]), f * MAX_CARDS * CAP_DIM);
      }
      feeds.location = new ort.Tensor('float32', loc, [1, F, MAX_CARDS, LOC_DIM]);
      feeds.used_pile = new ort.Tensor('float32', usp, [1, F, MAX_CARDS, USP_DIM]);
      feeds.capability = new ort.Tensor('float32', cap, [1, F, MAX_CARDS, CAP_DIM]);
    }
    root.delete();
    return feeds;
  }

  /* Build the feeds + capture per-combo part locations and the attack/defense flag,
   * so runFeeds() can score the outputs. Mirrors NetBot.buildFeeds' signature. */
  buildFeeds(rawPhase, combos, priorHistory = [], opts = {}) {
    const reshuffle = opts.reshuffle !== false;
    const attacking = rawPhase.phase_attacking;
    const K = combos.size();
    const comboLocs = [];
    for (let i = 0; i < K; i++) comboLocs.push(comboLocations(combos.get(i))); // consumes handles
    const feeds = this._featurize(rawPhase, priorHistory, null, reshuffle);
    return { feeds, K, comboLocs, attacking };
  }

  /* One forward pass -> the per-offered-combo scores runFeeds argmaxes over: for an
   * attack, the combomap grid prior a[loc*22+pst] (a is the flat, already-softmaxed
   * (1,1,56,22) grid; a combo with no cell scores 0, never for a legal attack); for a
   * defense, the keepy score max(0, 1 - Σ k[loc]) (prefer discarding cards the net
   * least wants to keep). Returned as a Float32Array of length K. */
  async scoreCombos(built) {
    const { feeds, K, comboLocs, attacking } = built;
    const out = await this.session.run(feeds);
    const scores = new Float32Array(K);
    if (attacking) {
      const a = out.a.data;
      for (let i = 0; i < K; i++) {
        const cell = this.comboMap[bitwiseOfLocations(comboLocs[i]).toString()];
        scores[i] = cell ? a[cell[0] * PLAYED_STATUS + cell[1]] : 0;
      }
    } else {
      const k = out.k.data;
      for (let i = 0; i < K; i++) {
        let wt = 0;
        for (const loc of comboLocs[i]) wt += k[loc];
        scores[i] = Math.max(0, 1 - wt);
      }
    }
    return scores;
  }

  /* Pick the highest-scoring offered combo (argmax; ties -> first, like np.argmax). */
  async runFeeds(built) {
    if (built.K === 0) return -1;
    const scores = await this.scoreCombos(built);
    let best = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
    return best;
  }

  async choose(rawPhase, combos, priorHistory = [], opts = {}) {
    return this.runFeeds(this.buildFeeds(rawPhase, combos, priorHistory, opts));
  }

  /* MCTS leaf eval (rl/az/explorer.py AlphaZeroNode): one forward pass -> the value
   * head (v, tanh) and the per-offered-combo priors -- the SAME scores the Direct bot
   * ranks (attack: combomap grid a[cell]; defense: keepy max(0,1-Σk)). The Explorer
   * smooths + normalizes these into child priors. Independent of scoreCombos so the
   * golden-checked Direct path is untouched. */
  async predictLeaf(built) {
    const { feeds, K, comboLocs, attacking } = built;
    const out = await this.session.run(feeds);
    const value = out.v.data[0];
    const priors = new Float32Array(K);
    if (attacking) {
      const a = out.a.data;
      for (let i = 0; i < K; i++) {
        const cell = this.comboMap[bitwiseOfLocations(comboLocs[i]).toString()];
        priors[i] = cell ? a[cell[0] * PLAYED_STATUS + cell[1]] : 0;
      }
    } else {
      const k = out.k.data;
      for (let i = 0; i < K; i++) {
        let wt = 0; for (const loc of comboLocs[i]) wt += k[loc];
        priors[i] = Math.max(0, 1 - wt);
      }
    }
    return { value, priors };
  }

  /* Jester redirect (rl/az/explorer.py getRedirectIndex): hand the turn to the other
   * seat whose position the net values highest. N-1 forward passes. */
  async chooseRedirect(rawPhase, priorHistory, numPlayers) {
    const active = rawPhase.active_player;
    let best = -1, bestVal = -Infinity;
    for (let i = 0; i < numPlayers; i++) {
      if (i === active) continue;
      const feeds = this._featurize(rawPhase, priorHistory, i, true);
      const out = await this.session.run(feeds);
      const v = out.v.data[0];
      if (v > bestVal) { bestVal = v; best = i; }
    }
    return best;
  }
}

/* Convenience loader: fetch the io-contract + create the ORT session; `comboMap` is
 * the already-loaded combomap (fetch tables/combomap.json once and share it). */
export async function loadAZBot(M, ort, baseUrl, netName, comboMap) {
  const contract = await fetch(`${baseUrl}/${netName}.io.json`).then((r) => r.json());
  const session = await ort.InferenceSession.create(`${baseUrl}/${netName}.onnx`);
  return new AZBot(M, ort, session, contract, comboMap);
}
