/* Net-guided MCTS Explorer (search) for the client-side bots -- the JS port of
 * strats/mcts_explorer.py MCTSNode + rl/{adz,az}/explorer.py's AlphaZeroNode/ADZNode.
 * One net forward pass per node (leaf value + child priors); PUCT selection; lazy,
 * prior-ordered expansion; visit-count backup; the move is the most-visited child.
 * ~iterations+1 forward passes per move (serial -- each expand awaits its leaf eval).
 *
 * Net-agnostic: MCTSNode drives search through a `bot` that exposes buildFeeds(phase,
 * combos, priorHistory, opts) -> built, and predictLeaf(built) -> { value, priors }
 * (priors are the per-offered-combo priors: ADZ softmax(cand_logits[:K]); AZ combomap
 * grid / keepy). So the same tree serves both paradigms -- ExplorerBot just wraps the
 * paradigm's Direct bot.
 *
 * Determinize ONCE at the search root (ExplorerBot perspectivizes there); every node
 * featurizes with reshuffle:false so children don't re-randomize hidden cards (that
 * would be an info leak). Phases are strings, combos are bitwise BigInts -- no live
 * engine handles are held across the async awaits. */

import { PhaseExpander } from './phase_expander.mjs';
import { bitwiseOfLocations } from './net_common.mjs';

/* rl/utils.normalize_probs: scale a nonnegative array (in place) to sum 1; an all-zero
 * sum dumps the mass on the last slot (terminal / no-move). */
function normalizeProbs(arr) {
  let t = 0; for (let i = 0; i < arr.length; i++) t += arr[i];
  if (t !== 0) { for (let i = 0; i < arr.length; i++) arr[i] /= t; }
  else if (arr.length) arr[arr.length - 1] = 1.0;
  return arr;
}

/* rl/utils.hp_loss_penalty: shaped reward for a LOST game, tiered by enemy HP left. */
function hpLossPenalty(hp) {
  if (hp > 320) return -1.0;
  if (hp > 280) return -0.9;
  return (160 - hp) / 160;
}

/* enemy_hp_left(phase): sum of max(hp, 0) over the enemy pile. */
function enemyHpLeft(M, phaseString) {
  const ph = M.PhaseInfo.from_string(phaseString);
  const ep = ph.enemy_pile;
  let hp = 0;
  for (let i = 0; i < ep.size(); i++) { const e = ep.get(i); if (e.hp > 0) hp += e.hp; e.delete(); }
  ep.delete(); ph.delete();
  return hp;
}

/* Deterministic in-place shuffle of `arr` via a small LCG on `seed` (so equal-prior
 * children get a stable-but-varied expansion order, mirroring the Python node's
 * random.shuffle before the ascending prior sort -- without touching Math.random). */
function seededShuffle(arr, seed) {
  let s = (seed >>> 0) || 1;
  for (let i = arr.length - 1; i > 0; i--) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const j = s % (i + 1);
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
  return arr;
}

export class MCTSNode {
  constructor(M, phaseString, { parent = null, prevBitwise = null, prevIndex = -1, prior = 1.0, weight = Math.SQRT2, seed = null } = {}) {
    this.M = M;
    this.phaseString = phaseString;
    this.parent = parent;
    this.prevBitwise = prevBitwise; // the combo (bitwise) played to reach this node
    this.prevIndex = prevIndex;
    this.prior = prior;
    this.weight = weight;
    this.seed = seed;
    this.visits = 0;
    this.value = 0.0;       // backed-up reward sum W (kept separate from leafValue)
    this.leafValue = 0.0;   // the net's value estimate for this leaf
    this.children = [];
    this.childmap = new Map(); // bitwise string -> child node
    this.nextCombos = [];      // offered-combo descriptors at this node
    this.nextPriors = null;    // Float32Array, aligned to nextCombos
    this.remExpInd = [];       // unexpanded child indices, ascending by prior (pop = highest)
    this.expander = null;
    this.priorHistory = [];    // decision phase strings BEFORE this node (buildFeeds trims)
    const ph = M.PhaseInfo.from_string(phaseString);
    this.endValue = ph.game_endvalue; // != 0 -> terminal
    ph.delete();
  }

  isTerminal() { return this.endValue !== 0; }
  canExpandFurther() { return this.remExpInd.length !== 0; }

  /* PUCT: Q + prior*weight*sqrt(parent.N)/(1+N); inf at N==0 so unvisited children
   * are tried first (rl/{adz,az}/explorer.py ADZNode/AlphaZeroNode.ucb1). */
  ucb1() {
    if (this.visits === 0) return Infinity;
    const q = this.value / this.visits;
    const u = this.parent ? Math.sqrt(this.parent.visits) / (1 + this.visits) : 0;
    return q + this.prior * this.weight * u;
  }

  bestChildNode() {
    let best = this.children[0];
    for (const c of this.children) if (c.visits > best.visits) best = c; // ties -> first
    return best;
  }

  /* Leaf eval: list this node's offered combos, capture the net feeds while the combos
   * are alive, run ONE forward pass, and set leafValue + the smoothed/normalized child
   * priors (expansion ordered by prior). `priorHistory` = the decision phase strings
   * leading here. Terminal / no-move nodes skip the net. */
  async evaluate(bot, priorHistory) {
    this.priorHistory = priorHistory;
    if (this.isTerminal()) { this.leafValue = 0.0; return; }
    this.expander = new PhaseExpander(this.M, this.phaseString, { seed: this.seed });
    const M = this.M;
    this.nextCombos = this.expander.offered((phase, combos) => {
      const hist = priorHistory.map((s) => M.PhaseInfo.from_string(s));
      const built = bot.buildFeeds(phase, combos, hist, { reshuffle: false });
      hist.forEach((p) => p.delete());
      return built;
    });
    const K = this.nextCombos.length;
    if (K === 0) { this.leafValue = 0.0; this.expander = null; return; }
    const { value, priors } = await bot.predictLeaf(this.expander.capture);
    this.expander.capture = null; // release the leaf feeds (tensors) now the pass is done
    this.leafValue = value;
    const np = new Float32Array(K);
    for (let i = 0; i < K && i < priors.length; i++) np[i] = priors[i];
    for (let i = 0; i < K; i++) np[i] += 1e-3; // smooth, then normalize (matches Python)
    normalizeProbs(np);
    this.nextPriors = np;
    this.remExpInd = Array.from({ length: K }, (_, i) => i);
    // shuffle (stable-but-varied for equal priors), then sort ascending by prior so
    // pop() expands the highest-prior child first
    seededShuffle(this.remExpInd, (this.seed || 1) + K);
    this.remExpInd.sort((a, b) => np[a] - np[b]);
  }

  /* Materialize the highest-prior unexpanded child (one expander.step + one leaf eval). */
  async expand(bot) {
    const i = this.remExpInd.pop();
    const combo = this.nextCombos[i];
    const child = this.expander.step(combo.bitwise); // { phaseString, endValue }
    const node = new MCTSNode(this.M, child.phaseString, {
      parent: this, prevBitwise: combo.bitwise, prevIndex: i, prior: this.nextPriors[i],
      weight: this.weight, seed: this.seed,
    });
    await node.evaluate(bot, [...this.priorHistory, this.phaseString]);
    this.children.push(node);
    this.childmap.set(combo.bitwise.toString(), node);
    if (this.remExpInd.length === 0) this.expander = null; // fully expanded: drop the throwaway game
    return node;
  }

  /* Leaf backup value on the [-1,1] scale: win 1, loss shaped, else the net estimate. */
  simulate() {
    if (this.endValue === 1) return 1.0;
    if (this.endValue === -1) return hpLossPenalty(enemyHpLeft(this.M, this.phaseString));
    return this.leafValue;
  }

  /* Descend by PUCT to a node with room to expand (or a terminal / no-move leaf). */
  static select(node) {
    while (!node.canExpandFurther() && !node.isTerminal()) {
      if (node.children.length === 0) break; // a no-move leaf (K==0): nothing to descend into
      let best = node.children[0];
      for (const c of node.children) if (c.ucb1() > best.ucb1()) best = c;
      node = best;
    }
    return node;
  }

  /* Co-op backup: add the reward to every node from `node` up to the root (no sign flip). */
  static update(node, reward) {
    while (node) { node.visits += 1; node.value += reward; node = node.parent; }
  }
}

/* Run `iterations` of net-guided MCTS from an already-evaluated `root` (in place). */
export async function simulateNode(bot, root, iterations) {
  for (let i = 0; i < iterations; i++) {
    let node = MCTSNode.select(root);
    if (!node.isTerminal() && node.canExpandFurther()) node = await node.expand(bot);
    MCTSNode.update(node, node.simulate());
  }
  return root;
}

/* A search bot: wraps a Direct bot (NetBot / AZBot) and plays the most-visited move
 * from an `iterations`-deep MCTS. The wrapped bot supplies buildFeeds + predictLeaf
 * (the paradigm-specific leaf eval); this class is otherwise net-agnostic. */
export class ExplorerBot {
  constructor(M, netBot, { iterations = 64, weight = Math.SQRT2, seed = 1 } = {}) {
    this.M = M;
    this.net = netBot;
    this.iterations = iterations;
    this.weight = weight;
    this.seed = seed >>> 0 || 1;
    this.maxHistory = netBot.maxHistory;
    this.isExplorer = true;
    // AZ Direct bots redirect by value-argmax (chooseRedirect); ADZ have none (random
    // fallback). Delegate to the wrapped bot when present so the joker handoff matches.
    if (typeof netBot.chooseRedirect === 'function') {
      this.chooseRedirect = (...a) => netBot.chooseRedirect(...a);
    }
  }

  /* Search from `decisionPhaseString` (the driver's captured decision) and return the
   * OFFERED index (into `comboData`) of the most-visited move. `historyStrings` is the
   * real past decision phases (strings). Perspectivizes ONCE here (the search root). */
  async search(decisionPhaseString, historyStrings, comboData) {
    if (!comboData.length) return -1;
    const M = this.M;
    M.seed(this.seed); // reproducible root determinization (nodes re-seed per re-seat)
    const raw = M.PhaseInfo.from_string(decisionPhaseString);
    const persp = M.PhaseInfo.randomize_from(raw, raw.active_player); // determinize once
    const rootString = persp.to_string();
    persp.delete(); raw.delete();

    const root = new MCTSNode(M, rootString, { prior: 1.0, weight: this.weight, seed: this.seed });
    await root.evaluate(this.net, historyStrings);
    await simulateNode(this.net, root, this.iterations);
    if (root.children.length === 0) return 0; // nothing searched (e.g. single forced move)

    const best = root.bestChildNode().prevBitwise;
    for (const c of comboData) if (bitwiseOfLocations(c.locations) === best) return c.index;
    return 0; // best combo not in the offered set (shouldn't happen: same hand, same offers)
  }
}
