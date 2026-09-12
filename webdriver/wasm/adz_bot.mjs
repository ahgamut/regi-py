/* ADZ Direct-net bot: the client-side analogue of ADZDirectStrategy._policy_index
 * (rl/adz/explorer.py). Given the current phase + its offered combos, it featurizes
 * with the WASM kernels (identical to training), runs ONE onnxruntime forward pass,
 * and returns the argmax offered-combo index -- NO search. Attack and defense share
 * this one path (the ADZ policy is uniform over offered subsets).
 *
 * Framework-agnostic: the caller injects the WASM module `M`, the `ort` namespace
 * (onnxruntime-web), an already-created `session`, and the net's io-contract JSON
 * (the <net>.io.json the exporter emits). See loadNetBot() for a convenience loader.
 *
 * Wire contract (must match trainers/export_onnx.py):
 *   inputs  tokens (1,56,264) f32, cand_feats (1,128,9) f32, cand_mask (1,128) f32,
 *           + membership: adzmulti cand_members (1,128,56) f32
 *                         adzpool  cand_idx (1,128,7) i64 + cand_partmask (1,128,7) f32
 *   outputs value (1,1), cand_logits (1,128) raw (-inf on pads), keepy (1,56)
 * JS reads cand_logits[0:K] over the K real candidates and argmaxes. */

const MAX_CARDS = 56;
const MAX_CANDIDATES = 128;
const MAX_PARTS = 7; // adzpool: max member locations in an offered subset (7-card hand)
const CAND_FEATURE_DIM = 9;

/* trimmed_history (rl/adz/explorer.py): the last `maxhist` phases ending at `phase`,
 * left-padded with the oldest frame when short. */
function trimmedHistory(history, phase, maxhist) {
  const tmp = [...history, phase];
  if (tmp.length >= maxhist) return tmp.slice(tmp.length - maxhist);
  return new Array(maxhist - tmp.length).fill(tmp[0]).concat(tmp);
}

/* Member card locations of a combo (== set bits of its bitwise); yield -> [].
 * CONSUMES `combo` (deletes the handle), since callers pass a fresh combos.get(i).*/
function comboLocations(combo) {
  const parts = combo.parts;
  const locs = [];
  for (let i = 0; i < parts.size(); i++) {
    const card = parts.get(i);
    locs.push(card.location);
    card.delete();
  }
  parts.delete();
  combo.delete();
  return locs;
}

export class NetBot {
  /* `contract` is the parsed <net>.io.json ({net, paradigm, inputs:[{name,...}],
   * outputs, max_history}). Only ADZ nets are supported here (the Direct MVP). */
  constructor(M, ort, session, contract) {
    if (contract.paradigm !== 'adz') {
      throw new Error(`NetBot supports ADZ nets only; got ${contract.paradigm} (${contract.net})`);
    }
    this.M = M;
    this.ort = ort;
    this.session = session;
    this.contract = contract;
    this.maxHistory = contract.max_history;
    this.inputNames = contract.inputs.map((i) => i.name);
    this.logitsName = contract.outputs.includes('cand_logits') ? 'cand_logits' : contract.outputs[1];
  }

  /* Build the onnxruntime feeds for one decision. `rawPhase` is the current phase
   * (from GameState.export_phaseinfo()); `combos` its offered VectorCombo;
   * `priorHistory` the real past phases (may be empty). Mirrors tensorify_predict:
   * perspectivize the root, window the history, featurize with the WASM kernels. */
  buildFeeds(rawPhase, combos, priorHistory = [], opts = {}) {
    const { M, ort } = this;
    const reshuffle = opts.reshuffle !== false; // default: perspectivize (real play)
    const K = combos.size();
    // perspectivize: reshuffle hidden cards from the active player's view. Disabled
    // (reshuffle:false) for the golden check, so both sides featurize the SAME phase
    // with no RNG divergence; `root` is then a faithful string round-trip copy (what
    // the Python fixture does with PhaseInfo.from_string). Either way `root` is a
    // fresh, deletable object.
    const root = reshuffle
      ? M.PhaseInfo.randomize_from(rawPhase, rawPhase.active_player)
      : M.PhaseInfo.from_string(rawPhase.to_string());
    const persp = root.active_player;
    const window = trimmedHistory(priorHistory, root, this.maxHistory);

    // tokens (1,56,264): fuse the card-token window
    const wv = new M.VectorPhaseInfo();
    for (const p of window) wv.push_back(p);
    const tokens = M.features_fuse_card_tokens(wv, persp); // Float32Array(56*264)
    wv.delete();

    // per-candidate semantics (K,9), padded to MAX_CANDIDATES
    const candRaw = M.features_candidate_semantics(root, combos); // Float32Array(K*9)
    const candFeats = new Float32Array(MAX_CANDIDATES * CAND_FEATURE_DIM);
    candFeats.set(candRaw.subarray(0, K * CAND_FEATURE_DIM));
    const candMask = new Float32Array(MAX_CANDIDATES);
    for (let i = 0; i < K; i++) candMask[i] = 1;

    const feeds = {};
    if (this.inputNames.includes('tokens')) {
      feeds.tokens = new ort.Tensor('float32', tokens, [1, MAX_CARDS, tokens.length / MAX_CARDS]);
    }
    feeds.cand_feats = new ort.Tensor('float32', candFeats, [1, MAX_CANDIDATES, CAND_FEATURE_DIM]);
    feeds.cand_mask = new ort.Tensor('float32', candMask, [1, MAX_CANDIDATES]);

    // membership: adzmulti multi-hot(56), or adzpool index+mask(7)
    if (this.inputNames.includes('cand_members')) {
      const members = new Float32Array(MAX_CANDIDATES * MAX_CARDS);
      for (let i = 0; i < K; i++) {
        for (const loc of comboLocations(combos.get(i))) members[i * MAX_CARDS + loc] = 1;
      }
      feeds.cand_members = new ort.Tensor('float32', members, [1, MAX_CANDIDATES, MAX_CARDS]);
    }
    if (this.inputNames.includes('cand_idx')) {
      const idx = new BigInt64Array(MAX_CANDIDATES * MAX_PARTS);
      const partmask = new Float32Array(MAX_CANDIDATES * MAX_PARTS);
      for (let i = 0; i < K; i++) {
        const locs = comboLocations(combos.get(i));
        for (let j = 0; j < locs.length && j < MAX_PARTS; j++) {
          idx[i * MAX_PARTS + j] = BigInt(locs[j]);
          partmask[i * MAX_PARTS + j] = 1;
        }
      }
      feeds.cand_idx = new ort.Tensor('int64', idx, [1, MAX_CANDIDATES, MAX_PARTS]);
      feeds.cand_partmask = new ort.Tensor('float32', partmask, [1, MAX_CANDIDATES, MAX_PARTS]);
    }

    root.delete();
    return { feeds, K };
  }

  /* Pick the highest-scoring offered-combo index (argmax over the K real
   * candidates' logits; softmax preserves the argmax, so scoring the raw logits is
   * equivalent). Returns -1 for an empty offer. */
  async choose(rawPhase, combos, priorHistory = [], opts = {}) {
    const { K, feeds } = this.buildFeeds(rawPhase, combos, priorHistory, opts);
    if (K === 0) return -1;
    const out = await this.session.run(feeds);
    const logits = out[this.logitsName].data; // Float32Array length MAX_CANDIDATES
    let best = 0, bestVal = logits[0];
    for (let i = 1; i < K; i++) {
      if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
    }
    return best;
  }
}

/* Convenience loader: fetch the io-contract + create the ORT session from a base
 * URL that holds `<net>.onnx` and `<net>.io.json` (e.g. the dist/ dir). The caller
 * passes the already-imported `M` (WASM module) and `ort` (onnxruntime-web). */
export async function loadNetBot(M, ort, baseUrl, netName) {
  const contract = await fetch(`${baseUrl}/${netName}.io.json`).then((r) => r.json());
  const session = await ort.InferenceSession.create(`${baseUrl}/${netName}.onnx`);
  return new NetBot(M, ort, session, contract);
}
