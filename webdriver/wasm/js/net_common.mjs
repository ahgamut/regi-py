/* Shared helpers for the client-side net bots (ADZ + AZ, Direct + MCTS Explorer).
 * Kept framework-agnostic: callers inject the WASM module `M`. */

export const MAX_CARDS = 56;

/* trimmed_history (rl/.../explorer.py _trimmed_history): the last `maxhist` phases
 * ending at `phase`, left-padded with the oldest frame when short. */
export function trimmedHistory(history, phase, maxhist) {
  const tmp = [...history, phase];
  if (tmp.length >= maxhist) return tmp.slice(tmp.length - maxhist);
  return new Array(maxhist - tmp.length).fill(tmp[0]).concat(tmp);
}

/* Member card locations of a combo (== set bits of its bitwise); yield -> [].
 * CONSUMES `combo` (deletes the handle), since callers pass a fresh combos.get(i). */
export function comboLocations(combo) {
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

/* perspectivize (strats/phase_utils.py): view the phase from `persp`'s seat. In real
 * play (reshuffle=true) hidden cards are re-randomized from that seat (randomize_from);
 * for determinism (reshuffle=false, e.g. golden checks) it's a faithful string
 * round-trip copy instead. Returns a fresh, deletable PhaseInfo. `persp` defaults to
 * the active player. */
export function perspectivize(M, rawPhase, persp = null, reshuffle = true) {
  const p = persp == null ? rawPhase.active_player : persp;
  return reshuffle
    ? M.PhaseInfo.randomize_from(rawPhase, p)
    : M.PhaseInfo.from_string(rawPhase.to_string());
}

/* The combo's bitwise identity (u64 location bitmask) as a BigInt: OR of 1<<loc over
 * its card locations. Yield / empty -> 0n. Matches core Combo.bitwise (and the keys
 * of tables/combomap.json, which are BigInt(...).toString()). 32-bit shifts corrupt
 * locations >= 32, so BigInt is mandatory. */
export function bitwiseOfLocations(locs) {
  let b = 0n;
  for (const loc of locs) b |= (1n << BigInt(loc));
  return b;
}

/* Find the offered-combo index whose bitwise matches `target` (BigInt); `def` (-1)
 * if none. Port of phase_utils.index_of_bitwise -- bitwise is the canonical identity,
 * robust to enumeration order. `combosBitwise` is an array of BigInt. */
export function indexOfBitwise(combosBitwise, target, def = -1) {
  for (let i = 0; i < combosBitwise.length; i++) {
    if (combosBitwise[i] === target) return i;
  }
  return def;
}

/* softmax over the first K entries of `logits` (a typed array), returning a plain
 * Float32Array of length K. Numerically stable (subtract max). */
export function softmaxK(logits, K) {
  const out = new Float32Array(K);
  if (K === 0) return out;
  let mx = logits[0];
  for (let i = 1; i < K; i++) if (logits[i] > mx) mx = logits[i];
  let sum = 0;
  for (let i = 0; i < K; i++) { const e = Math.exp(logits[i] - mx); out[i] = e; sum += e; }
  if (sum > 0) for (let i = 0; i < K; i++) out[i] /= sum;
  return out;
}
