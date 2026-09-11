/* loadBot: the single entry point app.mjs uses to build a bot for a seat. It reads
 * the <net>.io.json contract, creates the onnxruntime session ONCE, and dispatches
 * on `paradigm` (adz -> NetBot, az -> AZBot) and `iters` (0 -> Direct-net; >0 ->
 * MCTS Explorer, added in Part B). AZ nets need the combomap (bitwise -> grid cell);
 * it's small + build-derived, so we take `M.combo_map()` unless the caller shares a
 * pre-built one via opts.comboMap.
 *
 * Framework-agnostic: the caller injects `M` (WASM module) and `ort` (onnxruntime). */

import { NetBot } from './adz_bot.mjs';
import { AZBot } from './az_bot.mjs';
import { ExplorerBot } from './mcts.mjs';

/* Pure dispatch (no IO): pick the bot for an already-loaded `contract` + `session`.
 * Split from loadBot so node callers (which read the files themselves, not over fetch)
 * share the exact same paradigm/iters logic app.mjs uses.
 * opts: { iters = 0, comboMap = null }. `iters` 0 = the Direct-net bot; > 0 wraps it in
 * an `iters`-deep MCTS Explorer (both paradigms; the leaf eval is the Direct bot's). */
export function buildBot(M, ort, session, contract, opts = {}) {
  const { iters = 0, comboMap = null } = opts;
  const direct = contract.paradigm === 'az'
    ? new AZBot(M, ort, session, contract, comboMap || M.combo_map())
    : new NetBot(M, ort, session, contract);
  return iters > 0 ? new ExplorerBot(M, direct, { iterations: iters }) : direct;
}

/* Browser loader: fetch the <net>.io.json contract + create the ORT session, then
 * dispatch via buildBot. (Uses fetch -- relative to the page -- so it's browser-only;
 * node smokes call buildBot with files they read via readFileSync.) */
export async function loadBot(M, ort, baseUrl, netName, opts = {}) {
  const contract = await fetch(`${baseUrl}/${netName}.io.json`).then((r) => r.json());
  const session = await ort.InferenceSession.create(`${baseUrl}/${netName}.onnx`);
  return buildBot(M, ort, session, contract, opts);
}
