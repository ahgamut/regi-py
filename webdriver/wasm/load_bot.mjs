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

/* Pure dispatch (no IO): pick the bot class for an already-loaded `contract` +
 * `session`. Split from loadBot so node callers (which read the files themselves,
 * not over fetch) share the exact same paradigm/iters logic app.mjs uses.
 * opts: { iters = 0, comboMap = null }. Only iters === 0 (Direct-net) is supported
 * for now; a non-zero iters throws until the Explorer lands (Part B). */
export function buildBot(M, ort, session, contract, opts = {}) {
  const { iters = 0, comboMap = null } = opts;
  if (iters !== 0) {
    throw new Error(`buildBot: MCTS Explorer (iters=${iters}) not implemented yet; use iters=0`);
  }
  if (contract.paradigm === 'az') {
    return new AZBot(M, ort, session, contract, comboMap || M.combo_map());
  }
  return new NetBot(M, ort, session, contract);
}

/* Browser loader: fetch the <net>.io.json contract + create the ORT session, then
 * dispatch via buildBot. (Uses fetch -- relative to the page -- so it's browser-only;
 * node smokes call buildBot with files they read via readFileSync.) */
export async function loadBot(M, ort, baseUrl, netName, opts = {}) {
  const contract = await fetch(`${baseUrl}/${netName}.io.json`).then((r) => r.json());
  const session = await ort.InferenceSession.create(`${baseUrl}/${netName}.onnx`);
  return buildBot(M, ort, session, contract, opts);
}
