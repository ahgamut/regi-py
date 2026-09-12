/* Generate tables/presets_{2,3,4}p.json -- fixed starter deals the menu offers as
 * "known scenarios" so a player can replay a curated opening instead of a random one.
 * Each file holds three difficulty tiers (Easy/Medium/Hard), 5 presets each = 15/count.
 *
 * A preset is just an opening phase string (a fresh GameState.initialize() for that
 * player count, exported), replayed later via GameDriver.newGame(startPhase) ->
 * init_string. This mirrors repeaters/make_phases.py, but is driven from the WASM
 * engine in node because the pybind ext can't be imported in this env (same reason
 * gen_combomap.mjs exists instead of a gen_combomap.py). Deterministic: a fixed
 * per-count seed makes regeneration reproducible.
 *
 * Run: node gen_presets.mjs   (writes ./tables/presets_{2,3,4}p.json, committed).
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import RegiModule from './dist/regicore.mjs';

const M = await RegiModule();
const COUNTS = [2, 3, 4];
const TIERS = ['Easy', 'Medium', 'Hard']; // difficulty buckets shown in the menu
const PER_TIER = 5;           // presets per tier => 15 per player count
const BASE_SEED = 20260911;   // fixed => reproducible regeneration

/* One fresh deal for `numPlayers`, exported as its opening phase string. Seeds are
 * consumed from the core RNG, which make_phases lets advance across deals. */
function dealPhase(numPlayers) {
  const log = new M.NoOpLog();
  const g = new M.GameState(log);
  const seats = [];
  for (let i = 0; i < numPlayers; i++) { const s = new M.RandomStrategy(); seats.push(s); g.add_player(s); }
  g.initialize();
  const phase = g.export_string();
  seats.forEach((s) => s.delete());
  g.delete(); log.delete();
  return phase;
}

mkdirSync('./tables', { recursive: true });
for (const numPlayers of COUNTS) {
  M.seed(BASE_SEED + numPlayers); // per-count seed; deals advance the RNG in turn
  // NOTE: these are arbitrary placeholder deals for now -- the Easy/Medium/Hard split
  // is NOT yet meaningful; curate real per-difficulty openings into each tier later.
  const tiers = TIERS.map((name) => ({
    name,
    phases: Array.from({ length: PER_TIER }, () => dealPhase(numPlayers)),
  }));
  const out = { num_players: numPlayers, tiers };
  const path = `./tables/presets_${numPlayers}p.json`;
  writeFileSync(path, JSON.stringify(out, null, 2));
  console.log(`wrote ${path}: ${TIERS.length * PER_TIER} presets (${TIERS.length} tiers x ${PER_TIER}, ${numPlayers}p)`);
}
