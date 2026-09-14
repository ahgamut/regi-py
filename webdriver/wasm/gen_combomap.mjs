/* Generate tables/combomap.json (the combo bitwise -> [loc, pst] cell map) from the
 * WASM engine -- the AZ card-space nets need it to index their (56,22) policy grid.
 * Run: node gen_combomap.mjs   (writes ./tables/combomap.json). Fully in-env; the
 * table only changes if the engine's combo space changes. */
import { mkdirSync, writeFileSync } from 'node:fs';
import RegiModule from './dist/regicore.mjs';
const M = await RegiModule();
const map = M.combo_map(); // { "<bitwise>": [loc, pst] }
const keys = Object.keys(map);
mkdirSync('./tables', { recursive: true });
writeFileSync('./tables/combomap.json', JSON.stringify(map));
console.log(`wrote tables/combomap.json: ${keys.length} entries`);
console.log('yield ->', map['0']);
// sanity: a few single-card cells (loc == entry+14*suit, PLAYED_SELF=0)
for (const k of keys.slice(0, 3)) console.log(k, '->', map[k]);
