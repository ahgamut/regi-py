/* One entry point that runs every smoke suite by IMPORTING each and CALLING its
 * runSmoke() in-process (not by spawning node subprocesses). Each suite returns its
 * failure count; this sums them and exits non-zero if any failed.
 *
 *   node smoke_all.mjs        (or `npm run smoke`)
 *
 * To add a suite: create smoke_<x>.mjs exporting `runSmoke()` and list it below. */
import { runSmoke as engine } from './smoke.mjs';
import { runSmoke as features } from './smoke_features.mjs';
import { runSmoke as bot } from './smoke_bot.mjs';
import { runSmoke as expander } from './smoke_expander.mjs';
import { runSmoke as driver } from './smoke_driver.mjs';
import { runSmoke as az } from './smoke_az.mjs';
import { runSmoke as mcts } from './smoke_mcts.mjs';

const suites = [
  ['smoke', engine],
  ['smoke_features', features],
  ['smoke_bot', bot],
  ['smoke_expander', expander],
  ['smoke_driver', driver],
  ['smoke_az', az],
  ['smoke_mcts', mcts],
];

let totalFailures = 0;
const failedSuites = [];
for (const [name, run] of suites) {
  console.log(`\n===== ${name} =====`);
  const failures = await run();
  totalFailures += failures;
  if (failures) failedSuites.push(`${name} (${failures})`);
}

console.log('\n========================================');
console.log(`${suites.length} suite(s): ${suites.length - failedSuites.length} passed, ${failedSuites.length} failed`);
if (failedSuites.length) { console.log('FAILED: ' + failedSuites.join(', ')); process.exit(1); }
console.log('ALL SMOKE SUITES PASSED');
