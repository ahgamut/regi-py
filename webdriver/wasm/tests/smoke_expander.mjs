/* Headless smoke for phase_expander.mjs (Part B's missing engine primitive), pure
 * WASM engine -- no onnxruntime. Build the module first, then:
 *   node smoke_expander.mjs        (or all suites via `npm run smoke`)
 *
 * Checks: (1) offered() lists a decision's combos and each descriptor's bitwise ==
 * OR of its card locations; (2) that set MATCHES GameDriver's own enumeration of the
 * same decision (same engine, same phase); (3) step(bitwise) returns a well-formed
 * child that is a real decision node (offered non-empty) or terminal (endValue +-1),
 * never an empty non-terminal; (4) step is deterministic under a fixed seed; (5) the
 * expander alone can drive a game from a root to a terminal state (loops past auto-
 * resolved onePhases to the next decision); (6) onOffer fires once with live combos.
 *
 * Exports runSmoke() so smoke_all.mjs can call it as a function (returns the failure
 * count); running this file directly runs just this suite and exits non-zero on fail. */
import { pathToFileURL } from 'node:url';
import RegiModule from '../dist/regicore.mjs';
import { GameDriver } from '../js/game_driver.mjs';
import { PhaseExpander } from '../js/phase_expander.mjs';
import { bitwiseOfLocations } from '../js/net_common.mjs';

const bitwiseSet = (arr) => new Set(arr.map((x) => x.toString()));
const eqSet = (a, b) => a.size === b.size && [...a].every((x) => b.has(x));

/* Advance a driver (all-human seats, so every turn is a captured decision) to the
 * next real decision, returning that dec (or null if the game ended). */
function nextDecision(driver) {
  for (let i = 0; i < 500; i++) {
    const dec = driver.prepare();
    if (dec.kind === 'ended') return null;
    if (dec.kind === 'auto') { driver.commit(-1); continue; }
    return dec;
  }
  return null;
}

export async function runSmoke() {
  const M = await RegiModule();
  let failures = 0;
  const check = (name, cond) => { console.log(`${cond ? 'ok  ' : 'FAIL'}  ${name}`); if (!cond) failures++; };

  // Collect several real decision phases across a few games / seat counts.
  const decisions = []; // { phaseString, comboBitwise:Set<string>, numPlayers }
  for (const [numPlayers, seed] of [[2, 11], [3, 22], [4, 33]]) {
    const driver = new GameDriver(M, { numPlayers, seatBots: new Array(numPlayers).fill(null), seed });
    driver.newGame();
    let got = 0;
    while (got < 6) {
      const dec = nextDecision(driver);
      if (!dec) { driver.newGame(); continue; }
      const bw = dec.comboData.map((c) => bitwiseOfLocations(c.locations));
      decisions.push({ phaseString: dec.decisionPhaseString, comboBitwise: bitwiseSet(bw), numPlayers });
      driver.commit(0); // play the first offered combo to move on
      got++;
    }
    driver.dispose();
  }
  check(`collected decision phases across 2/3/4p (n=${decisions.length})`, decisions.length >= 15);

  // 1 + 2) offered() well-formed and consistent with GameDriver's enumeration.
  let badBitwise = 0, mismatchOffers = 0, emptyOffers = 0;
  for (const d of decisions) {
    const exp = new PhaseExpander(M, d.phaseString, { seed: 7 });
    const offered = exp.offered();
    if (offered.length === 0) emptyOffers++;
    for (const o of offered) {
      if (bitwiseOfLocations(o.locations) !== o.bitwise) badBitwise++;
    }
    if (!eqSet(bitwiseSet(offered.map((o) => o.bitwise)), d.comboBitwise)) mismatchOffers++;
    exp.dispose();
  }
  check('every offered descriptor bitwise == OR of its card locations', badBitwise === 0);
  check('no decision root offered an empty combo set', emptyOffers === 0);
  check('expander offers == GameDriver offers for the same decision', mismatchOffers === 0);

  // 3) step(bitwise) children are well-formed: real decision node OR terminal.
  let badChild = 0, stepped = 0, terminals = 0;
  for (const d of decisions) {
    const exp = new PhaseExpander(M, d.phaseString, { seed: 7 });
    for (const o of exp.offered()) {
      const child = exp.step(o.bitwise);
      stepped++;
      if (typeof child.phaseString !== 'string' || ![-1, 0, 1].includes(child.endValue)) { badChild++; continue; }
      const cexp = new PhaseExpander(M, child.phaseString, { seed: 7 });
      if (child.endValue === 0) {
        // a non-terminal child MUST sit at a genuine decision (non-empty offers)
        if (cexp.isTerminal() || cexp.offered().length === 0) badChild++;
      } else {
        terminals++;
        if (!cexp.isTerminal()) badChild++;
      }
      cexp.dispose();
    }
    exp.dispose();
  }
  check(`step() children all well-formed (stepped=${stepped}, terminals=${terminals})`, badChild === 0 && stepped > 0);

  // 4) determinism: same root + same seed -> identical child for the same combo.
  let nondet = 0;
  for (const d of decisions) {
    const a = new PhaseExpander(M, d.phaseString, { seed: 123 });
    const b = new PhaseExpander(M, d.phaseString, { seed: 123 });
    for (const o of a.offered()) {
      if (a.step(o.bitwise).phaseString !== b.step(o.bitwise).phaseString) nondet++;
    }
    a.dispose(); b.dispose();
  }
  check('step() is deterministic under a fixed seed', nondet === 0);

  // 5) the expander alone drives a game to a terminal state (greedy first-combo walk).
  {
    let phase = decisions[0].phaseString;
    let ended = false, steps = 0;
    for (; steps < 2000; steps++) {
      const exp = new PhaseExpander(M, phase, { seed: 99 });
      if (exp.isTerminal()) { ended = true; exp.dispose(); break; }
      const offered = exp.offered();
      const child = exp.step(offered[0].bitwise);
      exp.dispose();
      phase = child.phaseString;
      if (child.endValue !== 0) { ended = true; break; }
    }
    check(`expander walks a game to a terminal state (steps=${steps})`, ended);
  }

  // 6) onOffer fires exactly once with the live combos + a usable phase.
  {
    const exp = new PhaseExpander(M, decisions[0].phaseString, { seed: 5 });
    let calls = 0, sawCombos = 0, phaseOk = false;
    const offered = exp.offered((phase, combos, descriptors) => {
      calls++;
      sawCombos = combos.size();
      phaseOk = phase.num_players >= 2 && typeof phase.to_string() === 'string';
      return { k: descriptors.length };
    });
    check('onOffer fired once with live combos matching offered()', calls === 1 && sawCombos === offered.length);
    check('onOffer received a usable live phase', phaseOk);
    check('onOffer return value stashed on exp.capture', exp.capture && exp.capture.k === offered.length);
    exp.dispose();
  }

  console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
  return failures;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exit((await runSmoke()) ? 1 : 0);
}
