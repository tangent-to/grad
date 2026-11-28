/**
 * Compare regenerated reference fixtures with the committed ones, ignoring
 * the `meta` block (it records library versions, so it always differs across
 * machines). Prints, per file, whether the cases are identical and the
 * largest relative difference between numbers, and fails if any number moved
 * by more than the tolerance the tests use.
 *
 *   node tests/references/compare_fixtures.mjs <committed-dir> <regenerated-dir>
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const [committedDir, regeneratedDir] = process.argv.slice(2);
const FILES = ['nd-cases.json', 'nd-torch.json', 'nd-r.json'];
const TOL = 1e-12;

let worst = 0;
let structural = [];

function walk(a, b, path) {
  if (typeof a === 'number' && typeof b === 'number') {
    worst = Math.max(worst, Math.abs(a - b) / (1 + Math.abs(b)));
    return;
  }
  if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) {
    a.forEach((x, i) => walk(x, b[i], `${path}[${i}]`));
    return;
  }
  if (a && b && typeof a === 'object' && typeof b === 'object' && !Array.isArray(a) && !Array.isArray(b)) {
    const ka = Object.keys(a).sort().join();
    if (ka === Object.keys(b).sort().join()) {
      for (const k of Object.keys(a)) walk(a[k], b[k], `${path}.${k}`);
      return;
    }
  }
  if (a !== b) structural.push(path);
}

let failed = false;
for (const name of FILES) {
  const { meta: m1, ...a } = JSON.parse(readFileSync(join(committedDir, name), 'utf8'));
  const { meta: m2, ...b } = JSON.parse(readFileSync(join(regeneratedDir, name), 'utf8'));
  worst = 0;
  structural = [];
  walk(a, b, name);
  const identical = JSON.stringify(a) === JSON.stringify(b);
  const versions = (m) => Object.entries(m).filter(([k]) => /torch|numpy|python|R$/.test(k))
    .map(([k, v]) => `${k}: ${v}`).join(', ');
  console.log(`${name}: ${identical ? 'identical' : `largest relative difference ${worst.toExponential(2)}`}`
    + `${structural.length ? `, ${structural.length} structural differences (first: ${structural[0]})` : ''}`);
  console.log(`  committed:   ${versions(m1)}\n  regenerated: ${versions(m2)}`);
  if (structural.length || worst > TOL) failed = true;
}
if (failed) {
  console.error(`regenerated references moved by more than ${TOL}`);
  process.exit(1);
}
