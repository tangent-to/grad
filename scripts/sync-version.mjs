// Keep deno.json's version in lockstep with package.json.
// Run automatically by `npm version` (see the "version" script in package.json
// and RELEASING.md), so a single `npm version <patch|minor|major>` bumps both
// manifests in one commit + tag. package.json is the single source of truth.
import { readFileSync, writeFileSync } from 'node:fs';

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
const deno = JSON.parse(readFileSync('deno.json', 'utf8'));

if (deno.version === pkg.version) {
  console.log(`deno.json already at ${pkg.version}`);
} else {
  const from = deno.version;
  deno.version = pkg.version;
  writeFileSync('deno.json', JSON.stringify(deno, null, 2) + '\n');
  console.log(`deno.json ${from} -> ${pkg.version}`);
}
