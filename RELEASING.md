# Releasing

This package publishes to **npm** (`@tangent.to/grad`) and **JSR**
(`@tangent/grad`) from one GitHub Actions workflow, `.github/workflows/release.yml`,
triggered by a version tag. Both registries get a provenance attestation.

## One-time setup

- **npm trusted publisher (no token)** — on npmjs.com -> this package -> Settings
  -> Trusted Publisher -> GitHub Actions: org `tangent-to`, this repo, workflow
  file `release.yml`. The package must already exist; for a brand-new package do
  one manual `npm publish --access public` first, then add the trusted publisher.
- **JSR** needs no secret: the workflow authenticates over GitHub OIDC. The JSR
  package must exist and be linked to this GitHub repo (jsr.io → package →
  Settings → link the repository) once.

## Cutting a release

`package.json` is the single source of truth for the version; `npm version`
bumps it and the `version` script syncs `deno.json`, so both land in one commit
and tag.

```bash
npm version patch   # or: minor | major | 1.2.3
git push --follow-tags
```

Pushing the `v*` tag runs the Release workflow, which publishes to npm and JSR.
The npm job is idempotent — it skips a version already on the registry — so a
failed JSR half can be re-run without a version collision.

## Notes

- Version lives in exactly two files, kept in lockstep: `package.json` (source of
  truth) and `deno.json` (synced by `scripts/sync-version.mjs`). There is no
  `jsr.json`; `deno publish` reads `deno.json`.
- To dry-run JSR locally: `npm run jsr:publish:dry-run`.
- `npm run test:dist` runs the whole suite against the minified bundle that
  npm publishes, rather than the sources. Run it before tagging.
- The PyTorch and R reference fixtures are committed and checked by the
  ordinary test run. After changing an op, regenerate them with
  `npm run references` (needs `python3` with `numpy` and `torch`, and `Rscript`
  with `jsonlite`) and commit the result; the References workflow reruns them
  on pull requests and weekly.
- The JSR job disables Deno's 24-hour dependency cooldown (`deno cache
  --minimum-dependency-age=0`) so this package can be released the same day as an
  intra-suite dependency it was published alongside.
