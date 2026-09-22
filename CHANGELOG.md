# Changelog

Notable changes to `@tangent.to/grad`. This file starts at 0.3.0; for earlier
releases see the [git history](https://github.com/tangent-to/grad/commits/main)
and the [release tags](https://github.com/tangent-to/grad/releases).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Inputs.** An objective may take a second argument, a `{name: value}` map
  of leaves that are not differentiated and change between calls: a
  mini-batch, a dropout mask, a per-fit coefficient. `valueAndGrad(f)(x,
  inputs)` and `compile(f)(x, inputs)` evaluate `f(params, inputs)`; a
  compiled plan writes new inputs into its leaves the way it writes new
  parameters, instead of freezing them as constants.
- **Forward-only replay.** `compile(f).value(x, inputs)` and
  `valueAndGrad(f).value(x, inputs)` evaluate the objective without the
  backward sweep. The root need not be a scalar there: a network's
  predictions replay through the same plan as its loss.
- **A plan per shape.** `compile` keeps a handful of plans keyed by the
  shapes of parameters and inputs, so a loop that alternates a full batch
  with a partial last batch builds each plan once.
- **Row broadcasting.** The binary elementwise ops (`add`, `sub`, `mul`,
  `div`, `maximum`, `minimum`) accept a `[n, d]` operand with a `[d]` operand
  on either side: a bias added to every row of `X·W`. The vector's adjoint is
  the column sum.
- **Tensors as leaves.** A parameter given as `{ data: Float64Array, shape }`
  gets its gradient back in that form, a copy, so a training loop keeps its
  weights and optimizer state as typed arrays. `isTensor` is exported.

### Changed
- Serialized plans are written as `version: 2`, which adds the `input` node
  kind; `compileFromJSON` reads versions 1 and 2. A rebuilt plan's shape
  error now says "arguments shaped" rather than "parameters shaped", since
  inputs are part of the signature.
- `compile(f).toJSON()` writes the plan of the shapes most recently
  evaluated, there being several.

### Fixed
- The zero gradient reported for a matrix parameter the objective never reads
  was a flat array; it is now nested rows, the form the parameter arrived in.
