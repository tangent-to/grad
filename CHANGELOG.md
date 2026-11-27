# Changelog

Notable changes to `@tangent.to/grad`. This file starts at 0.3.0; for earlier
releases see the [git history](https://github.com/tangent-to/grad/commits/main)
and the [release tags](https://github.com/tangent-to/grad/releases).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Rank-3 tensors**, for a batch of sequences `[n, T, F]`. Nested
  `[n][T][F]` arrays are read and written like rows; plans are keyed by the
  full shape. `reshape` accepts targets up to rank 3; `slice` takes windows of
  rank-3 tensors, one time step being `slice(x, [0, t, 0], [n, 1, F])`; and
  `concat` joins rank-3 parts along any axis, axis 1 stacking per-step states
  back into a sequence. The elementwise ops accept rank 3 unchanged.
- **Column broadcasting.** The binary elementwise ops accept an `[n, d]`
  operand with an `[n, 1]` operand on either side, one weight per row: a
  padding mask applied to a recurrent state. The column's adjoint is the row
  sum. Row broadcasting of a `[d]` vector now also reaches the last axis of a
  rank-3 tensor, a bias added at every time step.
- **Axis reductions.** `sum`, `mean` and the new `max` take
  `{ axis, keepDims }`; with `keepDims` the reduced axis stays at length 1 and
  broadcasts back, which is what a stable softmax or log-sum-exp needs.
  Without options they reduce to a scalar as before, and serialize as before.
  `max` sends its adjoint to the first maximal entry and propagates NaN.
- **`gather(table, index)`** picks rows of a table by integer index, shaped as
  the index followed by a row, with a scatter-add adjoint: an embedding
  lookup, or a hierarchical model's `z[site]`. The index may be a constant or
  an input leaf, and is validated on every forward pass.
- `concat` joins matrices as well as vectors: along rows (`axis: 0`), or
  side by side along columns (`axis: 1`), which is how a network joins two
  branches for every row of a batch. The axis is serialized with the plan.
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
- A plan for a rank-3 argument no longer shares a cache key with a matrix of
  the same leading shape; before, `[2, 3, 4]` and `[2, 3]` both signed as
  `2x3` and the second replayed the first's graph.
- `slice` no longer allocates and accumulates a gradient buffer the size of
  its whole parent: its adjoint adds into the touched entries only. A
  recurrent network slicing T steps from one batch paid O(T²) for that in the
  reverse pass.
- `matmul`'s reverse pass reads and writes contiguously; about 15% faster on
  a 300 × 300 product, and on the recurrent shapes above.
- `compileFromJSON` defers op evaluation until the restored plan receives real
  parameters and inputs, so domain-sensitive ops such as `cholesky` and
  `solveGeneral` succeed on serialized positive-definite matrix objectives.
- The shared value and gradient cache compares matrix entries and tensor
  shapes by value, including after in-place parameter edits.
- The zero gradient reported for a matrix parameter the objective never reads
  was a flat array; it is now nested rows, the form the parameter arrived in.
