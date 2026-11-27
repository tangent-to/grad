# Changelog

Notable changes to `@tangent.to/grad`. This file starts at 0.3.0; for earlier
releases see the [git history](https://github.com/tangent-to/grad/commits/main)
and the [release tags](https://github.com/tangent-to/grad/releases).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Tensors of any rank.** Nested arrays of any depth are read and written
  like rows, and plans are keyed by the full shape. The elementwise ops,
  `reshape`, `slice`, `concat` (along any axis), the reductions and `gather`
  accept any rank; one time step of a batch of sequences `[n, T, F]` is
  `slice(x, [0, t, 0], [n, 1, F])`, and `concat` along axis 1 stacks
  per-step states back into a sequence.
- **numpy broadcasting.** The binary elementwise ops align shapes from the
  right and repeat length-1 or missing axes: a bias `[d]` on `[n, T, d]`, a
  per-row weight `[n, 1]` on `[n, d]`, a mask `[n, 1, T]` on `[n, T, T]`. The
  adjoint sums back over the repeated axes. Unlike numpy, the result must be
  one operand's shape: `[n]` against `[n, 1]` is refused instead of becoming
  an `[n, n]` table, the shape a residual meeting an observation by mistake
  takes.
- **`transpose(x, perm)`** permutes any number of axes; without `perm` it
  reverses them, which for a matrix is the transpose as before.
- **Batched `matmul`.** The product runs over the last two axes:
  `[..., m, k] · [k, n]` applies one matrix to every leading index (a weight
  matrix to every time step) as a single product, and
  `[..., m, k] · [..., k, n]` multiplies block by block.
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
- A compiled plan checks the shape of nested arrays as it writes them. The
  plan is chosen by the first entry at each level, and the write trusted it,
  so a ragged later row (`[[1, 2], [3]]` after `[[1, 2], [3, 4]]`) silently
  reused the previous call's values; it now throws, naming the parameter or
  input and the row. This dates from 0.3's matrices.
- `valueAndGradFns`'s cache copies parameters at every depth. It copied two
  levels, so editing a rank-3 parameter in place changed the cached copy too
  and returned the old value and gradient.
- A plan for an argument of rank 3 or more no longer shares a cache key
  with a matrix of the same leading shape; before, `[2, 3, 4]` and `[2, 3]`
  both signed as `2x3` and the second replayed the first's graph.

### Validation
- The n-dimensional ops are checked against PyTorch (values and autograd
  gradients) and base R (values), in float64, on 102 cases with asymmetric
  shapes; the fixtures are committed and `npm run references` regenerates
  them. A scheduled workflow reruns the references themselves.
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
