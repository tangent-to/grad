/**
 * Tensor storage.
 *
 * A tensor is `{ data: Float64Array, shape: number[] }` in row-major
 * (C-order) storage, of any rank. Most of the suite — GP marginal
 * likelihoods, GLM IRLS, SEM discrepancy functions, MCMC potentials — lives
 * in matrices and vectors, and the linear algebra stays there. Higher ranks
 * are for networks: a batch of sequences `[n, T, F]`, a batch of images
 * `[n, H, W, C]`. The elementwise ops, broadcasting, the structural ops
 * (`reshape`, `slice`, `concat`, `transpose`, `gather`), the reductions and
 * `matmul` accept any rank; `cholesky` and the solves are matrix-only.
 *
 * Public entry points take and return the suite's boundary currency: plain
 * numbers, nested arrays, and typed arrays. This module is the only place that
 * converts between that and flat storage.
 */

/** @typedef {{ data: Float64Array, shape: number[] }} Tensor */

/**
 * Number of elements a shape holds.
 * @param {number[]} shape
 * @returns {number}
 */
export function sizeOf(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/**
 * Are two shapes identical?
 * @param {number[]} a
 * @param {number[]} b
 * @returns {boolean}
 */
export function sameShape(a, b) {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}

/**
 * Render a shape for an error message: `[]` reads as "scalar".
 * @param {number[]} shape
 * @returns {string}
 */
export function shapeStr(shape) {
  return shape.length === 0 ? 'scalar' : `[${shape.join('×')}]`;
}

/**
 * Build a tensor from raw storage, without copying.
 * @param {Float64Array|number[]} data
 * @param {number[]} shape
 * @returns {Tensor}
 */
export function tensor(data, shape) {
  const d = data instanceof Float64Array ? data : Float64Array.from(data);
  if (d.length !== sizeOf(shape)) {
    throw new Error(`data length ${d.length} does not match shape ${shapeStr(shape)}`);
  }
  return { data: d, shape: shape.slice() };
}

/**
 * Zero tensor of a given shape.
 * @param {number[]} shape
 * @returns {Tensor}
 */
export function zeros(shape) {
  return { data: new Float64Array(sizeOf(shape)), shape: shape.slice() };
}

/**
 * Identity matrix as a tensor.
 * @param {number} n
 * @returns {Tensor}
 */
export function eye(n) {
  const t = zeros([n, n]);
  for (let i = 0; i < n; i++) t.data[i * n + i] = 1;
  return t;
}

/**
 * Is this already a tensor, `{ data: Float64Array, shape }`? The form a caller
 * may use for a parameter when it wants its gradient back as the same thing,
 * with no conversion to nested arrays on either side.
 * @param {*} x
 * @returns {boolean}
 */
export function isTensor(x) {
  return x !== null && typeof x === 'object' && x.data instanceof Float64Array && Array.isArray(x.shape);
}

/**
 * Coerce user input to a tensor: a number becomes a scalar, a flat array a
 * vector, rows a matrix, and arrays nested deeper a tensor of that rank. An
 * existing tensor passes through untouched.
 *
 * @param {number|number[]|number[][]|Float64Array|Tensor} x
 * @param {string} [name='value'] - argument name for error messages
 * @returns {Tensor}
 */
export function asTensor(x, name = 'value') {
  if (isTensor(x)) return x;
  if (typeof x === 'number') {
    // NaN and ±Infinity pass through deliberately. A sampler legitimately
    // probes outside a parameter's support — NUTS steps past σ = 0 on its way
    // to rejecting the trajectory — and needs the density to answer -Infinity
    // there, not to throw. Rejecting non-finite input would also be
    // inconsistent: the array branches below never checked.
    return { data: Float64Array.of(x), shape: [] };
  }
  if (x instanceof Float64Array) {
    return { data: x, shape: [x.length] };
  }
  if (!Array.isArray(x) || x.length === 0) {
    throw new Error(`${name} must be a number, a non-empty array, or a tensor`);
  }
  if (typeof x[0] === 'number') {
    return { data: Float64Array.from(x), shape: [x.length] };
  }
  const shape = nestedShape(x);
  if (shape.length === 2) {
    const [m, n] = shape;
    if (n === 0) throw new Error(`${name} must have at least one column`);
    const data = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      const row = x[i];
      if (!isRow(row) || row.length !== n) {
        throw new Error(`${name} is not rectangular (row ${i} has length ${row?.length ?? 0}, expected ${n})`);
      }
      for (let j = 0; j < n; j++) data[i * n + j] = row[j];
    }
    return { data, shape };
  }
  if (shape.length > 2) {
    if (shape.includes(0)) throw new Error(`${name} has an empty axis: ${shapeStr(shape)}`);
    const data = new Float64Array(sizeOf(shape));
    fillNested(data, x, shape, name);
    return { data, shape };
  }
  throw new Error(`${name} must be a number, an array of numbers, or nested arrays of numbers`);
}

const isRow = (v) => Array.isArray(v) || v instanceof Float64Array;

/**
 * The shape nested arrays claim, read down the first element of each level.
 * Not validated: {@link asTensor} checks every level against it.
 * @param {Array} x
 * @returns {number[]}
 */
export function nestedShape(x) {
  const shape = [];
  let v = x;
  while (isRow(v)) {
    shape.push(v.length);
    if (v.length === 0) break;
    v = v[0];
  }
  return shape;
}

/**
 * Write nested arrays into flat row-major storage, checking that every level
 * has the length `shape` says. @private
 */
function fillNested(data, x, shape, name) {
  let o = 0;
  const walk = (v, depth, path) => {
    const want = shape[depth];
    if (!isRow(v) || v.length !== want) {
      throw new Error(
        `${name} is not rectangular (${path || 'the top level'} has length ${isRow(v) ? v.length : 0}, expected ${want})`,
      );
    }
    if (depth === shape.length - 1) {
      for (let k = 0; k < want; k++) {
        if (typeof v[k] !== 'number') throw new Error(`${name}${path}[${k}] is not a number`);
        data[o++] = v[k];
      }
      return;
    }
    for (let k = 0; k < want; k++) walk(v[k], depth + 1, `${path}[${k}]`);
  };
  walk(x, 0, '');
}

/**
 * Write nested arrays of a known shape into existing flat storage, without
 * checks: the caller has already matched the shape. What a compiled plan
 * uses to write a new value into a leaf.
 * @param {Float64Array} data
 * @param {Array} x
 */
export function writeNested(data, x) {
  let o = 0;
  const walk = (v) => {
    if (typeof v[0] === 'number') {
      for (let k = 0; k < v.length; k++) data[o++] = v[k];
      return;
    }
    for (let k = 0; k < v.length; k++) walk(v[k]);
  };
  walk(x);
}

/**
 * Convert a tensor back to the boundary currency: a number for rank 0, a flat
 * array for rank 1, nested rows for rank 2, and arrays nested one level per
 * axis beyond that.
 *
 * @param {Tensor} t
 * @returns {number|number[]|number[][]}
 */
export function toNested(t) {
  const { shape, data } = t;
  if (shape.length === 0) return data[0];
  if (shape.length === 1) return Array.from(data);
  let o = 0;
  const build = (depth) => {
    const n = shape[depth];
    if (depth === shape.length - 1) {
      const row = Array.from(data.subarray(o, o + n));
      o += n;
      return row;
    }
    const out = new Array(n);
    for (let k = 0; k < n; k++) out[k] = build(depth + 1);
    return out;
  };
  return build(0);
}
