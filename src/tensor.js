/**
 * Tensor storage.
 *
 * A tensor is `{ data: Float64Array, shape: number[] }` in row-major order,
 * with rank 0 (scalar), 1 (vector), 2 (matrix) or 3. Most of the suite —
 * GP marginal likelihoods, GLM IRLS, SEM discrepancy functions, MCMC
 * potentials — lives in matrices and vectors, and the linear algebra stays
 * there. Rank 3 exists for one consumer: a batch of sequences, `[n, T, F]`,
 * which a recurrent network slices one time step at a time. Only the
 * structural ops (`reshape`, `slice`, `concat`, the reductions) and the
 * elementwise ops accept it; rank-N would cost broadcasting complexity in
 * every adjoint for no consumer.
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

/** The highest rank a tensor may have. */
export const MAX_RANK = 3;

/**
 * Coerce user input to a tensor: a number becomes a scalar, a flat array a
 * vector, rows a matrix, and an array of matrices a rank-3 tensor. An
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
  if (Array.isArray(x[0]) && x[0].length > 0 && (Array.isArray(x[0][0]) || x[0][0] instanceof Float64Array)) {
    return nested3(x, name);
  }
  if (Array.isArray(x[0]) || x[0] instanceof Float64Array) {
    const m = x.length;
    const n = x[0].length;
    if (n === 0) throw new Error(`${name} must have at least one column`);
    const data = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      const row = x[i];
      if (row.length !== n) {
        throw new Error(`${name} is not rectangular (row ${i} has length ${row.length}, expected ${n})`);
      }
      for (let j = 0; j < n; j++) data[i * n + j] = row[j];
    }
    return { data, shape: [m, n] };
  }
  throw new Error(`${name} must be a number, an array of numbers, or an array of rows`);
}

/** An array of matrices, `[n][T][F]`, flattened row-major. @private */
function nested3(x, name) {
  const n = x.length;
  const t = x[0].length;
  const f = x[0][0].length;
  if (f === 0) throw new Error(`${name} must have at least one column`);
  const data = new Float64Array(n * t * f);
  let o = 0;
  for (let i = 0; i < n; i++) {
    const mat = x[i];
    if (!Array.isArray(mat) || mat.length !== t) {
      throw new Error(`${name} is not rectangular (block ${i} has ${mat?.length ?? 0} rows, expected ${t})`);
    }
    for (let j = 0; j < t; j++) {
      const row = mat[j];
      if (!(Array.isArray(row) || row instanceof Float64Array) || row.length !== f) {
        throw new Error(
          `${name} is not rectangular (block ${i}, row ${j} has length ${row?.length ?? 0}, expected ${f})`,
        );
      }
      for (let k = 0; k < f; k++) data[o++] = row[k];
    }
  }
  return { data, shape: [n, t, f] };
}

/**
 * Convert a tensor back to the boundary currency: a number for rank 0, a flat
 * array for rank 1, nested rows for rank 2, an array of matrices for rank 3.
 *
 * @param {Tensor} t
 * @returns {number|number[]|number[][]}
 */
export function toNested(t) {
  if (t.shape.length === 0) return t.data[0];
  if (t.shape.length === 1) return Array.from(t.data);
  if (t.shape.length === 2) {
    const [m, n] = t.shape;
    const out = new Array(m);
    for (let i = 0; i < m; i++) out[i] = Array.from(t.data.subarray(i * n, i * n + n));
    return out;
  }
  const [b, m, n] = t.shape;
  const out = new Array(b);
  for (let k = 0; k < b; k++) {
    const mat = new Array(m);
    for (let i = 0; i < m; i++) {
      const o = (k * m + i) * n;
      mat[i] = Array.from(t.data.subarray(o, o + n));
    }
    out[k] = mat;
  }
  return out;
}
