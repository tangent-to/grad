/**
 * Central finite differences, the reference every adjoint here is checked
 * against. Deliberately the crude method the package exists to replace: if the
 * tape agrees with it to ~1e-8 on smooth objectives, the adjoints are right.
 */

/**
 * Gradient of a scalar function of a flat parameter vector.
 * @param {(x: number[]) => number} f
 * @param {number[]} x
 * @param {number} [h=1e-6]
 * @returns {number[]}
 */
export function fdGrad(f, x, h = 1e-6) {
  const g = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const a = x.slice();
    const b = x.slice();
    a[i] += h;
    b[i] -= h;
    g[i] = (f(a) - f(b)) / (2 * h);
  }
  return g;
}

/** Flatten nested rows to a vector. */
export const flat = (A) => (Array.isArray(A[0]) ? A.flat() : A.slice());

/** Rebuild n×n nested rows from a flat vector. */
export const square = (v, n) =>
  Array.from({ length: n }, (_, i) => v.slice(i * n, i * n + n));

/**
 * A symmetric positive-definite matrix with a deterministic, well-conditioned
 * spectrum — squeezed enough to exercise the solves, not so much that finite
 * differences lose their own accuracy.
 */
export function spd(n, seed = 7) {
  let s = seed >>> 0;
  const u = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296) - 0.5;
  const B = Array.from({ length: n }, () => Array.from({ length: n }, u));
  const A = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let k = 0; k < n; k++) acc += B[i][k] * B[j][k];
      A[i][j] = acc;
    }
    A[i][i] += n; // keep it comfortably positive definite
  }
  return A;
}
