/**
 * The ops a structural equation model needs, which the SPD-only path could not
 * supply.
 *
 * `@tangent.to/sem` minimizes F_ML(θ) = log|Σ| + tr(S Σ⁻¹) − log|S| − p with
 * L-BFGS on finite-difference gradients, where
 *
 *     Σ(θ) = F (I − A)⁻¹ Ψ (I − A)⁻ᵀ Fᵀ, observed block only.
 *
 * (I − A) is a matrix of directed paths and is NOT symmetric, so `solvePSD`
 * cannot touch it; and only a sub-block of the full Σ is compared with the
 * data. Hence `solveGeneral`/`inv` and `slice`.
 */

import { describe, expect, it } from 'vitest';
import { inv as linaInv } from '@tangent.to/lina';
import {
  add, inv, logdetPSD, matmul, mul, slice, solveGeneral, solvePSD, sub, sum, trace,
  transpose, valueAndGrad,
} from '../src/index.js';
import { fdGrad, flat, square as reshapeSquare } from './_fd.js';

/** A deliberately NON-symmetric, well-conditioned matrix. */
const A = [
  [3.0, 0.5, -1.2],
  [0.2, 2.5, 0.7],
  [-0.4, 1.1, 4.0],
];
const B = [[1, 0.5], [-2, 1.5], [0.5, -1]];

/** Compare an adjoint in a full (unsymmetrized) matrix argument with FD. */
function agrees(build, M, digits = 6) {
  const n = M.length;
  const { gradient } = valueAndGrad(build)(M);
  const fd = fdGrad((v) => valueAndGrad(build)(reshapeSquare(v, n)).value, flat(M));
  flat(gradient).forEach((g, i) => expect(g).toBeCloseTo(fd[i], digits));
}

describe('solveGeneral', () => {
  it('solves a non-symmetric system', () => {
    const X = solveGeneral(A, B);
    const AX = matmul(A, X);
    AX.data.forEach((v, i) => expect(v).toBeCloseTo(B[Math.floor(i / 2)][i % 2], 9));
  });

  it('matches finite differences in the matrix', () => {
    agrees((M) => sum(mul(solveGeneral(M, B), solveGeneral(M, B))), A);
  });

  it('matches finite differences in the right-hand side', () => {
    const { gradient } = valueAndGrad((p) => sum(solveGeneral(A, p.b)))({ b: [1, -2, 0.5] });
    const f = (v) => valueAndGrad((p) => sum(solveGeneral(A, p.b)))({ b: v }).value;
    fdGrad(f, [1, -2, 0.5]).forEach((fi, i) => expect(gradient.b[i]).toBeCloseTo(fi, 7));
  });

  it('agrees with solvePSD on a symmetric positive-definite system', () => {
    const S = [[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 2]];
    const b = [1, 2, 3];
    const viaLU = solveGeneral(S, b);
    const viaChol = solvePSD(S, b);
    viaLU.data.forEach((v, i) => expect(v).toBeCloseTo(viaChol.data[i], 10));
  });
});

describe('inv', () => {
  it('reproduces lina inv', () => {
    const got = inv(A);
    const ref = linaInv(A);
    got.data.forEach((v, i) => expect(v).toBeCloseTo(ref[Math.floor(i / 3)][i % 3], 10));
  });

  it('matches finite differences', () => {
    agrees((M) => sum(mul(inv(M), inv(M))), A);
  });
});

describe('slice', () => {
  const M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];

  it('extracts a submatrix and routes the gradient to its source cells', () => {
    const { value, gradient } = valueAndGrad((X) => sum(mul(slice(X, [0, 0], [2, 2]), 10)))(M);
    expect(value).toBe(120); // 10*(1+2+4+5)
    expect(gradient).toEqual([[10, 10, 0], [10, 10, 0], [0, 0, 0]]);
  });

  it('extracts a subvector', () => {
    const { gradient } = valueAndGrad((v) => sum(mul(slice(v, [1], [2]), 3)))([1, 2, 3, 4]);
    expect(gradient).toEqual([0, 3, 3, 0]);
  });

  it('rejects an out-of-range window, naming the axis', () => {
    expect(() => slice(M, [1, 1], [3, 1])).toThrow(/exceeds axis 0/);
  });
});

describe('the SEM discrepancy function end to end', () => {
  // A miniature RAM model: 2 latent + 3 observed, Sigma over all 5, observed
  // block compared with the data. Exactly the shape sem/fit.js builds.
  const nAll = 5;
  const p = 3;
  const S_obs = [[2.0, 0.7, 0.4], [0.7, 1.8, 0.5], [0.4, 0.5, 1.5]];

  /** Sigma(theta) = (I - A(theta))^-1 Psi(theta) (I - A(theta))^-T, observed block. */
  const sigma = (t) => {
    // A: directed paths, non-symmetric, parameterized by two loadings.
    const rows = [];
    for (let i = 0; i < nAll; i++) rows.push(new Array(nAll).fill(0));
    const Aconst = rows.map((r) => r.slice());
    Aconst[0][3] = 1; // fixed loading
    Aconst[1][3] = 1;
    Aconst[2][4] = 1;
    // I - A, with the two free loadings added in
    const Amat = add(Aconst, scaledUnit(nAll, 1, 3, t.l1));
    const Amat2 = add(Amat, scaledUnit(nAll, 2, 4, t.l2));
    const IminusA = sub(identity(nAll), Amat2);
    const Psi = diagFrom(nAll, [t.e1, t.e2, t.e3, 1, 1]);
    const Minv = inv(IminusA);
    const full = matmul(matmul(Minv, Psi), transpose(Minv));
    return slice(full, [0, 0], [p, p]);
  };

  const fml = (t) => {
    const Sig = sigma(t);
    return add(logdetPSD(Sig), trace(solvePSD(Sig, S_obs)));
  };

  const at = { l1: 0.8, l2: 0.6, e1: 0.9, e2: 1.1, e3: 1.0 };

  it('differentiates through a non-symmetric inverse and a sub-block', () => {
    const { value, gradient } = valueAndGrad(fml)(at);
    expect(Number.isFinite(value)).toBe(true);

    const keys = Object.keys(at);
    const f = (v) => valueAndGrad(fml)(Object.fromEntries(keys.map((k, i) => [k, v[i]]))).value;
    const fd = fdGrad(f, keys.map((k) => at[k]));
    keys.forEach((k, i) => expect(gradient[k]).toBeCloseTo(fd[i], 5));
  });
});

/** Constant identity, as a plain nested matrix. */
function identity(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}
/** A matrix that is `v` at (r, c) and zero elsewhere, differentiable in v. */
function scaledUnit(n, r, c, v) {
  const mask = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === r && j === c ? 1 : 0)));
  return mul(v, mask);
}
/** Diagonal matrix from a mixed list of Vars and constants. */
function diagFrom(n, entries) {
  let M = mul(0, identity(n));
  for (let i = 0; i < n; i++) {
    const mask = Array.from({ length: n }, (_, r) =>
      Array.from({ length: n }, (_, c) => (r === i && c === i ? 1 : 0)));
    M = add(M, mul(entries[i], mask));
  }
  return M;
}
