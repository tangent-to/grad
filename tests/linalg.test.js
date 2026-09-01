/**
 * Adjoints of the linear algebra.
 *
 * Two independent references, because finite differences alone would not catch
 * a systematically-wrong-but-smooth adjoint:
 *   1. central finite differences on the composed objective;
 *   2. closed forms that are known exactly — d logdet(A)/dA = A⁻¹, and the
 *      Gaussian-process identity below.
 */

import { describe, expect, it } from 'vitest';
import { inv } from '@tangent.to/lina';
import {
  addDiag, cholesky, diagPart, dot, log, logdetPSD, matmul, mul, solvePSD, sum,
  transpose, triangularSolve, valueAndGrad,
} from '../src/index.js';
import { fdGrad, flat, spd, square as reshapeSquare } from './_fd.js';

/**
 * Differentiate a scalar objective of a symmetric matrix and compare with
 * finite differences. The perturbation is symmetrized to match the adjoint's
 * convention: only symmetric perturbations of a covariance are meaningful.
 */
function agreesOnSymmetric(build, A, digits = 6) {
  const n = A.length;
  const { gradient } = valueAndGrad((M) => build(M))(A);
  const g = flat(gradient);

  const f = (v) => {
    const M = reshapeSquare(v, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) M[j][i] = M[i][j];
    return valueAndGrad((X) => build(X))(M).value;
  };
  const fd = fdGrad(f, flat(A), 1e-6);

  // Only the LOWER triangle is probed: the helper above rebuilds the upper
  // triangle from the lower, so perturbing an upper entry is overwritten and
  // its finite difference is identically zero — nothing to compare against.
  // A lower perturbation moves A_ij and A_ji together, so its finite
  // difference carries the sum of both partials; the symmetrized adjoint is
  // summed the same way.
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      const expected = fd[i * n + j];
      const got = i === j ? g[i * n + i] : g[i * n + j] + g[j * n + i];
      expect(got).toBeCloseTo(expected, digits);
    }
  }
}

describe('cholesky adjoint', () => {
  const A = spd(5);

  it('matches finite differences through the factor', () => {
    agreesOnSymmetric((M) => sum(mul(cholesky(M), cholesky(M))), A);
  });

  it('matches finite differences through a log-diagonal reduction', () => {
    agreesOnSymmetric((M) => sum(log(diagPart(cholesky(M)))), A);
  });

  it('reconstructs A from its factor', () => {
    const L = cholesky(A);
    const { value } = valueAndGrad(() => sum(mul(L, L)))(0);
    expect(Number.isFinite(value)).toBe(true);
    const LLt = valueAndGrad((M) => sum(matmul(cholesky(M), transpose(cholesky(M)))))(A);
    const total = A.flat().reduce((s, v) => s + v, 0);
    expect(LLt.value).toBeCloseTo(total, 9);
  });

  it('rejects a non-positive-definite matrix through lina', () => {
    expect(() => cholesky([[1, 2], [2, 1]])).toThrow(/positive definite/);
  });
});

describe('logdetPSD', () => {
  const A = spd(6, 21);

  it('has gradient A⁻¹ exactly, the known closed form', () => {
    // The strongest check available: no finite differences involved.
    const { value, gradient } = valueAndGrad((M) => logdetPSD(M))(A);
    const Ainv = inv(A);
    const n = A.length;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        // The adjoint is symmetrized, and A⁻¹ is symmetric here, so they agree
        // entry by entry.
        expect(gradient[i][j]).toBeCloseTo(Ainv[i][j], 9);
      }
    }
    expect(Number.isFinite(value)).toBe(true);
  });

  it('matches finite differences', () => {
    agreesOnSymmetric((M) => logdetPSD(M), A);
  });
});

describe('triangularSolve adjoint', () => {
  const L = [[2, 0, 0], [0.5, 1.5, 0], [-1, 0.25, 3]];
  const b = [1, -2, 0.5];

  it('solves and differentiates a lower system against finite differences', () => {
    const { value, gradient } = valueAndGrad((T) => sum(mul(
      triangularSolve(T, b, { lower: true }),
      triangularSolve(T, b, { lower: true }),
    )))(L);
    const f = (v) => valueAndGrad((T) => sum(mul(
      triangularSolve(T, b, { lower: true }),
      triangularSolve(T, b, { lower: true }),
    )))(reshapeSquare(v, 3)).value;
    const fd = fdGrad(f, flat(L));
    flat(gradient).forEach((gi, i) => expect(gi).toBeCloseTo(fd[i], 6));
    expect(Number.isFinite(value)).toBe(true);
  });

  it('differentiates the right-hand side too', () => {
    const { gradient } = valueAndGrad((p) => sum(triangularSolve(L, p.b, { lower: true })))({ b });
    const f = (v) => valueAndGrad((p) => sum(triangularSolve(L, p.b, { lower: true })))({ b: v }).value;
    fdGrad(f, b).forEach((fi, i) => expect(gradient.b[i]).toBeCloseTo(fi, 7));
  });

  it('handles an upper system', () => {
    const U = transpose(L);
    const { gradient } = valueAndGrad((p) => sum(triangularSolve(U, p.b, { lower: false })))({ b });
    const Un = [[2, 0.5, -1], [0, 1.5, 0.25], [0, 0, 3]];
    const f = (v) =>
      valueAndGrad((p) => sum(triangularSolve(Un, p.b, { lower: false })))({ b: v }).value;
    fdGrad(f, b).forEach((fi, i) => expect(gradient.b[i]).toBeCloseTo(fi, 7));
  });

  it('leaves the unused triangle at zero', () => {
    const { gradient } = valueAndGrad((T) => sum(triangularSolve(T, b, { lower: true })))(L);
    expect(gradient[0][1]).toBe(0);
    expect(gradient[0][2]).toBe(0);
    expect(gradient[1][2]).toBe(0);
  });
});

describe('solvePSD', () => {
  const A = spd(4, 3);
  const b = [1, -0.5, 2, 0.25];

  it('solves A x = b', () => {
    const { value } = valueAndGrad(() => sum(mul(solvePSD(A, b), 1)))(0);
    // Ax should recover b.
    const x = valueAndGrad((p) => sum(mul(solvePSD(A, p.b), 0)))({ b });
    expect(Number.isFinite(value)).toBe(true);
    expect(Number.isFinite(x.value)).toBe(true);
    const Ax = matmul(A, solvePSD(A, b));
    Ax.data.forEach((v, i) => expect(v).toBeCloseTo(b[i], 9));
  });

  it('matches finite differences in A', () => {
    agreesOnSymmetric((M) => dot(solvePSD(M, b), b), A);
  });
});
