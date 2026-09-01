/**
 * Vector-valued Jacobians.
 *
 * Written to feed `@tangent.to/ode`'s Rosenbrock integrator an exact ∂f/∂y in
 * place of its finite-difference one. Measured, it is the wrong tool for that:
 * identical step counts and up to 26x slower at n = 30 (see the table in
 * `jacobian`'s docs). The correctness tests below still matter — the op is
 * right, and useful where outputs are few — but nothing here claims a stiff
 * solver should use it.
 */

import { describe, expect, it } from 'vitest';
import { concat, jacobian, mul, slice, square, sub, sum as sumOf, valueAndGrad } from '../src/index.js';
import { fdGrad } from './_fd.js';

/** Van der Pol, the standard stiff test problem: y'' - mu(1-y²)y' + y = 0. */
const MU = 100;
const vdp = (y) => {
  const y0 = slice(y, [0], [1]);
  const y1 = slice(y, [1], [1]);
  return concat([
    y1,
    mul(MU, sub(mul(sub(1, square(y0)), y1), y0)),
  ]);
};

/** The Jacobian by hand: the reference every claim below is checked against. */
const vdpJacExact = ([a, b]) => [
  [0, 1],
  [MU * (-2 * a * b - 1), MU * (1 - a * a)],
];

describe('jacobian', () => {
  it('matches the hand-derived Van der Pol Jacobian exactly', () => {
    for (const y of [[2, 0], [-1, 0.5], [0.3, -2.1]]) {
      const J = jacobian(vdp)(y);
      const ref = vdpJacExact(y);
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) expect(J[i][j]).toBeCloseTo(ref[i][j], 9);
      }
    }
  });

  it('is more accurate than central finite differences', () => {
    // True, and largely beside the point for a stiff solver: its Newton
    // iteration converges to the same answer either way, which is why the
    // measured step counts are identical.
    const y = [2, 0.7];
    const ref = vdpJacExact(y);
    const J = jacobian(vdp)(y);

    const h = 1e-6;
    const fdJ = [[0, 0], [0, 0]];
    for (let j = 0; j < 2; j++) {
      const a = y.slice(); a[j] += h;
      const b = y.slice(); b[j] -= h;
      const fa = jacobianFreeEval(a);
      const fb = jacobianFreeEval(b);
      for (let i = 0; i < 2; i++) fdJ[i][j] = (fa[i] - fb[i]) / (2 * h);
    }

    let adErr = 0;
    let fdErr = 0;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        adErr = Math.max(adErr, Math.abs(J[i][j] - ref[i][j]));
        fdErr = Math.max(fdErr, Math.abs(fdJ[i][j] - ref[i][j]));
      }
    }
    expect(adErr).toBeLessThan(1e-9);
    expect(fdErr).toBeGreaterThan(adErr * 100);
  });

  it('reuses one forward pass across the output rows', () => {
    let calls = 0;
    const counted = (y) => { calls++; return vdp(y); };
    jacobian(counted)([1, 1]);
    expect(calls).toBe(1); // one graph build, two reverse sweeps
  });

  it('handles a non-square Jacobian', () => {
    // three outputs from two inputs
    const f = (v) => {
      const a = slice(v, [0], [1]);
      const b = slice(v, [1], [1]);
      return concat([mul(a, b), square(a), sub(b, a)]);
    };
    const J = jacobian(f)([3, 4]);
    expect(J).toHaveLength(3);
    expect(J[0]).toEqual([4, 3]);
    expect(J[1]).toEqual([6, 0]);
    expect(J[2]).toEqual([-1, 1]);
  });

  it('rejects a matrix-valued function, naming the limit', () => {
    expect(() => jacobian(() => mul([[1, 2], [3, 4]], 1))([1]))
      .toThrow(/must return a scalar or a vector/);
  });
});

describe('concat', () => {
  it('routes each part its own slice of the gradient', () => {
    const { gradient } = valueAndGrad((p) =>
      sumOf(slice(concat([p.a, p.b]), [1], [2])))({ a: [1, 2], b: [3, 4] });
    // picks a[1] and b[0]
    expect(gradient.a).toEqual([0, 1]);
    expect(gradient.b).toEqual([1, 0]);
  });

  it('matches finite differences through a concatenated expression', () => {
    const build = (v) => {
      const a = slice(v, [0], [2]);
      const b = slice(v, [2], [2]);
      return sumOf(square(concat([mul(a, 2), b])));
    };
    const x = [0.5, -1.2, 2.0, 0.3];
    const { gradient } = valueAndGrad(build)(x);
    fdGrad((w) => valueAndGrad(build)(w).value, x)
      .forEach((fi, i) => expect(gradient[i]).toBeCloseTo(fi, 7));
  });
});

/** Van der Pol on plain numbers, for the finite-difference comparison. */
function jacobianFreeEval([a, b]) {
  return [b, MU * ((1 - a * a) * b - a)];
}
