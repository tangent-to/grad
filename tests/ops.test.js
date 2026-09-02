/**
 * Elementwise ops, reductions and matrix products, each checked against
 * central finite differences.
 */

import { describe, expect, it } from 'vitest';
import {
  add, addDiag, diagPart, div, dot, exp, log, matmul, mean, mul, neg, pow,
  lgamma, maximum, minimum, relu, reshape, sigmoid, sqrt, square, sub, sum, tanh, trace, transpose,
  valueAndGrad, variable,
} from '../src/index.js';
import { fdGrad } from './_fd.js';

/** Check grad(f) against finite differences at x. */
function agrees(build, x, digits = 7) {
  const { value, gradient } = valueAndGrad(build)(x);
  const flatX = Array.isArray(x) ? x : [x];
  const scalarIn = !Array.isArray(x);
  const fd = fdGrad((v) => valueAndGrad(build)(scalarIn ? v[0] : v).value, flatX);
  const g = Array.isArray(gradient) ? gradient : [gradient];
  g.forEach((gi, i) => expect(gi).toBeCloseTo(fd[i], digits));
  return value;
}

describe('elementwise ops', () => {
  const x = [0.7, -1.3, 2.1, 0.4];

  it.each([
    ['add + mul', (v) => sum(mul(add(v, 2), v))],
    ['sub + div', (v) => sum(div(sub(v, 0.25), add(square(v), 3)))],
    ['neg', (v) => sum(neg(v))],
    ['exp', (v) => sum(exp(v))],
    ['log of a positive expression', (v) => sum(log(add(square(v), 1)))],
    ['sqrt', (v) => sum(sqrt(add(square(v), 0.5)))],
    ['square', (v) => sum(square(v))],
    ['tanh', (v) => sum(tanh(v))],
    ['sigmoid', (v) => sum(sigmoid(v))],
    ['pow', (v) => sum(pow(add(square(v), 1), 1.5))],
    ['mean', (v) => mean(exp(v))],
    ['deep chain', (v) => sum(tanh(mul(exp(div(v, 3)), log(add(square(v), 2)))))],
  ])('%s matches finite differences', (_name, build) => {
    agrees(build, x);
  });

  it('broadcasts a scalar against a vector, both ways', () => {
    // The scalar's gradient must be the SUM over the elements it was spread to.
    const { gradient } = valueAndGrad((p) => sum(mul(p.k, [1, 2, 3])))({ k: 2 });
    expect(gradient.k).toBeCloseTo(6, 12);

    const g2 = valueAndGrad((p) => sum(div([2, 4, 6], p.k)))({ k: 2 });
    expect(g2.gradient.k).toBeCloseTo(-3, 10); // d/dk Σ(cᵢ/k) = -Σcᵢ/k² = -12/4
  });

  it('rejects a non-constant exponent with a usable message', () => {
    expect(() => pow(variable([1, 2]), variable(2))).toThrow(/exponent must be a finite number/);
  });
});

describe('reductions and matrix ops', () => {
  it('matmul (matrix × matrix) matches finite differences', () => {
    const B = [[0.5, -1], [2, 0.25], [1, 1.5]];
    const build = (v) => sum(matmul(reshape(v, [2, 3]), B));
    agrees(build, [1, 2, 3, 4, 5, 6]);
  });

  it('matmul (matrix × vector) matches finite differences', () => {
    const build = (v) => sum(square(matmul(reshape(v, [2, 3]), [1, -2, 0.5])));
    agrees(build, [1, 2, 3, 4, 5, 6]);
  });

  it('propagates into BOTH matmul operands', () => {
    // A one-sided adjoint still passes a test that only varies one operand.
    const { gradient } = valueAndGrad((p) => sum(matmul(p.A, p.B)))({
      A: [[1, 2], [3, 4]],
      B: [[5, 6], [7, 8]],
    });
    // d/dA Σ(AB) = 1·Bᵀ summed over rows; d/dB = Aᵀ·1
    expect(gradient.A).toEqual([[11, 15], [11, 15]]);
    expect(gradient.B).toEqual([[4, 4], [6, 6]]);
  });

  it('dot, transpose, diagPart and trace match finite differences', () => {
    agrees((v) => dot(v, [1, -2, 0.5, 3]), [0.7, -1.3, 2.1, 0.4]);
    agrees((v) => sum(square(transpose(reshape(v, [2, 3])))), [1, 2, 3, 4, 5, 6]);
    agrees((v) => sum(exp(diagPart(reshape(v, [3, 3])))), [1, 2, 3, 4, 5, 6, 7, 8, 9].map((z) => z / 10));
    agrees((v) => trace(reshape(v, [3, 3])), [1, 2, 3, 4, 5, 6, 7, 8, 9].map((z) => z / 10));
  });

  it('addDiag differentiates in the matrix and in the scalar', () => {
    const A = [[2, 0.3], [0.3, 1.5]];
    const { gradient } = valueAndGrad((p) => sum(square(addDiag(A, p.alpha))))({ alpha: 0.1 });
    // d/dα Σ(A+αI)² = 2·Σ_diag (A_ii + α) = 2·(2.1 + 1.6)
    expect(gradient.alpha).toBeCloseTo(2 * (2.1 + 1.6), 10);
  });

  it('addDiag takes a per-row noise vector', () => {
    const A = [[2, 0.3], [0.3, 1.5]];
    const { gradient } = valueAndGrad((p) => sum(square(addDiag(A, p.a))))({ a: [0.1, 0.2] });
    expect(gradient.a[0]).toBeCloseTo(2 * 2.1, 10);
    expect(gradient.a[1]).toBeCloseTo(2 * 1.7, 10);
  });
});

describe('clamps: maximum, minimum, relu', () => {
  // Deliberately NOT in the elementwise it.each table above. Those cases run
  // through fdGrad, which straddles each point by +/-1e-6; a test point at or
  // near a kink would report a mismatch that is the probe's fault, not the
  // adjoint's. Finite differences are used here only well clear of the tie.

  describe('away from the kink', () => {
    it.each([
      ['maximum against a scalar', (v) => sum(maximum(v, 0))],
      ['minimum against a scalar', (v) => sum(minimum(v, 0))],
      ['relu', (v) => sum(relu(v))],
      ['relu composed', (v) => sum(square(relu(sub(v, 0.25))))],
      ['maximum of two expressions', (v) => sum(maximum(square(v), mul(v, 2)))],
    ])('%s matches finite differences', (_name, build) => {
      // Every element sits on one branch or the other, none within 1e-6 of a
      // tie, and both branches are active within the same vector.
      agrees(build, [-1.7, 0.9, -0.4, 2.3]);
    });

    it('routes the adjoint to whichever operand is active, per element', () => {
      const { gradient } = valueAndGrad((p) => sum(maximum(p.a, p.b)))({
        a: [3, -1, 5],
        b: [1, 2, -4],
      });
      expect(gradient.a).toEqual([1, 0, 1]);
      expect(gradient.b).toEqual([0, 1, 0]);
    });
  });

  describe('at the kink', () => {
    // The subgradient here is a documented convention, not something finite
    // differences could resolve: a central difference straddling the tie
    // reports the average of the two branches, which is not what any of these
    // ops promise. Asserted directly instead.

    it("relu'(0) is 0, matching JAX and PyTorch", () => {
      expect(valueAndGrad((p) => relu(p.x))({ x: 0 }).gradient.x).toBe(0);
      expect(valueAndGrad((p) => relu(p.x))({ x: 0 }).value).toBe(0);
    });

    it('maximum sends the whole adjoint left on a tie', () => {
      const { value, gradient } = valueAndGrad((p) => maximum(p.a, p.b))({ a: 2, b: 2 });
      expect(value).toBe(2);
      expect(gradient).toEqual({ a: 1, b: 0 });
    });

    it('minimum sends the whole adjoint left on a tie', () => {
      const { value, gradient } = valueAndGrad((p) => minimum(p.a, p.b))({ a: 2, b: 2 });
      expect(value).toBe(2);
      expect(gradient).toEqual({ a: 1, b: 0 });
    });

    it('is not a half-and-half split', () => {
      // Guards the convention against a future "fairer" rewrite: 0.5/0.5 is
      // defensible in the abstract and is not what is documented.
      const { gradient } = valueAndGrad((p) => maximum(p.a, p.b))({ a: 1, b: 1 });
      expect(gradient.a).not.toBeCloseTo(0.5, 6);
    });

    it('ties inside a vector all fall the same way', () => {
      const { gradient } = valueAndGrad((p) => sum(maximum(p.v, 0)))({ v: [0, 0, 0] });
      expect(gradient.v).toEqual([1, 1, 1]);
    });
  });

  describe('broadcasting', () => {
    it('folds a scalar operand back to a scalar gradient', () => {
      // The scalar was spread across four elements, so its adjoint is the SUM
      // of the four contributions, not one of them.
      const { gradient } = valueAndGrad((p) => sum(maximum([-2, 3, -1, 5], p.k)))({ k: 0 });
      expect(gradient.k).toBe(2); // two elements below the threshold
    });

    it('folds a scalar operand on the left too', () => {
      const { gradient } = valueAndGrad((p) => sum(minimum(p.k, [-2, 3, -1, 5])))({ k: 0 });
      expect(gradient.k).toBe(2); // k is the smaller one twice
    });

    it('keeps a matrix operand at its own shape', () => {
      const { gradient } = valueAndGrad((p) => sum(relu(p.m)))({
        m: [[-1, 2], [3, -4]],
      });
      expect(gradient.m).toEqual([[0, 1], [1, 0]]);
    });

    it('matches finite differences with a scalar threshold as the variable', () => {
      const build = (p) => sum(square(maximum([-2.3, 0.7, 1.9, -0.6], p.k)));
      const { gradient } = valueAndGrad(build)({ k: 0.4 });
      const fd = fdGrad((v) => valueAndGrad(build)({ k: v[0] }).value, [0.4]);
      expect(gradient.k).toBeCloseTo(fd[0], 6);
    });
  });

  describe('the case these ops exist for', () => {
    // A quadratic-plateau dose response: g*(1 - relu(1 - x/n)^2). The join sits
    // at the PARAMETER n, so which observations fall below it moves as the
    // sampler does, and no precomputed mask can stand in for the clamp. This is
    // also where a sign or chain-rule slip would hide: the per-op tests above
    // would all still pass.
    const X = [0.2, 0.9, 1.6, 2.4, 3.1, 4.2];

    const response = (p) =>
      mul(p.g, sub(1, square(relu(sub(1, div(X, p.n))))));

    it('differentiates through a parameter-valued join', () => {
      // n = 2.0 puts three of the six observations below the join and three
      // above, none closer than 0.4 to it.
      const at = { g: 1.4, n: 2.0 };
      const build = (p) => sum(response(p));
      const { gradient } = valueAndGrad(build)(at);

      const keys = ['g', 'n'];
      const f = (v) => valueAndGrad(build)({ g: v[0], n: v[1] }).value;
      const fd = fdGrad(f, keys.map((k) => at[k]));
      keys.forEach((k, i) => expect(gradient[k]).toBeCloseTo(fd[i], 6));
    });

    it('gives the plateau a zero gradient in the join', () => {
      // With every observation above the join the clamp is saturated, so the
      // response no longer depends on n at all.
      const { gradient } = valueAndGrad((p) => sum(response(p)))({ g: 1.4, n: 0.1 });
      expect(gradient.n).toBe(0);
      expect(gradient.g).toBeCloseTo(X.length, 10);
    });

    it('still matches finite differences when the join moves', () => {
      const build = (p) => sum(square(response(p)));
      for (const n of [0.7, 1.35, 2.85, 3.7]) {
        const { gradient } = valueAndGrad(build)({ g: 1.4, n });
        const fd = fdGrad((v) => valueAndGrad(build)({ g: v[0], n: v[1] }).value, [1.4, n]);
        expect(gradient.g).toBeCloseTo(fd[0], 5);
        expect(gradient.n).toBeCloseTo(fd[1], 5);
      }
    });
  });
});

describe('valueAndGrad', () => {
  it('returns the gradient in the shape the input arrived in', () => {
    const r = valueAndGrad((p) => add(square(p.mu), square(p.sigma)))({ mu: 3, sigma: 4 });
    expect(r.value).toBe(25);
    expect(r.gradient).toEqual({ mu: 6, sigma: 8 });
  });

  it('gives an untouched parameter a zero gradient rather than omitting it', () => {
    // A sampler indexes every parameter by name; a missing key would crash it.
    const r = valueAndGrad((p) => square(p.used))({ used: 2, unused: [1, 2] });
    expect(r.gradient.used).toBe(4);
    expect(r.gradient.unused).toEqual([0, 0]);
  });

  it('refuses a non-scalar objective, naming the fix', () => {
    expect(() => valueAndGrad((v) => square(v))([1, 2])).toThrow(/must return a scalar/);
  });

  it('refuses an objective that bypassed the ops', () => {
    expect(() => valueAndGrad(() => 42)([1, 2])).toThrow(/must return a Var/);
  });
});

describe('variadic add and mul', () => {
  // JavaScript cannot overload `+`, so a long sum has to be a call. Making the
  // call take every term at once is the part that IS available, and it has to
  // be exactly the nested form it replaces: same value, same gradient, same
  // graph.
  const P = { a: 1.3, b: -0.7, c: 2.9, d: 0.45 };

  it('add(a, b, c, d) equals the left-nested form', () => {
    const flat = valueAndGrad((p) => add(p.a, p.b, p.c, p.d))(P);
    const nested = valueAndGrad((p) => add(add(add(p.a, p.b), p.c), p.d))(P);
    expect(flat.value).toBe(nested.value);
    expect(flat.gradient).toEqual(nested.gradient);
  });

  it('mul(a, b, c, d) equals the left-nested form', () => {
    const flat = valueAndGrad((p) => mul(p.a, p.b, p.c, p.d))(P);
    const nested = valueAndGrad((p) => mul(mul(mul(p.a, p.b), p.c), p.d))(P);
    expect(flat.value).toBeCloseTo(nested.value, 14);
    for (const k of Object.keys(P)) {
      expect(flat.gradient[k]).toBeCloseTo(nested.gradient[k], 12);
    }
  });

  it('still broadcasts a scalar across the terms', () => {
    const { value, gradient } = valueAndGrad(
      (p) => sum(add(p.k, [1, 2, 3], mul(p.k, [10, 10, 10]))),
    )({ k: 2 });
    // (2+1+20) + (2+2+20) + (2+3+20) = 72; d/dk = 3·(1 + 10) = 33.
    expect(value).toBe(72);
    expect(gradient.k).toBe(33);
  });

  it('mixes shapes the way the nested form did', () => {
    // A vector, a scalar Var and a raw array in one call: each pair broadcasts
    // as it folds, which is the property that makes the flat form a drop-in.
    const { value, gradient } = valueAndGrad(
      (p) => sum(add(p.v, p.s, [0.5, 0.5, 0.5])),
    )({ v: [1, 2, 3], s: 4 });
    expect(value).toBe(1 + 2 + 3 + 12 + 1.5);
    expect(gradient.v).toEqual([1, 1, 1]);
    expect(gradient.s).toBe(3);
  });

  it('rejects fewer than two operands, rather than quietly passing one through', () => {
    expect(() => add(1)).toThrow(/needs at least two operands/);
    expect(() => mul()).toThrow(/needs at least two operands/);
  });

  it('takes the two-operand form unchanged', () => {
    expect(add(2, 3).data[0]).toBe(5);
    expect(mul(2, 3).data[0]).toBe(6);
  });

  it('a variadic sum over MIXED shapes keeps every term', () => {
    // The scalar cases above would still pass if the fold silently stopped
    // after two operands, as long as it stopped consistently. This one would
    // not: dropping the tail leaves a vector short of two whole terms.
    const v = [1, 2, 3];
    expect(Array.from(add(0.5, v, v, v).data)).toEqual([3.5, 6.5, 9.5]);
    expect(Array.from(mul(2, v, v).data)).toEqual([2, 8, 18]);
  });

  it('a strictly binary op refuses a third operand instead of dropping it', () => {
    // Before add and mul were variadic, every binary op accepted extra
    // arguments and ignored them: sub(a, b, c) returned a - b. A wrong number
    // with no error is worse than a failure, and the habit of reaching for a
    // third operand is exactly what the variadic pair encourages.
    expect(() => sub(1, 2, 3)).toThrow(/takes exactly two operands, got 3/);
    expect(() => div(1, 2, 3)).toThrow(/takes exactly two operands/);
    expect(() => exp(1, 2)).toThrow(/takes exactly one operand, got 2/);
  });
});

describe('lgamma', () => {
  it('matches proba\'s lgamma, which uses the same Lanczos approximation', async () => {
    const { special } = await import('@tangent.to/proba').catch(() => ({ special: null }));
    if (!special) return; // proba is not a dependency of grad; the check runs where it is installed
    for (const x of [0.3, 0.5, 1, 1.5, 2.5, 7, 12.25, 40, 150.5]) {
      expect(lgamma(x).data[0]).toBeCloseTo(special.lgamma(x), 13);
    }
  });

  it('is exact at the integers, where Γ(n) = (n-1)!', () => {
    expect(lgamma(1).data[0]).toBeCloseTo(0, 14);
    expect(lgamma(5).data[0]).toBeCloseTo(Math.log(24), 13);
    expect(lgamma(11).data[0]).toBeCloseTo(Math.log(3628800), 12);
  });

  it('differentiates to the digamma function, checked by finite differences', () => {
    for (const x of [0.7, 1.5, 3.2, 9, 25]) {
      const { gradient } = valueAndGrad((p) => lgamma(p.x))({ x });
      const [fd] = fdGrad(([v]) => lgamma(v).data[0], [x]);
      expect(gradient.x).toBeCloseTo(fd, 6);
    }
  });

  it('broadcasts over a vector', () => {
    const { value, gradient } = valueAndGrad((p) => sum(lgamma(p.v)))({ v: [1, 2, 3] });
    expect(value).toBeCloseTo(Math.log(1) + Math.log(1) + Math.log(2), 13);
    expect(gradient.v).toHaveLength(3);
  });
});
