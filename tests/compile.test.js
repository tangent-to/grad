/**
 * Tape reuse.
 *
 * `compile(f)` keeps the graph and replays it at new parameters. The whole
 * contract is that it must be indistinguishable from rebuilding, so almost
 * every test here is the same assertion: run the same objective through
 * `compile` and through `valueAndGrad` and demand they agree.
 *
 * The interesting part is that the agreement has to hold on the SECOND and
 * THIRD call, not just the first. A replayed op that cached something derived
 * from its build-time inputs — a transpose, a factorization, a buffer it
 * reallocated instead of refilling — passes call one and quietly serves a
 * gradient computed at the wrong point afterwards. So every case below sweeps
 * several parameter points through one compiled closure.
 */

import { describe, expect, it } from 'vitest';
import {
  add, addDiag, cholesky, compile, lgamma, concat, diagPart, div, dot, exp, inv, log,
  logdetPSD, matmul, maximum, mean, minimum, mul, neg, pow, relu, reshape,
  sigmoid, slice, solveGeneral, solvePSD, sqrt, square, sub, sum, tanh, trace,
  transpose, triangularSolve, valueAndGrad, valueAndGradFns, variable, Var,
} from '../src/index.js';

/** Every leaf of a gradient, flattened, so nested and flat shapes compare alike. */
function flatten(g) {
  if (typeof g === 'number') return [g];
  if (Array.isArray(g)) return g.flatMap(flatten);
  return Object.keys(g).sort().flatMap((k) => flatten(g[k]));
}

/**
 * The core assertion: replaying agrees with rebuilding, at every point in
 * sequence through ONE compiled closure.
 */
function agreesOver(f, points) {
  const compiled = compile(f);
  const plain = valueAndGrad(f);
  points.forEach((p, i) => {
    const got = compiled(p);
    const want = plain(p);
    expect(got.value, `value at point ${i}`).toBeCloseTo(want.value, 12);
    const a = flatten(got.gradient);
    const b = flatten(want.gradient);
    expect(a.length, `gradient width at point ${i}`).toBe(b.length);
    a.forEach((v, j) => expect(v, `gradient[${j}] at point ${i}`).toBeCloseTo(b[j], 10));
  });
}

const scalarPoints = [
  { a: 1.3, b: 0.7 },
  { a: 2.9, b: -1.4 },
  { a: 0.4, b: 3.1 },
];

describe('compile: elementwise ops replay', () => {
  const cases = {
    add: (p) => add(p.a, p.b),
    sub: (p) => sub(p.a, p.b),
    mul: (p) => mul(p.a, p.b),
    div: (p) => div(p.a, add(p.b, 5)),
    neg: (p) => neg(p.a),
    exp: (p) => exp(p.b),
    log: (p) => log(add(p.a, 4)),
    sqrt: (p) => sqrt(add(p.a, 4)),
    square: (p) => square(p.b),
    pow: (p) => pow(add(p.a, 2), 3),
    tanh: (p) => tanh(p.b),
    sigmoid: (p) => sigmoid(p.b),
    maximum: (p) => maximum(p.a, p.b),
    minimum: (p) => minimum(p.a, p.b),
    relu: (p) => relu(sub(p.a, p.b)),
    'scalar broadcast': (p) => sum(mul(p.a, [1, 2, 3])),
    'variadic add': (p) => add(p.a, p.b, 2, mul(p.a, p.b)),
    'variadic mul': (p) => mul(p.a, p.b, 3, add(p.a, 1)),
    lgamma: (p) => lgamma(add(p.a, 2)),
  };
  for (const [name, f] of Object.entries(cases)) {
    it(name, () => agreesOver(f, scalarPoints));
  }
});

describe('compile: reductions and shape ops replay', () => {
  const vecPoints = [
    { v: [1, 2, 3, 4], m: [[1, 2], [3, 4]] },
    { v: [-2, 0.5, 7, 1], m: [[5, -1], [0.5, 2]] },
    { v: [0.1, 0.2, 0.3, 0.4], m: [[2, 3], [1, 9]] },
  ];
  const cases = {
    sum: (p) => sum(p.v),
    mean: (p) => mean(p.v),
    dot: (p) => dot(p.v, [1, -1, 2, 0.5]),
    matmul: (p) => sum(matmul(p.m, [[1, 0], [0, 2]])),
    'matmul with a vector': (p) => sum(matmul(p.m, [2, -1])),
    transpose: (p) => sum(mul(transpose(p.m), [[1, 2], [3, 4]])),
    diagPart: (p) => sum(square(diagPart(p.m))),
    trace: (p) => trace(p.m),
    addDiag: (p) => sum(addDiag(p.m, 3)),
    'addDiag per row': (p) => sum(addDiag(p.m, [1, 2])),
    reshape: (p) => sum(square(reshape(p.v, [2, 2]))),
    'slice a vector': (p) => sum(square(slice(p.v, [1], [2]))),
    'slice a matrix': (p) => sum(square(slice(p.m, [0, 0], [2, 1]))),
    concat: (p) => sum(square(concat([p.v, slice(p.v, [0], [1]), [9, 8]]))),
  };
  for (const [name, f] of Object.entries(cases)) {
    it(name, () => agreesOver(f, vecPoints));
  }
});

describe('compile: linear algebra replays', () => {
  // Diagonally dominant by construction, so every point stays factorizable.
  // Built through concat/reshape because a matrix of Vars is not a Var: the
  // ops are what put the parameters inside the tape.
  const spd = (p) => reshape(concat([add(4, p.a), 1, 1, add(3, p.b)]), [2, 2]);
  const points = [
    { a: 0.5, b: 0.2, rhs: [1, 2] },
    { a: 2.0, b: 1.7, rhs: [-3, 0.5] },
    { a: 0.1, b: 0.9, rhs: [4, 4] },
  ];
  const cases = {
    cholesky: (p) => sum(square(cholesky(spd(p)))),
    logdetPSD: (p) => logdetPSD(spd(p)),
    solvePSD: (p) => sum(square(solvePSD(spd(p), p.rhs))),
    solveGeneral: (p) => sum(square(solveGeneral(spd(p), p.rhs))),
    inv: (p) => sum(square(inv(spd(p)))),
    triangularSolve: (p) => sum(square(triangularSolve(cholesky(spd(p)), p.rhs))),
  };
  for (const [name, f] of Object.entries(cases)) {
    it(name, () => agreesOver(f, points));
  }

  it('refreshes what a backward closure derived from the forward values', () => {
    // solveGeneral factors Aᵀ once and its adjoint closes over the result. If a
    // replay skipped that refresh, the value would track A while the gradient
    // stayed pinned to the matrix from the first call — the exact failure this
    // whole file exists to catch, and invisible on call one.
    const f = (p) => sum(solveGeneral(reshape(concat([p.a, 1, 0.5, p.b]), [2, 2]), [1, 2]));
    const compiled = compile(f);
    compiled({ a: 3, b: 4 });
    const replayed = compiled({ a: 9, b: 2 });
    const fresh = valueAndGrad(f)({ a: 9, b: 2 });
    expect(replayed.gradient.a).toBeCloseTo(fresh.gradient.a, 10);
    expect(replayed.gradient.b).toBeCloseTo(fresh.gradient.b, 10);
  });
});

describe('compile: a realistic objective', () => {
  it('matches on a Gaussian process marginal likelihood', () => {
    const X = [0, 0.4, 1.1, 1.9, 2.6, 3.4];
    const y = [0.1, 0.7, 1.2, 0.4, -0.5, -1.1];
    const f = (p) => {
      const s2 = exp(mul(2, p.logAmp));
      const ls = exp(p.logLen);
      const rows = X.map((xi) => X.map((xj) => -0.5 * (xi - xj) ** 2));
      const K = addDiag(mul(s2, exp(div(rows, square(ls)))), exp(p.logNoise));
      const a = solvePSD(K, y);
      return mul(-0.5, add(dot(y, a), logdetPSD(K)));
    };
    agreesOver(f, [
      { logAmp: 0.0, logLen: 0.0, logNoise: -2 },
      { logAmp: 0.6, logLen: -0.3, logNoise: -4 },
      { logAmp: -0.4, logLen: 0.9, logNoise: -1 },
    ]);
  });

  it('matches on a quadratic-plateau dose response, the shape relu was added for', () => {
    const x = [0, 25, 50, 75, 100, 140, 180, 220];
    const y = [2.1, 5.4, 8.0, 10.2, 11.6, 12.8, 13.0, 12.9];
    const f = (p) => {
      const g = exp(p.logG);
      const ns = exp(p.logNstar);
      const mu = mul(g, sub(1, square(relu(sub(1, div(x, ns))))));
      return neg(sum(square(sub(y, mu))));
    };
    agreesOver(f, [
      { logG: Math.log(13), logNstar: Math.log(150) },
      { logG: Math.log(9), logNstar: Math.log(90) },
      // Past the largest x, so every observation sits on the rising arm and
      // relu's branch falls the other way. A compiled tape must follow the
      // kernel's branch even though the graph is fixed.
      { logG: Math.log(20), logNstar: Math.log(400) },
    ]);
  });
});

describe('compile: input shapes', () => {
  it('takes a bare array, and returns the gradient in that shape', () => {
    const f = (v) => sum(square(v));
    const compiled = compile(f);
    expect(compiled([1, 2, 3]).gradient).toEqual([2, 4, 6]);
    expect(compiled([4, 5, 6]).gradient).toEqual([8, 10, 12]);
  });

  it('takes a bare number', () => {
    const compiled = compile((x) => square(x));
    expect(compiled(3)).toEqual({ value: 9, gradient: 6 });
    expect(compiled(5)).toEqual({ value: 25, gradient: 10 });
  });

  it('reports a zero gradient for a parameter the objective never reads', () => {
    const compiled = compile((p) => square(p.used));
    expect(compiled({ used: 2, spare: [1, 2] }).gradient).toEqual({ used: 4, spare: [0, 0] });
  });

  it('rebuilds when a parameter changes shape instead of replaying the wrong graph', () => {
    const f = (p) => sum(square(p.v));
    const compiled = compile(f);
    expect(compiled({ v: [1, 2] }).gradient).toEqual({ v: [2, 4] });
    expect(compiled({ v: [1, 2, 3] }).gradient).toEqual({ v: [2, 4, 6] });
    expect(compiled({ v: [5, 5] }).gradient).toEqual({ v: [10, 10] });
  });

  it('does not alias a Float64Array the caller passed in', () => {
    // variable() wraps a Float64Array without copying. The plan writes into its
    // leaves on every call, so without a copy the second evaluation would
    // overwrite the caller's array from the first.
    const held = Float64Array.of(1, 2, 3);
    const compiled = compile((v) => sum(square(v)));
    compiled(held);
    compiled([9, 9, 9]);
    expect(Array.from(held)).toEqual([1, 2, 3]);
  });
});

describe('compile: what it refuses to do', () => {
  it('falls back to rebuilding when the graph holds a node it cannot replay', () => {
    // A Var built by hand carries a backward closure but no way to recompute
    // its forward value. Rather than refuse to differentiate, compile() hands
    // the objective to valueAndGrad.
    const opaque = (a) => new Var({ data: Float64Array.of(a.data[0] * 3), shape: [] }, [a],
      (g) => [Float64Array.of(g[0] * 3)]);
    const f = (p) => opaque(mul(p.x, 1));
    const compiled = compile(f);
    expect(compiled({ x: 2 })).toEqual({ value: 6, gradient: { x: 3 } });
    expect(compiled({ x: 7 })).toEqual({ value: 21, gradient: { x: 3 } });
  });

  it('rejects a non-scalar objective', () => {
    expect(() => compile((v) => square(v))([1, 2])).toThrow(/must return a scalar/);
  });

  it('rejects a non-function', () => {
    expect(() => compile(42)).toThrow(/expected a function/);
  });

  it('propagates a non-finite density instead of throwing, as a sampler needs', () => {
    // NUTS steps outside a support and reads back -Infinity to reject the
    // trajectory. Replaying must not turn that into an exception, and must not
    // leave the plan poisoned for the next in-support point.
    const compiled = compile((p) => log(p.sigma));
    expect(compiled({ sigma: -1 }).value).toBeNaN();
    expect(compiled({ sigma: 0 }).value).toBe(-Infinity);
    expect(compiled({ sigma: 2 }).value).toBeCloseTo(Math.log(2), 12);
    expect(compiled({ sigma: 2 }).gradient.sigma).toBeCloseTo(0.5, 12);
  });
});

describe('valueAndGradFns({ compile: true })', () => {
  const build = (p) => add(square(p.mu), mul(3, p.sigma));

  it('agrees with the uncompiled pair', () => {
    const plain = valueAndGradFns(build);
    const fast = valueAndGradFns(build, { compile: true });
    for (const p of [{ mu: 2, sigma: 5 }, { mu: -1, sigma: 0.5 }, { mu: 4, sigma: 3 }]) {
      expect(fast.value(p)).toBeCloseTo(plain.value(p), 12);
      expect(fast.gradient(p)).toEqual(plain.gradient(p));
    }
  });

  it('is off by default', () => {
    // The static-graph assumption is the caller's to make, so it has to be
    // asked for. A hand-built node would be replayed by neither, but a plan
    // that captured mutable closed-over data would differ silently — which is
    // why this default is not a performance question.
    const mutable = [1, 1];
    const f = (p) => mul(p.x, sum(mutable));
    const plain = valueAndGradFns(f);
    expect(plain.gradient({ x: 1 }).x).toBe(2);
    mutable[0] = 5;
    expect(plain.gradient({ x: 2 }).x).toBe(6);
  });
});
