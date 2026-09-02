/**
 * A compiled plan as data.
 *
 * `compile(f).toJSON()` writes the graph out; `compileFromJSON` rebuilds it.
 * The contract is that the rebuilt plan is the original plan: same value,
 * same gradient, to the bit, at every point. Every op has to survive the
 * round trip, so the cases below cover the whole surface, and each is checked
 * at several points through one rebuilt closure, where a static argument
 * recorded wrong (a slice offset, a pow exponent, a solve's triangle) would
 * show up as a gradient that drifts from the original.
 *
 * The JSON goes through `JSON.parse(JSON.stringify(...))` and through
 * `structuredClone` on the way, because "plain data" is the property that
 * lets it cross into a worker, and a Float64Array or a closure hiding inside
 * would pass a naive equality test and fail there.
 */

import { describe, expect, it } from 'vitest';
import {
  add, addDiag, cholesky, compile, compileFromJSON, concat, diagPart, div, dot, exp, inv,
  log, logdetPSD, matmul, maximum, mean, minimum, mul, neg, pow, relu, reshape, sigmoid,
  slice, solveGeneral, solvePSD, sqrt, square, sub, sum, tanh, trace, transpose,
  triangularSolve, Var,
} from '../src/index.js';

function flatten(g) {
  if (typeof g === 'number') return [g];
  if (Array.isArray(g)) return g.flatMap(flatten);
  return Object.keys(g).sort().flatMap((k) => flatten(g[k]));
}

/** Compile, build at the first point, round-trip, then demand bit equality everywhere. */
function roundTrips(f, points) {
  const original = compile(f);
  original(points[0]);
  const json = structuredClone(JSON.parse(JSON.stringify(original.toJSON())));
  const rebuilt = compileFromJSON(json);
  points.forEach((p, i) => {
    const a = original(p);
    const b = rebuilt(p);
    expect(b.value, `value at point ${i}`).toBe(a.value);
    const ga = flatten(a.gradient);
    const gb = flatten(b.gradient);
    expect(gb.length, `gradient width at point ${i}`).toBe(ga.length);
    gb.forEach((v, j) => expect(v, `gradient[${j}] at point ${i}`).toBe(ga[j]));
  });
  return json;
}

const scalarPoints = [{ a: 1.3, b: 0.7 }, { a: 2.9, b: -1.4 }, { a: 0.4, b: 3.1 }];

describe('serialize: elementwise ops', () => {
  const cases = {
    add: (p) => add(p.a, p.b),
    'add variadic': (p) => add(p.a, p.b, 2, p.a),
    sub: (p) => sub(p.a, p.b),
    mul: (p) => mul(p.a, p.b, 3),
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
  };
  for (const [name, f] of Object.entries(cases)) it(name, () => roundTrips(f, scalarPoints));
});

describe('serialize: reductions and shape ops', () => {
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
  for (const [name, f] of Object.entries(cases)) it(name, () => roundTrips(f, vecPoints));

  it('records slice offsets, not just shapes', () => {
    // Two slices of the same size at different offsets have identical shapes
    // everywhere. Only the recorded args tell them apart after a round trip.
    const f = (p) => sub(sum(slice(p.v, [2], [2])), sum(slice(p.v, [0], [2])));
    roundTrips(f, vecPoints);
  });
});

describe('serialize: linear algebra', () => {
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
    'triangularSolve lower': (p) => sum(square(triangularSolve(cholesky(spd(p)), p.rhs))),
    'triangularSolve upper': (p) =>
      sum(square(triangularSolve(transpose(cholesky(spd(p))), p.rhs, { lower: false }))),
  };
  for (const [name, f] of Object.entries(cases)) it(name, () => roundTrips(f, points));
});

describe('serialize: realistic objectives', () => {
  it('a Gaussian process marginal likelihood', () => {
    const X = [0, 0.4, 1.1, 1.9, 2.6, 3.4];
    const y = [0.1, 0.7, 1.2, 0.4, -0.5, -1.1];
    const rows = X.map((xi) => X.map((xj) => -0.5 * (xi - xj) ** 2));
    const f = (p) => {
      const K = addDiag(mul(exp(mul(2, p.logAmp)), exp(div(rows, square(exp(p.logLen))))), exp(p.logNoise));
      return mul(-0.5, add(dot(y, solvePSD(K, y)), logdetPSD(K)));
    };
    roundTrips(f, [
      { logAmp: 0.0, logLen: 0.0, logNoise: -2 },
      { logAmp: 0.6, logLen: -0.3, logNoise: -4 },
      { logAmp: -0.4, logLen: 0.9, logNoise: -1 },
    ]);
  });

  it('a quadratic-plateau dose response, data captured as constants', () => {
    const x = [0, 25, 50, 75, 100, 140, 180, 220];
    const y = [2.1, 5.4, 8.0, 10.2, 11.6, 12.8, 13.0, 12.9];
    const f = (p) => {
      const mu = mul(exp(p.logG), sub(1, square(relu(sub(1, div(x, exp(p.logNstar)))))));
      return neg(sum(square(sub(y, mu))));
    };
    const json = roundTrips(f, [
      { logG: Math.log(13), logNstar: Math.log(150) },
      { logG: Math.log(9), logNstar: Math.log(90) },
      { logG: Math.log(20), logNstar: Math.log(400) },
    ]);
    // The closure captured x and y. They are in the plan as constants, which
    // is what lets a worker evaluate this without ever seeing the closure.
    const consts = json.nodes.filter((n) => n.kind === 'const' && n.shape[0] === 8);
    expect(consts.map((c) => c.data)).toEqual(expect.arrayContaining([x, y]));
  });
});

describe('serialize: input shapes', () => {
  it('a bare array', () => roundTrips((v) => sum(square(v)), [[1, 2, 3], [4, 5, 6]]));
  it('a bare number', () => roundTrips((x) => square(x), [3, 5]));

  it('keeps a parameter the objective never reads, and reports its zero gradient', () => {
    const f = (p) => square(p.used);
    const json = roundTrips(f, [{ used: 2, spare: [1, 2] }, { used: 3, spare: [5, 5] }]);
    const rebuilt = compileFromJSON(json);
    expect(rebuilt({ used: 2, spare: [1, 2] }).gradient).toEqual({ used: 4, spare: [0, 0] });
  });

  it('is plain data with no typed arrays or functions inside', () => {
    const c = compile((p) => sum(mul(p.v, [1, 2, 3])));
    c({ v: [1, 1, 1] });
    const json = c.toJSON();
    const walk = (x) => {
      expect(typeof x).not.toBe('function');
      expect(x instanceof Float64Array).toBe(false);
      if (x && typeof x === 'object') Object.values(x).forEach(walk);
    };
    walk(json);
    expect(json.version).toBe(1);
  });
});

describe('serialize: refusals', () => {
  it('toJSON before the first call has no graph to write', () => {
    expect(() => compile((x) => square(x)).toJSON()).toThrow(/call the compiled function once/);
  });

  it('toJSON on an objective with a hand-built node', () => {
    const opaque = (a) => new Var({ data: Float64Array.of(a.data[0] * 3), shape: [] }, [a],
      (g) => [Float64Array.of(g[0] * 3)]);
    const c = compile((p) => opaque(mul(p.x, 1)));
    c({ x: 2 });
    expect(() => c.toJSON()).toThrow(/cannot be replayed or serialized/);
  });

  it('a rebuilt plan refuses a different shape instead of adapting', () => {
    const c = compile((p) => sum(square(p.v)));
    c({ v: [1, 2] });
    const rebuilt = compileFromJSON(c.toJSON());
    expect(() => rebuilt({ v: [1, 2, 3] })).toThrow(/built for parameters shaped v:v2, got v:v3/);
  });

  it('rejects something that is not a plan', () => {
    expect(() => compileFromJSON({ nodes: 'no' })).toThrow(/not a serialized plan/);
    expect(() => compileFromJSON({ version: 1, nodes: [{ kind: 'op', op: 'frobnicate', parents: [] }], root: 0, input: 'single' }))
      .toThrow(/unknown op "frobnicate"/);
  });
});

describe('serialize: across a worker boundary', () => {
  it('a worker rebuilds the plan from a posted message and returns the same gradient', async () => {
    // The whole reason the format exists. The closure below captures data it
    // could never send to a worker. The plan carries that data as constants,
    // crosses postMessage's structured clone, and evaluates on the other side.
    const { Worker } = await import('node:worker_threads');
    const x = [0, 25, 50, 75, 100, 140, 180, 220];
    const y = [2.1, 5.4, 8.0, 10.2, 11.6, 12.8, 13.0, 12.9];
    const f = (p) => {
      const mu = mul(exp(p.logG), sub(1, square(relu(sub(1, div(x, exp(p.logNstar)))))));
      return neg(sum(square(sub(y, mu))));
    };
    const local = compile(f);
    const p0 = { logG: Math.log(13), logNstar: Math.log(150) };
    const p1 = { logG: Math.log(9), logNstar: Math.log(90) };
    local(p0);

    const src = new URL('../src/index.js', import.meta.url).href;
    const worker = new Worker(
      `const { parentPort, workerData } = require('node:worker_threads');
       import(${JSON.stringify(src)}).then(({ compileFromJSON }) => {
         const vg = compileFromJSON(workerData.plan);
         parentPort.postMessage(vg(workerData.at));
       });`,
      { eval: true, workerData: { plan: local.toJSON(), at: p1 } },
    );
    const remote = await new Promise((resolve, reject) => {
      worker.once('message', resolve);
      worker.once('error', reject);
    });
    await worker.terminate();

    const here = local(p1);
    expect(remote.value).toBe(here.value);
    expect(remote.gradient.logG).toBe(here.gradient.logG);
    expect(remote.gradient.logNstar).toBe(here.gradient.logNstar);
  }, 20000);
});
