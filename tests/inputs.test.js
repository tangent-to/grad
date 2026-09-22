/**
 * Inputs: leaves that are not differentiated and change between calls.
 *
 * The contract has two halves. First, an objective evaluated with inputs
 * through `compile` must agree with the same objective evaluated with the
 * same data closed over as constants, at every batch in sequence through one
 * compiled closure: a plan that kept the first batch would pass the first
 * call and drift after. Second, no gradient is reported for an input, and the
 * forward-only replay returns the root alone, scalar or not.
 */

import { describe, expect, it } from 'vitest';
import {
  add, compile, compileFromJSON, exp, matmul, mean, mul, relu, square, sub, sum, valueAndGrad,
} from '../src/index.js';
import { fdGrad } from './_fd.js';

const X1 = [[1, 2], [3, 4], [5, 6]];
const X2 = [[0.5, -1], [2, 2], [-3, 1]];
const X3 = [[1, 1]];                          // a partial last batch
const y1 = [1, 2, 3], y2 = [0, 0, 1], y3 = [2];
const p = { W: [[0.1, -0.2], [0.3, 0.4]], b: [0.05, -0.05], w2: [1, -1] };

// A small network: relu(X·W + b)·w2, squared error against y.
const net = (q, d) => matmul(relu(add(matmul(d.X, q.W), q.b)), q.w2);
const loss = (q, d) => mean(square(sub(net(q, d), d.y)));
const closedOver = (X, y) => (q) => loss(q, { X, y });

function flatten(g) {
  if (typeof g === 'number') return [g];
  if (Array.isArray(g)) return g.flatMap(flatten);
  return Object.keys(g).sort().flatMap((k) => flatten(g[k]));
}

describe('inputs: valueAndGrad', () => {
  it('evaluates f(params, inputs) and reports no gradient for an input', () => {
    const vg = valueAndGrad(loss);
    const got = vg(p, { X: X1, y: y1 });
    const ref = valueAndGrad(closedOver(X1, y1))(p);
    expect(got.value).toBeCloseTo(ref.value, 12);
    expect(flatten(got.gradient)).toEqual(flatten(ref.gradient));
    expect(Object.keys(got.gradient).sort()).toEqual(['W', 'b', 'w2']);
  });

  it('matches finite differences with the data as inputs', () => {
    const vg = valueAndGrad(loss);
    const flat = [0.1, -0.2, 0.3, 0.4, 0.05, -0.05, 1, -1];
    const unflat = (v) => ({ W: [[v[0], v[1]], [v[2], v[3]]], b: [v[4], v[5]], w2: [v[6], v[7]] });
    const fd = fdGrad((v) => vg.value(unflat(v), { X: X2, y: y2 }), flat);
    const got = flatten(vg(unflat(flat), { X: X2, y: y2 }).gradient);
    fd.forEach((g, i) => expect(got[i]).toBeCloseTo(g, 6));
  });

  it('.value returns the root alone, and it may be a vector or a matrix', () => {
    const vg = valueAndGrad(net);
    const out = vg.value(p, { X: X1 });
    expect(out).toHaveLength(3);
    expect(() => vg(p, { X: X1 })).toThrow(/must return a scalar/);
  });

  it('refuses inputs that are not a map', () => {
    expect(() => valueAndGrad(loss)(p, [1, 2, 3])).toThrow(/inputs must be a \{name: value\} map/);
  });
});

describe('inputs: compile', () => {
  it('replays on a new batch instead of the one the plan was built with', () => {
    const c = compile(loss);
    for (const [X, y] of [[X1, y1], [X2, y2], [X1, y2], [X2, y1]]) {
      const got = c(p, { X, y });
      const ref = valueAndGrad(closedOver(X, y))(p);
      expect(got.value).toBeCloseTo(ref.value, 12);
      flatten(ref.gradient).forEach((g, i) => expect(flatten(got.gradient)[i]).toBeCloseTo(g, 12));
    }
  });

  it('keeps a plan per shape, so a partial last batch does not evict the full one', () => {
    const c = compile(loss);
    const seq = [[X1, y1], [X3, y3], [X2, y2], [X3, y3], [X1, y1]];
    for (const [X, y] of seq) {
      const got = c(p, { X, y });
      const ref = valueAndGrad(closedOver(X, y))(p);
      expect(got.value).toBeCloseTo(ref.value, 12);
    }
  });

  it('.value is the forward replay: the same number, no gradient computed', () => {
    const c = compile(loss);
    const v = c.value(p, { X: X2, y: y2 });
    expect(v).toBeCloseTo(valueAndGrad(closedOver(X2, y2))(p).value, 12);
    // and the gradient path still works afterwards on the same plan
    expect(c(p, { X: X2, y: y2 }).value).toBeCloseTo(v, 12);
  });

  it('.value on a non-scalar objective returns the root as nested rows', () => {
    const c = compile((q, d) => add(matmul(d.X, q.W), q.b));
    const out = c.value(p, { X: X1 });
    expect(out).toHaveLength(3);
    expect(out[0]).toHaveLength(2);
    expect(out[1][0]).toBeCloseTo(3 * 0.1 + 4 * 0.3 + 0.05, 12);
    expect(() => c(p, { X: X1 })).toThrow(/must return a scalar/);
  });

  it('does not alias the caller\'s input arrays', () => {
    const c = compile(loss);
    const X = X1.map((r) => Float64Array.from(r));
    const before = c(p, { X, y: y1 }).value;
    X[0][0] = 100;                        // mutate after the plan was built
    const after = c(p, { X: X1, y: y1 }).value;
    expect(after).toBeCloseTo(before, 12); // the plan read X1 fresh, not a poisoned alias
  });

  it('a scalar input works, for a per-fit coefficient', () => {
    const c = compile((q, d) => mul(d.lambda, sum(square(q.w2))));
    expect(c(p, { lambda: 0.5 }).value).toBeCloseTo(1, 12);
    expect(c(p, { lambda: 2 }).value).toBeCloseTo(4, 12);
    expect(c(p, { lambda: 2 }).gradient.w2).toEqual([4, -4]);
  });

  it('falls back to rebuilding when the graph cannot be replayed, inputs included', () => {
    const c = compile((q, d) => {
      const v = exp(q.w2);
      // A hand-built node has no recompute; the plan is refused and the eager path
      // must still see the inputs.
      const hand = new (v.constructor)(v.value, [v], () => [null]);
      return sum(mul(hand, d.s));
    });
    expect(c(p, { s: [1, 2] }).value).toBeCloseTo(Math.exp(1) + 2 * Math.exp(-1), 12);
    expect(c(p, { s: [2, 2] }).value).toBeCloseTo(2 * Math.exp(1) + 2 * Math.exp(-1), 12);
  });
});

describe('inputs: serialization', () => {
  it('a plan with inputs round-trips and asks for them again by name', () => {
    const c = compile(loss);
    c(p, { X: X1, y: y1 });
    const json = JSON.parse(JSON.stringify(c.toJSON()));
    expect(json.version).toBe(2);
    expect(json.nodes.filter((n) => n.kind === 'input').map((n) => n.name).sort()).toEqual(['X', 'y']);
    const again = compileFromJSON(json);
    for (const [X, y] of [[X2, y2], [X1, y1]]) {
      const a = c(p, { X, y }), b = again(p, { X, y });
      expect(b.value).toBe(a.value);
      expect(flatten(b.gradient)).toEqual(flatten(a.gradient));
      expect(again.value(p, { X, y })).toBe(a.value);
    }
  });

  it('a rebuilt plan refuses inputs of another shape, or a missing input', () => {
    const c = compile(loss);
    c(p, { X: X1, y: y1 });
    const again = compileFromJSON(c.toJSON());
    expect(() => again(p, { X: X3, y: y3 })).toThrow(/built for arguments shaped/);
    expect(() => again(p, { X: X1 })).toThrow(/built for arguments shaped/);
    expect(() => again(p)).toThrow(/built for arguments shaped/);
  });

  it('serializes the plan of the shapes most recently evaluated', () => {
    const c = compile(loss);
    c(p, { X: X1, y: y1 });
    c(p, { X: X3, y: y3 });
    const inputs = c.toJSON().nodes.filter((n) => n.kind === 'input');
    expect(inputs.find((n) => n.name === 'X').shape).toEqual([1, 2]);
  });

  it('still reads a version-1 plan', () => {
    const c = compile((q) => sum(square(q.w2)));
    c(p);
    const json = c.toJSON();
    json.version = 1;
    expect(compileFromJSON(json)(p).value).toBe(2);
  });
});
