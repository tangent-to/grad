/**
 * n-dimensional ops against references in float64.
 *
 * tests/references/nd_reference.py defines the cases and computes each op's
 * output, and the gradient of L = sum(weight · op(inputs)), with PyTorch;
 * tests/references/nd_reference.R computes the outputs again with base R's
 * column-major arrays. Both write committed fixtures, so this runs without
 * Python or R; regenerate them with `npm run references`.
 *
 * Every case is also held to the package's own checks: finite differences,
 * a compiled plan replayed at a second point, and that plan rebuilt from JSON.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as g from '../src/index.js';
import { checkOp } from './_check.js';

const load = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const { cases } = load('nd-cases.json');
const torch = load('nd-torch.json').results;
const R = load('nd-r.json').results;

// jsonlite writes a length-1 array as a bare value.
const arr = (x) => [].concat(x);
const tensorOf = (t) => ({ data: Float64Array.from(arr(t.data)), shape: arr(t.shape) });

/** The op a case names, as a function of `{ a, b }` Vars and the index input. */
function opOf(c) {
  const { op, args } = c;
  switch (op) {
    case 'add': case 'sub': case 'mul': case 'div': case 'maximum':
      return (p) => g[op](p.a, p.b);
    case 'sum': case 'mean': case 'max':
      return (p) => g[op](p.a, args.axis === undefined ? undefined : { axis: args.axis, keepDims: args.keepDims });
    case 'transpose':
      return (p) => g.transpose(p.a, args.perm ?? undefined);
    case 'matmul':
      return (p) => g.matmul(p.a, p.b);
    case 'slice':
      return (p) => g.slice(p.a, args.start, args.size);
    case 'concat':
      return (p) => g.concat([p.a, p.b], { axis: args.axis });
    case 'reshape':
      return (p) => g.reshape(p.a, args.shape);
    case 'gather':
      return (p, i) => g.gather(p.a, i.index);
    default:
      throw new Error(`no op ${op}`);
  }
}

/** Parameters (differentiated) and inputs (not) of a case, as tensors. */
function split(c) {
  const params = {};
  const feed = {};
  for (const [k, v] of Object.entries(c.inputs)) (c.nondiff.includes(k) ? feed : params)[k] = tensorOf(v);
  return { params, feed: Object.keys(feed).length ? feed : undefined };
}

/** Largest relative error of `got` against `want`. */
const worst = (got, want) => {
  let e = 0;
  for (let i = 0; i < want.length; i++) e = Math.max(e, Math.abs(got[i] - want[i]) / (1 + Math.abs(want[i])));
  return e;
};

const TOL = 1e-12;

describe('n-dimensional ops against PyTorch and R (float64)', () => {
  it('has the same cases in all three fixtures', () => {
    expect(torch.map((r) => r.name)).toEqual(cases.map((c) => c.name));
    expect(R.map((r) => r.name)).toEqual(cases.map((c) => c.name));
  });

  cases.forEach((c, n) => {
    it(c.name, () => {
      const f = opOf(c);
      const { params, feed } = split(c);
      const vars = Object.fromEntries(Object.entries(params).map(([k, v]) => [k, g.variable(v)]));
      const inputs = feed && Object.fromEntries(Object.entries(feed).map(([k, v]) => [k, g.variable(v)]));
      const y = f(vars, inputs);

      // Output: shape and values against both references.
      for (const ref of [torch[n].output, R[n].output]) {
        expect(y.shape).toEqual(arr(ref.shape));
        expect(worst(y.data, arr(ref.data))).toBeLessThan(TOL);
      }

      // Gradient of sum(weight · op) against torch autograd.
      const w = tensorOf(c.weight);
      const { gradient } = g.valueAndGrad((p, i) => g.sum(g.mul(f(p, i), w)))(params, feed);
      for (const [k, ref] of Object.entries(torch[n].grads)) {
        expect(gradient[k].shape).toEqual(arr(ref.shape));
        expect(worst(gradient[k].data, arr(ref.data))).toBeLessThan(TOL);
      }

      // Finite differences, compiled replay at a second point, JSON rebuild.
      const nested = (ps, s) => Object.fromEntries(Object.entries(ps).map(([k, v]) =>
        [k, g.toNested({ data: v.data.map((x) => x * s + 0.1 * Math.sign(x)), shape: v.shape })]));
      const feedNested = feed && { index: g.toNested(feed.index) };
      checkOp((p, i) => g.sum(g.mul(f(p, i), w)), nested(params, 1), nested(params, 0.7), feedNested, feedNested);
    });
  });
});
