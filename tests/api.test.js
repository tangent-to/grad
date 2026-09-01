/**
 * The (value, gradient) function pair, and its shared-evaluation cache.
 *
 * This is the shape `@tangent.to/mc`'s `model.potential(name, fn, gradFn)`
 * takes, so the contract here is what makes a probabilistic model
 * differentiable instead of finite-differenced.
 */

import { describe, expect, it, vi } from 'vitest';
import { add, div, log, mul, square, sub, sum, valueAndGradFns } from '../src/index.js';

describe('valueAndGradFns', () => {
  const build = (p) => add(square(p.mu), mul(3, p.sigma));

  it('returns a value function and a gradient function', () => {
    const { value, gradient } = valueAndGradFns(build);
    expect(value({ mu: 2, sigma: 5 })).toBe(19);
    expect(gradient({ mu: 2, sigma: 5 })).toEqual({ mu: 4, sigma: 3 });
  });

  it('evaluates the objective ONCE for a value/gradient pair at the same point', () => {
    // mc's logProbAndGradient calls the value pass and the gradient pass in
    // turn; without sharing, every gradient would cost two sweeps over the data.
    const spy = vi.fn(build);
    const { value, gradient } = valueAndGradFns(spy);
    const p = { mu: 2, sigma: 5 };
    value(p);
    gradient(p);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('recomputes at a different point', () => {
    const spy = vi.fn(build);
    const { gradient } = valueAndGradFns(spy);
    gradient({ mu: 1, sigma: 1 });
    gradient({ mu: 2, sigma: 1 });
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it('recomputes when a parameter array is mutated IN PLACE', () => {
    // The trap: a sampler that reuses its position buffer would otherwise get a
    // stale gradient forever. The cache compares against a defensive copy.
    const f = (p) => sum(square(p.beta));
    const spy = vi.fn(f);
    const { gradient } = valueAndGradFns(spy);
    const beta = [1, 2, 3];
    const g1 = gradient({ beta });
    expect(g1).toEqual({ beta: [2, 4, 6] });

    beta[0] = 10; // same object, different contents
    const g2 = gradient({ beta });
    expect(spy).toHaveBeenCalledTimes(2);
    expect(g2).toEqual({ beta: [20, 4, 6] });
  });

  it('does not hand out a gradient that a later call can mutate', () => {
    const f = (p) => sum(square(p.beta));
    const { gradient } = valueAndGradFns(f);
    const g = gradient({ beta: [1, 2, 3] });
    gradient({ beta: [4, 5, 6] });
    expect(g).toEqual({ beta: [2, 4, 6] });
  });

  it('propagates a non-finite parameter instead of throwing', () => {
    // A sampler steps outside the support on its way to rejecting a
    // trajectory: NUTS pushes σ past 0, reads back -Infinity, and stops the
    // tree. Throwing there would kill the whole run instead of one proposal.
    const { value, gradient } = valueAndGradFns((p) => log(p.sigma));
    // The VALUE is what drives rejection: the sampler's Hamiltonian goes
    // non-finite and the tree stops. (The gradient need not be non-finite —
    // d/dx log(x) = 1/x is perfectly finite at x = -0.5 — so nothing is
    // asserted about it beyond not throwing.)
    expect(value({ sigma: -0.5 })).toBeNaN();
    expect(value({ sigma: 0 })).toBe(-Infinity);
    expect(() => gradient({ sigma: -0.5 })).not.toThrow();
    expect(() => gradient({ sigma: NaN })).not.toThrow();
  });

  it('carries a real likelihood term: Gaussian regression', () => {
    // The canonical mc potential, written in ops and differentiated exactly.
    const X = [-2, -1, 0, 1, 2];
    const Y = [-3.1, -0.9, 1.0, 3.2, 5.1];
    const n = X.length;
    const logLik = (p) => {
      const resid = sub(Y, add(mul(p.slope, X), p.intercept));
      return sub(mul(-0.5, sum(square(div(resid, p.sigma)))), mul(n, log(p.sigma)));
    };
    const at = { slope: 2, intercept: 1, sigma: 0.5 };
    const { gradient } = valueAndGradFns(logLik);
    const g = gradient(at);

    // Hand-derived, independently.
    let gS = 0, gI = 0, gSig = 0;
    for (let i = 0; i < n; i++) {
      const r = Y[i] - (at.slope * X[i] + at.intercept);
      gS += (r * X[i]) / at.sigma ** 2;
      gI += r / at.sigma ** 2;
      gSig += (r * r) / at.sigma ** 3;
    }
    expect(g.slope).toBeCloseTo(gS, 10);
    expect(g.intercept).toBeCloseTo(gI, 10);
    expect(g.sigma).toBeCloseTo(gSig - n / at.sigma, 10);
  });
});
