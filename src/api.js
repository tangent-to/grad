/**
 * High-level differentiation API.
 *
 * `valueAndGrad(f)` turns a scalar objective written in this package's ops into
 * a function returning both its value and its exact gradient. Two argument
 * shapes are accepted, because the suite has two callers:
 *
 *   - a plain array or number — what an optimizer (`@tangent.to/opt`) passes;
 *   - a `{name: number|number[]}` map — what a probabilistic model
 *     (`@tangent.to/mc`) passes, and what its samplers expect back.
 *
 * The gradient comes back in the same shape it went in, so it drops straight
 * into a leapfrog step or an L-BFGS iteration with no marshalling. A parameter
 * given as a tensor, `{ data: Float64Array, shape }`, gets its gradient back
 * as a tensor, which is the form a training loop keeps its state in.
 *
 * An objective may take a second argument, a map of INPUTS: leaves of the tape
 * that are not differentiated and change between calls — a mini-batch, a
 * dropout mask, a forcing term. `f(params, inputs)` is evaluated at
 * `vg(params, inputs)`, and a compiled plan writes the new inputs into its
 * leaves the way it writes the new parameters, instead of freezing them as
 * constants.
 */

import { Var, topoOrder, variable } from './tape.js';
import { isTensor, shapeStr, sizeOf, toNested } from './tensor.js';
import * as ops from './ops.js';
import * as linalg from './linalg.js';

/**
 * Every op a serialized plan may name, keyed by its exported name. A node's
 * `spec.op` is looked up here when a plan is rebuilt.
 * @private
 */
const REGISTRY = Object.fromEntries(
  Object.entries({ ...ops, ...linalg }).filter(([, v]) => typeof v === 'function'),
);

/** Is this a `{name: value}` parameter map rather than a single value? @private */
function isParamMap(x) {
  return x !== null && typeof x === 'object' && !Array.isArray(x) &&
    !(x instanceof Float64Array) && !(x instanceof Var) && !isTensor(x);
}

/**
 * Read a leaf's accumulated gradient back into the form its input arrived in:
 * a number, nested arrays, or a tensor for a tensor.
 * @private
 */
function gradOf(v, original) {
  // The objective may never have touched this parameter. A zero gradient is
  // the honest answer, and silently omitting the key would break a sampler
  // that indexes every parameter by name.
  const g = v.grad === null ? new Float64Array(v.value.data.length) : v.grad;
  if (isTensor(original)) {
    // A copy: a plan reuses its gradient buffers on the next call, and the
    // caller's Adam state must not be reading one that is being refilled.
    return { data: Float64Array.from(g), shape: v.shape.slice() };
  }
  if (typeof original === 'number') return g[0];
  return toNested({ data: g, shape: v.shape });
}

/**
 * The inputs argument must be a `{name: value}` map, or absent. @private
 */
function checkInputs(feed, where) {
  if (feed === undefined) return;
  if (!isParamMap(feed)) {
    throw new Error(`${where}: inputs must be a {name: value} map, got ${typeof feed}`);
  }
}

/**
 * A leaf a plan will write into on every call must own its storage:
 * `variable()` aliases a Float64Array or a tensor rather than copying it, and
 * evaluating at new values would otherwise scribble over the caller's array
 * from the first call. @private
 */
function ownedCopy(v) {
  if (typeof v === 'number') return v;
  if (isTensor(v)) return { data: Float64Array.from(v.data), shape: v.shape.slice() };
  return Array.from(v, (e) => (Array.isArray(e) || e instanceof Float64Array ? Array.from(e) : e));
}

/**
 * Wrap parameters as leaves, in the structure they arrived in. @private
 * @returns {{ isMap: boolean, leaves: Var|Object<string,Var> }}
 */
function wrapParams(x, own) {
  if (isParamMap(x)) {
    const leaves = {};
    for (const [k, v] of Object.entries(x)) leaves[k] = variable(own ? ownedCopy(v) : v, `parameter "${k}"`);
    return { isMap: true, leaves };
  }
  return { isMap: false, leaves: variable(own ? ownedCopy(x) : x, 'parameter') };
}

/** Wrap inputs as leaves that no gradient is read from. @private */
function wrapInputs(feed, own) {
  if (feed === undefined) return null;
  const leaves = {};
  for (const [k, v] of Object.entries(feed)) leaves[k] = variable(own ? ownedCopy(v) : v, `input "${k}"`);
  return leaves;
}

/** The root's value in the boundary currency: a number for a scalar, nested rows otherwise. @private */
function rootValue(root) {
  return root.isScalar ? root.data[0] : toNested(root.value);
}

/**
 * The contract both entry points enforce on the objective's return value.
 * @private
 */
function requireScalarObjective(out, where) {
  if (!(out instanceof Var)) {
    throw new Error(
      `${where}: the objective must return a Var built from this package's ops; ` +
        `got ${out === null ? 'null' : typeof out}. A plain number means the ops were ` +
        'bypassed somewhere, which breaks the chain.',
    );
  }
  if (!out.isScalar) {
    throw new Error(
      `${where}: the objective must return a scalar, got ${shapeStr(out.shape)}. ` +
        'Reduce it with sum() or mean() first.',
    );
  }
}

/**
 * Differentiate a scalar objective, returning both value and gradient.
 *
 * @param {(x: any, inputs?: any) => Var} f - objective, built from this
 *   package's ops. It receives `Var`s in the same structure as the parameters,
 *   and, when the returned function is called with a second argument, a map of
 *   input `Var`s as its own second argument. It must return a scalar `Var`.
 * @returns {(x: any, inputs?: Object) => { value: number, gradient: any }}
 *   with a `.value(x, inputs)` that evaluates the objective alone, and may
 *   return a non-scalar in the boundary currency.
 *
 * @example
 * const f = (p) => add(square(p.mu), square(p.sigma));
 * valueAndGrad(f)({ mu: 3, sigma: 4 });
 * // { value: 25, gradient: { mu: 6, sigma: 8 } }
 *
 * @example
 * // Data as inputs rather than closed-over constants: the same objective
 * // evaluates on any batch.
 * const sse = (p, d) => sum(square(sub(d.y, mul(p.slope, d.x))));
 * valueAndGrad(sse)({ slope: 2 }, { x: [1, 2], y: [2, 5] });
 */
export function valueAndGrad(f) {
  if (typeof f !== 'function') throw new Error('valueAndGrad: expected a function');

  const trace = (x, feed, where) => {
    checkInputs(feed, where);
    const { isMap, leaves } = wrapParams(x, false);
    const inputs = wrapInputs(feed, false);
    const out = inputs === null ? f(leaves) : f(leaves, inputs);
    if (!(out instanceof Var)) requireScalarObjective(out, where);
    return { isMap, leaves, out };
  };

  const vg = (x, feed) => {
    const { isMap, leaves, out } = trace(x, feed, 'valueAndGrad');
    requireScalarObjective(out, 'valueAndGrad');
    out.backward();
    let gradient;
    if (isMap) {
      gradient = {};
      for (const k of Object.keys(x)) gradient[k] = gradOf(leaves[k], x[k]);
    } else {
      gradient = gradOf(leaves, x);
    }
    return { value: out.data[0], gradient };
  };
  vg.value = (x, feed) => rootValue(trace(x, feed, 'valueAndGrad.value').out);
  return vg;
}

/**
 * Gradient only, discarding the objective's value.
 *
 * @param {(x: any, inputs?: any) => Var} f
 * @returns {(x: any, inputs?: Object) => any} gradient, shaped like the parameters
 */
export function grad(f) {
  const vg = valueAndGrad(f);
  return (x, feed) => vg(x, feed).gradient;
}

/** Structural equality over a `{name: number|number[]}` parameter map. @private */
function sameParams(a, b) {
  if (a === undefined || b === undefined) return false;
  if (typeof a === 'number') return a === b;
  if (isTensor(a)) {
    if (!isTensor(b) || a.data.length !== b.data.length) return false;
    for (let i = 0; i < a.data.length; i++) if (a.data[i] !== b.data[i]) return false;
    return true;
  }
  if (Array.isArray(a) || a instanceof Float64Array) {
    if (!(Array.isArray(b) || b instanceof Float64Array) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  if (a === null || typeof a !== 'object') return false;
  const ka = Object.keys(a);
  if (ka.length !== Object.keys(b).length) return false;
  for (const k of ka) if (!sameParams(a[k], b[k])) return false;
  return true;
}

/** Defensive copy of a parameter map, so a caller mutating in place cannot
 * poison the cache below. @private */
function copyParams(x) {
  if (typeof x === 'number') return x;
  if (isTensor(x)) return { data: Float64Array.from(x.data), shape: x.shape.slice() };
  if (Array.isArray(x) || x instanceof Float64Array) return Array.from(x);
  const out = {};
  for (const [k, v] of Object.entries(x)) out[k] = copyParams(v);
  return out;
}

/**
 * Split an objective into the SEPARATE value and gradient functions that an
 * API taking a `(fn, gradFn)` pair expects — `@tangent.to/mc`'s
 * `model.potential(name, fn, gradFn)` is the case this exists for.
 *
 * The two share one evaluation: calling `.value(p)` then `.gradient(p)` on the
 * same parameters runs the tape once, not twice. That matters because a
 * sampler's value-and-gradient path calls both in turn, and the forward pass
 * is a full sweep over the data.
 *
 * The cache holds exactly one entry and compares parameters structurally
 * against a defensive copy, so mutating a parameter array in place invalidates
 * it correctly rather than returning a stale gradient. A call that passes
 * inputs bypasses the cache: the same parameters on a different batch are a
 * different evaluation, and copying a batch to compare it would cost what the
 * cache saves.
 *
 * @param {(x: any) => Var} f - objective built from this package's ops
 * @param {Object} [options]
 * @param {boolean} [options.compile=false] - build the tape once and replay it,
 *   via {@link compile}. Worth an order of magnitude on a sampler, which calls
 *   this thousands of times at the same shapes; read `compile`'s constraint
 *   before turning it on. Off by default: a static graph is an assumption about
 *   your objective, and one this package cannot check for you.
 * @returns {{ value: (x:any) => number, gradient: (x:any) => any, compiled?: Function }}
 *   With `compile: true`, `compiled` is the underlying {@link compile} closure,
 *   so its `toJSON()` is reachable: what lets a model send its likelihood to a
 *   worker as data.
 *
 * @example
 * const { value, gradient } = valueAndGradFns((p) => logLik(p), { compile: true });
 * model.potential('y', value, gradient);
 */
export function valueAndGradFns(f, options = {}) {
  const vg = options.compile ? compile(f) : valueAndGrad(f);
  const fns = splitValueAndGrad(vg);
  if (options.compile) fns.compiled = vg;
  return fns;
}

/**
 * Split any `(x) => { value, gradient }` function into the separate value and
 * gradient functions a `(fn, gradFn)` API takes, sharing one evaluation
 * between them exactly as {@link valueAndGradFns} does. For a function that
 * already exists, such as one rebuilt by {@link compileFromJSON}.
 *
 * @param {(x: any) => { value: number, gradient: any }} vg
 * @returns {{ value: (x:any) => number, gradient: (x:any) => any }}
 */
export function splitValueAndGrad(vg) {
  let lastInput;
  let lastResult;
  const evaluate = (x, feed) => {
    if (feed !== undefined) return vg(x, feed);
    if (lastResult !== undefined && sameParams(lastInput, x)) return lastResult;
    lastResult = vg(x);
    lastInput = copyParams(x);
    return lastResult;
  };
  return {
    value: (x, feed) => evaluate(x, feed).value,
    gradient: (x, feed) => evaluate(x, feed).gradient,
  };
}

/**
 * Jacobian of a VECTOR-valued function: `J[i][j] = ∂f(x)ᵢ / ∂xⱼ`.
 *
 * Cost is one forward pass plus one reverse pass per OUTPUT — the tape is
 * built once and seeded m times.
 *
 * DO NOT reach for this to supply a stiff ODE solver's ∂f/∂y. It was written
 * for that and measured against `@tangent.to/ode`'s finite-difference
 * Jacobian on a stiff reaction-diffusion system; it lost, and lost worse as
 * the system grew:
 *
 *     n     FD      exact     steps (FD / exact)
 *     2    15 ms    27 ms          175 / 175
 *    10    23 ms   218 ms          171 / 171
 *    30    60 ms  1559 ms          171 / 171
 *
 * The step counts are identical, which is the whole story: a Newton iteration
 * converges to the same answer with an approximate Jacobian — the residual is
 * still evaluated exactly — so finite-difference error costs nothing there.
 * Meanwhile a square Jacobian is the worst case for reverse mode: n sweeps
 * over an n-node graph, against n+1 evaluations of cheap scalar arithmetic.
 * Forward mode, or finite differences with sparsity colouring, is the right
 * tool for that shape.
 *
 * Reverse mode pays when outputs are FEW and the map to them is expensive —
 * a delta-method standard error, the sensitivity of a handful of summaries to
 * many inputs.
 *
 * @param {(x: Var) => Var} f - vector-valued function built from these ops
 * @returns {(x: number[]) => number[][]} m × n Jacobian
 *
 * @example
 * const J = jacobian((y) => stack([mul(-2, y0(y)), sub(y0(y), y1(y))]));
 */
export function jacobian(f) {
  if (typeof f !== 'function') throw new Error('jacobian: expected a function');

  return (x) => {
    const leaf = variable(x, 'input');
    const out = f(leaf);
    if (!(out instanceof Var)) {
      throw new Error('jacobian: the function must return a Var built from this package\'s ops');
    }
    if (out.shape.length > 1) {
      throw new Error(
        `jacobian: the function must return a scalar or a vector, got ${shapeStr(out.shape)}`,
      );
    }
    const m = out.value.data.length;
    const n = leaf.value.data.length;
    const J = new Array(m);
    const seed = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      // One reverse sweep per output row, reusing the forward pass. backward()
      // reallocates every node's gradient, so the sweeps do not contaminate
      // each other.
      seed.fill(0);
      seed[i] = 1;
      out.backward(seed);
      J[i] = leaf.grad === null ? new Array(n).fill(0) : Array.from(leaf.grad);
    }
    return J;
  };
}

/**
 * Structural fingerprint of an input: which parameters, and what shape each is.
 * Keys are sorted so two maps built in different orders match. @private
 */
function signatureOf(x) {
  if (typeof x === 'number') return 's';
  if (isTensor(x)) return shapeSig(x.shape);
  if (Array.isArray(x) || x instanceof Float64Array) {
    const first = x[0];
    if (Array.isArray(first) || first instanceof Float64Array) {
      return `${x.length}x${first.length}`;
    }
    return `v${x.length}`;
  }
  return Object.keys(x).sort().map((k) => `${k}:${signatureOf(x[k])}`).join(',');
}

/** The fingerprint a plan is cached under: parameters, then inputs. @private */
function planSignature(x, feed) {
  return feed === undefined ? signatureOf(x) : `${signatureOf(x)}|${signatureOf(feed)}`;
}

/** Overwrite a leaf's storage with a fresh value of the same shape. @private */
function writeLeaf(leaf, v) {
  const d = leaf.value.data;
  if (typeof v === 'number') {
    d[0] = v;
    return;
  }
  if (isTensor(v)) {
    d.set(v.data);
    return;
  }
  const first = v[0];
  if (Array.isArray(first) || first instanceof Float64Array) {
    let o = 0;
    for (let i = 0; i < v.length; i++) {
      const row = v[i];
      for (let j = 0; j < row.length; j++) d[o++] = row[j];
    }
    return;
  }
  for (let i = 0; i < v.length; i++) d[i] = v[i];
}

/**
 * Build the reusable plan: run `f` once, then keep the graph.
 * Returns null if any node cannot be replayed, which sends the caller back to
 * the ordinary rebuild-every-time path. The root is not required to be a
 * scalar here: a forward-only replay (`compiled.value`) may return a vector or
 * a matrix, and the gradient path checks for itself. @private
 */
function buildPlan(f, x, feed) {
  const { isMap, leaves } = wrapParams(x, true);
  const inputs = wrapInputs(feed, true);
  const out = inputs === null ? f(leaves) : f(leaves, inputs);
  if (!(out instanceof Var)) requireScalarObjective(out, 'compile');

  const order = topoOrder(out);
  for (const nd of order) {
    if (nd.parents.length > 0 && !nd._recompute) return null; // a hand-built node
    nd.grad = new Float64Array(nd.value.data.length);
  }
  return { isMap, leaves, inputs, root: out, order, signature: planSignature(x, feed) };
}

/** Write new parameters and inputs into a plan's leaves and recompute. @private */
function replayForward(plan, x, feed) {
  const { isMap, leaves, inputs, order } = plan;
  if (isMap) {
    for (const k of Object.keys(leaves)) writeLeaf(leaves[k], x[k]);
  } else {
    writeLeaf(leaves, x);
  }
  if (inputs !== null) {
    // The signature matched, so every input the plan has is in `feed`.
    for (const k of Object.keys(inputs)) writeLeaf(inputs[k], feed[k]);
  }
  for (let i = 0; i < order.length; i++) {
    const r = order[i]._recompute;
    if (r !== null) r();
  }
}

/** Evaluate a built plan at new parameters, with the gradient. @private */
function runPlan(plan, x, feed) {
  const { isMap, leaves, root, order } = plan;
  replayForward(plan, x, feed);
  requireScalarObjective(root, 'compile');

  for (let i = 0; i < order.length; i++) order[i].grad.fill(0);
  root.grad[0] = 1;
  for (let i = order.length - 1; i >= 0; i--) {
    const nd = order[i];
    if (nd._backward === null) continue;
    const contribs = nd._backward(nd.grad);
    for (let k = 0; k < nd.parents.length; k++) {
      const c = contribs[k];
      if (!c) continue;
      const g = nd.parents[k].grad;
      for (let j = 0; j < g.length; j++) g[j] += c[j];
    }
  }

  let gradient;
  if (isMap) {
    gradient = {};
    for (const k of Object.keys(x)) gradient[k] = gradOf(leaves[k], x[k]);
  } else {
    gradient = gradOf(leaves, x);
  }
  return { value: root.data[0], gradient };
}

/** Evaluate a built plan at new parameters, forward only. @private */
function runPlanValue(plan, x, feed) {
  replayForward(plan, x, feed);
  return rootValue(plan.root);
}

/**
 * How many plans one compiled objective keeps, by shape. A training loop has
 * two (the batch and the partial last batch) and a `predict` a third; a
 * sampler has one. Beyond a handful, something is varying that should not be.
 * @private
 */
const MAX_PLANS = 8;

/**
 * Like {@link valueAndGrad}, but the tape is built once and replayed.
 *
 * `valueAndGrad` reconstructs the whole graph on every call: a `Var` and a
 * closure per operation, a topological sort, a fresh gradient buffer per node.
 * On a 340-observation regression that bookkeeping is 92% of the runtime — the
 * arithmetic itself is the small part. None of it changes between calls, since
 * the shapes are fixed and the sequence of operations is the same; only the
 * parameter values move. So this keeps the graph, writes the new values into
 * its leaves, and replays it. Measured on that model: 0.59 ms per gradient
 * becomes 0.024 ms.
 *
 * THE CONSTRAINT. The graph must be the same on every call. Two ways to break
 * that, both of them things you have to go out of your way to write:
 *
 *   - branching on a parameter's numeric value, by reaching into `.data`, so
 *     that different inputs take different paths through the objective;
 *   - closing over data that is mutated between calls, which the plan captured
 *     as a constant when it was built.
 *
 * A branch INSIDE an op is fine, and is the reason `relu` and `maximum` exist:
 * the kernel picks a side per element, while the graph stays put. If your
 * objective needs a genuine structural branch, use `valueAndGrad`.
 *
 * Data that changes between calls is not a constant: pass it as INPUTS, the
 * second argument. The plan writes new inputs into its leaves exactly as it
 * writes new parameters, and reads no gradient from them. A mini-batch, a
 * dropout mask, a per-fit coefficient all go this way.
 *
 * A change in a parameter's or an input's SHAPE builds another plan, and the
 * plans are kept by shape (a handful of them), so a loop that alternates a
 * full batch and a partial last batch pays for each shape once.
 *
 * @param {(x: any, inputs?: any) => Var} f - objective, as for {@link valueAndGrad}
 * @returns {(x: any, inputs?: Object) => { value: number, gradient: any }}
 *   with `.value(x, inputs)`, the forward replay alone, which returns the
 *   root's value and may be a vector or a matrix; and `.toJSON()`, the plan as
 *   data.
 *
 * @example
 * const vg = compile((p) => negLogLik(p));
 * for (const p of chain) vg(p);   // one graph, many evaluations
 *
 * @example
 * const step = compile((p, d) => loss(net(p, d.X), d.y));
 * for (const [X, y] of batches) update(p, step(p, { X, y }).gradient);
 * step.value(p, { X: Xval, y: yval });   // the validation loss, no backward sweep
 */
export function compile(f) {
  if (typeof f !== 'function') throw new Error('compile: expected a function');
  const fallback = valueAndGrad(f);
  const plans = new Map();
  let last;
  let refused = false;

  const planFor = (x, feed) => {
    checkInputs(feed, 'compile');
    const sig = planSignature(x, feed);
    let plan = plans.get(sig);
    if (plan === undefined) {
      plan = buildPlan(f, x, feed);
      if (plan === null) {
        // The objective reached the tape through something other than this
        // package's ops. Nothing is wrong with that graph, it just cannot be
        // replayed, so fall back rather than refuse to differentiate.
        refused = true;
        return null;
      }
      plans.set(sig, plan);
      if (plans.size > MAX_PLANS) plans.delete(plans.keys().next().value);
    }
    last = plan;
    return plan;
  };

  const compiled = (x, feed) => {
    if (refused) return fallback(x, feed);
    const plan = planFor(x, feed);
    return plan === null ? fallback(x, feed) : runPlan(plan, x, feed);
  };

  compiled.value = (x, feed) => {
    if (refused) return fallback.value(x, feed);
    const plan = planFor(x, feed);
    return plan === null ? fallback.value(x, feed) : runPlanValue(plan, x, feed);
  };

  /**
   * The graph as data: every node's op and static arguments, every constant's
   * values, every parameter's and input's name and shape. See
   * {@link compileFromJSON}. The graph exists only after a first call, since
   * its shapes come from the arguments, and it is the graph of the shapes
   * most recently evaluated.
   */
  compiled.toJSON = () => {
    if (refused) {
      throw new Error(
        'toJSON: this objective holds a node built outside this package\'s ops, ' +
          'which cannot be replayed or serialized.',
      );
    }
    if (last === undefined) {
      throw new Error('toJSON: call the compiled function once first, so the graph is built.');
    }
    return serializePlan(last);
  };
  return compiled;
}

/** Shape → the signature fragment `signatureOf` would produce for it. @private */
function shapeSig(shape) {
  if (shape.length === 0) return 's';
  if (shape.length === 1) return `v${shape[0]}`;
  return `${shape[0]}x${shape[1]}`;
}

/** @private */
function serializePlan(plan) {
  const { isMap, leaves, inputs, root, order } = plan;
  const index = new Map();
  const nodes = [];
  const push = (nd, entry) => {
    index.set(nd, nodes.length);
    nodes.push(entry);
  };
  // Parameters first, whether or not the objective reads them: a parameter
  // the graph never touches still needs a zero gradient reported under its
  // name, so it must survive the round trip.
  if (isMap) {
    for (const [name, v] of Object.entries(leaves)) push(v, { kind: 'param', name, shape: v.shape.slice() });
  } else {
    push(leaves, { kind: 'param', shape: leaves.shape.slice() });
  }
  // Inputs likewise: one the graph never reads is still part of the call
  // signature, and a rebuilt plan must ask for the same map.
  if (inputs !== null) {
    for (const [name, v] of Object.entries(inputs)) push(v, { kind: 'input', name, shape: v.shape.slice() });
  }
  for (const nd of order) {
    if (index.has(nd)) continue;
    if (nd.parents.length === 0) {
      push(nd, { kind: 'const', shape: nd.shape.slice(), data: Array.from(nd.value.data) });
      continue;
    }
    const entry = { kind: 'op', op: nd.spec.op, parents: nd.parents.map((p) => index.get(p)) };
    if (nd.spec.args) entry.args = nd.spec.args;
    if (nd.spec.list) entry.list = true;
    push(nd, entry);
  }
  // Version 2 adds the `input` node kind; a reader of version 1 would not
  // know what to do with one. A plan without inputs is still written as 2.
  return { version: 2, input: isMap ? 'map' : 'single', nodes, root: index.get(root) };
}

/**
 * Rebuild a compiled objective from the data {@link compile}'s `toJSON`
 * produced, on this thread or another.
 *
 * What comes back behaves like the output of `compile`, with one difference:
 * it has no objective function to fall back to, so it evaluates only at the
 * shapes it was built for and throws on any other. That is the point. A
 * worker cannot receive a closure, but it can receive this, and the data the
 * closure captured travels inside it as constant leaves; the data the closure
 * took as inputs is asked for again, by name, on every call.
 *
 * @param {Object} json - the value `compiled.toJSON()` returned
 * @returns {(x: any, inputs?: Object) => { value: number, gradient: any }}
 *   with `.value(x, inputs)` as on {@link compile}
 *
 * @example
 * const vg = compile(negLogLik);
 * vg(p0);                                   // builds the graph
 * const json = vg.toJSON();                 // structured-clonable
 * const again = compileFromJSON(json);      // in a worker, say
 * again(p1);                                // same gradient the original gives
 */
export function compileFromJSON(json) {
  if (!json || !(json.version === 1 || json.version === 2) || !Array.isArray(json.nodes)) {
    throw new Error('compileFromJSON: not a serialized plan');
  }
  const isMap = json.input === 'map';
  const vars = new Array(json.nodes.length);
  const leaves = isMap ? {} : null;
  let single = null;
  let inputs = null;
  const sigParts = [];
  const inputParts = [];

  json.nodes.forEach((n, i) => {
    if (n.kind === 'param') {
      const v = variable({ data: new Float64Array(sizeOf(n.shape)), shape: n.shape.slice() });
      vars[i] = v;
      if (isMap) {
        leaves[n.name] = v;
        sigParts.push([n.name, shapeSig(n.shape)]);
      } else {
        single = v;
        sigParts.push(['', shapeSig(n.shape)]);
      }
    } else if (n.kind === 'input') {
      const v = variable({ data: new Float64Array(sizeOf(n.shape)), shape: n.shape.slice() });
      vars[i] = v;
      if (inputs === null) inputs = {};
      inputs[n.name] = v;
      inputParts.push([n.name, shapeSig(n.shape)]);
    } else if (n.kind === 'const') {
      vars[i] = variable({ data: Float64Array.from(n.data), shape: n.shape.slice() });
    } else {
      const fn = REGISTRY[n.op];
      if (!fn) throw new Error(`compileFromJSON: unknown op "${n.op}"`);
      const parents = n.parents.map((p) => vars[p]);
      vars[i] = n.list ? fn(parents, ...(n.args ?? [])) : fn(...parents, ...(n.args ?? []));
    }
  });

  const root = vars[json.root];
  const order = topoOrder(root);
  for (const nd of order) nd.grad = new Float64Array(nd.value.data.length);
  const joined = (parts) =>
    parts.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([k, v]) => `${k}:${v}`).join(',');
  const paramSig = isMap ? joined(sigParts) : sigParts[0][1];
  const signature = inputs === null ? paramSig : `${paramSig}|${joined(inputParts)}`;
  const plan = { isMap, leaves: isMap ? leaves : single, inputs, root, order, signature };

  const check = (x, feed) => {
    checkInputs(feed, 'compileFromJSON');
    const got = planSignature(x, feed);
    if (got !== signature) {
      throw new Error(
        `compileFromJSON: this plan was built for arguments shaped ${signature}, ` +
          `got ${got}. A rebuilt plan has no objective to re-trace, so it cannot adapt.`,
      );
    }
  };
  const rebuilt = (x, feed) => {
    check(x, feed);
    return runPlan(plan, x, feed);
  };
  rebuilt.value = (x, feed) => {
    check(x, feed);
    return runPlanValue(plan, x, feed);
  };
  return rebuilt;
}
