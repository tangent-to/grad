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
 * into a leapfrog step or an L-BFGS iteration with no marshalling.
 */

import { Var, topoOrder, variable } from './tape.js';
import { shapeStr, sizeOf, toNested } from './tensor.js';
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
    !(x instanceof Float64Array) && !(x instanceof Var);
}

/**
 * Read a leaf's accumulated gradient back into the shape its input arrived in.
 * @private
 */
function gradOf(v, original) {
  if (v.grad === null) {
    // The objective never touched this parameter. A zero gradient is the honest
    // answer, and silently omitting the key would break a sampler that indexes
    // every parameter by name.
    return typeof original === 'number' ? 0 : new Array(v.value.data.length).fill(0);
  }
  const t = { data: v.grad, shape: v.shape };
  return toNested(t);
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
 * @param {(x: any) => Var} f - objective, built from this package's ops. It
 *   receives `Var`s in the same structure as the input and must return a
 *   scalar `Var`.
 * @returns {(x: any) => { value: number, gradient: any }}
 *
 * @example
 * const f = (p) => add(square(p.mu), square(p.sigma));
 * valueAndGrad(f)({ mu: 3, sigma: 4 });
 * // { value: 25, gradient: { mu: 6, sigma: 8 } }
 */
export function valueAndGrad(f) {
  if (typeof f !== 'function') throw new Error('valueAndGrad: expected a function');

  return (x) => {
    let wrapped;
    let leaves;
    if (isParamMap(x)) {
      wrapped = {};
      leaves = {};
      for (const [k, v] of Object.entries(x)) {
        leaves[k] = variable(v, `parameter "${k}"`);
        wrapped[k] = leaves[k];
      }
    } else {
      leaves = variable(x, 'parameter');
      wrapped = leaves;
    }

    const out = f(wrapped);
    requireScalarObjective(out, 'valueAndGrad');

    out.backward();

    let gradient;
    if (isParamMap(x)) {
      gradient = {};
      for (const k of Object.keys(x)) gradient[k] = gradOf(leaves[k], x[k]);
    } else {
      gradient = gradOf(leaves, x);
    }
    return { value: out.data[0], gradient };
  };
}

/**
 * Gradient only, discarding the objective's value.
 *
 * @param {(x: any) => Var} f
 * @returns {(x: any) => any} gradient, shaped like the input
 */
export function grad(f) {
  const vg = valueAndGrad(f);
  return (x) => vg(x).gradient;
}

/** Structural equality over a `{name: number|number[]}` parameter map. @private */
function sameParams(a, b) {
  if (a === undefined || b === undefined) return false;
  if (typeof a === 'number') return a === b;
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
 * it correctly rather than returning a stale gradient.
 *
 * @param {(x: any) => Var} f - objective built from this package's ops
 * @param {Object} [options]
 * @param {boolean} [options.compile=false] - build the tape once and replay it,
 *   via {@link compile}. Worth an order of magnitude on a sampler, which calls
 *   this thousands of times at the same shapes; read `compile`'s constraint
 *   before turning it on. Off by default: a static graph is an assumption about
 *   your objective, and one this package cannot check for you.
 * @returns {{ value: (x:any) => number, gradient: (x:any) => any }}
 *
 * @example
 * const { value, gradient } = valueAndGradFns((p) => logLik(p), { compile: true });
 * model.potential('y', value, gradient);
 */
export function valueAndGradFns(f, options = {}) {
  const vg = options.compile ? compile(f) : valueAndGrad(f);
  let lastInput;
  let lastResult;

  const evaluate = (x) => {
    if (lastResult !== undefined && sameParams(lastInput, x)) return lastResult;
    lastResult = vg(x);
    lastInput = copyParams(x);
    return lastResult;
  };

  return {
    value: (x) => evaluate(x).value,
    gradient: (x) => evaluate(x).gradient,
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
  if (Array.isArray(x) || x instanceof Float64Array) {
    const first = x[0];
    if (Array.isArray(first) || first instanceof Float64Array) {
      return `${x.length}x${first.length}`;
    }
    return `v${x.length}`;
  }
  return Object.keys(x).sort().map((k) => `${k}:${signatureOf(x[k])}`).join(',');
}

/** Overwrite a leaf's storage with a fresh value of the same shape. @private */
function writeLeaf(leaf, v) {
  const d = leaf.value.data;
  if (typeof v === 'number') {
    d[0] = v;
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
 * the ordinary rebuild-every-time path. @private
 */
function buildPlan(f, x) {
  const isMap = isParamMap(x);
  let wrapped;
  let leaves;
  if (isMap) {
    wrapped = {};
    leaves = {};
    for (const [k, v] of Object.entries(x)) {
      // variable() aliases a Float64Array argument rather than copying it. The
      // plan writes into its leaves on every call, so it must own their
      // storage — otherwise evaluating at new parameters would scribble over
      // the caller's array from the first call.
      leaves[k] = variable(typeof v === 'number' ? v : Array.from(v, (e) => (Array.isArray(e) || e instanceof Float64Array ? Array.from(e) : e)), `parameter "${k}"`);
      wrapped[k] = leaves[k];
    }
  } else {
    leaves = variable(typeof x === 'number' ? x : Array.from(x, (e) => (Array.isArray(e) || e instanceof Float64Array ? Array.from(e) : e)), 'parameter');
    wrapped = leaves;
  }

  const out = f(wrapped);
  requireScalarObjective(out, 'compile');

  const order = topoOrder(out);
  for (const nd of order) {
    if (nd.parents.length > 0 && !nd._recompute) return null; // a hand-built node
    nd.grad = new Float64Array(nd.value.data.length);
  }
  return { isMap, leaves, root: out, order, signature: signatureOf(x) };
}

/** Evaluate a built plan at new parameters. @private */
function runPlan(plan, x) {
  const { isMap, leaves, root, order } = plan;
  if (isMap) {
    for (const k of Object.keys(leaves)) writeLeaf(leaves[k], x[k]);
  } else {
    writeLeaf(leaves, x);
  }

  for (let i = 0; i < order.length; i++) {
    const r = order[i]._recompute;
    if (r !== null) r();
  }

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
 * A change in a parameter's SHAPE is detected and rebuilds the plan, so
 * varying dimensions cost a rebuild rather than a wrong answer.
 *
 * @param {(x: any) => Var} f - objective, as for {@link valueAndGrad}
 * @returns {(x: any) => { value: number, gradient: any }}
 *
 * @example
 * const vg = compile((p) => negLogLik(p));
 * for (const p of chain) vg(p);   // one graph, many evaluations
 */
export function compile(f) {
  if (typeof f !== 'function') throw new Error('compile: expected a function');
  const fallback = valueAndGrad(f);
  let plan;
  let refused = false;

  const compiled = (x) => {
    if (refused) return fallback(x);
    if (plan === undefined || plan.signature !== signatureOf(x)) {
      plan = buildPlan(f, x);
      if (plan === null) {
        // The objective reached the tape through something other than this
        // package's ops. Nothing is wrong with that graph, it just cannot be
        // replayed, so fall back rather than refuse to differentiate.
        refused = true;
        return fallback(x);
      }
    }
    return runPlan(plan, x);
  };

  /**
   * The graph as data: every node's op and static arguments, every constant's
   * values, every parameter's name and shape. See {@link compileFromJSON}.
   * The graph exists only after a first call, since its shapes come from the
   * input.
   */
  compiled.toJSON = () => {
    if (refused) {
      throw new Error(
        'toJSON: this objective holds a node built outside this package\'s ops, ' +
          'which cannot be replayed or serialized.',
      );
    }
    if (plan === undefined) {
      throw new Error('toJSON: call the compiled function once first, so the graph is built.');
    }
    return serializePlan(plan);
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
  const { isMap, leaves, root, order } = plan;
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
  return { version: 1, input: isMap ? 'map' : 'single', nodes, root: index.get(root) };
}

/**
 * Rebuild a compiled objective from the data {@link compile}'s `toJSON`
 * produced, on this thread or another.
 *
 * What comes back behaves like the output of `compile`, with one difference:
 * it has no objective function to fall back to, so it evaluates only at the
 * shapes it was built for and throws on any other. That is the point. A
 * worker cannot receive a closure, but it can receive this, and the data the
 * closure captured travels inside it as constant leaves.
 *
 * @param {Object} json - the value `compiled.toJSON()` returned
 * @returns {(x: any) => { value: number, gradient: any }}
 *
 * @example
 * const vg = compile(negLogLik);
 * vg(p0);                                   // builds the graph
 * const json = vg.toJSON();                 // structured-clonable
 * const again = compileFromJSON(json);      // in a worker, say
 * again(p1);                                // same gradient the original gives
 */
export function compileFromJSON(json) {
  if (!json || json.version !== 1 || !Array.isArray(json.nodes)) {
    throw new Error('compileFromJSON: not a serialized plan');
  }
  const isMap = json.input === 'map';
  const vars = new Array(json.nodes.length);
  const leaves = isMap ? {} : null;
  let single = null;
  const sigParts = [];

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
  const signature = isMap
    ? sigParts.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([k, v]) => `${k}:${v}`).join(',')
    : sigParts[0][1];
  const plan = { isMap, leaves: isMap ? leaves : single, root, order, signature };

  return (x) => {
    const got = signatureOf(x);
    if (got !== signature) {
      throw new Error(
        `compileFromJSON: this plan was built for parameters shaped ${signature}, ` +
          `got ${got}. A rebuilt plan has no objective to re-trace, so it cannot adapt.`,
      );
    }
    return runPlan(plan, x);
  };
}
