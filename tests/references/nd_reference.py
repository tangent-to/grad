"""
Reference values and gradients for grad's n-dimensional ops, from PyTorch in
float64.

Writes two fixtures next to the tests:

- fixtures/nd-cases.json  the cases: op, inputs, static arguments and a fixed
                          weight tensor. R reads these to compute its own
                          forward values (nd_reference.R).
- fixtures/nd-torch.json  for each case, the output of the op and the gradient
                          of  L = sum(weight * op(inputs))  w.r.t. every
                          differentiated input, by torch autograd.

Every array is stored flat in row-major (C) order with its shape. Run from the
package root:  .venv/bin/python tests/references/nd_reference.py
"""
import json
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(20261127)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "fixtures")


def values(shape, positive=False):
    # Distinct, asymmetric values so a swapped axis cannot pass by accident;
    # `positive` keeps a divisor well away from zero.
    x = rng.uniform(-2.0, 2.0, size=shape)
    return np.abs(x) + 0.5 if positive else x


def enc(a):
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "data": a.reshape(-1).tolist()}


cases = []


def case(name, op, inputs, args=None, nondiff=()):
    cases.append({"name": name, "op": op, "inputs": inputs, "args": args or {}, "nondiff": list(nondiff)})


# Broadcasting, one-sided, across ranks and singleton axes.
pairs = [
    ([2, 3, 4], [4]),
    ([2, 3, 4], [3, 1]),
    ([2, 3, 4], [2, 1, 4]),
    ([1, 3, 1], [2, 3, 4]),
    ([2, 3, 2, 2], [3, 1, 2]),
    ([5, 1], [5, 3]),
    ([2, 1, 3, 1], [2, 4, 3, 5]),
]
for op in ["add", "sub", "mul", "div", "maximum"]:
    for sa, sb in pairs:
        case(f"{op} {sa} {sb}", op, {"a": values(sa), "b": values(sb, positive=(op == "div"))})

# Reductions along every axis, with and without keepDims.
for shape in ([2, 3, 4], [2, 3, 2, 2]):
    for op in ["sum", "mean", "max"]:
        for axis in range(len(shape)):
            for keep in (False, True):
                case(f"{op} {shape} axis={axis} keepDims={keep}", op, {"a": values(shape)}, {"axis": axis, "keepDims": keep})
    for op in ["sum", "mean", "max"]:
        case(f"{op} {shape} whole", op, {"a": values(shape)})

# Axis permutations; no perm reverses the axes.
for shape, perm in [([3, 4], None), ([2, 3, 4], None), ([2, 3, 4], [0, 2, 1]), ([2, 3, 4], [2, 0, 1]),
                    ([2, 3, 2, 5], [3, 1, 0, 2])]:
    case(f"transpose {shape} perm={perm}", "transpose", {"a": values(shape)}, {"perm": perm})

# Products: shared weights, batched blocks, vector on the right.
for sa, sb in [([2, 3, 4], [4, 5]), ([2, 3, 4], [2, 4, 5]), ([2, 2, 3, 4], [2, 2, 4, 3]),
               ([3, 4], [4]), ([2, 3, 4], [4]), ([3, 4], [4, 2])]:
    case(f"matmul {sa} {sb}", "matmul", {"a": values(sa), "b": values(sb)})

# Structural ops.
case("slice [2,3,2,5] window", "slice", {"a": values([2, 3, 2, 5])}, {"start": [1, 0, 1, 2], "size": [1, 2, 1, 3]})
case("slice [2,3,4] one step", "slice", {"a": values([2, 3, 4])}, {"start": [0, 1, 0], "size": [2, 1, 4]})
case("concat axis 1 rank 4", "concat", {"a": values([2, 1, 2, 2]), "b": values([2, 3, 2, 2])}, {"axis": 1})
case("concat axis 2 rank 4", "concat", {"a": values([2, 3, 1, 2]), "b": values([2, 3, 2, 2])}, {"axis": 2})
case("concat axis 0 rank 3", "concat", {"a": values([1, 3, 4]), "b": values([2, 3, 4])}, {"axis": 0})
case("reshape [2,3,4] to [4,3,2]", "reshape", {"a": values([2, 3, 4])}, {"shape": [4, 3, 2]})
case("gather rank 3 table, rank 2 index", "gather",
     {"a": values([5, 2, 3]), "index": np.array([[0, 4], [2, 2]], dtype=np.float64)}, nondiff=["index"])
case("gather matrix rows, repeated", "gather",
     {"a": values([4, 3]), "index": np.array([3, 0, 3], dtype=np.float64)}, nondiff=["index"])


def run(c):
    t = {k: torch.tensor(v, requires_grad=k not in c["nondiff"]) for k, v in c["inputs"].items()}
    op, args = c["op"], c["args"]
    a = t["a"]
    if op in ("add", "sub", "mul", "div", "maximum"):
        y = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div,
             "maximum": torch.maximum}[op](a, t["b"])
    elif op in ("sum", "mean"):
        f = torch.sum if op == "sum" else torch.mean
        y = f(a) if "axis" not in args else f(a, dim=args["axis"], keepdim=args["keepDims"])
    elif op == "max":
        y = torch.amax(a) if "axis" not in args else torch.amax(a, dim=args["axis"], keepdim=args["keepDims"])
    elif op == "transpose":
        perm = args["perm"] if args["perm"] is not None else list(reversed(range(a.dim())))
        y = a.permute(*perm)
    elif op == "matmul":
        y = torch.matmul(a, t["b"])
    elif op == "slice":
        y = a
        for d, (s0, n0) in enumerate(zip(args["start"], args["size"])):
            y = y.narrow(d, s0, n0)
    elif op == "concat":
        y = torch.cat([a, t["b"]], dim=args["axis"])
    elif op == "reshape":
        y = a.reshape(args["shape"])
    elif op == "gather":
        y = a[t["index"].long()]
    else:
        raise ValueError(op)
    w = torch.tensor(values(list(y.shape)))
    (w * y).sum().backward()
    c["inputs"] = {k: enc(v) for k, v in c["inputs"].items()}
    c["weight"] = enc(w.numpy())
    grads = {k: enc(v.grad.numpy()) for k, v in t.items() if v.requires_grad}
    return {"name": c["name"], "output": enc(y.detach().numpy()), "grads": grads}


results = [run(c) for c in cases]
meta = {"generator": "nd_reference.py", "torch": torch.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0], "dtype": "float64", "order": "row-major",
        "objective": "L = sum(weight * op(inputs))"}
with open(os.path.join(OUT, "nd-cases.json"), "w") as f:
    json.dump({"meta": meta, "cases": cases}, f)
with open(os.path.join(OUT, "nd-torch.json"), "w") as f:
    json.dump({"meta": meta, "results": results}, f)
print(f"{len(cases)} cases written")
