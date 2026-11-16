"""Shape-level differential test of the Lean model against the pinned JAX.

JAX side: every case is traced with `jax.make_jaxpr` on abstract inputs, so nothing is evaluated.
For each modelled function it records the primal output shape,
the output shape of each registered JVP and transpose rule, called directly inside the trace,
the user-facing result of `jax.jvp` and `jax.vjp`, and whether tracing raises.

Lean side: the recorded results become one Lean `example` per case in `gen/shapes/`,
checked by `decide` against the shapes in the types of the model's definitions.
If JAX traces a case, the example states that the proof arguments hold and that every shape agrees.
If JAX raises, it states that a proof argument is false.

A transpose rule with several linear inputs is also called for every choice of linear inputs,
and a batching rule for every choice of batched arguments and batch axes:
the example states whether the model's rule raises, and what it returns.

A case passes if Lean accepts its example.
An expected disagreement is listed in `EXPECTED`: for those cases the example states the model's
side of the disagreement instead, so Lean still checks that the disagreement is exactly the expected one.
On failure, the failing cases are checked again assertion by assertion, and the model's shapes are printed.

Usage: `python scripts/shapes.py [--jobs N] [--only PRIM]`.
"""

import argparse
import functools
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent.parent
GEN = ROOT / "gen" / "shapes"
SIZES = (0, 1, 2, 3)
COMPARE = ("eq", "ne", "ge", "gt", "le", "lt")


def shapes(rank, sizes=SIZES):
    return [tuple(s) for s in itertools.product(sizes, repeat=rank)]


def all_shapes(max_rank=3, sizes=SIZES):
    return [s for r in range(max_rank + 1) for s in shapes(r, sizes)]


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def generate_cases():
    """All cases, as `(prim, params)` pairs. Deterministic."""
    rng = random.Random(0)
    cases = []
    for prim in ("sin", "cos", "log", "sqrt", "exp"):
        cases += [(prim, {"s": s}) for s in all_shapes()]
    cases += [("integer_pow", {"s": s, "y": y}) for s in all_shapes() for y in range(-4, 6)]
    # Helpers of rules that have no rules of their own in the model.
    for prim in ("sign", "floor", "abs"):
        cases += [(prim, {"s": s}) for s in all_shapes()]
    for prim in ("mul", "div", "max", "pow", "rem") + COMPARE:
        cases += [
            (prim, {"sx": a, "sy": b}) for r in range(4) for a in shapes(r) for b in shapes(r)
        ]
    # select_n with a boolean `which`: cases of equal or different shapes, of the same or another rank,
    # and `which` a scalar, of the shape of the first case, of another shape or of another rank.
    for r in range(4):
        ss = shapes(r)
        for s0 in ss:
            other_rank = [rng.choice(shapes((r + 1) % 4))] if s0[:1] != (0,) else []
            for s1 in [s0] + rng.sample(ss, min(2, len(ss))) + other_rank:
                whiches = {(), s0, rng.choice(ss), rng.choice(shapes((r + 1) % 4))}
                cases += [("select_n", {"sw": sw, "s0": s0, "s1": s1}) for sw in sorted(whiches)]
    # clamp: bounds that are scalars, of the shape of `x`, of another shape or of another rank.
    # The model has rules for bounds of the shape of `x`, and for scalar bounds.
    for r in range(4):
        ss = shapes(r)
        for sx in ss:
            bounds = sorted({(), sx, rng.choice(ss), rng.choice(shapes((r + 1) % 4))})
            cases += [("clamp", {"sl": sl, "s": sx, "sh": sh}) for sl in bounds for sh in bounds]
    # Polynomial: degrees 0 to 3, all shapes of rank at most 1,
    # and sizes 1 and 3 at rank 2 and 1 and 2 at rank 3.
    # At most 256 combinations of shapes per degree and rank, sampled.
    for deg in range(4):
        for r, sizes in ((0, SIZES), (1, SIZES), (2, (1, 3)), (3, (1, 2))):
            if r == 3 and deg > 1:
                continue
            combos = list(itertools.product(shapes(r, sizes), repeat=deg + 2))
            if len(combos) > 256:
                combos = rng.sample(combos, 256)
            cases += [("polynomial", {"sx": ss[0], "sc": list(ss[1:])}) for ss in combos]
    # reduce_sum: every subset of axes, and every order of it.
    for s in all_shapes():
        for k in range(len(s) + 1):
            for axes in itertools.permutations(range(len(s)), k):
                cases.append(("reduce_sum", {"s": s, "axes": list(axes)}))
    # reduce: every subset of axes in every order, and some dimensions with repetitions.
    for s in all_shapes():
        for k in range(len(s) + 1):
            for axes in itertools.permutations(range(len(s)), k):
                cases.append(("reduce", {"s": s, "dims": list(axes)}))
        if s:
            cases.append(("reduce", {"s": s, "dims": [0, 0]}))
            cases.append(("reduce", {"s": s, "dims": [len(s) - 1, 0, len(s) - 1]}))
    # broadcast_in_dim: every map `broadcast_dimensions` of rank at most 3,
    # all operand and result shapes up to rank 2, and a sample at rank 3.
    for n in range(4):
        for m in range(n + 1):
            for bd in itertools.product(range(n), repeat=m):
                pairs = [(so, sh) for sh in shapes(n) for so in shapes(m)]
                if n == 3:
                    # Half of the sample broadcasts correctly.
                    good = [
                        (so, sh)
                        for so, sh in pairs
                        if all(so[i] in (1, sh[bd[i]]) for i in range(m))
                    ]
                    pairs = rng.sample(good, min(24, len(good))) + rng.sample(pairs, 24)
                cases += [
                    ("broadcast_in_dim", {"so": so, "shape": sh, "bd": list(bd)})
                    for so, sh in pairs
                ]
    # expand_dims: operands up to rank 2 and dimensions in 0..4 without repetitions, in any order.
    for s in all_shapes(2):
        for k in range(4):
            for dims in itertools.permutations(range(5), k):
                if list(dims) == sorted(dims) or k == 2:
                    cases.append(("expand_dims", {"s": s, "dims": list(dims)}))
    # reshape without dimensions: all pairs of shapes up to rank 2, and a sample at rank 3.
    for s in all_shapes(3):
        for t in all_shapes(2):
            if len(s) < 3 or rng.random() < 0.25:
                cases.append(("reshape", {"s": s, "new": t}))
    for t in shapes(3):
        for s in rng.sample(all_shapes(2), 6):
            cases.append(("reshape", {"s": s, "new": t}))
    # _unbroadcast: all pairs of shapes of equal rank up to 2, with and without enable_checks.
    for r in range(3):
        for sa in shapes(r):
            for sx in shapes(r):
                for checks in (False, True):
                    cases.append(("unbroadcast", {"sa": sa, "sx": sx, "checks": checks}))
    # cumsum: every axis, forward and reverse.
    for s in all_shapes():
        for axis in range(len(s)):
            cases += [("cumsum", {"s": s, "axis": axis, "reverse": rev}) for rev in (False, True)]
    # pad: low and high padding in -2..2 and interior padding in 0..2,
    # every configuration at rank at most 1, and a sample at ranks 2 and 3.
    for r in range(4):
        configs = list(
            itertools.product(
                shapes(r),
                *[list(itertools.product(range(-2, 3), range(-2, 3), range(3)))] * r,
            )
        )
        if r >= 2:
            configs = rng.sample(configs, 300 if r == 2 else 60)
        cases += [("pad", {"s": s, "sv": (), "config": list(config)}) for s, *config in configs]
    # slice: sizes in 0..5, start and limit indices in 0..6 and strides in 0..3,
    # every combination at rank at most 1, and a sample at rank 2, half of which JAX accepts.
    for r in range(3):
        params = list(
            itertools.product(
                shapes(r, range(6)),
                *[list(itertools.product(range(7), range(7), range(4)))] * r,
            )
        )
        if r == 2:
            good = [
                (s, *axes)
                for s, *axes in params
                if all(a <= b <= d and k > 0 for d, (a, b, k) in zip(s, axes))
            ]
            params = rng.sample(good, 200) + rng.sample(params, 200)
        for s, *axes in params:
            start, limit, strides = ([a[i] for a in axes] for i in range(3))
            cases.append(("slice", {"s": s, "start": start, "limit": limit, "strides": strides}))
    # _maybe_broadcast: all pairs of shapes of equal rank up to 2.
    for r in range(3):
        for sx in shapes(r):
            for target in shapes(r):
                cases.append(("maybe_broadcast", {"sx": sx, "target": target}))
    # Batching rule of clamp: operands up to rank 2, bounds that are scalars or of the shape of `x`,
    # every choice of batched arguments and batch axes, and batches of size 2 and 4. Sampled.
    batch_cases = []
    for s in all_shapes(2, (1, 2, 3)):
        for sl, sh in itertools.product({(), s}, repeat=2):
            for batched in itertools.product((False, True), repeat=3):
                if not any(batched):
                    continue
                axes = [range(len(a) + 1) if b else [None] for a, b in zip((sl, s, sh), batched)]
                for bdims in itertools.product(*axes):
                    for size in (2, 4):
                        ins = [
                            a if d is None else (*a[:d], size, *a[d:])
                            for a, d in zip((sl, s, sh), bdims)
                        ]
                        batch_cases.append(("clamp_batch", {"shapes": ins, "bdims": list(bdims)}))
    cases += rng.sample(batch_cases, 1500)
    # Drop repeated cases, keeping the order.
    unique = {json.dumps(c, sort_keys=True): c for c in cases}
    return list(unique.values())


# ---------------------------------------------------------------------------
# JAX side
# ---------------------------------------------------------------------------


def jax_record(case):
    """Trace one case on the pinned JAX. Returns a dict from observation names to results.

    A result is `{"shape": [...]}`, `{"shapes": [[...], ...]}` or `{"error": "..."}`.
    The observation `primal` is always present.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax._src import config, core
    from jax._src.interpreters import ad
    from jax._src.lax import lax as L

    prim, p = case

    def sds(s):
        if isinstance(s, jax.ShapeDtypeStruct):
            return s
        return jax.ShapeDtypeStruct(tuple(s), jnp.float32)

    def bools(s):
        return jax.ShapeDtypeStruct(tuple(s), jnp.bool_)

    def trace(fn, *ins):
        try:
            closed = jax.make_jaxpr(fn)(*map(sds, ins))
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()
            return {"error": f"{type(e).__name__}: {msg[0] if msg else ''}"}
        outs = [list(a.shape) for a in closed.out_avals]
        return {"shape": outs[0]} if len(outs) == 1 else {"shapes": outs}

    def params_of(fn, primitive, *ins):
        jaxpr = jax.make_jaxpr(fn)(*map(sds, ins)).jaxpr
        return next(e.params for e in jaxpr.eqns if e.primitive is primitive)

    def undefined(s):
        return ad.UndefinedPrimal(core.ShapedArray(tuple(s), jnp.float32))

    def jvp_rules(primitive):
        # `ad.defjvp` registers a partial application of `ad.standard_jvp` to the rules.
        jvp = ad.primitive_jvps[primitive]
        assert isinstance(jvp, functools.partial)
        return jvp.args[0]

    rec = {}

    def unary(f, primitive, s, ans=False, **kw):
        rec["primal"] = trace(lambda x: f(x, **kw), s)
        if "error" in rec["primal"]:
            return
        params = params_of(lambda x: f(x, **kw), primitive, s)
        (rule,) = jvp_rules(primitive)
        if ans:
            # `ad.defjvp2` passes the primal result `ans` after the tangent.
            rec["jvp"] = trace(lambda g, x: rule(g, f(x, **kw), x, **params), s, s)
        else:
            rec["jvp"] = trace(lambda g, x: rule(g, x, **params), s, s)
        rec["user_jvp"] = trace(lambda x, t: jax.jvp(lambda x: f(x, **kw), (x,), (t,))[1], s, s)
        rec["user_vjp"] = trace(
            lambda x, ct: jax.vjp(lambda x: f(x, **kw), x)[1](ct), s, rec["primal"]["shape"]
        )

    if prim in ("sin", "cos", "log", "sqrt", "exp"):
        unary(getattr(lax, prim), getattr(L, f"{prim}_p"), p["s"], ans=prim in ("sqrt", "exp"))
    elif prim == "integer_pow":
        unary(lax.integer_pow, L.integer_pow_p, p["s"], y=p["y"])
    elif prim in ("sign", "floor", "abs"):
        rec["primal"] = trace(getattr(lax, prim), p["s"])
    elif prim in ("mul", "div", "max", "pow", "rem"):
        op, primitive = getattr(lax, prim), getattr(L, f"{prim}_p")
        sx, sy = p["sx"], p["sy"]
        rec["primal"] = trace(op, sx, sy)
        if "error" not in rec["primal"]:
            out = rec["primal"]["shape"]
            params = params_of(op, primitive, sx, sy)
            left, right = jvp_rules(primitive)
            if prim == "pow":
                # `ad.defjvp2` passes the primal result `ans` after the tangent.
                rec["jvp_left"] = trace(
                    lambda g, x, y: left(g, op(x, y), x, y, **params), sx, sx, sy
                )
                rec["jvp_right"] = trace(
                    lambda g, x, y: right(g, op(x, y), x, y, **params), sy, sx, sy
                )
            else:
                rec["jvp_left"] = trace(lambda g, x, y: left(g, x, y, **params), sx, sx, sy)
                rec["jvp_right"] = trace(lambda g, x, y: right(g, x, y, **params), sy, sx, sy)
            if prim == "mul":
                tr = ad.primitive_transposes[primitive]
                rec["transpose_left"] = trace(
                    lambda ct, y: tr(ct, undefined(sx), y, **params)[0], out, sy
                )
                rec["transpose_right"] = trace(
                    lambda ct, x: tr(ct, x, undefined(sy), **params)[1], out, sx
                )
            rec["user_jvp"] = trace(
                lambda x, y, tx, ty: jax.jvp(op, (x, y), (tx, ty))[1], sx, sy, sx, sy
            )
            rec["user_vjp"] = trace(lambda x, y, ct: jax.vjp(op, x, y)[1](ct), sx, sy, out)
    elif prim in COMPARE:
        # The rule is the zero tangent of `ad.defjvp_zero`, with dtype float0 and the output shape.
        f = getattr(lax, prim)
        sx, sy = p["sx"], p["sy"]
        rec["primal"] = trace(f, sx, sy)
        if "error" not in rec["primal"]:
            rec["user_jvp"] = trace(
                lambda x, y, tx, ty: jax.jvp(f, (x, y), (tx, ty))[1], sx, sy, sx, sy
            )
    elif prim == "select_n":
        sw, s0, s1 = bools(p["sw"]), p["s0"], p["s1"]
        rec["primal"] = trace(lax.select_n, sw, s0, s1)
        if "error" not in rec["primal"]:
            out = rec["primal"]["shape"]
            jvp = ad.primitive_jvps[L.select_n_p]
            tr = ad.primitive_transposes[L.select_n_p]
            rec["jvp"] = trace(
                lambda w, a, b, ga, gb: jvp((w, a, b), (None, ga, gb))[1], sw, s0, s1, s0, s1
            )
            rec["transpose0"] = trace(
                lambda ct, w: tr(ct, w, undefined(s0), undefined(s1))[1], out, sw
            )
            rec["transpose1"] = trace(
                lambda ct, w: tr(ct, w, undefined(s0), undefined(s1))[2], out, sw
            )
            rec["user_jvp"] = trace(
                lambda w, a, b, ta, tb: jax.jvp(
                    lambda a, b: lax.select_n(w, a, b), (a, b), (ta, tb)
                )[1],
                sw,
                s0,
                s1,
                s0,
                s1,
            )
            rec["user_vjp"] = trace(
                lambda w, a, b, ct: jax.vjp(lambda a, b: lax.select_n(w, a, b), a, b)[1](ct),
                sw,
                s0,
                s1,
                out,
            )
    elif prim == "clamp":
        sl, sx, sh = p["sl"], p["s"], p["sh"]
        rec["primal"] = trace(lax.clamp, sl, sx, sh)
        if "error" not in rec["primal"] and tuple(sl) == tuple(sx) == tuple(sh):
            out = rec["primal"]["shape"]
            rules = jvp_rules(L.clamp_p)
            for name, rule, sg in zip(("jvp_min", "jvp_operand", "jvp_max"), rules, (sl, sx, sh)):
                rec[name] = trace(
                    lambda g, lo, x, hi, rule=rule: rule(g, lo, x, hi), sg, sl, sx, sh
                )
            rec["user_jvp"] = trace(
                lambda lo, x, hi, tl, tx, th: jax.jvp(lax.clamp, (lo, x, hi), (tl, tx, th))[1],
                sl,
                sx,
                sh,
                sl,
                sx,
                sh,
            )
            rec["user_vjp"] = trace(
                lambda lo, x, hi, ct: jax.vjp(lax.clamp, lo, x, hi)[1](ct), sl, sx, sh, out
            )
        if "error" not in rec["primal"] and tuple(sl) == tuple(sh) == ():
            # Scalar bounds: the rules for `min` and `max` raise unless `x` is a scalar.
            rule_min, _, rule_max = jvp_rules(L.clamp_p)
            rec["jvp_min_scalar"] = trace(rule_min, sl, sl, sx, sh)
            rec["jvp_max_scalar"] = trace(rule_max, sh, sl, sx, sh)
    elif prim == "polynomial":
        sx, sc = p["sx"], p["sc"]
        poly = lambda x, *cs: lax.polynomial(x, list(cs))
        rec["primal"] = trace(poly, sx, *sc)
        if "error" not in rec["primal"] and len(sc) >= 2:
            params = params_of(poly, L.polynomial_p, sx, *sc)
            rule = ad.primitive_jvps[L.polynomial_p]

            def jvp(dx, x, *cs):
                zeros = [ad.Zero(core.typeof(c).to_tangent_aval()) for c in cs]
                return rule((x, *cs), (dx, *zeros), **params)[1]

            rec["jvp"] = trace(jvp, sx, sx, *sc)
            rec["user_jvp"] = trace(
                lambda x, t, *cs: jax.jvp(lambda x: poly(x, *cs), (x,), (t,))[1], sx, sx, *sc
            )
            rec["user_vjp"] = trace(
                lambda x, ct, *cs: jax.vjp(lambda x: poly(x, *cs), x)[1](ct),
                sx,
                rec["primal"]["shape"],
                *sc,
            )
    elif prim in ("reduce_sum", "broadcast_in_dim", "cumsum"):
        if prim == "reduce_sum":
            s = p["s"]
            f = lambda x: lax.reduce_sum(x, p["axes"])
            primitive = L.reduce_sum_p
        elif prim == "cumsum":
            from jax._src.lax.control_flow import loops

            s = p["s"]
            f = lambda x: lax.cumsum(x, p["axis"], p["reverse"])
            primitive = loops.cumsum_p
        else:
            s = p["so"]
            f = lambda x: lax.broadcast_in_dim(x, p["shape"], p["bd"])
            primitive = L.broadcast_in_dim_p
        rec["primal"] = trace(f, s)
        if "error" not in rec["primal"]:
            out = rec["primal"]["shape"]
            if prim in ("reduce_sum", "cumsum"):
                params = params_of(f, primitive, s)
            else:
                # The forwarding rule may drop the equation, so the parameters are built here.
                params = {
                    "shape": tuple(p["shape"]),
                    "broadcast_dimensions": tuple(p["bd"]),
                    "sharding": None,
                }
                rec["fwd"] = {"forwarded": not jax.make_jaxpr(f)(sds(s)).eqns}
            jvp = ad.primitive_jvps[primitive]
            tr = ad.primitive_transposes[primitive]
            rec["jvp"] = trace(lambda g, x: jvp((x,), (g,), **params)[1], s, s)
            rec["transpose"] = trace(lambda ct: tr(ct, undefined(s), **params)[0], out)
            rec["user_jvp"] = trace(lambda x, t: jax.jvp(f, (x,), (t,))[1], s, s)
            rec["user_vjp"] = trace(lambda x, ct: jax.vjp(f, x)[1](ct), s, out)
    elif prim == "reduce":
        s = p["s"]
        # A non-commutative monoid, so that `lax.reduce` binds `reduce_p`.
        op = lambda x, y: lax.select(y == 1.0, x, y)
        f = lambda x: lax.reduce(x, 1.0, op, p["dims"])
        rec["primal"] = trace(f, s)
        if "error" not in rec["primal"]:
            out = rec["primal"]["shape"]
            params = params_of(f, L.reduce_p, s)
            jvp = ad.primitive_jvps[L.reduce_p]
            init_dot = ad.Zero(core.ShapedArray((), jnp.float32))
            rec["jvp"] = trace(
                lambda g, x: jvp((x, jnp.float32(1.0)), (g, init_dot), **params)[1][0], s, s
            )
            rec["user_jvp"] = trace(lambda x, t: jax.jvp(f, (x,), (t,))[1], s, s)
            rec["user_vjp"] = trace(lambda x, ct: jax.vjp(f, x)[1](ct), s, out)
    elif prim == "pad":
        s, sv, config = p["s"], p["sv"], [tuple(c) for c in p["config"]]
        f = lambda x, v: lax.pad(x, v, config)
        rec["primal"] = trace(f, s, sv)
        if "error" not in rec["primal"]:
            out = rec["primal"]["shape"]
            params = {"padding_config": tuple(config)}
            jvp = ad.primitive_jvps[L.pad_p]
            tr = ad.primitive_transposes[L.pad_p]
            rec["jvp"] = trace(
                lambda gx, gv, x, v: jvp((x, v), (gx, gv), **params)[1], s, sv, s, sv
            )
            # Both arguments are linear inputs.
            rec["transpose_operand"] = trace(
                lambda ct: tr(ct, undefined(s), undefined(sv), **params)[0], out
            )
            rec["transpose_padding_value"] = trace(
                lambda ct: tr(ct, undefined(s), undefined(sv), **params)[1], out
            )
            # The transpose rule for every choice of linear inputs: "1" marks a linear input.
            partial_cts: dict[str, list[bool] | None] = {}
            rec["partial"] = partial_cts
            for lin in ("11", "10", "01"):

                def partial_rule(ct, x, v, lin=lin):
                    cts = tr(
                        ct,
                        undefined(s) if lin[0] == "1" else x,
                        undefined(sv) if lin[1] == "1" else v,
                        **params,
                    )
                    partial_cts[lin] = [c is not None for c in cts]
                    return [c for c in cts if c is not None]

                if "error" in trace(partial_rule, out, s, sv):
                    partial_cts[lin] = None
            rec["user_jvp"] = trace(
                lambda x, v, tx, tv: jax.jvp(f, (x, v), (tx, tv))[1], s, sv, s, sv
            )
            rec["user_vjp"] = trace(lambda x, v, ct: jax.vjp(f, x, v)[1](ct), s, sv, out)
    elif prim == "slice":
        from jax._src.lax import slicing

        s = p["s"]
        f = lambda x: lax.slice(x, p["start"], p["limit"], p["strides"])
        rec["primal"] = trace(f, s)
        if "error" not in rec["primal"]:
            out = rec["primal"]["shape"]
            params = params_of(f, slicing.slice_p, s)
            jvp = ad.primitive_jvps[slicing.slice_p]
            rule = ad.fancy_transposes[slicing.slice_p]

            def transpose(ct):
                # The rule adds the cotangent of the operand to an accumulator.
                accum = ad.ValAccum(core.ShapedArray(tuple(s), jnp.float32))
                rule(ct, accum, **params)
                return accum.freeze()

            rec["jvp"] = trace(lambda g, x: jvp((x,), (g,), **params)[1], s, s)
            rec["transpose"] = trace(transpose, out)
            rec["user_jvp"] = trace(lambda x, t: jax.jvp(f, (x,), (t,))[1], s, s)
            rec["user_vjp"] = trace(lambda x, ct: jax.vjp(f, x)[1](ct), s, out)
    elif prim == "expand_dims":
        rec["primal"] = trace(lambda x: lax.expand_dims(x, p["dims"]), p["s"])
    elif prim == "reshape":
        rec["primal"] = trace(lambda x: lax.reshape(x, p["new"]), p["s"])
    elif prim == "maybe_broadcast":
        rec["primal"] = trace(lambda x: L._maybe_broadcast(tuple(p["target"]), x, None), p["sx"])
    elif prim == "unbroadcast":
        aval = core.ShapedArray(tuple(p["sa"]), jnp.float32)
        with config.enable_checks(p["checks"]):
            rec["primal"] = trace(lambda x: L._unbroadcast(aval, x), p["sx"])
    elif prim == "clamp_batch":
        bdims = tuple(p["bdims"])
        rec["primal"] = {"shape": []}
        out_bdim = []

        def rule(lo, x, hi):
            out, bdim = L._clamp_batch_rule((lo, x, hi), bdims)
            out_bdim.append(bdim)
            return out

        res = trace(rule, *p["shapes"])
        user = trace(jax.vmap(lax.clamp, in_axes=bdims), *p["shapes"])
        rec["batch"] = {
            "result": None if "error" in res else [res["shape"], int(out_bdim[0])],
            # `vmap` moves the batch axis of the result to the front.
            "user": user.get("shape"),
            "error": res.get("error"),
        }
    else:
        raise ValueError(prim)
    return rec


def jax_records(cases, jobs):
    chunk = max(1, len(cases) // (jobs * 8))
    with ProcessPoolExecutor(jobs) as pool:
        return list(pool.map(jax_record, cases, chunksize=chunk))


# ---------------------------------------------------------------------------
# Lean side
# ---------------------------------------------------------------------------


# Lists with an explicit element type: numerals of unknown type make elaboration several times slower.
def lshape(s):
    return f"(sh [{', '.join(map(str, s))}])"


def llist(s):
    return f"([{', '.join(map(str, s))}] : List ℕ)"


def lfinset(xs, ty):
    return (
        f"({{{', '.join(map(str, sorted(set(xs))))}}} : Finset {ty})"
        if xs
        else f"(∅ : Finset {ty})"
    )


def by(*unfold):
    return f"(by unfold {' '.join(unfold)}; decide)" if unfold else "(by decide)"


def lean_spec(prim, p):
    """The Lean side of a case.

    Returns `(args, terms, vjp)`:
    `args` are the proof arguments of the modelled function as `(proposition, unfold)` pairs,
    `terms` maps observation names to Lean terms whose type contains the model's shape,
    and `vjp` lists the pairs of a term and a term or shape that must have the same shape
    for reverse mode to succeed.
    The terms contain `PROOF i` for the proof of argument `i`.
    """
    if prim in ("sin", "cos", "log", "sqrt", "exp"):
        S, f = lshape(p["s"]), f"Lax.{prim}"
        primal = f"{f} (0 : Tensor (Index {S}))"
        # The rules of `ad.defjvp2` also take the primal result.
        rule_args = "0 0 0" if prim in ("sqrt", "exp") else "0 0"
        jvp = f"{prim}Jvp (s := {S}) {rule_args}"
        return [], {"primal": primal, "jvp": jvp}, [("jvp", "primal")]
    if prim == "integer_pow":
        S, y = lshape(p["s"]), f"({p['y']})"
        primal = f"integerPow {y} (0 : Tensor (Index {S}))"
        jvp = f"integerPowJvp (s := {S}) {y} 0 0"
        return [], {"primal": primal, "jvp": jvp}, [("jvp", "primal")]
    if prim in ("sign", "floor", "abs"):
        return [], {"primal": f"Lax.{prim} (0 : Tensor (Index {lshape(p['s'])}))"}, []
    if prim in ("mul", "div", "max", "pow", "rem"):
        SX, SY = lshape(p["sx"]), lshape(p["sy"])
        imp = f"(sx := {SX}) (sy := {SY}) PROOF0"
        name = prim
        primal = f"Lax.{name} {imp} 0 0"
        # The rules of `ad.defjvp2` also take the primal result.
        rule_args = "0 0 0 0" if prim == "pow" else "0 0 0"
        terms = {
            "primal": primal,
            "jvp_left": f"{name}JvpLeft {imp} {rule_args}",
            "jvp_right": f"{name}JvpRight {imp} {rule_args}",
        }
        vjp = [("jvp_left", "primal"), ("jvp_right", "primal")]
        if prim == "mul":
            terms["transpose_left"] = f"mulTransposeLeft {imp} 0 0"
            terms["transpose_right"] = f"mulTransposeRight {imp} 0 0"
            vjp += [("transpose_left", p["sx"]), ("transpose_right", p["sy"])]
        return [(f"Broadcastable {SX} {SY}", ("Broadcastable",))], terms, vjp
    if prim in COMPARE:
        SX, SY = lshape(p["sx"]), lshape(p["sy"])
        primal = f"Lax.{prim} (sx := {SX}) (sy := {SY}) PROOF0 0 0"
        return [(f"Broadcastable {SX} {SY}", ("Broadcastable",))], {"primal": primal}, []
    if prim == "select_n":
        SW, S0, S1 = lshape(p["sw"]), lshape(p["s0"]), lshape(p["s1"])
        which = f"(fun _ => true : Index {SW} → Bool)"
        imp = f"(sw := {SW}) (s0 := {S0}) (s1 := {S1}) PROOF0 PROOF1 {which}"
        tr = f"(sw := {SW}) (s0 := {S0}) PROOF1 {which} 0"
        terms = {
            "primal": f"selectN {imp} 0 0",
            "jvp": f"selectNJvp {imp} 0 0 0 0",
            "transpose0": f"selectNTranspose0 {tr}",
            "transpose1": f"selectNTranspose1 {tr}",
        }
        args = [
            (f"List.ofFn {S1} = List.ofFn {S0}", ()),
            (f"WhichShape {SW} {S0}", ("WhichShape",)),
        ]
        return args, terms, [("jvp", "primal"), ("transpose0", p["s0"]), ("transpose1", p["s1"])]
    if prim == "clamp":
        SL, S, SH = lshape(p["sl"]), lshape(p["s"]), lshape(p["sh"])
        primal = f"Lax.clamp (sl := {SL}) (sh := {SH}) (s := {S}) PROOF0 PROOF1 0 0 0"
        terms = {
            "primal": primal,
            "jvp_min": f"clampJvpMin (s := {S}) 0 0 0 0",
            "jvp_operand": f"clampJvpOperand (s := {S}) 0 0 0 0",
            "jvp_max": f"clampJvpMax (s := {S}) 0 0 0 0",
        }
        args: list[tuple[str, tuple[str, ...]]] = [
            (f"WhichShape {SL} {S}", ("WhichShape",)),
            (f"WhichShape {SH} {S}", ("WhichShape",)),
        ]
        if tuple(p["sl"]) == tuple(p["sh"]) == ():
            imp = f"(sl := {SL}) (sh := {SH}) (s := {S})"
            terms["jvp_min_scalar"] = f"clampJvpMinScalar {imp} PROOF2 0 0 0 0"
            terms["jvp_max_scalar"] = f"clampJvpMaxScalar {imp} PROOF3 0 0 0 0"
            args += [
                (f"List.ofFn {SL} = List.ofFn {S}", ()),
                (f"List.ofFn {SH} = List.ofFn {S}", ()),
            ]
        return (
            args,
            terms,
            [("jvp_min", "primal"), ("jvp_operand", "primal"), ("jvp_max", "primal")],
        )
    if prim == "polynomial":
        SX = lshape(p["sx"])
        n, k = len(p["sx"]), len(p["sc"])
        SC = f"(![{', '.join(map(lshape, p['sc']))}] : Fin {k} → Shape {n})"
        imp = f"(sx := {SX}) (sc := {SC}) PROOF0"
        primal = f"Lax.polynomial {imp} 0 (fun _ => 0)"
        terms = {"primal": primal}
        vjp = []
        if k >= 2:
            terms["jvp"] = f"polynomialJvp {imp} (fun _ => 0) 0 0"
            vjp = [("jvp", "primal")]
        arg = f"BroadcastableAll ({SX} :: List.ofFn {SC})"
        return [(arg, ("BroadcastableAll", "Broadcastable"))], terms, vjp
    if prim == "reduce_sum":
        S, A = lshape(p["s"]), lfinset(p["axes"], f"(Fin {len(p['s'])})")
        primal = f"reduceSum (s := {S}) {A} 0"
        terms = {
            "primal": primal,
            "jvp": f"reduceSumJvp (s := {S}) {A} 0 0",
            "transpose": f"reduceSumTranspose (s := {S}) {A} 0",
        }
        return [], terms, [("jvp", "primal"), ("transpose", p["s"])]
    if prim == "cumsum":
        S = lshape(p["s"])
        imp = f"(s := {S}) ({p['axis']} : Fin {len(p['s'])}) {str(p['reverse']).lower()}"
        terms = {
            "primal": f"Lax.cumsum {imp} 0",
            "jvp": f"cumsumJvp {imp} 0 0",
            "transpose": f"cumsumTranspose {imp} 0",
        }
        return [], terms, [("jvp", "primal"), ("transpose", p["s"])]
    if prim == "reduce":
        n = len(p["s"])
        S, D = lshape(p["s"]), f"([{', '.join(map(str, p['dims']))}] : List (Fin {n}))"
        primal = f"Lax.reduce (s := {S}) lastNotOne 1 {D} PROOF0 0"
        terms = {"primal": primal, "jvp": f"reduceJvp (s := {S}) lastNotOne 1 {D} PROOF0 0 0"}
        return [(f"List.Nodup {D}", ())], terms, [("jvp", "primal")]
    if prim == "broadcast_in_dim":
        n = len(p["shape"])
        SO, SH = lshape(p["so"]), lshape(p["shape"])
        BD = f"(dims {n} [{', '.join(map(str, p['bd']))}])"
        primal = f"broadcastInDim (so := {SO}) {SH} {BD} PROOF0 PROOF1 0"
        terms = {
            "primal": primal,
            "jvp": f"broadcastInDimJvp (so := {SO}) {SH} {BD} PROOF0 PROOF1 0 0",
            "transpose": f"broadcastInDimTranspose {SO} {SH} {BD} PROOF1 0",
        }
        args = [
            (f"BroadcastsTo {SO} ({SH} ∘ {BD})", ("BroadcastsTo",)),
            (f"Function.Injective {BD}", ()),
        ]
        return args, terms, [("jvp", "primal"), ("transpose", p["so"])]
    if prim == "pad":
        n = len(p["s"])
        S = lshape(p["s"])
        lo, hi, interior = (
            f"(![{', '.join(str(c[i]) for c in p['config'])}] : Fin {n} → {ty})"
            for i, ty in enumerate("ℤℤℕ")
        )
        imp = f"{lo} {hi} {interior} PROOF0"
        terms = {
            "primal": f"Lax.pad (s := {S}) (sv := {lshape(p['sv'])}) {imp} 0 0",
            "jvp": f"padJvp (s := {S}) (sv := {lshape(p['sv'])}) {imp} 0 0 0 0",
            "transpose_operand": f"padTransposeOperand {S} {imp} 0",
            "transpose_padding_value": f"padTransposePaddingValue {S} {imp} 0",
        }
        arg = f"∀ d, 0 ≤ {lo} d + {hi} d + dilateDim ({S} d) ({interior} d + 1)"
        vjp = [
            ("jvp", "primal"),
            ("transpose_operand", p["s"]),
            ("transpose_padding_value", p["sv"]),
        ]
        return [(arg, ())], terms, vjp
    if prim == "slice":
        n = len(p["s"])
        S = lshape(p["s"])
        start, limit, strides = (
            f"(![{', '.join(map(str, p[k]))}] : Fin {n} → ℕ)" for k in ("start", "limit", "strides")
        )
        imp = f"{start} {limit} {strides}"
        terms = {
            "primal": f"Lax.slice (s := {S}) {imp} PROOF0 PROOF1 PROOF2 0",
            "jvp": f"sliceJvp (s := {S}) {imp} PROOF0 PROOF1 PROOF2 0 0",
            "transpose": f"sliceTranspose {S} {imp} PROOF2 0",
        }
        args = [
            (f"∀ d, {limit} d ≤ {S} d", ()),
            (f"∀ d, {start} d ≤ {limit} d", ()),
            (f"∀ d, 0 < {strides} d", ()),
        ]
        return args, terms, [("jvp", "primal"), ("transpose", p["s"])]
    if prim == "expand_dims":
        S, D = lshape(p["s"]), lfinset(p["dims"], "ℕ")
        arg = f"∀ i ∈ {D}, i < {len(p['s'])} + {D}.card"
        return [(arg, ())], {"primal": f"expandDims (s := {S}) {D} PROOF0 0"}, []
    if prim == "reshape":
        S, T = lshape(p["s"]), lshape(p["new"])
        arg = f"∏ d, {S} d = ∏ d, {T} d"
        return [(arg, ())], {"primal": f"Lax.reshape (s := {S}) {T} PROOF0 0"}, []
    if prim == "maybe_broadcast":
        SX, T = lshape(p["sx"]), lshape(p["target"])
        arg = f"∏ d, {SX} d = ∏ a, ({SX} ∘ keptAxes {{d | {SX} d ≠ {T} d}}) a"
        primal = f"maybeBroadcast (sx := {SX}) {T} PROOF0 0"
        return [(arg, ())], {"primal": primal}, []
    if prim == "unbroadcast":
        SA, SX = lshape(p["sa"]), lshape(p["sx"])
        arg = f"BroadcastsTo {SA} {SX}"
        return [(arg, ("BroadcastsTo",))], {"primal": f"unbroadcast (sx := {SX}) {SA} PROOF0 0"}, []
    raise ValueError(prim)


# Forwarding rules: the Lean term that is `true` if the rule replaces the equation by its operand.
# JAX forwards exactly when the traced jaxpr has no equations.
FWD = {
    "broadcast_in_dim": lambda p: (
        f"broadcastInDimFwd {lshape(p['so'])} {lshape(p['shape'])} "
        f"(dims {len(p['shape'])} [{', '.join(map(str, p['bd']))}])"
    ),
}


def pad_partial(p, lin):
    n = len(p["s"])
    lo, hi, interior = (
        f"(![{', '.join(str(c[i]) for c in p['config'])}] : Fin {n} → {ty})"
        for i, ty in enumerate("ℤℤℕ")
    )
    x, v = ("none" if c == "1" else "(some 0)" for c in lin)
    return (
        f"(padTranspose (sv := {lshape(p['sv'])}) {lshape(p['s'])} {lo} {hi} {interior} PROOF0 0 "
        f"{x} {v}).map (fun r => [r.1.isSome, r.2.isSome])"
    )


# Transpose rules for every choice of linear inputs: the Lean term that is `none` if the rule raises,
# and otherwise lists which arguments get a cotangent.
PARTIAL = {"pad": pad_partial}


def larr(s):
    return f"(⟨sh [{', '.join(map(str, s))}], 0⟩ : Arr)"


def lbatched(s, bdim):
    return f"⟨{larr(s)}, {'none' if bdim is None else f'some {bdim}'}⟩"


# Batching rules: the Lean term that is `none` if the rule raises,
# and otherwise the shape and the batch axis of its result.
BATCH = {
    "clamp_batch": lambda p: (
        f"(clampBatch {' '.join(lbatched(s, d) for s, d in zip(p['shapes'], p['bdims']))})"
        ".map (fun y => (y.val.shapeList, y.bdim))"
    ),
}


def fill(term, args):
    for i, (_, unfold) in enumerate(args):
        term = term.replace(f"PROOF{i}", by(*unfold))
    return term


def shape_eq(term, args, shape):
    return f"shapeOf ({fill(term, args)}) = {llist(shape)}"


def neg_args(args):
    props = " ∧ ".join(f"({a})" for a, _ in args)
    unfold = list(dict.fromkeys(u for _, us in args for u in us))
    tac = f"unfold {' '.join(unfold)}; decide" if unfold else "decide"
    return f"¬ ({props})", tac


# Expected disagreements: JAX traces, and a proof argument of the model is false.
# Each entry gives the reason, the cases it covers, and the shape argument of the model's result type.
# For these cases the example states that the proof argument is false
# and that JAX returns the shape that the model's result type would have.
EXPECTED = {
    "reshape-size": (
        (
            "`_reshape_shape_rule` does not check that the sizes agree; "
            "it is a constraint of StableHLO `reshape`, a proof argument in the model"
        ),
        lambda prim, p, rec: (
            prim == "reshape"
            and "error" not in rec["primal"]
            and math.prod(p["s"]) != math.prod(p["new"])
        ),
        lambda p: lshape(p["new"]),
    ),
    "reduce-unique": (
        (
            "`_reduce_shape_rule` does not check that `dimensions` has no repetitions; "
            "it is constraint (C5) of StableHLO `reduce`, a proof argument in the model"
        ),
        lambda prim, p, rec: (
            prim == "reduce"
            and "error" not in rec["primal"]
            and len(set(p["dims"])) != len(p["dims"])
        ),
        lambda p: (
            f"({lshape(p['s'])} ∘ keptAxes "
            f"([{', '.join(map(str, p['dims']))}] : List (Fin {len(p['s'])})).toFinset)"
        ),
    ),
    "maybe-broadcast-size": (
        (
            "`_maybe_broadcast` drops axes with `reshape`, whose shape rule does not check "
            "that the sizes agree; it is a constraint of StableHLO `reshape`, a proof argument in the model"
        ),
        lambda prim, p, rec: (
            prim == "maybe_broadcast"
            and "error" not in rec["primal"]
            and math.prod(p["sx"]) != math.prod(a for a, b in zip(p["sx"], p["target"]) if a == b)
        ),
        lambda p: lshape(p["target"]),
    ),
    "unbroadcast-checks": (
        (
            "the assertion of `_unbroadcast` only runs under `jax_enable_checks`, "
            "a proof argument in the model"
        ),
        lambda prim, p, rec: (
            prim == "unbroadcast"
            and not p["checks"]
            and "error" not in rec["primal"]
            and not all(a in (1, b) for a, b in zip(p["sa"], p["sx"]))
        ),
        lambda p: lshape(p["sa"]),
    ),
}

# The inputs that `jax.vjp` differentiates, whose shapes the cotangents must have.
VJP_INPUTS = {
    "sin": ["s"],
    "cos": ["s"],
    "log": ["s"],
    "sqrt": ["s"],
    "exp": ["s"],
    "integer_pow": ["s"],
    "select_n": ["s0", "s1"],
    "clamp": ["sl", "s", "sh"],
    "mul": ["sx", "sy"],
    "div": ["sx", "sy"],
    "max": ["sx", "sy"],
    "pow": ["sx", "sy"],
    "rem": ["sx", "sy"],
    "polynomial": ["sx"],
    "reduce_sum": ["s"],
    "reduce": ["s"],
    "broadcast_in_dim": ["so"],
    "pad": ["s", "sv"],
    "cumsum": ["s"],
    "slice": ["s"],
}


def assertions(prim, p, rec):
    """The Lean propositions that the case asserts, with names, and the tactic that proves them.

    Returns `(expected, [(name, proposition, tactic)])`, where `expected` names an expected disagreement.
    """
    if prim in BATCH:
        res = rec["batch"]["result"]
        want = "none" if res is None else f"some ({llist(res[0])}, some {res[1]})"
        out = [("batch", f"{BATCH[prim](p)} = {want}", "decide")]
        # `vmap` raises exactly when the rule does.
        if (res is None) != (rec["batch"]["user"] is None):
            out.append(("batch user", "False", "decide"))
        return None, out
    args, terms, vjp = lean_spec(prim, p)
    for key, (_, applies, result_shape) in EXPECTED.items():
        if applies(prim, p, rec):
            prop, tac = neg_args(args)
            jax_shape = f"{llist(rec['primal']['shape'])} = List.ofFn {result_shape(p)}"
            return key, [("model rejects", prop, tac), ("primal", jax_shape, "decide")]
    if "error" in rec["primal"]:
        # JAX rejects: one of the proof arguments of the modelled function must be false.
        primal_args = [a for i, a in enumerate(args) if f"PROOF{i}" in terms["primal"]]
        prop, tac = neg_args(primal_args)
        return None, [("rejects", prop, tac)]
    out = []
    # The shapes of the terms that are already compared with JAX, so that every term is elaborated once.
    known = {}

    def ref(name):
        return llist(known[name]) if name in known else f"shapeOf ({fill(terms[name], args)})"

    for name, term in terms.items():
        if name not in rec:
            continue
        if "error" in rec[name]:
            # The rule raised: one of its proof arguments must be false.
            rule_args = [a for i, a in enumerate(args) if f"PROOF{i}" in term]
            prop, tac = neg_args(rule_args)
            out.append((f"{name} rejects", prop, tac))
        else:
            out.append((name, shape_eq(term, args, rec[name]["shape"]), "decide"))
            known[name] = rec[name]["shape"]
    if "fwd" in rec:
        forwarded = "true" if rec["fwd"]["forwarded"] else "false"
        out.append(("fwd", f"{FWD[prim](p)} = {forwarded}", "decide"))
    for lin, cts in rec.get("partial", {}).items():
        want = "none" if cts is None else f"some [{', '.join(str(c).lower() for c in cts)}]"
        out.append((f"partial {lin}", f"{fill(PARTIAL[prim](p, lin), args)} = {want}", "decide"))
    if "user_jvp" in rec:
        # A comparison has no rule term: its zero tangent has the shape of the primal output.
        jvps = [n for n in terms if n.startswith("jvp")] or ["primal"]
        if "error" in rec["user_jvp"]:
            out.append(("user_jvp rejects", "False", "decide"))
        else:
            for n in jvps:
                out.append(
                    (f"user_jvp = {n}", f"{llist(rec['user_jvp']['shape'])} = {ref(n)}", "decide")
                )
    if "user_vjp" in rec and vjp:
        conj = " ∧ ".join(
            f"({ref(t)} = {ref(o) if isinstance(o, str) else llist(o)})" for t, o in vjp
        )
        if "error" in rec["user_vjp"]:
            out.append(("user_vjp rejects", f"¬ ({conj})", "decide"))
        else:
            # Reverse mode succeeds: the model's rules have the right shapes,
            # and the cotangents have the shapes of the inputs.
            got = rec["user_vjp"].get("shapes") or [rec["user_vjp"]["shape"]]
            want = [p[k] for k in VJP_INPUTS[prim]]
            cts = " ∧ ".join(f"({llist(g)} = {llist(w)})" for g, w in zip(got, want))
            if len(got) != len(want):
                cts = "False"
            out.append(("user_vjp", f"{conj} ∧ {cts}", "decide"))
    return None, out


HEADER = """import Jax

open Jax Jax.Lax

set_option linter.all false

/-- The shape in the type of an array of scalars or booleans, as a list. -/
def shapeOf {n : ℕ} {s : Shape n} {α : Type} (_ : Index s → α) : List ℕ := List.ofFn s

/-- The shape with the sizes `l`. -/
abbrev sh (l : List ℕ) : Shape l.length := fun d => l.get d

/-- The map from the axes of an operand to the axes `l` of rank `n`. -/
abbrev dims (n : ℕ) (l : List (Fin n)) : Fin l.length → Fin n := fun d => l.get d
"""


def case_id(prim, p):
    return f"{prim} {json.dumps(p, separators=(',', ':'))}"


def write_lean(path, items, split):
    """Write one example per item, or one per assertion if `split`. Returns line → (case index, name)."""
    lines: list[str] = [*HEADER.splitlines()]
    where = {}
    for idx, (cid, asserts) in items:
        lines.append(f"-- {cid}")
        for group in ([a] for a in asserts) if split else [asserts]:
            props = " ∧ ".join(f"({prop})" for _, prop, _ in group)
            tacs = [tac for _, _, tac in group]
            proof = (
                f"by {tacs[0]}"
                if len(set(tacs)) == 1
                else "⟨" + ", ".join(f"by {t}" for t in tacs) + "⟩"
            )
            lines.append(f"example : {props} := {proof}")
            where[len(lines)] = (idx, " ∧ ".join(name for name, _, _ in group))
    path.write_text("\n".join(lines) + "\n")
    return where


def lean(path):
    """Check a Lean file, reporting every error."""
    return subprocess.run(
        ["lake", "env", "lean", "-DmaxErrors=1000000", str(path.relative_to(ROOT))],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def run_lean(path):
    run = lean(path)
    errors = {}
    pattern = re.compile(rf"^{re.escape(str(path.relative_to(ROOT)))}:(\d+):\d+: error: (.*)$")
    current = None
    for line in run.stdout.splitlines():
        m = pattern.match(line)
        if m:
            current = int(m.group(1))
            errors.setdefault(current, []).append(m.group(2))
        elif current is not None and line.startswith("  "):
            errors[current].append(line)
    if run.returncode != 0 and not errors:
        raise RuntimeError(f"lean failed on {path}:\n{run.stdout}\n{run.stderr}")
    return errors


def run_lean_files(files, jobs):
    with ProcessPoolExecutor(jobs) as pool:
        return dict(zip(files, pool.map(run_lean, files)))


def reduce_file(path, items):
    """Print the model's shapes for the failing cases with `#reduce`."""
    lines = [HEADER]
    for _, (cid, terms) in items:
        lines.append(f'#eval IO.println "-- {cid.replace('"', "")}"')
        for name, term in terms:
            lines.append(f'#eval IO.println "{name}:"')
            lines.append(f"#reduce shapeOf ({term})")
    path.write_text("\n".join(lines) + "\n")
    run = lean(path)
    return run.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--only", help="comma-separated primitives")
    args = ap.parse_args()

    t0 = time.time()
    cases = generate_cases()
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c[0] in keep]
    records = jax_records(cases, args.jobs)
    t1 = time.time()

    items, expected = [], {}
    for idx, ((prim, p), rec) in enumerate(zip(cases, records)):
        key, asserts = assertions(prim, p, rec)
        if key:
            expected.setdefault(key, []).append(idx)
        items.append((idx, (case_id(prim, p), asserts)))

    GEN.mkdir(parents=True, exist_ok=True)
    for old in GEN.glob("*.lean"):
        old.unlink()
    nfiles = args.jobs
    files, wheres = [], {}
    for i in range(nfiles):
        path = GEN / f"Shapes{i}.lean"
        wheres[path] = write_lean(path, items[i::nfiles], split=False)
        files.append(path)
    (GEN / "records.json").write_text(
        json.dumps([{"case": case_id(*c), "jax": r} for c, r in zip(cases, records)], indent=1)
    )
    results = run_lean_files(files, args.jobs)
    t2 = time.time()

    stray = [
        (f, line, errs[line][0])
        for f, errs in results.items()
        for line in errs
        if line not in wheres[f]
    ]
    if stray:
        for f, line, err in stray:
            print(f"{f.relative_to(ROOT)}:{line}: {err}", file=sys.stderr)
        print("shapes: Lean errors outside the cases", file=sys.stderr)
        return 1
    failing = sorted({wheres[f][line][0] for f, errs in results.items() for line in errs})
    (GEN / "failing.txt").write_text("".join(f"{items[i][1][0]}\n" for i in failing))
    by_prim = {}
    for prim, _ in cases:
        by_prim[prim] = by_prim.get(prim, 0) + 1
    print(f"shapes: {len(cases)} cases, JAX {t1 - t0:.0f}s, Lean {t2 - t1:.0f}s")
    for prim, count in by_prim.items():
        print(f"  {prim}: {count}")
    for key, idxs in expected.items():
        bad = [i for i in idxs if i in failing]
        print(
            f"  expected disagreement {key}: {len(idxs)} cases ({EXPECTED[key][0]})"
            + (f", {len(bad)} not as expected" if bad else "")
        )
    if not failing:
        print("shapes: ok")
        return 0

    # Diagnose: one example per assertion, and the model's shapes.
    diag = GEN / "Diagnose.lean"
    sel = [items[i] for i in failing[:40]]
    where = write_lean(diag, sel, split=True)
    errs = run_lean(diag)
    red = []
    for idx, (cid, _) in sel:
        prim, p = cases[idx]
        a, terms, _ = lean_spec(prim, p)
        red.append((idx, (cid, [(n, fill(t, a)) for n, t in terms.items()])))
    shown = reduce_file(GEN / "Reduce.lean", red)
    for idx, (cid, asserts) in sel:
        print(f"\nFAIL {cid}\n  JAX: {json.dumps(records[idx])}")
        for line, (i, name) in where.items():
            if i == idx and line in errs:
                print(f"  failed: {name}: {errs[line][0][:200]}")
    print("\nModel shapes (#reduce):\n" + shown[:20000])
    print(f"\nshapes: {len(failing)} failing cases", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
