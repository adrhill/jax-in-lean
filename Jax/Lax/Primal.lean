module

public import Jax.Core

/-!
# Primal semantics

What each `lax` primitive computes, and the helper functions that differentiation rules call.
The semantics are those of the `lax` docstrings and the [StableHLO spec](https://openxla.org/stablehlo/spec).
They live in one file because the rules of one primitive call other primitives.
-/

@[expose] public section

namespace Jax.Lax

variable {ι : Type} {m n : ℕ}

/-! ## Elementwise unary primitives -/

/-- `lax.neg`, elementwise negation. `jax/_src/lax/lax.py:248` -/
def neg (x : Tensor ι) : Tensor ι := fun i => -x i

/-- `lax.sin`, elementwise sine. `jax/_src/lax/lax.py:735`; `accuracy` is dropped -/
noncomputable def sin (x : Tensor ι) : Tensor ι := fun i => (x i).sin

/-- `lax.cos`, elementwise cosine. `jax/_src/lax/lax.py:768`; `accuracy` is dropped -/
noncomputable def cos (x : Tensor ι) : Tensor ι := fun i => (x i).cos

/-- `lax.log`, elementwise natural logarithm. `jax/_src/lax/lax.py:570`; `accuracy` is dropped -/
noncomputable def log (x : Tensor ι) : Tensor ι := fun i => (x i).log

/-- `lax.exp`, elementwise exponential. `jax/_src/lax/lax.py:475`; `accuracy` is dropped.
It lowers to [StableHLO `exponential`](https://openxla.org/stablehlo/spec#exponential). -/
noncomputable def exp (x : Tensor ι) : Tensor ι := fun i => (x i).exp

/-- `lax.sqrt`, elementwise square root. `jax/_src/lax/lax.py:990`; `accuracy` is dropped.
It lowers to [StableHLO `sqrt`](https://openxla.org/stablehlo/spec#sqrt),
which is `squareRoot` of IEEE 754: NaN at negative numbers. -/
noncomputable def sqrt (x : Tensor ι) : Tensor ι := fun i => (x i).sqrt

/-- `lax.sign`, the elementwise sign. `jax/_src/lax/lax.py:268`.
It lowers to [StableHLO `sign`](https://openxla.org/stablehlo/spec#sign). -/
noncomputable def sign (x : Tensor ι) : Tensor ι := fun i => (x i).sign

/-- `lax.floor`, the elementwise floor. `jax/_src/lax/lax.py:342`.
It lowers to [StableHLO `floor`](https://openxla.org/stablehlo/spec#floor),
which is `roundToIntegralTowardNegative` of IEEE 754. -/
noncomputable def floor (x : Tensor ι) : Tensor ι := fun i => (x i).floor

/-- `lax.abs`, the elementwise absolute value. `jax/_src/lax/lax.py:922`.
It lowers to [StableHLO `abs`](https://openxla.org/stablehlo/spec#abs). -/
noncomputable def abs (x : Tensor ι) : Tensor ι := fun i => (x i).abs

/-- The loop of `_integer_pow`, which multiplies `acc` by `x ^ y` by repeated squaring.
`none` is the empty product. -/
noncomputable def integerPowLoop (acc : Option Scalar) (x : Scalar) (y : ℕ) : Option Scalar :=
  if y = 0 then acc else
    integerPowLoop (if y % 2 = 1 then some (match acc with | none => x | some a => a * x) else acc)
      (if y / 2 > 0 then x * x else x) (y / 2)
termination_by y
decreasing_by omega

/-- `_integer_pow` on one entry. The assertion `acc is not None` holds because `y ≠ 0`. -/
noncomputable def integerPowScalar (y : ℤ) (x : Scalar) : Scalar :=
  if y = 0 then 1 else
    let acc := (integerPowLoop none x y.natAbs).getD 0
    if y < 0 then 1 / acc else acc

/-- `lax.integer_pow`, the elementwise power `x ^ y` for an integer `y`. `jax/_src/lax/lax.py:968`.
XLA computes it as `_integer_pow` does, `jax/_src/lax/lax.py:5393`:
```
def _integer_pow(x, *, y):
  # This should be kept in sync with the jax2tf translation rule.
  if y == 0:
    return full_like(x, 1)
  is_reciprocal = y < 0
  if is_reciprocal:
    y = -y
  acc: Array | None = None
  while y > 0:
    if y & 1:
      acc = x if acc is None else mul(acc, x)
    y >>= 1
    if y > 0:
      # We don't call square because it calls integer_pow.
      x = mul(x, x)
  assert acc is not None
  return div(full_like(acc, 1), acc) if is_reciprocal else acc
```
`_integer_pow_lowering` emits `x`, `x * x`, `x * x * x` and `1 / x` for `y = 1, 2, 3, -1`,
which agree with `_integer_pow` because multiplication is commutative. -/
noncomputable def integerPow (y : ℤ) (x : Tensor ι) : Tensor ι := fun i => integerPowScalar y (x i)

/-- `mul(_const(x, c), x)`, the product with the rank-0 constant `c`.
`broadcasting_shape_rule` ignores operands of rank 0, `jax/_src/lax/lax.py:4508`,
so the result has the shape of `x`. -/
noncomputable def constMul (c : Scalar) (x : Tensor ι) : Tensor ι := fun i => c * x i

/-- `div(_const(x, c), y)`, the quotient of the rank-0 constant `c` by `y`.
`broadcasting_shape_rule` ignores operands of rank 0, `jax/_src/lax/lax.py:4508`,
so the result has the shape of `y`. -/
noncomputable def constDiv (c : Scalar) (y : Tensor ι) : Tensor ι := fun i => c / y i

/-! ## Elementwise binary primitives -/

/-- An elementwise binary primitive, which broadcasts its arguments against each other.
Its entries are scalars, or booleans for a comparison.
The proof `h` is the shape check of `broadcasting_shape_rule`, `jax/_src/lax/lax.py:4503`. -/
def binop {α : Type} {sx sy : Shape n} (op : Scalar → Scalar → α) (h : Broadcastable sx sy)
    (x : Tensor (Index sx)) (y : Tensor (Index sy)) : Index (broadcastShape sx sy) → α :=
  fun j => op (x (j.unbroadcast (broadcastsTo_left sx sy))) (y (j.unbroadcast h.right))

variable {sx sy : Shape n}

/-- `lax.mul`, elementwise product. `jax/_src/lax/lax.py:1275`; `out_dtype` is dropped -/
noncomputable def mul (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  binop (· * ·) h x y

/-- `lax.div`, elementwise quotient. `jax/_src/lax/lax.py:1327` -/
noncomputable def div (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  binop (· / ·) h x y

/-- `lax.sub`, elementwise difference. `jax/_src/lax/lax.py:1251` -/
noncomputable def sub (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  binop (fun a b => a + -b) h x y

/-- `lax.pow`, elementwise power `x ^ y` for a floating-point `y`. `jax/_src/lax/lax.py:942`.
It lowers to [StableHLO `power`](https://openxla.org/stablehlo/spec#power),
which is `pow` of IEEE 754 for floating-point arguments.
Integer exponents are not modelled. -/
noncomputable def pow (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  binop Scalar.pow h x y

/-- `lax.rem`, elementwise remainder with the sign of the dividend `x`. `jax/_src/lax/lax.py:1357`.
It lowers to [StableHLO `remainder`](https://openxla.org/stablehlo/spec#remainder):
`lhs - d * rhs`, where `d` is the quotient of IEEE 754 with the rounding `roundTowardZero`.
Integer arguments are not modelled. -/
noncomputable def rem (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  binop Scalar.rem h x y

/-- `lax.max`, elementwise maximum. `jax/_src/lax/lax.py:1385` -/
noncomputable def max (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  binop Scalar.max h x y

/-- An elementwise binary primitive whose first argument has rank 0.
`broadcasting_shape_rule` ignores the operands of rank 0, so the result has the shape of `y`.
The entries are scalars or booleans.
`jax/_src/lax/lax.py:4508`:
```
  shapes = [aval.shape for aval in avals if aval.shape]
  if not shapes:
    return ()
``` -/
def binopScalarLeft {α β γ : Type} {sx : Shape 0} {sy : Shape n} (op : α → β → γ)
    (x : Index sx → α) (y : Index sy → β) : Index sy → γ :=
  fun j => op (x fun d => d.elim0) (y j)

/-- An elementwise binary primitive whose second argument has rank 0, as `binopScalarLeft`.
The result has the shape of `x`. -/
def binopScalarRight {α β γ : Type} {sx : Shape n} {sy : Shape 0} (op : α → β → γ)
    (x : Index sx → α) (y : Index sy → β) : Index sx → γ :=
  fun j => op (x j) (y fun d => d.elim0)

/-! ## Comparisons

A comparison returns an array of booleans, `Index s → Bool`.
For floating-point arguments, [StableHLO `compare`](https://openxla.org/stablehlo/spec#compare)
with `compare_type = FLOAT` implements the quiet comparisons of IEEE 754:
`ne` is true and the others are false if an argument is NaN.
The lowerings pass `total_order=False` to `_compare_lower_hlo`, as in `jax/_src/lax/lax.py:5718`:
`mlir.register_lowering(eq_p, partial(_compare_lower_hlo, "EQ", False))`. -/

open scoped Classical in
/-- `lax.eq`, elementwise `x == y`. `jax/_src/lax/lax.py:1517` -/
noncomputable def eq (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Index (broadcastShape sx sy) → Bool :=
  binop (fun a b => decide (a.eq b)) h x y

open scoped Classical in
/-- `lax.ne`, elementwise `x != y`. `jax/_src/lax/lax.py:1548` -/
noncomputable def ne (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Index (broadcastShape sx sy) → Bool :=
  binop (fun a b => decide (a.ne b)) h x y

open scoped Classical in
/-- `lax.ge`, elementwise `x >= y`. `jax/_src/lax/lax.py:1579` -/
noncomputable def ge (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Index (broadcastShape sx sy) → Bool :=
  binop (fun a b => decide (a.ge b)) h x y

open scoped Classical in
/-- `lax.gt`, elementwise `x > y`. `jax/_src/lax/lax.py:1610` -/
noncomputable def gt (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Index (broadcastShape sx sy) → Bool :=
  binop (fun a b => decide (a.gt b)) h x y

open scoped Classical in
/-- `lax.le`, elementwise `x <= y`. `jax/_src/lax/lax.py:1641` -/
noncomputable def le (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Index (broadcastShape sx sy) → Bool :=
  binop (fun a b => decide (a.le b)) h x y

open scoped Classical in
/-- `lax.lt`, elementwise `x < y`. `jax/_src/lax/lax.py:1672` -/
noncomputable def lt (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Index (broadcastShape sx sy) → Bool :=
  binop (fun a b => decide (a.lt b)) h x y

/-- `lax.bitwise_and` on booleans, elementwise `x and y`. `jax/_src/lax/lax.py:1105`.
It broadcasts its arguments like `binop`. -/
def bitwiseAnd (h : Broadcastable sx sy) (x : Index sx → Bool) (y : Index sy → Bool) :
    Index (broadcastShape sx sy) → Bool :=
  fun j => x (j.unbroadcast (broadcastsTo_left sx sy)) && y (j.unbroadcast h.right)

/-! ## Selection -/

/-- `which` of `select_n` has shape `sw`: it is a scalar or has the shape `s` of the cases.
`_select_shape_rule`, `jax/_src/lax/lax.py:8856`:
```
  if which.shape and which.shape != cases[0].shape:
``` -/
def WhichShape {k : ℕ} (sw : Shape k) (s : Shape n) : Prop := k = 0 ∨ List.ofFn sw = List.ofFn s

theorem WhichShape.rank {k : ℕ} {sw : Shape k} {s : Shape n} (h : WhichShape sw s) (hk : k ≠ 0) :
    k = n := by
  simpa using congrArg List.length (h.resolve_left hk)

theorem WhichShape.size {k : ℕ} {sw : Shape k} {s : Shape n} (h : WhichShape sw s) (hk : k ≠ 0)
    (d : Fin k) : sw d = s (Fin.cast (h.rank hk) d) := by
  have := congrArg (fun l => l[(d : ℕ)]?) (h.resolve_left hk)
  simp [List.getElem?_ofFn] at this
  exact this.2

/-- The entry of `which` that `select_n` reads at position `j` of the cases:
the only entry of a scalar, and entry `j` otherwise. -/
def Index.ofWhich {k : ℕ} {sw : Shape k} {s : Shape n} (h : WhichShape sw s) (j : Index s) :
    Index sw :=
  if hk : k = 0 then fun d => (Fin.cast hk d).elim0
  else fun d => Fin.cast (h.size hk d).symm (j (Fin.cast (h.rank hk) d))

/-- Shapes with the same sizes have the same rank. -/
theorem rank_eq_of_ofFn_eq {n' : ℕ} {s1 : Shape n'} {s0 : Shape n}
    (h : List.ofFn s1 = List.ofFn s0) : n' = n := by
  simpa using congrArg List.length h

theorem size_eq_of_ofFn_eq {n' : ℕ} {s1 : Shape n'} {s0 : Shape n}
    (h : List.ofFn s1 = List.ofFn s0) (d : Fin n') :
    s1 d = s0 (Fin.cast (rank_eq_of_ofFn_eq h) d) := by
  have := congrArg (fun l => l[(d : ℕ)]?) h
  simp [List.getElem?_ofFn] at this
  exact this.2

/-- The index of shape `s1` with the values of the index `j` of the equal shape `s0`. -/
def Index.castShape {n' : ℕ} {s1 : Shape n'} {s0 : Shape n} (h : List.ofFn s1 = List.ofFn s0)
    (j : Index s0) : Index s1 :=
  fun d => Fin.cast (size_eq_of_ofFn_eq h d).symm (j (Fin.cast (rank_eq_of_ofFn_eq h) d))

@[simp]
theorem Index.castShape_self {s : Shape n} (h : List.ofFn s = List.ofFn s) (j : Index s) :
    Index.castShape h j = j :=
  rfl

/-- `lax.select_n` with a boolean `which` and two cases: `case1` where `which` is true,
and `case0` where it is false. `lax.select(pred, on_true, on_false)` is `select_n(pred, on_false, on_true)`.
`jax/_src/lax/lax.py:3209`; an integer `which` and other numbers of cases are not modelled.
It lowers to [StableHLO `select`](https://openxla.org/stablehlo/spec#select),
which broadcasts a scalar `which`, `jax/_src/lax/lax.py:9018`:
```
  if which_aval.dtype == np.dtype(np.bool_):
    assert len(cases) <= 2
    if len(cases) == 1: return cases
    op = hlo.select(which, cases[1], cases[0])
```
The proofs `hc` and `hw` are the shape checks of `_select_shape_rule`, `jax/_src/lax/lax.py:8850`,
that the cases have the same shape, of the same rank, and that `which` is a scalar or has their shape:
```
  if any(case.shape != cases[0].shape for case in cases[1:]):
    msg = "select cases must have the same shapes, got [{}]."
``` -/
def selectN {k n' : ℕ} {sw : Shape k} {s0 : Shape n} {s1 : Shape n'}
    (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0) (which : Index sw → Bool)
    (case0 : Tensor (Index s0)) (case1 : Tensor (Index s1)) : Tensor (Index s0) :=
  fun j => if which (Index.ofWhich hw j) then case1 (Index.castShape hc j) else case0 j

/-- `lax.clamp(min, x, max)`, which clamps `x` between `lo = min` and `hi = max`.
`jax/_src/lax/lax.py:1883`.
It lowers to [StableHLO `clamp`](https://openxla.org/stablehlo/spec#clamp),
`minimum(maximum(operand, min), max)`, so the result is `max` wherever `min > max`.
The docstring of `lax.clamp` instead gives `min` where `x < min`; XLA follows StableHLO.
The proofs `hl` and `hh` are the shape checks of `_clamp_shape_rule`, `jax/_src/lax/lax.py:7719`,
that the bounds are scalars or have the shape of `x`, which `WhichShape` states:
```
  if min.shape and min.shape != operand.shape:
    raise TypeError(...)
  if max.shape and max.shape != operand.shape:
    raise TypeError(...)
``` -/
noncomputable def clamp {k k' : ℕ} {sl : Shape k} {sh : Shape k'} {s : Shape n}
    (hl : WhichShape sl s) (hh : WhichShape sh s) (lo : Tensor (Index sl)) (x : Tensor (Index s))
    (hi : Tensor (Index sh)) : Tensor (Index s) :=
  fun j => ((x j).max (lo (Index.ofWhich hl j))).min (hi (Index.ofWhich hh j))

/-- `lax.reshape` without `dimensions`: the entries of the operand in row-major order,
laid out in the shape `newSizes`.
`jax/_src/lax/lax.py:3053`; `out_sharding` is dropped, and `dimensions` other than `None` are not modelled.
The proof `h` is the constraint of [StableHLO `reshape`](https://openxla.org/stablehlo/spec#reshape)
that the sizes agree. `_reshape_shape_rule`, `jax/_src/lax/lax.py:8360`, does not check it. -/
def reshape {s : Shape m} (newSizes : Shape n) (h : ∏ d, s d = ∏ d, newSizes d)
    (operand : Tensor (Index s)) : Tensor (Index newSizes) :=
  fun j => operand (Index.rowMajor.symm (Fin.cast h.symm (Index.rowMajor j)))

/-! ## Polynomials -/

/-- Horner's rule for the coefficients `a 0, …, a K` at `x`, in the order of `_polynomial_lower`:
`acc` starts at the coefficient of highest degree, and each step computes `add(mul(acc, x), c)`.
`jax/_src/lax/lax.py:5297`:
```
  if len(coeffs) <= unroll_threshold:
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
      acc = add(mul(acc, x), c)
    return acc
``` -/
noncomputable def horner (x : Scalar) : {K : ℕ} → (Fin (K + 1) → Scalar) → Scalar
  | 0, a => a 0
  | _ + 1, a => horner x (Fin.tail a) * x + a 0

/-- `lax.polynomial` with at least one coefficient, `a 0 + x * (a 1 + … + x * a K)`,
which broadcasts `x` and all coefficients against each other.
With two or more coefficients this is the primitive `polynomial_p`.
The proof `h` is the shape check of `broadcasting_shape_rule`, `jax/_src/lax/lax.py:4503`.
`unroll` is dropped, and the semantics are those of the unrolled loop in `_polynomial_lower`.
`jax/_src/lax/lax.py:4104`:
```
  if len(c_args) == 1:
    out_shape = broadcasting_shape_rule('polynomial', typeof(x), typeof(c_args[0]))
    out_sharding = broadcasting_sharding_rule('polynomial', typeof(x), typeof(c_args[0]))
    return _maybe_broadcast(out_shape, c_args[0], out_sharding)
  kwargs = {} if unroll is None else {'unroll': unroll}
  return polynomial_p.bind(x, *c_args, **kwargs)
``` -/
noncomputable def polynomial {K : ℕ} {sx : Shape n} {sc : Fin (K + 1) → Shape n}
    (h : BroadcastableAll (sx :: List.ofFn sc)) (x : Tensor (Index sx))
    (coeffs : (k : Fin (K + 1)) → Tensor (Index (sc k))) :
    Tensor (Index (broadcastShapes (sx :: List.ofFn sc))) :=
  fun j => horner (x (j.unbroadcast (broadcastsTo_broadcastShapes h List.mem_cons_self)))
    fun k => coeffs k (j.unbroadcast (broadcastsTo_broadcastShapes h
      (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨k, rfl⟩))))

/-! ## Structural primitives -/

/-- `lax.broadcast_in_dim`: axis `d` of the operand becomes axis `broadcastDimensions d` of the result.
`jax/_src/lax/lax.py:2946`; `out_sharding` is dropped.
The proofs `h` and `hd` are the shape checks of `_broadcast_in_dim_shape_rule`
that the sizes agree, `jax/_src/lax/lax.py:7474`,
and that `broadcast_dimensions` has no duplicates, `jax/_src/lax/lax.py:7485`.
The semantics do not need `hd`. -/
def broadcastInDim {so : Shape m} (shape : Shape n) (broadcastDimensions : Fin m → Fin n)
    (h : BroadcastsTo so (shape ∘ broadcastDimensions))
    (_hd : Function.Injective broadcastDimensions) (operand : Tensor (Index so)) :
    Tensor (Index shape) :=
  fun j => operand ((j.proj broadcastDimensions).unbroadcast h)

/-- `lax.transpose`, which permutes the axes: axis `d` of the result is axis `permutation d` of the operand.
`jax/_src/lax/lax.py:3236`.
It lowers to [StableHLO `transpose`](https://openxla.org/stablehlo/spec#transpose):
`result[result_index] = operand[operand_index]` where `result_index[d] = operand_index[permutation[d]]`.
The proof `h` is the shape check of `_transpose_shape_rule`, `jax/_src/lax/lax.py:8803`,
that `permutation` is a permutation of the axes:
```
  if tuple(sorted(permutation)) != tuple(range(operand.ndim)):
``` -/
noncomputable def transpose {s : Shape n} (permutation : Fin n → Fin n)
    (h : Function.Injective permutation) (operand : Tensor (Index s)) :
    Tensor (Index (s ∘ permutation)) :=
  let e := Equiv.ofBijective permutation (Finite.injective_iff_bijective.1 h)
  fun j => operand fun d => Fin.cast (congrArg s (e.apply_symm_apply d)) (j (e.symm d))

/-- `lax.reduce_sum`, the sum over `axes`. The result keeps the other axes in order.
`jax/_src/lax/lax.py:3412`; `out_sharding` is dropped -/
def reduceSum {s : Shape n} (axes : Finset (Fin n)) (operand : Tensor (Index s)) :
    Tensor (Index (s ∘ keptAxes axes)) :=
  fun j => ∑ i with i.proj (keptAxes axes) = j, operand i

/-- `lax.reduce_max`, the maximum over `axes`. The result keeps the other axes in order.
`jax/_src/lax/lax.py:3466`; `out_sharding` is dropped.
It lowers to [StableHLO `reduce`](https://openxla.org/stablehlo/spec#reduce) with `maximum`
and the initial value `_get_max_identity`, which is `-∞` for floating-point dtypes.
`jax/_src/lax/lax.py:9503`:
```
mlir.register_lowering(reduce_max_p, partial(_unary_reduce_lower, mlir.max_hlo,
                                             _get_max_identity))
```
So the maximum over an empty axis is `-∞`, and over an axis with a NaN it is NaN.
`maximum` is associative and commutative (`Scalar.max`), so the order of the reduction does not matter.
Duplicate and out-of-bounds axes, which `_reduce_op_shape_rule` rejects, are not representable. -/
def reduceMax {s : Shape n} (axes : Finset (Fin n)) (operand : Tensor (Index s)) :
    Tensor (Index (s ∘ keptAxes axes)) :=
  fun j => ({i | i.proj (keptAxes axes) = j} : Finset (Index s)).fold Scalar.max .negInf operand

/-- `lax.convert_element_type` from `bool` to a floating-point dtype: `1` for true and `0` for false.
`jax/_src/lax/lax.py:1713`. It lowers to [StableHLO `convert`](https://openxla.org/stablehlo/spec#convert). -/
def convertBool {ι : Type} (x : ι → Bool) : Tensor ι := fun i => if x i then 1 else 0

/-- `lax.cumsum`, the cumulative sum along `axis`:
entry `j` is the sum of the entries at or before `j` on `axis`, and at or after `j` if `reverse`.
`jax/_src/lax/control_flow/loops.py:3161`.
It lowers to a `reduce_window` sum over the window that ends at `j`, or starts at `j` if `reverse`
(`cumred_reduce_window_impl`, `jax/_src/lax/control_flow/loops.py:3200`).
Negative and out-of-bounds axes, which `_cumred_shape_rule` rejects, are not representable. -/
def cumsum {s : Shape n} (axis : Fin n) (reverse : Bool) (operand : Tensor (Index s)) :
    Tensor (Index s) :=
  fun j => ∑ i with (∀ d, d ≠ axis → i d = j d) ∧
    (if reverse then j axis ≤ i axis else i axis ≤ j axis), operand i

/-! ## Padding and slicing -/

/-- `core.dilate_dim`: the size of an axis of size `d`
once `dilation - 1` entries are inserted between neighbouring entries.
`jax/_src/core.py:3253`:
```
def dilate_dim(d: DimSize, dilation: DimSize) -> DimSize:
  """max(0, 1 + dilation * (d - 1)).

  Assumes dilation >= 1.
  """
  if definitely_equal(dilation, 1):  # fast path
    return d
  return max_dim(1 + dilation * (d - 1), 0)
``` -/
def dilateDim (d dilation : ℕ) : ℕ := (1 + (dilation : ℤ) * ((d : ℤ) - 1)).toNat

/-- The shape of the result of `pad`. `_pad_shape_rule`, `jax/_src/lax/lax.py:8195`:
```
  result = tuple(l + h + core.dilate_dim(d, i + 1)
                 for (l, h, i), d in zip(padding_config, op_shape))
``` -/
def padShape (s : Shape n) (lo hi : Fin n → ℤ) (interior : Fin n → ℕ) : Shape n :=
  fun d => (lo d + hi d + dilateDim (s d) (interior d + 1)).toNat

/-- `lax.pad`, which inserts `interior d` copies of the rank-0 `paddingValue` between the entries
of axis `d`, and then adds `lo d` copies before and `hi d` copies after them.
Negative `lo d` and `hi d` remove entries instead.
`padding_config` is the sequence of the triples `(lo d, hi d, interior d)`.
`jax/_src/lax/lax.py:3115`.
It lowers to [StableHLO `pad`](https://openxla.org/stablehlo/spec#pad):
`result[result_index] = operand[operand_index]` if
`result_index = edge_padding_low + operand_index * (interior_padding + 1)`, and `padding_value` otherwise.
The proof `_h` is the shape check of `_pad_shape_rule`, `jax/_src/lax/lax.py:8197`,
that no axis of the result has a negative size.
Negative interior padding, which `_pad_shape_rule` rejects, is not representable. -/
noncomputable def pad {s : Shape n} {sv : Shape 0} (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (_h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1))
    (operand : Tensor (Index s)) (paddingValue : Tensor (Index sv)) :
    Tensor (Index (padShape s lo hi interior)) :=
  fun j => if h : ∃ i : Index s, ∀ d, ((j d : ℕ) : ℤ) = lo d + (i d : ℕ) * ((interior d : ℤ) + 1)
    then operand h.choose else paddingValue fun d => d.elim0

/-- `core.stride_dim`: the number of windows of size `windowSize` at distance `windowStride`
that fit in an axis of size `d`.
For a positive `windowStride`, `/` on `ℤ` is the floor division `//` of Python.
`jax/_src/core.py:3262`:
```
def stride_dim(d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
  """max(0, (d - window_size) // window_stride + 1)
  ...
  return max_dim((d - window_size) // window_stride + 1, 0)
``` -/
def strideDim (d windowSize windowStride : ℕ) : ℕ :=
  (((d : ℤ) - (windowSize : ℤ)) / (windowStride : ℤ) + 1).toNat

/-- The shape of the result of `slice`. `_slice_shape_rule`, `jax/_src/lax/slicing.py:1363`:
```
  diff = tuple(map(operator.sub, limit_indices, start_indices))
  if strides is None or tuple(strides) == (1,) * len(operand.shape):
    return diff
  ...
  return tuple(core.stride_dim(d, window_size=1, window_stride=s)
               for d, s in zip(diff, strides))
```
For strides 1, `stride_dim` returns `diff`. -/
def sliceShape (startIndices limitIndices strides : Fin n → ℕ) : Shape n :=
  fun d => strideDim (limitIndices d - startIndices d) 1 (strides d)

theorem slice_index_lt {size start limit stride j : ℕ} (hl : limit ≤ size) (hst : 0 < stride)
    (hj : j < strideDim (limit - start) 1 stride) : start + j * stride < size := by
  unfold strideDim at hj
  have hq : (j : ℤ) ≤ (((limit - start : ℕ) : ℤ) - ((1 : ℕ) : ℤ)) / (stride : ℤ) := by omega
  have := (Int.le_ediv_iff_mul_le (by exact_mod_cast hst)).1 hq
  have h2 : ((j * stride : ℕ) : ℤ) = (j : ℤ) * stride := by push_cast; rfl
  omega

/-- `lax.slice`, the entries of the operand at the positions `start`, `start + stride`, …
below `limit` on every axis. `strides=None` is the strides 1.
`jax/_src/lax/slicing.py:60`.
It lowers to [StableHLO `slice`](https://openxla.org/stablehlo/spec#slice):
`result[result_index] = operand[operand_index]` where
`operand_index = start_indices + result_index * strides`.
The proofs are the shape checks of `_slice_shape_rule`, `jax/_src/lax/slicing.py:1351`:
`limit_indices` is at most the shape, `start_indices` is at most `limit_indices`,
and no stride is zero, for which `stride_dim` raises `ZeroDivisionError`.
Negative indices and strides, which `_slice_shape_rule` rejects, are not representable. -/
def slice {s : Shape n} (startIndices limitIndices strides : Fin n → ℕ)
    (hl : ∀ d, limitIndices d ≤ s d) (_hs : ∀ d, startIndices d ≤ limitIndices d)
    (hst : ∀ d, 0 < strides d) (operand : Tensor (Index s)) :
    Tensor (Index (sliceShape startIndices limitIndices strides)) :=
  fun j => operand fun d =>
    ⟨startIndices d + j d * strides d, slice_index_lt (hl d) (hst d) (j d).isLt⟩

/-! ## Reductions -/

/-- The `computation` of `lax.reduce` with one operand, traced to the jaxpr `jaxpr` of `reduce_p`. -/
structure Computation where
  /-- The function of two scalars that the jaxpr computes. -/
  op : Scalar → Scalar → Scalar
  /-- The JVP `jvp x y dx dy` of `op` that `api.jvp` derives from the jaxpr. -/
  jvp : Scalar → Scalar → Scalar → Scalar → Scalar

/-- `op` is associative with the two-sided identity `e`:
the condition of the `lax.reduce` docstring that `computation` and `init_values` form a monoid. -/
structure IsMonoidOp {α : Type} (op : α → α → α) (e : α) : Prop where
  assoc : ∀ a b c, op (op a b) c = op a (op b c)
  left_id : ∀ a, op e a = a
  right_id : ∀ a, op a e = a

/-- The JVP of `computation` is correct at the finite pair `(a, b)`:
near `(a, b)`, `op` computes a real function `F`, which is differentiable at `(a, b)`,
and `jvp` applies its derivative to finite tangents. -/
def Computation.IsJvpAt (computation : Computation) (a b : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ → ℝ) (L : ℝ × ℝ →L[ℝ] ℝ),
    (∀ᶠ p in nhds (a, b), computation.op (.fin p.1) (.fin p.2) = .fin (F p)) ∧
    HasFDerivAt F L (a, b) ∧
    ∀ da db, computation.jvp (.fin a) (.fin b) (.fin da) (.fin db) = .fin (L (da, db))

/-- The reduction of the entries `xs p` with `p ∈ A`: the left fold with `op`,
in which the other entries are replaced by `e`. -/
def reduceOn {α : Type} {k : ℕ} (op : α → α → α) (e : α) (xs : Fin k → α) (A : Finset (Fin k)) : α :=
  (List.ofFn fun p => if p ∈ A then xs p else e).foldl op e

/-- The axes in `D` in increasing order. -/
def sortedAxes (D : Finset (Fin n)) : List (Fin n) := (List.finRange n).filter (· ∈ D)

theorem mem_sortedAxes {D : Finset (Fin n)} {d : Fin n} : d ∈ sortedAxes D ↔ d ∈ D := by
  simp [sortedAxes]

theorem nodup_sortedAxes (D : Finset (Fin n)) : (sortedAxes D).Nodup :=
  (List.nodup_finRange n).filter _

/-- The entries of `x` along the axes `ax` at the position `j` of the other axes,
in row-major order of the axes `ax`, in the order in which `ax` lists them. -/
def reductionSlice {s : Shape n} {D : Finset (Fin n)} (ax : List (Fin n)) (hax : ∀ d, d ∈ ax ↔ d ∈ D)
    (x : Tensor (Index s)) (j : Index (s ∘ keptAxes D)) : Fin (∏ k, (s ∘ ax.get) k) → Scalar :=
  fun p => x (Index.merge ax hax (Index.rowMajor.symm p) j)

/-- `lax.reduce` with one operand and one initial value, which reduces the operand over `dimensions`
with `computation.op`. The result keeps the other axes in order.
`jax/_src/lax/lax.py:3260`; `out_sharding` is dropped.
Reductions that `_get_monoid_reducer` recognizes, `jax/_src/lax/lax.py:3351`, such as `add` with `0`,
bind `reduce_sum` and the like instead; this is the primitive `reduce_p` of all other reductions.

[StableHLO `reduce`](https://openxla.org/stablehlo/spec#reduce) evaluates an implementation-defined
binary tree whose in-order traversal is the entries of the slice in ascending lexicographic order of their
index, interspersed with `init_values` at implementation-defined positions.
The model takes the left fold `body(… body(body(init, x₀), x₁) …, xₖ)`, which is such a tree.
The `lax.reduce` docstring requires that `computation` and `init_values` form a monoid;
then every such tree gives the left fold.
The proof `_hd` is constraint (C5) of StableHLO `reduce`, `is_unique(dimensions)`;
`_reduce_shape_rule`, `jax/_src/lax/lax.py:9055`, does not check it. -/
def reduce {s : Shape n} (computation : Computation) (initValue : Scalar) (dimensions : List (Fin n))
    (_hd : dimensions.Nodup) (operand : Tensor (Index s)) :
    Tensor (Index (s ∘ keptAxes dimensions.toFinset)) :=
  fun j => (List.ofFn (reductionSlice (sortedAxes _) (fun _ => mem_sortedAxes) operand j)).foldl
    computation.op initValue

/-! ## Gather and scatter

An array of indices of shape `s` is a function `Index s → ℤ`, with no tangent.
Dimension numbers are lists of natural numbers, as in JAX. Negative dimensions, which the shape rules reject,
are not representable.
`unique_indices` and `indices_are_sorted` are dropped: they are promises that let XLA pick a faster implementation,
and the semantics are those without the promises.
-/

section GatherScatter

variable {ni nu : ℕ}

/-- `GatherScatterMode`, `jax/_src/lax/slicing.py:280`:
what `gather` and `scatter` do with a window that does not lie inside the operand.

* `clip`: both clamp the start of the window so that it does.
* `fillOrDrop`: `gather` returns `fill_value` in the whole window, and `scatter` drops the whole window.
* `promiseInBounds`: nothing is checked. With XLA, `gather` clamps and `scatter` drops:
  "In practice, with the current XLA implementation this means that out-of-bounds gathers will be clamped
  but out-of-bounds scatters will be discarded. Gradients will not be correct if indices are out-of-bounds."

`ONE_HOT` is not modelled: `_gather_lower` raises for it. -/
inductive GatherScatterMode where
  | clip
  | fillOrDrop
  | promiseInBounds
  deriving DecidableEq

/-- `GatherDimensionNumbers`, `jax/_src/lax/slicing.py:241`.
The index vector is the last axis of the indices. -/
structure GatherDimensionNumbers where
  offsetDims : List ℕ
  collapsedSliceDims : List ℕ
  startIndexMap : List ℕ
  operandBatchingDims : List ℕ := []
  startIndicesBatchingDims : List ℕ := []

/-- `ScatterDimensionNumbers`, `jax/_src/lax/slicing.py:432`.
The index vector is the last axis of the indices. -/
structure ScatterDimensionNumbers where
  updateWindowDims : List ℕ
  insertedWindowDims : List ℕ
  scatterDimsToOperandDims : List ℕ
  operandBatchingDims : List ℕ := []
  scatterIndicesBatchingDims : List ℕ := []

/-- The dimension numbers of the gather that reads the windows that the scatter writes.
The rules of `scatter` build them, as in `jax/_src/lax/slicing.py:2888`:
```
      gather_dnums = GatherDimensionNumbers(
          offset_dims=dimension_numbers.update_window_dims,
          collapsed_slice_dims=dimension_numbers.inserted_window_dims,
          start_index_map=dimension_numbers.scatter_dims_to_operand_dims,
          operand_batching_dims=dimension_numbers.operand_batching_dims,
          start_indices_batching_dims=dimension_numbers.scatter_indices_batching_dims,
      )
``` -/
def ScatterDimensionNumbers.toGather (dn : ScatterDimensionNumbers) : GatherDimensionNumbers :=
  ⟨dn.updateWindowDims, dn.insertedWindowDims, dn.scatterDimsToOperandDims, dn.operandBatchingDims,
    dn.scatterIndicesBatchingDims⟩

/-- The dimension numbers of the scatter that writes the windows that the gather reads.
`_gather_transpose_fancy` builds them, `jax/_src/lax/slicing.py:2246`:
```
  scatter_dnums = ScatterDimensionNumbers(
      update_window_dims=dimension_numbers.offset_dims,
      inserted_window_dims=dimension_numbers.collapsed_slice_dims,
      scatter_dims_to_operand_dims=dimension_numbers.start_index_map,
      operand_batching_dims=dimension_numbers.operand_batching_dims,
      scatter_indices_batching_dims=dimension_numbers.start_indices_batching_dims,
  )
``` -/
def GatherDimensionNumbers.toScatter (dn : GatherDimensionNumbers) : ScatterDimensionNumbers :=
  ⟨dn.offsetDims, dn.collapsedSliceDims, dn.startIndexMap, dn.operandBatchingDims,
    dn.startIndicesBatchingDims⟩

/-- The batch axes of the result of a gather of rank `r`: the axes that are not in `offsetDims`, in order.
Batch axis `a` of the result is axis `a` of the indices.
These are also the scatter axes of the updates of a scatter. -/
def batchDims (r : ℕ) (offsetDims : List ℕ) : List ℕ := (List.range r).filter (· ∉ offsetDims)

/-- The window axes of an operand of rank `n`: the axes that are neither collapsed nor batching, in order.
Window axis `q` of the operand is axis `offsetDims[q]` of the result. -/
def windowDims (n : ℕ) (dn : GatherDimensionNumbers) : List ℕ :=
  (List.range n).filter fun d => d ∉ dn.collapsedSliceDims ∧ d ∉ dn.operandBatchingDims

/-- The shape of the result of `gather`: axis `offsetDims[q]` has the slice size of window axis `q` of the operand,
and the other axes are those of the indices without the index vector, in order.
`_gather_shape_computation`, `jax/_src/lax/slicing.py:1986`:
```
  output_shape_rank = len(offset_dims) + _rank(indices) - 1
  ...
  expanded_indices_shape.pop(index_vector_dim)

  indices_shape_gen = iter(expanded_indices_shape)

  slice_sizes_gen = (
      s
      for i, s in enumerate(slice_sizes)
      if i not in collapsed_slice_dims and i not in operand_batching_dims
  )
  ans = tuple(next(slice_sizes_gen) if i in offset_dims
              else next(indices_shape_gen) for i in range(output_shape_rank))
``` -/
def gatherShape (si : Shape ni) (dn : GatherDimensionNumbers) (sliceSizes : List ℕ) :
    Shape (dn.offsetDims.length + (ni - 1)) :=
  fun i => if (i : ℕ) ∈ dn.offsetDims then
      sliceSizes.getD ((windowDims sliceSizes.length dn).getD (dn.offsetDims.idxOf (i : ℕ)) 0) 0
    else si.sizeD ((batchDims (dn.offsetDims.length + (ni - 1)) dn.offsetDims).idxOf (i : ℕ))

/-- The checks of `_gather_shape_rule`, `jax/_src/lax/slicing.py:1824`, in its order:

* the indices have an index vector axis,
* `offset_dims` is strictly increasing and lists axes of the result,
* `start_index_map` has one entry per entry of the index vector, lists axes of the operand and does not repeat,
* `collapsed_slice_dims` and `operand_batching_dims` are strictly increasing and list axes of the operand,
* `operand_batching_dims` shares no axis with `collapsed_slice_dims` or `start_index_map`,
* `start_indices_batching_dims` does not repeat and lists axes of the indices other than the index vector,
* the batching axes of the operand and of the indices have the same sizes,
* there is one slice size per axis of the operand,
  and every axis of the operand is an offset, collapsed or batching axis,
* no slice is larger than the operand, which is only checked if the indices are not empty:
  `if not core.is_empty_shape(indices.shape) and not (slice_size >= 0 and corresponding_input_size >= slice_size)`,
* collapsed axes have slice size 1, and batching axes at most 1. -/
def GatherValid (so : Shape n) (si : Shape ni) (dn : GatherDimensionNumbers) (sliceSizes : List ℕ) :
    Prop :=
  0 < ni ∧
  dn.offsetDims.Pairwise (· < ·) ∧
  (∀ d ∈ dn.offsetDims, d < dn.offsetDims.length + (ni - 1)) ∧
  dn.startIndexMap.length = si.sizeD (ni - 1) ∧
  (∀ d ∈ dn.startIndexMap, d < n) ∧
  dn.startIndexMap.Nodup ∧
  dn.collapsedSliceDims.Pairwise (· < ·) ∧
  (∀ d ∈ dn.collapsedSliceDims, d < n) ∧
  dn.operandBatchingDims.Pairwise (· < ·) ∧
  (∀ d ∈ dn.operandBatchingDims, d < n) ∧
  (∀ d ∈ dn.collapsedSliceDims, d ∉ dn.operandBatchingDims) ∧
  (∀ d ∈ dn.startIndexMap, d ∉ dn.operandBatchingDims) ∧
  dn.startIndicesBatchingDims.Nodup ∧
  (∀ d ∈ dn.startIndicesBatchingDims, d < ni - 1) ∧
  dn.operandBatchingDims.map so.sizeD = dn.startIndicesBatchingDims.map si.sizeD ∧
  sliceSizes.length = n ∧
  n = dn.offsetDims.length + dn.collapsedSliceDims.length + dn.operandBatchingDims.length ∧
  ((∀ d, si d ≠ 0) → ∀ d : Fin n, sliceSizes.getD d 0 ≤ so d) ∧
  (∀ d ∈ dn.collapsedSliceDims, sliceSizes.getD d 0 = 1) ∧
  (∀ d ∈ dn.operandBatchingDims, sliceSizes.getD d 0 ≤ 1)

instance {so : Shape n} {si : Shape ni} {dn : GatherDimensionNumbers} {sliceSizes : List ℕ} :
    Decidable (GatherValid so si dn sliceSizes) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- The constraints of XLA on a `gather` with a nonempty result that `_gather_shape_rule` does not check:

* no slice is larger than the operand, constraint (C13) of
  [StableHLO `gather`](https://openxla.org/stablehlo/spec#gather).
  The shape rule skips this check for empty indices, and compiling such a gather fails,
* batching axes have slice size 1. The shape rule and constraint (C12) also allow 0,
  for which XLA returns zeros where the formula of StableHLO reads the operand.

`_gather_lower`, `jax/_src/lax/slicing.py:2434`, returns zeros for an empty result and does not emit a gather. -/
def GatherLowers (so : Shape n) (si : Shape ni) (dn : GatherDimensionNumbers) (sliceSizes : List ℕ) :
    Prop :=
  (∀ d, gatherShape si dn sliceSizes d ≠ 0) →
    (∀ d : Fin n, sliceSizes.getD d 0 ≤ so d) ∧ ∀ d ∈ dn.operandBatchingDims, sliceSizes.getD d 0 = 1

instance {so : Shape n} {si : Shape ni} {dn : GatherDimensionNumbers} {sliceSizes : List ℕ} :
    Decidable (GatherLowers so si dn sliceSizes) :=
  inferInstanceAs (Decidable (_ → _))

/-- The entry of the array of integers `a` at the position `f`, and `0` if `f` is not a position of `a`. -/
def readD {s : Shape n} (a : Index s → ℤ) (f : ℕ → ℕ) : ℤ :=
  if h : ∀ d : Fin n, f d < s d then a fun d => ⟨f d, h d⟩ else 0

/-- The position `operand_index` of the operand that belongs to position `j` of the result of a gather,
or of the updates of a scatter. It is `none` if the window of `j` does not lie inside the operand.
[StableHLO `gather`](https://openxla.org/stablehlo/spec#gather) defines
`operand_index = full_start_index + full_batching_index + full_offset_index`:

* `batchIndex a` is `batch_index[a]`, the position of `j` on batch axis `a` of the result,
  which is its position on axis `a` of the indices.
* `start d` is `full_start_index[d]`. If `d` is `start_index_map[c]`, it is entry `c` of the index vector
  at `batch_index`, and 0 otherwise. If `clip`, it is clamped to `[0, dim(operand, d) - slice_sizes[d]]`.
* `batching d` is `full_batching_index[d]`. If `d` is `operand_batching_dims[b]`,
  it is `batch_index[start_indices_batching_dims[b]]`, and 0 otherwise.
* `offset d` is `full_offset_index[d]`. If `d` is window axis `q` of the operand,
  it is the position of `j` on axis `offset_dims[q]`, and 0 otherwise.

The window of `j` lies inside the operand if `0 ≤ start d` and `start d + slice_sizes[d] ≤ dim(operand, d)`
for every axis `d`. XLA clamps `start` in every gather, so `clip` is false only for the mask of `_gather_fill`.
XLA does not clamp in a scatter: it drops the whole window if it does not lie inside the operand,
where the formula of [StableHLO `scatter`](https://openxla.org/stablehlo/spec#scatter) drops single entries.
`scripts/shapes.py` compares these positions with those of JAX on concrete indices. -/
def gatherIndex (so : Shape n) (si : Shape ni) (dn : GatherDimensionNumbers) (sliceSizes : List ℕ)
    (clip : Bool) (indices : Index si → ℤ) (j : ℕ → ℕ) : Option (Index so) :=
  let batchIndex : ℕ → ℕ := fun a =>
    j ((batchDims (dn.offsetDims.length + (ni - 1)) dn.offsetDims).getD a 0)
  let start : ℕ → ℤ := fun d =>
    if d ∈ dn.startIndexMap then
      let v := readD indices fun a => if a = ni - 1 then dn.startIndexMap.idxOf d else batchIndex a
      if clip then (v ⊔ 0) ⊓ ((so.sizeD d : ℤ) - (sliceSizes.getD d 0 : ℤ)) else v
    else 0
  let batching : ℕ → ℕ := fun d =>
    if d ∈ dn.operandBatchingDims then
      batchIndex (dn.startIndicesBatchingDims.getD (dn.operandBatchingDims.idxOf d) 0)
    else 0
  let offset : ℕ → ℕ := fun d =>
    if d ∈ windowDims n dn then j (dn.offsetDims.getD ((windowDims n dn).idxOf d) 0) else 0
  if h : ∀ d : Fin n, 0 ≤ start d ∧ start d + (sliceSizes.getD d 0 : ℤ) ≤ (so d : ℤ) ∧
      start d + (batching d : ℤ) + (offset d : ℤ) < (so d : ℤ) then
    some fun d => ⟨(start d + (batching d : ℤ) + (offset d : ℤ)).toNat, by have := h d; omega⟩
  else none

/-- `lax.gather`: every position of the result reads the position `gatherIndex` of the operand.
`jax/_src/lax/slicing.py:322`; `unique_indices` and `indices_are_sorted` are dropped.
It lowers to [StableHLO `gather`](https://openxla.org/stablehlo/spec#gather), which clamps the windows
into the operand. In the mode `FILL_OR_DROP`, `_gather_fill` then replaces every window
that did not lie inside the operand by `fill_value`, `jax/_src/lax/slicing.py:2177`:
```
  mask = lax.bitwise_and(
      lax.ge(indices, index_dtype.type(0)),
      lax.le(indices, lax.expand_dims(upper_bound, tuple(range(num_batch_dims)))))
  mask = lax.reduce_and(mask, [num_batch_dims])
  ...
  return lax.select(
    lax.broadcast_in_dim(mask, output_shape, batch_dims_in_output, ...),
    gather_out, lax.full_like(gather_out, fill_value=fill_value))
```
`fillValue` is the `fill_value` of the mode `FILL_OR_DROP`, whose default for floating-point operands is NaN.
In the other modes it is `None`, and the clamped windows lie inside the operand.
The proof `_h` is the shape check of `_gather_shape_rule`, and `_hl` the constraints of XLA. -/
def gather {so : Shape n} {si : Shape ni} (dimensionNumbers : GatherDimensionNumbers)
    (sliceSizes : List ℕ) (mode : GatherScatterMode) (fillValue : Scalar)
    (_h : GatherValid so si dimensionNumbers sliceSizes)
    (_hl : GatherLowers so si dimensionNumbers sliceSizes)
    (operand : Tensor (Index so)) (startIndices : Index si → ℤ) :
    Tensor (Index (gatherShape si dimensionNumbers sliceSizes)) :=
  fun j =>
    match gatherIndex so si dimensionNumbers sliceSizes (mode != .fillOrDrop) startIndices j.val with
    | some i => operand i
    | none => fillValue

/-- The slice sizes of the windows of a scatter: the size of the updates on axis `update_window_dims[q]`
for window axis `q` of the operand, and 1 for the other axes.
The rules of `scatter` and `_clamp_scatter_indices` compute them, `jax/_src/lax/slicing.py:2821`:
```
  slice_sizes = []
  pos = 0
  for i in range(len(operand.shape)):
    if i in dnums.inserted_window_dims or i in dnums.operand_batching_dims:
      slice_sizes.append(1)
    else:
      slice_sizes.append(updates.shape[dnums.update_window_dims[pos]])
      pos += 1
``` -/
def scatterSliceSizes (n : ℕ) (su : Shape nu) (dn : ScatterDimensionNumbers) : List ℕ :=
  (List.range n).map fun d =>
    if d ∈ windowDims n dn.toGather then
      su.sizeD (dn.updateWindowDims.getD ((windowDims n dn.toGather).idxOf d) 0)
    else 1

/-- The checks of `_scatter_shape_rule`, `jax/_src/lax/slicing.py:2458`, in its order:

* the indices have an index vector axis,
* the updates have one axis per axis of the indices other than the index vector, and one per window axis,
* `update_window_dims` is strictly increasing and lists axes of the updates,
* `inserted_window_dims` and `operand_batching_dims` are strictly increasing and list axes of the operand,
* `operand_batching_dims` shares no axis with `inserted_window_dims` or `scatter_dims_to_operand_dims`,
* `scatter_indices_batching_dims` does not repeat and lists axes of the indices other than the index vector,
* the batching axes of the operand, of the indices and of the updates have the same sizes,
* every axis of the operand is a window, inserted or batching axis,
* `scatter_dims_to_operand_dims` has one entry per entry of the index vector,
  lists axes of the operand and does not repeat,
* no window of the updates is larger than the operand,
* the scatter axes of the updates have the sizes of the axes of the indices. -/
def ScatterValid (so : Shape n) (si : Shape ni) (su : Shape nu) (dn : ScatterDimensionNumbers) :
    Prop :=
  0 < ni ∧
  nu = ni - 1 + dn.updateWindowDims.length ∧
  dn.updateWindowDims.Pairwise (· < ·) ∧
  (∀ d ∈ dn.updateWindowDims, d < nu) ∧
  dn.insertedWindowDims.Pairwise (· < ·) ∧
  (∀ d ∈ dn.insertedWindowDims, d < n) ∧
  dn.operandBatchingDims.Pairwise (· < ·) ∧
  (∀ d ∈ dn.operandBatchingDims, d < n) ∧
  (∀ d ∈ dn.insertedWindowDims, d ∉ dn.operandBatchingDims) ∧
  (∀ d ∈ dn.scatterDimsToOperandDims, d ∉ dn.operandBatchingDims) ∧
  dn.scatterIndicesBatchingDims.Nodup ∧
  (∀ d ∈ dn.scatterIndicesBatchingDims, d < ni - 1) ∧
  dn.operandBatchingDims.map so.sizeD = dn.scatterIndicesBatchingDims.map si.sizeD ∧
  dn.operandBatchingDims.map so.sizeD = dn.scatterIndicesBatchingDims.map
    (fun a => su.sizeD ((batchDims nu dn.updateWindowDims).getD a 0)) ∧
  n = dn.updateWindowDims.length + dn.insertedWindowDims.length + dn.operandBatchingDims.length ∧
  dn.scatterDimsToOperandDims.length = si.sizeD (ni - 1) ∧
  (∀ d ∈ dn.scatterDimsToOperandDims, d < n) ∧
  dn.scatterDimsToOperandDims.Nodup ∧
  (∀ q < dn.updateWindowDims.length,
    su.sizeD (dn.updateWindowDims.getD q 0) ≤ so.sizeD ((windowDims n dn.toGather).getD q 0)) ∧
  (∀ a < ni - 1, su.sizeD ((batchDims nu dn.updateWindowDims).getD a 0) = si.sizeD a)

instance {so : Shape n} {si : Shape ni} {su : Shape nu} {dn : ScatterDimensionNumbers} :
    Decidable (ScatterValid so si su dn) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- `lax.scatter_add`: every entry of the updates is added to the position `gatherIndex` of the operand,
and the windows that do not lie inside the operand are dropped.
`jax/_src/lax/slicing.py:472`; `unique_indices` and `indices_are_sorted` are dropped.
It lowers to [StableHLO `scatter`](https://openxla.org/stablehlo/spec#scatter) with the update `add`.
In the mode `CLIP`, `_scatter_lower` first clamps the indices with `_clamp_scatter_indices`,
`jax/_src/lax/slicing.py:2830`:
```
  upper_bounds: core.Shape = tuple(operand.shape[i] - slice_sizes[i]
                                   for i in dnums.scatter_dims_to_operand_dims)
  ...
  return lax.clamp(indices.dtype.type(0), indices, upper_bound)
```
The proof `_h` is the shape check of `_scatter_shape_rule`.

XLA does not compute this for several empty index vectors, where `scatter_dims_to_operand_dims` is empty
and the indices have a shape such as `(3, 0)`: it fails to compile for an operand of rank 0
(`bugs/scatter-scalar-operand-compile.json`) and returns `operand + updates`
for updates of the shape of the operand (`bugs/scatter-empty-index-vector-elementwise.json`).
The model follows StableHLO there. -/
def scatterAdd {so : Shape n} {si : Shape ni} {su : Shape nu}
    (dimensionNumbers : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (_h : ScatterValid so si su dimensionNumbers)
    (operand : Tensor (Index so)) (scatterIndices : Index si → ℤ) (updates : Tensor (Index su)) :
    Tensor (Index so) :=
  fun k => operand k + ∑ u with gatherIndex so si dimensionNumbers.toGather
    (scatterSliceSizes n su dimensionNumbers) (mode == .clip) scatterIndices u.val = some k, updates u

/-- `lax.scatter_sub`: every entry of the updates is subtracted from the position `gatherIndex` of the operand,
and the windows that do not lie inside the operand are dropped, as in `scatterAdd`.
`jax/_src/lax/slicing.py:557`; `unique_indices` and `indices_are_sorted` are dropped.
It lowers to [StableHLO `scatter`](https://openxla.org/stablehlo/spec#scatter) with the update `subtract`. -/
def scatterSub {so : Shape n} {si : Shape ni} {su : Shape nu}
    (dimensionNumbers : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (_h : ScatterValid so si su dimensionNumbers)
    (operand : Tensor (Index so)) (scatterIndices : Index si → ℤ) (updates : Tensor (Index su)) :
    Tensor (Index so) :=
  fun k => operand k + ∑ u with gatherIndex so si dimensionNumbers.toGather
    (scatterSliceSizes n su dimensionNumbers) (mode == .clip) scatterIndices u.val = some k, -updates u

end GatherScatter

/-! ## Helpers of differentiation rules -/

/-- The axes of rank `n` that are listed in `dimensions`. -/
def dimsSet (n : ℕ) (dimensions : Finset ℕ) : Finset (Fin n) := {p | (p : ℕ) ∈ dimensions}

theorem card_dimsSet {N : ℕ} {dimensions : Finset ℕ} (h : ∀ i ∈ dimensions, i < N) :
    (dimsSet N dimensions).card = dimensions.card := by
  rw [← Finset.card_map Fin.valEmbedding]
  congr 1
  ext i
  simp only [Finset.mem_map, dimsSet, Finset.mem_filter, Finset.mem_univ, true_and,
    Fin.valEmbedding_apply]
  exact ⟨fun ⟨p, hp, hpi⟩ => hpi ▸ hp, fun hi => ⟨⟨i, h i hi⟩, hi, rfl⟩⟩

theorem card_compl_dimsSet {k : ℕ} {dimensions : Finset ℕ}
    (h : ∀ i ∈ dimensions, i < k + dimensions.card) :
    (dimsSet (k + dimensions.card) dimensions)ᶜ.card = k := by
  rw [Finset.card_compl, Fintype.card_fin, card_dimsSet h, Nat.add_sub_cancel]

/-- The shape `result_shape` of `expand_dims`: size 1 on the axes in `dims`,
and the axes of `s` in order on the others.
`jax/_src/lax/lax.py:3911`:
```
  result_shape = list(np.shape(array))
  for i in sorted(dims_set):
    result_shape.insert(i, 1)
``` -/
def expandDimsShape {k n : ℕ} (s : Shape k) (dims : Finset (Fin n))
    (h : dimsᶜ.card = k) : Shape n :=
  fun p => if hp : p ∈ dims then 1 else s (Fin.cast h (keptAxesIdx dims p hp))

theorem expandDimsShape_broadcastsTo {k n : ℕ} (s : Shape k) (dims : Finset (Fin n))
    (h : dimsᶜ.card = k) :
    BroadcastsTo s (expandDimsShape s dims h ∘ (keptAxes dims ∘ Fin.cast h.symm)) :=
  fun d => Or.inr <| by
    simp only [Function.comp_apply, expandDimsShape, keptAxes_notMem, ↓reduceDIte,
      keptAxesIdx_keptAxes, Fin.cast_cast, Fin.cast_eq_self]

/-- `lax.expand_dims`, which inserts axes of size 1 at the positions `dimensions` of the result.
Negative axes are not modelled.
The proof `h` is the bounds check of `canonicalize_axis`, `jax/_src/util.py:458`.
`jax/_src/lax/lax.py:3904`:
```
def expand_dims(array: ArrayLike, dimensions: Sequence[int]) -> Array:
  """Insert any number of size 1 dimensions into an array."""
  if len(set(dimensions)) != len(dimensions):
    raise ValueError(f'repeated axis in lax.expand_dims: {dimensions}')
  ndim_out = np.ndim(array) + len(dimensions)
  dims = [canonicalize_axis(i, ndim_out) for i in dimensions]
  if len(set(dims)) != len(dims):  # check again after canonicalizing
    raise ValueError(f'repeated axis in lax.expand_dims: {dims}')
  dims_set = frozenset(dims)
  result_shape = list(np.shape(array))
  for i in sorted(dims_set):
    result_shape.insert(i, 1)
  broadcast_dims = [i for i in range(ndim_out) if i not in dims_set]
  return broadcast_in_dim(array, result_shape, broadcast_dims)
``` -/
def expandDims {k : ℕ} {s : Shape k} (dimensions : Finset ℕ)
    (h : ∀ i ∈ dimensions, i < k + dimensions.card) (array : Tensor (Index s)) :
    Tensor (Index (expandDimsShape s (dimsSet _ dimensions) (card_compl_dimsSet h))) :=
  broadcastInDim _ (keptAxes (dimsSet _ dimensions) ∘ Fin.cast (card_compl_dimsSet h).symm)
    (expandDimsShape_broadcastsTo _ _ _) ((keptAxes_injective _).comp (Fin.cast_injective _)) array

open scoped Classical in
/-- `_replace_zero`, which replaces the zeros of `x` by ones.
The rank-0 constant broadcasts to the shape of `x`, so the result has the shape of `x`.
`jax/_src/lax/lax.py:5431`: `_replace_zero = lambda x: select(eq(x, _const(x, 0)), _ones(x), x)` -/
noncomputable def replaceZero (x : Tensor ι) : Tensor ι :=
  fun i => if (x i).eq 0 then 1 else x i

open scoped Classical in
/-- One entry of `_balanced_cmp`: `1` if `a > b`, `1/2` if `a == b`, `0` otherwise. -/
noncomputable def balancedCmpScalar (a b : Scalar) : Scalar :=
  if a.gt b then 1 else if a.eq b then .fin (1 / 2) else 0

/-- Helper of the JVP rules of `max` and `min`.
The boolean masks and `select` are folded into an `if`.
`jax/_src/lax/lax.py:10489`:
```
def _balanced_cmp(x, y):
  # 1.0 if x > y, 0.5 if x == y, 0.0 if x < y or NaN
  gt_mask = gt(x, y)
  eq_mask = eq(x, y)
  ones = full_like(gt_mask, 1, dtype=x.dtype)
  zeros = full_like(gt_mask, 0, dtype=x.dtype)
  half = full_like(gt_mask, 0.5, dtype=x.dtype)
  return select(gt_mask, ones, select(eq_mask, half, zeros))
``` -/
noncomputable def balancedCmp (h : Broadcastable sx sy) (x : Tensor (Index sx))
    (y : Tensor (Index sy)) : Tensor (Index (broadcastShape sx sy)) :=
  binop balancedCmpScalar h x y

theorem prod_unbroadcast {sa sx : Shape n} (h : BroadcastsTo sa sx) :
    ∏ a, (sx ∘ keptAxes {d | sx d ≠ sa d}) a = ∏ d, sa d := by
  refine Fintype.prod_of_injective _ (keptAxes_injective _) _ _ (fun d hd => ?_) fun a => ?_
  · obtain hd' : sx d ≠ sa d := by
      by_contra hc
      exact hd (keptAxes_surj (by simpa using hc))
    have := h d
    omega
  · have := keptAxes_notMem {d | sx d ≠ sa d} a
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_not] at this
    exact this

/-- `_unbroadcast`: the sum of `x` over the axes on which its shape differs from the shape `sa` of `aval`,
reshaped to `sa`. It transposes the implicit broadcast of an array of shape `sa` to the shape of `x`.
Only arrays of equal rank are modelled, and sharding is dropped.
If no axes differ, `_unbroadcast` skips `reduce_sum` or returns `x`, which computes the same array.
The proof `h` is the assertion under `enable_checks` that `aval` has size 1 on the axes in `dims`.
`jax/_src/lax/lax.py:4637`:
```
def _unbroadcast(aval, x):
  if not isinstance(aval, ShapedArray):
    raise TypeError(
        'transpose with implicit broadcasting of unshaped values. Got'
        f' {type(aval)}')
  x_shape = np.shape(x)
  if (core.definitely_equal_shape(aval.shape, x_shape) and
      aval.sharding == typeof(x).sharding):
    return x
  assert not aval.shape or len(x_shape) == len(aval.shape)
  if not aval.shape:
    return reduce_sum(x, list(range(len(x_shape))))
  else:
    dims = [i for i, (a, b) in enumerate(zip(x_shape, aval.shape))
            if not core.definitely_equal(a, b)]
    if config.enable_checks.value:
      assert all(aval.shape[i] == 1 for i in dims)
    x = reduce_sum(x, dims) if dims else x
    return reshape(x, aval.shape, out_sharding=aval.sharding)
``` -/
def unbroadcast {sx : Shape n} (sa : Shape n) (h : BroadcastsTo sa sx) (x : Tensor (Index sx)) :
    Tensor (Index sa) :=
  reshape sa (prod_unbroadcast h) (reduceSum {d | sx d ≠ sa d} x)

theorem prod_maybeBroadcast {sx target : Shape n} (h : BroadcastsTo sx target) :
    ∏ d, sx d = ∏ a, (sx ∘ keptAxes {d | sx d ≠ target d}) a := by
  refine (Fintype.prod_of_injective _ (keptAxes_injective _) _ _ (fun d hd => ?_) fun _ => rfl).symm
  by_contra hne
  refine hd (keptAxes_surj ?_)
  have := h d
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_not]
  omega

theorem maybeBroadcast_broadcastsTo (sx target : Shape n) :
    BroadcastsTo (sx ∘ keptAxes {d | sx d ≠ target d}) (target ∘ keptAxes {d | sx d ≠ target d}) :=
  fun a => Or.inr (by simpa using keptAxes_notMem {d | sx d ≠ target d} a)

/-- `_maybe_broadcast`, which broadcasts `x` to the shape `target`:
it drops the axes on which the shapes differ with `reshape` and adds them back with `broadcast_in_dim`.
Only arrays of the rank of `target` are modelled, and sharding is dropped.
If the shapes are equal, `_maybe_broadcast` returns `x`, which is the same array.
The proof `h` is the constraint of StableHLO `reshape` that the sizes agree,
which `_reshape_shape_rule` does not check.
It holds if `x` broadcasts to `target` (`prod_maybeBroadcast`): then the dropped axes have size 1.
`jax/_src/lax/lax.py:4657`:
```
def _maybe_broadcast(target_shape, x, target_sharding):
  x_shape = np.shape(x)
  x_sharding = typeof(x).sharding
  if (core.definitely_equal_shape(x_shape, target_shape) and
      x_sharding == target_sharding):
    return x
  elif not x_shape:
    return broadcast_in_dim(x, target_shape, (), out_sharding=target_sharding)
  else:
    dims = [i for i, (a, b) in enumerate(zip(x_shape, target_shape))
            if core.definitely_equal(a, b)]
    squeeze_shape = [x_shape[i] for i in dims]
    return broadcast_in_dim(reshape(x, squeeze_shape), target_shape, dims,
                            out_sharding=target_sharding)
``` -/
def maybeBroadcast {sx : Shape n} (target : Shape n)
    (h : ∏ d, sx d = ∏ a, (sx ∘ keptAxes {d | sx d ≠ target d}) a) (x : Tensor (Index sx)) :
    Tensor (Index target) :=
  broadcastInDim target (keptAxes {d | sx d ≠ target d})
    (maybeBroadcast_broadcastsTo sx target) (keptAxes_injective _)
    (reshape (sx ∘ keptAxes {d | sx d ≠ target d}) h x)

/-! ## Lemmas -/

/-- `_maybe_broadcast` reads the entry of `x` that is broadcast to `j`. -/
theorem maybeBroadcast_apply {sx target : Shape n} (h : BroadcastsTo sx target)
    (x : Tensor (Index sx)) (j : Index target) :
    maybeBroadcast target (prod_maybeBroadcast h) x j = x (j.unbroadcast h) := by
  let D : Finset (Fin n) := {d | sx d ≠ target d}
  simp only [maybeBroadcast, broadcastInDim, reshape]
  congr 1
  rw [Equiv.symm_apply_eq]
  refine Fin.ext ?_
  rw [Fin.val_cast, ← Index.rowMajor_proj (keptAxes_strictMono D) (fun d hd => ?_) (j.unbroadcast h)]
  · refine Index.rowMajor_congr (fun _ => rfl) fun a => ?_
    show (((j.proj _).unbroadcast (maybeBroadcast_broadcastsTo sx target) a : Fin _) : ℕ) =
      ((j.unbroadcast h) (keptAxes D a) : ℕ)
    rw [Index.unbroadcast_val, Index.unbroadcast_val]
    rfl
  · have hd' : sx d ≠ target d := by
      by_contra hc
      exact hd (keptAxes_surj (by simpa [D] using hc))
    have := h d
    omega

/-- A binary primitive on arrays of equal shape acts entry by entry. -/
theorem binop_refl_same {s : Shape n} (op : Scalar → Scalar → Scalar) (x y : Tensor (Index s))
    {c : Tensor (Index s)} (hc : ∀ i, op (x i) (y i) = c i) :
    Tensor.Same (binop op (Broadcastable.refl s) x y) c :=
  Tensor.same_of_shape_eq (fun d => by simp [broadcastShape]) fun i j hij => by
    rw [← hc j, binop, Index.unbroadcast_eq_of_val _ i j hij]

/-- The loop of `_integer_pow` on finite values multiplies `acc` by `x ^ y`. -/
theorem integerPowLoop_fin (x : ℝ) (y : ℕ) : ∀ acc : Option ℝ, (acc.isSome ∨ 0 < y) →
    integerPowLoop (acc.map .fin) (.fin x) y = some (.fin (acc.getD 1 * x ^ y)) := by
  induction y using Nat.strong_induction_on generalizing x with
  | _ y ih =>
  intro acc hacc
  rw [integerPowLoop.eq_def]
  by_cases hy : y = 0
  · subst hy
    cases acc <;> simp_all
  simp only [hy, ↓reduceIte]
  have key := ih (y / 2) (by omega) (if y / 2 > 0 then x * x else x)
    (if y % 2 = 1 then some (acc.getD 1 * x) else acc)
    (by split_ifs <;> simp_all; omega)
  have hacc' : (if y % 2 = 1 then some (match acc.map Scalar.fin with
      | none => Scalar.fin x | some a => a * Scalar.fin x) else acc.map .fin) =
      (if y % 2 = 1 then some (acc.getD 1 * x) else acc).map .fin := by
    cases acc <;> split_ifs <;> simp
  have hx : (if y / 2 > 0 then Scalar.fin x * .fin x else .fin x) =
      .fin (if y / 2 > 0 then x * x else x) := by
    split_ifs <;> rfl
  rw [hacc', hx, key]
  congr 2
  obtain ⟨k, hk⟩ : ∃ k, y / 2 = k := ⟨_, rfl⟩
  have hy' : y = 2 * k + y % 2 := by omega
  rw [hk]
  conv_rhs => rw [hy']
  rcases Nat.mod_two_eq_zero_or_one y with h2 | h2
  · have : k ≠ 0 := by omega
    simp only [h2, zero_ne_one, ↓reduceIte, Nat.pos_of_ne_zero this, add_zero]
    ring
  · simp only [h2, ↓reduceIte, Option.getD_some]
    split_ifs with hk0
    · ring
    · obtain rfl : k = 0 := by omega
      ring

/-- `integer_pow` of a finite value is finite where the power is defined. -/
theorem integerPowScalar_fin (y : ℤ) (x : ℝ) (h : 0 ≤ y ∨ x ≠ 0) :
    integerPowScalar y (.fin x) = .fin (x ^ y) := by
  have hloop := integerPowLoop_fin x y.natAbs none
  simp only [Option.map_none, Option.isSome_none, Bool.false_eq_true, false_or,
    Option.getD_none, one_mul] at hloop
  unfold integerPowScalar
  split_ifs with h0 hneg
  · subst h0
    rfl
  · have hx : x ≠ 0 := h.resolve_left (by omega)
    rw [hloop (by omega), Option.getD_some, Scalar.one_def,
      Scalar.fin_div_fin _ (pow_ne_zero _ hx)]
    congr 1
    rw [show y = -(y.natAbs : ℤ) by omega, zpow_neg, zpow_natCast, Int.natAbs_neg, Int.natAbs_natCast,
      one_div]
  · rw [hloop (by omega), Option.getD_some, show y = (y.natAbs : ℤ) by omega, zpow_natCast,
      Int.natAbs_natCast]

/-- `_unbroadcast` sums `x` over the indices that are broadcast from `i`. -/
theorem unbroadcast_apply {sa sx : Shape n} (h : BroadcastsTo sa sx) (x : Tensor (Index sx))
    (i : Index sa) : unbroadcast sa h x i = ∑ j with j.unbroadcast h = i, x j := by
  let D : Finset (Fin n) := {d | sx d ≠ sa d}
  have hkept : ∀ a, sx (keptAxes D a) = sa (keptAxes D a) := fun a => by
    have := keptAxes_notMem D a
    simpa [D] using this
  -- The index of the sum that `reshape` reads is `i` without the axes in `D`.
  let c : Index (sx ∘ keptAxes D) := fun a => ⟨i (keptAxes D a), by
    rw [Function.comp_apply, hkept]; exact (i _).isLt⟩
  have hc : Index.rowMajor.symm (Fin.cast (prod_unbroadcast h).symm (Index.rowMajor i)) = c := by
    rw [Equiv.symm_apply_eq]
    refine Fin.ext ?_
    rw [Fin.val_cast, Index.rowMajor_congr (t := sa ∘ keptAxes D) (fun a => hkept a)
      (j := i.proj (keptAxes D)) fun _ => rfl]
    refine (Index.rowMajor_proj (keptAxes_strictMono D) (fun d hd => ?_) i).symm
    have hd' : sx d ≠ sa d := by
      by_contra hc
      exact hd (keptAxes_surj (by simpa [D] using hc))
    have := h d
    omega
  simp only [unbroadcast, reshape, reduceSum]
  rw [hc]
  refine Finset.sum_congr (Finset.filter_congr fun j _ => ⟨fun hj => ?_, fun hj => ?_⟩) fun _ _ => rfl
  · refine Index.ext_val fun d => ?_
    rw [Index.unbroadcast_val]
    split_ifs with h1
    · have := (i d).isLt; omega
    · have hd : d ∉ D := by have := h d; simp [D]; omega
      obtain ⟨a, rfl⟩ := keptAxes_surj hd
      exact congrArg (fun k => ((k a : Fin _) : ℕ)) hj
  · refine Index.ext_val fun a => ?_
    have hja := congrArg (fun k => (k (keptAxes D a) : ℕ)) hj
    have h2 := hkept a
    have := (j (keptAxes D a)).isLt
    have := (i (keptAxes D a)).isLt
    simp only [Index.unbroadcast_val] at hja
    show (j (keptAxes D a) : ℕ) = i (keptAxes D a)
    split_ifs at hja <;> omega

end Jax.Lax
