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

/-! ## Elementwise binary primitives -/

/-- An elementwise binary primitive, which broadcasts its arguments against each other.
The proof `h` is the shape check of `broadcasting_shape_rule`, `jax/_src/lax/lax.py:4503`. -/
def binop {sx sy : Shape n} (op : Scalar → Scalar → Scalar) (h : Broadcastable sx sy)
    (x : Tensor (Index sx)) (y : Tensor (Index sy)) : Tensor (Index (broadcastShape sx sy)) :=
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

/-- `lax.max`, elementwise maximum. `jax/_src/lax/lax.py:1385` -/
noncomputable def max (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  binop Scalar.max h x y

/-! ## Structural primitives -/

/-- `lax.broadcast_in_dim`: axis `d` of the operand becomes axis `broadcastDimensions d` of the result.
`jax/_src/lax/lax.py:2946`; `out_sharding` is dropped.
The proof `h` is the shape check of `_broadcast_in_dim_shape_rule`, `jax/_src/lax/lax.py:7474`.
The rule also rejects duplicates in `broadcast_dimensions`, which the semantics do not need. -/
def broadcastInDim {so : Shape m} (shape : Shape n) (broadcastDimensions : Fin m → Fin n)
    (h : BroadcastsTo so (shape ∘ broadcastDimensions)) (operand : Tensor (Index so)) :
    Tensor (Index shape) :=
  fun j => operand ((j.proj broadcastDimensions).unbroadcast h)

/-- `lax.reduce_sum`, the sum over `axes`. The result keeps the other axes in order.
`jax/_src/lax/lax.py:3412`; `out_sharding` is dropped -/
def reduceSum {s : Shape n} (axes : Finset (Fin n)) (operand : Tensor (Index s)) :
    Tensor (Index (s ∘ keptAxes axes)) :=
  fun j => ∑ i with i.proj (keptAxes axes) = j, operand i

/-! ## Helpers of differentiation rules -/

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

/-! ## Lemmas -/

/-- A binary primitive on arrays of equal shape acts entry by entry. -/
theorem binop_refl_same {s : Shape n} (op : Scalar → Scalar → Scalar) (x y : Tensor (Index s))
    {c : Tensor (Index s)} (hc : ∀ i, op (x i) (y i) = c i) :
    (binop op (Broadcastable.refl s) x y).Same c :=
  Tensor.same_of_shape_eq (fun d => by simp [broadcastShape]) fun i j hij => by
    rw [← hc j, binop, Index.unbroadcast_eq_of_val _ i j hij]

end Jax.Lax
