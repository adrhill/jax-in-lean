module

public import Jax.Lax.Primal

/-!
# `mul`

Differentiation rules of the `mul` primitive of `jax/_src/lax/lax.py`.

The transpose rules are correct also for operands that are broadcast.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {sx sy : Shape n} {w : FloatWidth}

/-- The JVP rule of `mul` for its first argument.
`ad.defbilinear(mul_p, …)` at `jax/_src/lax/lax.py:5591` replaces the rules of `ad.defjvp` at line 5582.
`jax/_src/interpreters/ad.py:1006`: `lhs_jvp = lambda g, x, y, **kwargs: prim.bind(g, y, **kwargs)` -/
noncomputable def mulJvpLeft (h : Broadcastable sx sy) (xdot _x : Tensor (.float w) (Index sx))
    (y : Tensor (.float w) (Index sy)) : Tensor (.float w) (Index (broadcastShape sx sy)) :=
  mul h xdot y

/-- The JVP rule of `mul` for its second argument.
`ad.defbilinear(mul_p, …)` at `jax/_src/lax/lax.py:5591` replaces the rules of `ad.defjvp` at line 5582.
`jax/_src/interpreters/ad.py:1007`: `rhs_jvp = lambda g, x, y, **kwargs: prim.bind(x, g, **kwargs)` -/
noncomputable def mulJvpRight (h : Broadcastable sx sy) (ydot : Tensor (.float w) (Index sy))
    (x : Tensor (.float w) (Index sx)) (_y : Tensor (.float w) (Index sy)) : Tensor (.float w) (Index (broadcastShape sx sy)) :=
  mul h x ydot

/-- The JVP rules of `mul` compute the derivative of `mul`. -/
theorem mulJvp_correct (h : Broadcastable sx sy) :
    IsJvp₂Of (mulJvpLeft h) (mulJvpRight h) (mul (w := w) h) := by
  intro x y _
  let ux : Index (broadcastShape sx sy) → Index sx :=
    fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  refine ⟨fun p j => p.1 (ux j) * p.2 (uy j), ContinuousLinearMap.pi fun j =>
    x (ux j) • (ContinuousLinearMap.proj (uy j) ∘L ContinuousLinearMap.snd ℝ _ _) +
    y (uy j) • (ContinuousLinearMap.proj (ux j) ∘L ContinuousLinearMap.fst ℝ _ _), ?_, ?_, ?_, ?_⟩
  · exact .of_forall fun p => rfl
  · refine hasFDerivAt_pi.2 fun j => ?_
    exact (hasFDerivAt_fst_apply (ux j) (x, y)).mul (hasFDerivAt_snd_apply (uy j) (x, y))
  · intro g
    refine .of_eq (funext fun j => congrArg Scalar.fin ?_)
    simp [mul_comm]
    rfl
  · intro k
    refine .of_eq (funext fun j => congrArg Scalar.fin ?_)
    simp
    rfl

/-! ## Transpose rules -/

theorem mulTransposeLeft_broadcastsTo (h : Broadcastable sx sy) :
    BroadcastsTo sx (broadcastShape (broadcastShape sx sy) sy) := fun d => by
  have := h d
  simp only [broadcastShape]
  split_ifs <;> omega

theorem mulTransposeRight_broadcastsTo (h : Broadcastable sx sy) :
    BroadcastsTo sy (broadcastShape sx (broadcastShape sx sy)) := fun d => by
  have := h d
  simp only [broadcastShape]
  split_ifs <;> omega

/-- The transpose rule of `mul` for its first argument.
The undefined primal `x` is replaced by its shape `sx`, and `out_dtype` is dropped.
`jax/_src/lax/lax.py:5585`:
```
def _mul_transpose_left(ct, x, y, *, out_dtype):
  g = mul(ct, y, out_dtype=None if out_dtype is None else x.aval.dtype)
  return _unbroadcast(x.aval, g)
``` -/
noncomputable def mulTransposeLeft (h : Broadcastable sx sy) (y : Tensor (.float w) (Index sy))
    (ct : Tensor (.float w) (Index (broadcastShape sx sy))) : Tensor (.float w) (Index sx) :=
  unbroadcast sx (mulTransposeLeft_broadcastsTo h) (mul h.right.broadcastable.symm ct y)

/-- The transpose rule of `mul` for its second argument.
The undefined primal `y` is replaced by its shape `sy`, and `out_dtype` is dropped.
`jax/_src/lax/lax.py:5588`:
```
def _mul_transpose_right(ct, x, y, *, out_dtype):
  g = mul(x, ct, out_dtype=None if out_dtype is None else y.aval.dtype)
  return _unbroadcast(y.aval, g)
``` -/
noncomputable def mulTransposeRight (h : Broadcastable sx sy) (x : Tensor (.float w) (Index sx))
    (ct : Tensor (.float w) (Index (broadcastShape sx sy))) : Tensor (.float w) (Index sy) :=
  unbroadcast sy (mulTransposeRight_broadcastsTo h) (mul (broadcastable_left sx sy) x ct)

/-- The transpose rule of `mul` for its first argument transposes `mul` with a fixed second argument. -/
theorem mulTransposeLeft_correct (h : Broadcastable sx sy) (y : Index sy → ℝ) :
    IsTransposeOf (mulTransposeLeft h (ofReal y)) (mul (w := w) h · (ofReal y)) := by
  intro ct x
  let ux : Index (broadcastShape sx sy) → Index sx := fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  refine ⟨fun j => x (ux j) * y (uy j), fun i => ∑ j with ux j = i, ct j * y (uy j), rfl,
    .of_eq (funext fun i => ?_), ?_⟩
  · rw [mulTransposeLeft, unbroadcast_apply, ofReal, Scalar.fin_sum]
    refine sum_filter_index_congr (fun d => ?_) _ _ (fun j j' hjj => ?_) fun j j' hjj => ?_
    · have := h d
      simp only [broadcastShape]
      split_ifs <;> omega
    · rw [Index.unbroadcast_congr _ (broadcastsTo_left sx sy) j j' hjj]
    · simp only [mul, binop, ofReal, Scalar.fin_mul_fin, uy]
      rw [Index.unbroadcast_eq_of_val _ j j' hjj, Index.unbroadcast_congr _ h.right j j' hjj]
  · calc ∑ j, ct j * (x (ux j) * y (uy j))
        = ∑ i, ∑ j with ux j = i, ct j * (x (ux j) * y (uy j)) := (Finset.sum_fiberwise _ _ _).symm
      _ = ∑ i, (∑ j with ux j = i, ct j * y (uy j)) * x i := by
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun j hj => ?_
          rw [(Finset.mem_filter.1 hj).2]
          ring

/-- The transpose rule of `mul` for its second argument transposes `mul` with a fixed first argument. -/
theorem mulTransposeRight_correct (h : Broadcastable sx sy) (x : Index sx → ℝ) :
    IsTransposeOf (mulTransposeRight h (ofReal x)) (mul (w := w) h (ofReal x) ·) := by
  intro ct y
  let ux : Index (broadcastShape sx sy) → Index sx := fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  refine ⟨fun j => x (ux j) * y (uy j), fun i => ∑ j with uy j = i, x (ux j) * ct j, rfl,
    .of_eq (funext fun i => ?_), ?_⟩
  · rw [mulTransposeRight, unbroadcast_apply, ofReal, Scalar.fin_sum]
    refine sum_filter_index_congr (fun d => ?_) _ _ (fun j j' hjj => ?_) fun j j' hjj => ?_
    · have := h d
      simp only [broadcastShape]
      split_ifs <;> omega
    · rw [Index.unbroadcast_congr _ h.right j j' hjj]
    · simp only [mul, binop, ofReal, Scalar.fin_mul_fin, ux]
      rw [Index.unbroadcast_eq_of_val _ j j' hjj,
        Index.unbroadcast_congr _ (broadcastsTo_left sx sy) j j' hjj]
  · calc ∑ j, ct j * (x (ux j) * y (uy j))
        = ∑ i, ∑ j with uy j = i, ct j * (x (ux j) * y (uy j)) := (Finset.sum_fiberwise _ _ _).symm
      _ = ∑ i, (∑ j with uy j = i, x (ux j) * ct j) * y i := by
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun j hj => ?_
          rw [(Finset.mem_filter.1 hj).2]
          ring

end Jax.Lax
