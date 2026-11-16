module

public import Jax.Lax.Primal

/-!
# `mul`

Differentiation rules of the `mul` primitive of `jax/_src/lax/lax.py`.

The transpose rules `_mul_transpose_left` and `_mul_transpose_right` are not modelled yet:
they call `_unbroadcast`, which needs `reshape`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {sx sy : Shape n}

/-- The JVP rule of `mul` for its first argument.
`jax/_src/lax/lax.py:5583`: `lambda xdot, x, y, **kwargs: mul(xdot, y, **kwargs)` -/
noncomputable def mulJvpLeft (h : Broadcastable sx sy) (xdot _x : Tensor (Index sx))
    (y : Tensor (Index sy)) : Tensor (Index (broadcastShape sx sy)) :=
  mul h xdot y

/-- The JVP rule of `mul` for its second argument.
`jax/_src/lax/lax.py:5584`: `lambda ydot, x, y, **kwargs: mul(x, ydot, **kwargs)` -/
noncomputable def mulJvpRight (h : Broadcastable sx sy) (ydot : Tensor (Index sy))
    (x : Tensor (Index sx)) (_y : Tensor (Index sy)) : Tensor (Index (broadcastShape sx sy)) :=
  mul h x ydot

/-- The JVP rules of `mul` compute the derivative of `mul`. -/
theorem mulJvp_correct (h : Broadcastable sx sy) :
    IsJvp₂Of (mulJvpLeft h) (mulJvpRight h) (mul h) := by
  intro x y _
  let ux : Index (broadcastShape sx sy) → Index sx :=
    fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  refine ⟨fun p j => p.1 (ux j) * p.2 (uy j), ContinuousLinearMap.pi fun j =>
    x (ux j) • (ContinuousLinearMap.proj (uy j) ∘L ContinuousLinearMap.snd ℝ _ _) +
    y (uy j) • (ContinuousLinearMap.proj (ux j) ∘L ContinuousLinearMap.fst ℝ _ _), ?_, ?_, ?_⟩
  · exact .of_forall fun p => rfl
  · refine hasFDerivAt_pi.2 fun j => ?_
    exact (hasFDerivAt_fst_apply (ux j) (x, y)).mul (hasFDerivAt_snd_apply (uy j) (x, y))
  · intro g k
    refine ⟨fun j => g (ux j) * y (uy j), fun j => x (ux j) * k (uy j),
      .of_eq rfl, .of_eq rfl, ?_⟩
    ext j
    simp [mul_comm, add_comm]

end Jax.Lax
