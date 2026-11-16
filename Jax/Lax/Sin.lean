module

public import Jax.Lax.Primal
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# `sin`

Differentiation rules of the `sin` primitive of `jax/_src/lax/lax.py`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n}

/-- The JVP rule of `sin`.
`jax/_src/lax/lax.py:4928`: `ad.defjvp(sin_p, lambda g, x, accuracy: mul(g, cos(x, accuracy=accuracy)))` -/
noncomputable def sinJvp (g x : Tensor (Index s)) : Tensor (Index (broadcastShape s s)) :=
  mul (.refl s) g (cos x)

/-- The JVP rule of `sin` computes the derivative of `sin`. -/
theorem sinJvp_correct : IsJvpOf sinJvp (sin (ι := Index s)) := by
  intro x _
  refine ⟨fun y i => Real.sin (y i),
    ContinuousLinearMap.pi fun i => Real.cos (x i) • ContinuousLinearMap.proj i, ?_, ?_, ?_⟩
  · exact .of_forall fun y => rfl
  · refine hasFDerivAt_pi.2 fun i => ?_
    exact hasFDerivAt_elementwise x i (Real.hasDerivAt_sin (x i))
  · intro g
    exact binop_refl_same _ _ _ fun i => by simp [ofReal, cos, mul_comm]

end Jax.Lax
