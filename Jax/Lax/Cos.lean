module

public import Jax.Lax.Primal
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# `cos`

Differentiation rules of the `cos` primitive of `jax/_src/lax/lax.py`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n}

/-- The JVP rule of `cos`.
`jax/_src/lax/lax.py:4953`: `lambda g, x, accuracy: neg(mul(g, sin(x, accuracy=accuracy)))` -/
noncomputable def cosJvp (g x : Tensor (Index s)) : Tensor (Index (broadcastShape s s)) :=
  neg (mul (.refl s) g (sin x))

/-- The JVP rule of `cos` computes the derivative of `cos`. -/
theorem cosJvp_correct : IsJvpOf cosJvp (cos (ι := Index s)) := by
  intro x _
  refine ⟨fun y i => Real.cos (y i),
    ContinuousLinearMap.pi fun i => (-Real.sin (x i)) • ContinuousLinearMap.proj i, ?_, ?_, ?_⟩
  · exact .of_forall fun y => rfl
  · refine hasFDerivAt_pi.2 fun i => ?_
    exact hasFDerivAt_elementwise x i (Real.hasDerivAt_cos (x i))
  · intro g
    exact binop_refl_same (fun a b => -(a * b)) _ _ fun i => by simp [ofReal, sin, mul_comm]

end Jax.Lax
