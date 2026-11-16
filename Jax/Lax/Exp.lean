module

public import Jax.Lax.Primal
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# `exp`

Differentiation rules of the `exp` primitive of `jax/_src/lax/lax.py`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n}

/-- The JVP rule of `exp`.
`standard_jvp2` passes the primal result `ans` to the rule, `jax/_src/interpreters/ad.py:989`.
`jax/_src/lax/lax.py:4785`: `ad.defjvp2(exp_p, lambda g, ans, x, **kwargs: mul(g, ans))` -/
noncomputable def expJvp (g ans _x : Tensor (Index s)) : Tensor (Index (broadcastShape s s)) :=
  mul (.refl s) g ans

/-- The JVP rule of `exp` computes the derivative of `exp`. -/
theorem expJvp_correct : IsJvpOf (fun g x => expJvp g (exp x) x) (exp (ι := Index s)) := by
  intro x _
  refine ⟨fun y i => Real.exp (y i),
    ContinuousLinearMap.pi fun i => Real.exp (x i) • ContinuousLinearMap.proj i, ?_, ?_, ?_⟩
  · exact .of_forall fun y => rfl
  · refine hasFDerivAt_pi.2 fun i => ?_
    exact hasFDerivAt_elementwise x i (Real.hasDerivAt_exp (x i))
  · intro g
    exact binop_refl_same _ _ _ fun i => by simp [ofReal, exp, mul_comm]

end Jax.Lax
