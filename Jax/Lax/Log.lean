module

public import Jax.Lax.Primal
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# `log`

Differentiation rules of the `log` primitive of `jax/_src/lax/lax.py`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n}

/-- The JVP rule of `log`.
`jax/_src/lax/lax.py:4808`: `ad.defjvp(log_p, lambda g, x, **kwargs: div(g, x))` -/
noncomputable def logJvp (g x : Tensor (Index s)) : Tensor (Index (broadcastShape s s)) :=
  div (.refl s) g x

/-- The JVP rule of `log` computes the derivative of `log` at positive arrays. -/
theorem logJvp_correct : IsJvpOfOn logJvp (log (ι := Index s)) {x | ∀ i, 0 < x i} := by
  intro x hx
  refine ⟨fun y i => Real.log (y i),
    ContinuousLinearMap.pi fun i => (x i)⁻¹ • ContinuousLinearMap.proj i, ?_, ?_, ?_⟩
  · have hopen : IsOpen {y : Index s → ℝ | ∀ i, 0 < y i} := by
      simp only [Set.ofPred_forall]
      exact isOpen_iInter_of_finite fun i => isOpen_lt continuous_const (continuous_apply i)
    filter_upwards [hopen.mem_nhds hx] with y hy
    funext i
    simp [log, ofReal, Scalar.log_fin (hy i)]
  · refine hasFDerivAt_pi.2 fun i => ?_
    exact hasFDerivAt_elementwise x i (Real.hasDerivAt_log (hx i).ne')
  · intro g
    exact binop_refl_same _ _ _ fun i => by
      simp [ofReal, Scalar.fin_div_fin _ (hx i).ne', div_eq_inv_mul]

end Jax.Lax
