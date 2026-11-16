module

public import Jax.Lax.Primal
public import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# `sqrt`

Differentiation rules of the `sqrt` primitive of `jax/_src/lax/lax.py`.

The JVP rule is correct where every entry is positive.
Where an entry is zero or negative, no rule is correct:
arbitrarily close to such an array there are arrays with a negative entry, whose square root is NaN.
At `0` the rule returns `g * (0.5 / 0) = g * ∞`, which is NaN for `g = 0`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n}

/-- The JVP rule of `sqrt`.
`standard_jvp2` passes the primal result `ans` to the rule, `jax/_src/interpreters/ad.py:989`.
`_const(x, 0.5)` has rank 0, so `div` broadcasts it to the shape of `ans`.
`jax/_src/lax/lax.py:5181`:
`ad.defjvp2(sqrt_p, lambda g, ans, x, **kwargs: mul(g, div(_const(x, 0.5), ans)))` -/
noncomputable def sqrtJvp (g ans _x : Tensor (Index s)) : Tensor (Index (broadcastShape s s)) :=
  mul (.refl s) g (constDiv (.fin (1 / 2)) ans)

/-- The JVP rule of `sqrt` computes the derivative of `sqrt` at positive arrays. -/
theorem sqrtJvp_correct :
    IsJvpOfOn (fun g x => sqrtJvp g (sqrt x) x) (sqrt (ι := Index s)) {x | ∀ i, 0 < x i} := by
  intro x hx
  refine ⟨fun y i => Real.sqrt (y i),
    ContinuousLinearMap.pi fun i => (1 / (2 * Real.sqrt (x i))) • ContinuousLinearMap.proj i,
    ?_, ?_, ?_⟩
  · have hopen : IsOpen {y : Index s → ℝ | ∀ i, 0 < y i} := by
      simp only [Set.ofPred_forall]
      exact isOpen_iInter_of_finite fun i => isOpen_lt continuous_const (continuous_apply i)
    filter_upwards [hopen.mem_nhds hx] with y hy
    funext i
    simp [sqrt, ofReal, Scalar.sqrt_fin (hy i).le]
  · refine hasFDerivAt_pi.2 fun i => ?_
    exact hasFDerivAt_elementwise x i (Real.hasDerivAt_sqrt (hx i).ne')
  · intro g
    refine binop_refl_same _ _ _ fun i => ?_
    have hs : Real.sqrt (x i) ≠ 0 := (Real.sqrt_pos.2 (hx i)).ne'
    simp only [ofReal, sqrt, constDiv, Scalar.sqrt_fin (hx i).le, Scalar.fin_div_fin _ hs,
      Scalar.fin_mul_fin, ContinuousLinearMap.pi_apply, smul_apply,
      ContinuousLinearMap.proj_apply, smul_eq_mul]
    congr 1
    field_simp

/-- No JVP rule of `sqrt` is correct at an array with an entry that is not positive:
arbitrarily close to it there are arrays with a negative entry, whose square root is NaN. -/
theorem sqrt_not_isJvpOfOn (jvp : Tensor (Index s) → Tensor (Index s) → Tensor (Index s))
    {x : Index s → ℝ} (hx : ∃ i, x i ≤ 0) : ¬ IsJvpOfOn jvp (sqrt (ι := Index s)) {x} := by
  intro hj
  obtain ⟨i, hi⟩ := hx
  obtain ⟨F, -, hF, -⟩ := hj x rfl
  -- Move entry `i` of `x` to the left.
  let path : ℝ → Index s → ℝ := fun t => Function.update x i (x i - t)
  have hpath : Filter.Tendsto path (nhds 0) (nhds x) := by
    have hc : Continuous path := by
      refine continuous_pi fun k => ?_
      by_cases hk : k = i
      · subst hk
        simp only [path, Function.update_self]
        exact continuous_const.sub continuous_id
      · simp only [path, Function.update_of_ne hk]
        exact continuous_const
    have h0 : path 0 = x := by simp [path]
    exact h0 ▸ hc.tendsto 0
  obtain ⟨t, ht, ht0⟩ := ((hpath.eventually hF).filter_mono nhdsWithin_le_nhds).and
    (self_mem_nhdsWithin (s := Set.Ioi (0 : ℝ))) |>.exists
  have ht0 : 0 < t := ht0
  have := congrFun ht i
  simp only [sqrt, ofReal, path, Function.update_self] at this
  rw [Scalar.sqrt_fin_of_neg (by linarith)] at this
  cases this

end Jax.Lax
