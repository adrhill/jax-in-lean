module

public import Jax.Lax.Primal
public import Mathlib.Analysis.Calculus.Deriv.Inv

/-!
# `div`

Differentiation rules of the `div` primitive of `jax/_src/lax/lax.py`.

The transpose rule `_div_transpose_rule` is not modelled yet.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {sx sy : Shape n}

/-- The JVP rule of `div` for its first argument.
`jax/_src/lax/lax.py:5628`: `lambda g, x, y: div(g, y)` -/
noncomputable def divJvpLeft (h : Broadcastable sx sy) (g _x : Tensor (Index sx))
    (y : Tensor (Index sy)) : Tensor (Index (broadcastShape sx sy)) :=
  div h g y

/-- The JVP rule of `div` for its second argument.
`jax/_src/lax/lax.py:5629`: `lambda g, x, y: mul(mul(neg(g), x), integer_pow(y, -2))` -/
noncomputable def divJvpRight (h : Broadcastable sx sy) (g : Tensor (Index sy))
    (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape (broadcastShape sy sx) sy)) :=
  mul (broadcastsTo_left sy sx).broadcastable.symm (mul h.symm (neg g) x) (integerPow (-2) y)

/-- The right rule of `div` on finite arrays where `y` has no zeros. -/
private theorem divJvpRight_same (h : Broadcastable sx sy) (k : Index sy → ℝ) (x : Index sx → ℝ)
    (y : Index sy → ℝ) (hy : ∀ j : Index (broadcastShape sx sy), y (j.unbroadcast h.right) ≠ 0) :
    (divJvpRight h (ofReal k) (ofReal x) (ofReal y)).Same (ofReal fun j =>
      -k (j.unbroadcast h.right) * x (j.unbroadcast (broadcastsTo_left sx sy)) *
        y (j.unbroadcast h.right) ^ (-2 : ℤ)) := by
  refine Tensor.same_of_shape_eq (fun d => ?_) fun i j hij => ?_
  · have := h d
    simp only [broadcastShape]
    split_ifs <;> omega
  · have h1 : (i.unbroadcast (broadcastsTo_left sy sx).broadcastable.symm.right) =
        j.unbroadcast h.right := Index.unbroadcast_congr _ _ i j hij
    have h2 : (i.unbroadcast (broadcastsTo_left (broadcastShape sy sx) sy)).unbroadcast
        (broadcastsTo_left sy sx) = j.unbroadcast h.right := by
      refine Index.ext_val fun d => ?_
      have := h d; have := hij d; have := (j d).isLt
      simp only [Index.unbroadcast_val, broadcastShape] at this ⊢
      split_ifs at this ⊢ <;> omega
    have h3 : (i.unbroadcast (broadcastsTo_left (broadcastShape sy sx) sy)).unbroadcast
        h.symm.right = j.unbroadcast (broadcastsTo_left sx sy) := by
      refine Index.ext_val fun d => ?_
      have := h d; have := hij d; have := (j d).isLt
      simp only [Index.unbroadcast_val, broadcastShape] at this ⊢
      split_ifs at this ⊢ <;> omega
    simp only [divJvpRight, mul, binop, neg, integerPow, ofReal]
    rw [h1, h2, h3, integerPowScalar_fin _ _ (Or.inr (hy j))]
    rfl

/-- The JVP rules of `div` compute the derivative of `div` where the divisor has no zeros. -/
theorem divJvp_correct (h : Broadcastable sx sy) :
    IsJvp₂OfOn (divJvpLeft h) (divJvpRight h) (div h)
      {p | ∀ j : Index (broadcastShape sx sy), p.2 (j.unbroadcast h.right) ≠ 0} := by
  intro x y hy
  let ux : Index (broadcastShape sx sy) → Index sx :=
    fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  refine ⟨fun p j => p.1 (ux j) / p.2 (uy j), ContinuousLinearMap.pi fun j =>
    x (ux j) • ((ContinuousLinearMap.toSpanSingleton ℝ (-(y (uy j) ^ 2)⁻¹)).comp
      (ContinuousLinearMap.proj (uy j) ∘L ContinuousLinearMap.snd ℝ _ _)) +
    (y (uy j))⁻¹ • (ContinuousLinearMap.proj (ux j) ∘L ContinuousLinearMap.fst ℝ _ _), ?_, ?_, ?_⟩
  · have hopen : IsOpen {p : (Index sx → ℝ) × (Index sy → ℝ) | ∀ j, p.2 (uy j) ≠ 0} := by
      simp only [Set.ofPred_forall]
      exact isOpen_iInter_of_finite fun j => isOpen_ne_fun (by fun_prop) continuous_const
    filter_upwards [hopen.mem_nhds hy] with p hp
    funext j
    exact Scalar.fin_div_fin _ (hp j)
  · refine hasFDerivAt_pi.2 fun j => ?_
    have hinv := (hasFDerivAt_inv (hy j)).comp (x, y) (hasFDerivAt_snd_apply (ι := Index sx) (uy j) (x, y))
    have := (hasFDerivAt_fst_apply (κ := Index sy) (ux j) (x, y)).mul hinv
    convert this using 1
    · funext p
      simp [div_eq_mul_inv]
    · rfl
  · intro g k
    refine ⟨fun j => g (ux j) / y (uy j), _, .of_eq (funext fun j => ?_),
      divJvpRight_same h k x y hy, ?_⟩
    · exact Scalar.fin_div_fin _ (hy j)
    · funext j
      simp only [ContinuousLinearMap.pi_apply, add_apply, smul_apply, ux, uy,
        ContinuousLinearMap.comp_apply, ContinuousLinearMap.toSpanSingleton_apply,
        ContinuousLinearMap.proj_apply, ContinuousLinearMap.coe_snd', ContinuousLinearMap.coe_fst',
        smul_eq_mul, Pi.add_apply, zpow_neg]
      have := hy j
      field_simp
      ring

end Jax.Lax
