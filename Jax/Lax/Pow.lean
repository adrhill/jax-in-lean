module

public import Jax.Lax.Primal
public import Mathlib.Analysis.SpecialFunctions.Pow.Deriv

/-!
# `pow`

Differentiation rules of the `pow` primitive of `jax/_src/lax/lax.py`, for a floating-point exponent.

The JVP rules are correct where the base is positive.
For a constant exponent, the rule for the base is correct where `x ^ y` is differentiable in `x`,
except at `x = 0` for `y = 0`.
There `x ↦ x ^ 0` is the constant `1`, but the rule returns `0 * 0 ^ (-1) = 0 * ∞ = NaN`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {sx sy : Shape n}

theorem broadcastable_powJvpLeft₁ (h : Broadcastable sx sy) :
    Broadcastable sx (broadcastShape sy sy) :=
  fun d => by have := h d; simp only [broadcastShape]; split_ifs <;> omega

theorem broadcastable_powJvpLeft₂ (h : Broadcastable sx sy) :
    Broadcastable sy (broadcastShape sx (broadcastShape sy sy)) :=
  fun d => by have := h d; simp only [broadcastShape]; split_ifs <;> omega

theorem broadcastable_powJvpLeft₃ (h : Broadcastable sx sy) :
    Broadcastable sx (broadcastShape sy (broadcastShape sx (broadcastShape sy sy))) :=
  fun d => by have := h d; simp only [broadcastShape]; split_ifs <;> omega

theorem broadcastable_powJvpRight (h : Broadcastable sx sy) :
    Broadcastable sy (broadcastShape sx (broadcastShape sx sy)) :=
  fun d => by have := h d; simp only [broadcastShape]; split_ifs <;> omega

/-- The JVP rule of `pow` for its first argument, for a floating-point exponent.
`standard_jvp2` passes the primal result `ans` to the rule, `jax/_src/interpreters/ad.py:989`.
`x` and `y` have the same floating-point dtype, so `convert_element_type` does nothing,
and the branch for an integer exponent is not taken.
`jax/_src/lax/lax.py:5336`:
```
def _pow_jvp_lhs(g, ans, x, y):
  y_dtype = dtypes.dtype(y)
  result_dtype = dtypes.result_type(x, y)
  if result_dtype == bool:
    result_dtype = 'int32'
  x = convert_element_type(x, result_dtype)
  y = convert_element_type(y, result_dtype)
  if dtypes.issubdtype(y_dtype, np.integer):
    ...
  else:
    jac = mul(y, pow(x, sub(y, _ones(y))))
  return mul(g, jac)
``` -/
noncomputable def powJvpLeft (h : Broadcastable sx sy) (g : Tensor (Index sx))
    (_ans : Tensor (Index (broadcastShape sx sy))) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx (broadcastShape sy (broadcastShape sx (broadcastShape sy sy))))) :=
  mul (broadcastable_powJvpLeft₃ h) g
    (mul (broadcastable_powJvpLeft₂ h) y
      (pow (broadcastable_powJvpLeft₁ h) x (sub (.refl sy) y fun _ => 1)))

/-- The JVP rule of `pow` for its second argument.
`standard_jvp2` passes the primal result `ans` to the rule, `jax/_src/interpreters/ad.py:989`.
`convert_element_type` does nothing, because `y` has the dtype of the result.
`jax/_src/lax/lax.py:5355`:
```
def _pow_jvp_rhs(g, ans, x, y):
  y_dtype = dtypes.dtype(y)
  assert dtypes.issubdtype(y_dtype, np.inexact)
  return convert_element_type(mul(g, mul(log(_replace_zero(x)), ans)), y_dtype)
``` -/
noncomputable def powJvpRight (h : Broadcastable sx sy) (g : Tensor (Index sy))
    (ans : Tensor (Index (broadcastShape sx sy))) (x : Tensor (Index sx)) (_y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sy (broadcastShape sx (broadcastShape sx sy)))) :=
  mul (broadcastable_powJvpRight h) g (mul (broadcastable_left sx sy) (log (replaceZero x)) ans)

/-! ## Proofs -/

/-- The left rule of `pow` on finite arrays where `pow(x, y - 1)` is finite. -/
private theorem powJvpLeft_same (h : Broadcastable sx sy) (g : Index sx → ℝ)
    (ans : Tensor (Index (broadcastShape sx sy))) (x : Index sx → ℝ) (y : Index sy → ℝ)
    (hxy : ∀ j : Index (broadcastShape sx sy),
      (Scalar.fin (x (j.unbroadcast (broadcastsTo_left sx sy)))).pow
        (.fin (y (j.unbroadcast h.right) + -1)) =
      .fin (x (j.unbroadcast (broadcastsTo_left sx sy)) ^ (y (j.unbroadcast h.right) - 1))) :
    (powJvpLeft h (ofReal g) ans (ofReal x) (ofReal y)).Same (ofReal fun j =>
      g (j.unbroadcast (broadcastsTo_left sx sy)) * (y (j.unbroadcast h.right) *
        x (j.unbroadcast (broadcastsTo_left sx sy)) ^ (y (j.unbroadcast h.right) - 1))) := by
  refine Tensor.same_of_shape_eq (fun d => ?_) fun i j hij => ?_
  · have := h d
    simp only [broadcastShape]
    split_ifs <;> omega
  · simp only [powJvpLeft, mul, pow, sub, binop, ofReal, Index.unbroadcast_unbroadcast]
    rw [Index.unbroadcast_congr _ (broadcastsTo_left sx sy) i j hij,
      Index.unbroadcast_congr _ h.right i j hij, Scalar.one_def, Scalar.neg_fin, Scalar.fin_add_fin,
      hxy j]
    rfl

private theorem replaceZero_ofReal {ι : Type} (x : ι → ℝ) (i : ι) (hx : x i ≠ 0) :
    replaceZero (ofReal x) i = .fin (x i) := by
  simp [replaceZero, ofReal, Scalar.zero_def, hx]

/-- The right rule of `pow` on finite arrays with a positive base. -/
private theorem powJvpRight_same (h : Broadcastable sx sy) (k : Index sy → ℝ) (x : Index sx → ℝ)
    (y : Index sy → ℝ)
    (hx : ∀ j : Index (broadcastShape sx sy), 0 < x (j.unbroadcast (broadcastsTo_left sx sy))) :
    (powJvpRight h (ofReal k) (pow h (ofReal x) (ofReal y)) (ofReal x) (ofReal y)).Same
      (ofReal fun j =>
      k (j.unbroadcast h.right) * (Real.log (x (j.unbroadcast (broadcastsTo_left sx sy))) *
        x (j.unbroadcast (broadcastsTo_left sx sy)) ^ y (j.unbroadcast h.right))) := by
  refine Tensor.same_of_shape_eq (fun d => ?_) fun i j hij => ?_
  · have := h d
    simp only [broadcastShape]
    split_ifs <;> omega
  · simp only [powJvpRight, mul, pow, log, binop, Index.unbroadcast_unbroadcast]
    rw [Index.unbroadcast_congr _ (broadcastsTo_left sx sy) i j hij,
      Index.unbroadcast_congr _ h.right i j hij, replaceZero_ofReal x _ (hx j).ne']
    simp only [ofReal]
    rw [Scalar.log_fin (hx j), Scalar.pow_fin (.inl (hx j))]
    rfl

/-! ## Correctness -/

/-- The JVP rules of `pow` compute the derivative of `pow` where the base is positive. -/
theorem powJvp_correct (h : Broadcastable sx sy) :
    IsJvp₂OfOn (fun g x y => powJvpLeft h g (pow h x y) x y)
      (fun g x y => powJvpRight h g (pow h x y) x y) (pow h)
      {p | ∀ j : Index (broadcastShape sx sy), 0 < p.1 (j.unbroadcast (broadcastsTo_left sx sy))} := by
  intro x y hx
  let ux : Index (broadcastShape sx sy) → Index sx :=
    fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  refine ⟨fun p j => p.1 (ux j) ^ p.2 (uy j), ContinuousLinearMap.pi fun j =>
    (y (uy j) * x (ux j) ^ (y (uy j) - 1)) •
      (ContinuousLinearMap.proj (ux j) ∘L ContinuousLinearMap.fst ℝ _ _) +
    (x (ux j) ^ y (uy j) * Real.log (x (ux j))) •
      (ContinuousLinearMap.proj (uy j) ∘L ContinuousLinearMap.snd ℝ _ _), ?_, ?_, ?_⟩
  · have hopen : IsOpen {p : (Index sx → ℝ) × (Index sy → ℝ) | ∀ j, 0 < p.1 (ux j)} := by
      simp only [Set.ofPred_forall]
      exact isOpen_iInter_of_finite fun j => isOpen_lt continuous_const (by fun_prop)
    filter_upwards [hopen.mem_nhds hx] with p hp
    funext j
    exact Scalar.pow_fin (.inl (hp j))
  · refine hasFDerivAt_pi.2 fun j => ?_
    exact (hasFDerivAt_fst_apply (ux j) (x, y)).rpow (hasFDerivAt_snd_apply (uy j) (x, y)) (hx j)
  · intro g k
    refine ⟨_, _, powJvpLeft_same h g _ x y fun j => ?_, powJvpRight_same h k x y hx, ?_⟩
    · rw [← sub_eq_add_neg]
      exact Scalar.pow_fin (.inl (hx j))
    · funext j
      simp only [ContinuousLinearMap.pi_apply, add_apply,
        smul_apply, ContinuousLinearMap.comp_apply,
        ContinuousLinearMap.proj_apply, ContinuousLinearMap.coe_fst', ContinuousLinearMap.coe_snd',
        smul_eq_mul, Pi.add_apply, ux, uy]
      ring

/-- For a constant exponent `y`, the JVP rule of `pow` for the base computes the derivative of `x ↦ x ^ y`
where it is differentiable and finite near `x`, except at `x = 0` for `y = 0`:
where `x > 0`, or where `y` is an integer and `x ≠ 0` or `y > 0`.
The rule for `y` is not evaluated, because the tangent of a constant is a symbolic zero. -/
theorem powJvpLeft_correct (h : Broadcastable sx sy) (y : Index sy → ℝ) :
    IsJvpOfOn (fun g x => powJvpLeft h g (pow h x (ofReal y)) x (ofReal y)) (pow h · (ofReal y))
      {x | ∀ j : Index (broadcastShape sx sy),
        0 < x (j.unbroadcast (broadcastsTo_left sx sy)) ∨
        (∃ k : ℤ, y (j.unbroadcast h.right) = k) ∧
          (x (j.unbroadcast (broadcastsTo_left sx sy)) ≠ 0 ∨ 0 < y (j.unbroadcast h.right))} := by
  intro x hx
  let ux : Index (broadcastShape sx sy) → Index sx :=
    fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  -- On the domain, `pow` of finite values is the real power.
  have hfin : ∀ (a : ℝ) (j : Index (broadcastShape sx sy)),
      0 < a ∨ (∃ k : ℤ, y (uy j) = k) ∧ (a ≠ 0 ∨ 0 < y (uy j)) →
      (Scalar.fin a).pow (.fin (y (uy j))) = .fin (a ^ y (uy j)) :=
    fun a j ha => Scalar.pow_fin (ha.imp_right fun ⟨hk, ha⟩ => ⟨hk, ha.imp_right le_of_lt⟩)
  refine ⟨fun z j => z (ux j) ^ y (uy j),
    ContinuousLinearMap.pi fun j => (y (uy j) * x (ux j) ^ (y (uy j) - 1)) •
      ContinuousLinearMap.proj (ux j), ?_, ?_, fun g => ?_⟩
  · have hopen : IsOpen {z : Index sx → ℝ | ∀ j, 0 < z (ux j) ∨
        (∃ k : ℤ, y (uy j) = k) ∧ (z (ux j) ≠ 0 ∨ 0 < y (uy j))} := by
      simp only [Set.ofPred_forall, Set.ofPred_or, Set.ofPred_and]
      exact isOpen_iInter_of_finite fun j =>
        (isOpen_lt continuous_const (continuous_apply _)).union
          (isOpen_const.inter ((isOpen_ne_fun (continuous_apply _) continuous_const).union isOpen_const))
    filter_upwards [hopen.mem_nhds hx] with z hz
    funext j
    exact hfin _ j (hz j)
  · refine hasFDerivAt_pi.2 fun j => hasFDerivAt_elementwise x (ux j) (Real.hasDerivAt_rpow_const ?_)
    rcases hx j with hj | ⟨⟨k, hk⟩, hj | hj⟩
    · exact .inl hj.ne'
    · exact .inl hj
    · refine .inr ?_
      rw [hk] at hj ⊢
      exact_mod_cast (Int.cast_pos.1 hj : 0 < k)
  · have hxy : ∀ j, (Scalar.fin (x (ux j))).pow (.fin (y (uy j) + -1)) =
        .fin (x (ux j) ^ (y (uy j) - 1)) := fun j => by
      rw [← sub_eq_add_neg]
      refine Scalar.pow_fin ?_
      rcases hx j with hj | ⟨⟨k, hk⟩, hj⟩
      · exact .inl hj
      · refine .inr ⟨⟨k - 1, by rw [hk]; push_cast; ring⟩, hj.imp_right fun hj => ?_⟩
        rw [hk] at hj ⊢
        have : (1 : ℤ) ≤ k := Int.cast_pos.1 hj
        have : (1 : ℝ) ≤ k := by exact_mod_cast this
        linarith
    convert powJvpLeft_same h g _ x y hxy using 2
    funext j
    simp only [ContinuousLinearMap.pi_apply, smul_apply,
      ContinuousLinearMap.proj_apply, smul_eq_mul, ux, uy]
    ring

/-! ## Refutation -/

/-- The JVP rule of `pow` for the base is wrong at `x = 0` for the constant exponent `y = 0`.
`x ↦ x ^ 0` is the constant `1`, so its derivative is `0`,
but the rule computes `0 * 0 ^ (0 - 1) = 0 * ∞`, which is NaN. -/
theorem powJvpLeft_not_correct :
    ¬ IsJvpOf (fun g x => powJvpLeft (.refl ![1]) g (pow (.refl ![1]) x (ofReal 0)) x (ofReal 0))
      (pow (.refl ![1]) · (ofReal 0)) := by
  intro hj
  obtain ⟨_, L, _, _, hrule⟩ := hj 0 (Set.mem_univ _)
  obtain ⟨_, _, hv⟩ := hrule 0
  have := hv (fun _ => ⟨0, by simp [broadcastShape]⟩) (fun _ => ⟨0, by simp [broadcastShape]⟩)
    fun _ => rfl
  have e1 : (Scalar.fin 0).pow (.fin (0 + -1)) = .posInf := by
    simp [Scalar.pow, Scalar.zero_def, Scalar.one_def]
  have e2 : Scalar.fin 0 * .posInf = .nan := by
    show Scalar.infWithSign 0 = _
    simp [Scalar.infWithSign]
  simp only [powJvpLeft, mul, pow, sub, binop, ofReal, Pi.zero_apply, Scalar.one_def,
    Scalar.neg_fin, Scalar.fin_add_fin, e1, e2] at this
  cases this

end Jax.Lax
