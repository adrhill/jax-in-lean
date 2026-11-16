module

public import Jax.Lax.Primal

/-!
# `rem`

Differentiation rules of the `rem` primitive of `jax/_src/lax/lax.py`.

`rem(x, y)` is `x - d * y`, where `d` is `x / y` rounded toward zero.
It is differentiable where `y ≠ 0` and `x / y` is not a nonzero integer, and there the rules are correct
(`remJvp_correct`). At the nonzero integers `rem` jumps, and `rem(x, 0)` is NaN.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {sx sy : Shape n}

/-- The JVP rule of `rem` for its first argument, which broadcasts the tangent to the shape of the result.
The proof `_h` is the shape check of `broadcast_shapes`, and sharding is dropped.
`jax/_src/lax/lax.py:5636`:
```
lambda g, x, y: _maybe_broadcast(
    broadcast_shapes(np.shape(x), np.shape(y)), g,
    broadcast_shardings(typeof(x), typeof(y))),
``` -/
def remJvpLeft (_h : Broadcastable sx sy) (g _x : Tensor (Index sx)) (_y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  maybeBroadcast (broadcastShape sx sy) (prod_maybeBroadcast (broadcastsTo_left sx sy)) g

theorem remJvpRight_broadcastable (h : Broadcastable sx sy) :
    Broadcastable sy (broadcastShape (broadcastShape sx sy) (broadcastShape sx sy)) := fun d => by
  have := h d
  simp only [broadcastShape]
  split_ifs <;> omega

/-- The JVP rule of `rem` for its second argument.
`jax/_src/lax/lax.py:5639`:
`lambda g, x, y: mul(neg(g), mul(sign(div(x, y)), floor(abs(div(x, y)))))` -/
noncomputable def remJvpRight (h : Broadcastable sx sy) (g : Tensor (Index sy))
    (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sy
      (broadcastShape (broadcastShape sx sy) (broadcastShape sx sy)))) :=
  mul (remJvpRight_broadcastable h) (neg g)
    (mul (.refl _) (sign (div h x y)) (floor (abs (div h x y))))

/-! ## Proofs -/

/-- Rounding toward zero is the sign times the floor of the absolute value, as the rule computes it. -/
private theorem truncReal_eq (q : ℝ) : Scalar.truncReal q = Real.sign q * ⌊|q|⌋ := by
  unfold Scalar.truncReal
  rcases lt_trichotomy q 0 with hq | rfl | hq
  · simp only [not_le.2 hq, ↓reduceIte, Real.sign_of_neg hq, abs_of_neg hq, Int.floor_neg]
    push_cast
    ring
  · simp
  · simp only [hq.le, ↓reduceIte, Real.sign_of_pos hq, abs_of_pos hq, one_mul]

/-- Rounding toward zero is locally constant away from the nonzero integers. -/
private theorem truncReal_eventuallyEq {q : ℝ} (hq : ∀ k : ℤ, k ≠ 0 → q ≠ k) :
    ∀ᶠ r in nhds q, Scalar.truncReal r = Scalar.truncReal q := by
  unfold Scalar.truncReal
  rcases lt_trichotomy q 0 with hneg | rfl | hpos
  · -- Between `⌈q⌉ - 1` and `⌈q⌉ ≤ 0` the ceiling is `⌈q⌉`.
    have hz : (⌈q⌉ : ℝ) ≤ 0 := by exact_mod_cast Int.ceil_le.2 (by simpa using hneg.le)
    have hne : q ≠ ⌈q⌉ := fun heq => by
      refine hq ⌈q⌉ (fun h0 => ?_) heq
      rw [h0] at heq
      simp only [Int.cast_zero] at heq
      exact hneg.ne heq
    have hlt : q < ⌈q⌉ := lt_of_le_of_ne (Int.le_ceil q) hne
    filter_upwards [Ioo_mem_nhds (Int.ceil_lt_add_one q |> sub_lt_iff_lt_add.2) hlt] with r hr
    have hr0 : ¬ 0 ≤ r := not_le.2 (hr.2.trans_le hz)
    simp only [hr0, not_le.2 hneg, ↓reduceIte]
    exact_mod_cast Int.ceil_eq_iff.2 ⟨hr.1, hr.2.le⟩
  · -- Between `-1` and `1` rounding toward zero gives `0`.
    filter_upwards [Ioo_mem_nhds (show (-1 : ℝ) < 0 by norm_num) (show (0 : ℝ) < 1 by norm_num)]
      with r hr
    simp only [le_refl, ↓reduceIte, Int.floor_zero, Int.cast_zero]
    split_ifs with h0
    · exact_mod_cast Int.floor_eq_zero_iff.2 ⟨h0, hr.2⟩
    · exact_mod_cast Int.ceil_eq_zero_iff.2 ⟨hr.1, (not_le.1 h0).le⟩
  · -- Between `0 ≤ ⌊q⌋` and `⌊q⌋ + 1` the floor is `⌊q⌋`.
    have hz : (0 : ℝ) ≤ ⌊q⌋ := by exact_mod_cast Int.floor_nonneg.2 hpos.le
    have hne : q ≠ ⌊q⌋ := fun heq => by
      refine hq ⌊q⌋ (fun h0 => ?_) heq
      rw [h0] at heq
      simp only [Int.cast_zero] at heq
      exact hpos.ne' heq
    have hlt : (⌊q⌋ : ℝ) < q := lt_of_le_of_ne (Int.floor_le q) (Ne.symm hne)
    filter_upwards [Ioo_mem_nhds hlt (Int.lt_floor_add_one q)] with r hr
    have hr0 : 0 ≤ r := hz.trans hr.1.le
    simp only [hr0, hpos.le, ↓reduceIte]
    exact_mod_cast Int.floor_eq_iff.2 ⟨hr.1.le, hr.2⟩

/-- The right rule of `rem` on finite arrays where `y` has no zeros. -/
private theorem remJvpRight_same (h : Broadcastable sx sy) (k : Index sy → ℝ) (x : Index sx → ℝ)
    (y : Index sy → ℝ) (hy : ∀ j : Index (broadcastShape sx sy), y (j.unbroadcast h.right) ≠ 0) :
    (remJvpRight h (ofReal k) (ofReal x) (ofReal y)).Same (ofReal fun j =>
      -k (j.unbroadcast h.right) *
        (Real.sign (x (j.unbroadcast (broadcastsTo_left sx sy)) / y (j.unbroadcast h.right)) *
          ⌊|x (j.unbroadcast (broadcastsTo_left sx sy)) / y (j.unbroadcast h.right)|⌋)) := by
  refine Tensor.same_of_shape_eq (fun d => ?_) fun i j hij => ?_
  · have := h d
    simp only [broadcastShape]
    split_ifs <;> omega
  · have h1 : i.unbroadcast (broadcastsTo_left sy _) = j.unbroadcast h.right :=
      Index.unbroadcast_congr _ _ i j hij
    -- Both arguments of the inner `mul` read this entry.
    have h2 : (i.unbroadcast (remJvpRight_broadcastable h).right).unbroadcast
        (broadcastsTo_left (broadcastShape sx sy) (broadcastShape sx sy)) = j := by
      refine Index.ext_val fun d => ?_
      have := h d; have := hij d; have := (j d).isLt
      simp only [Index.unbroadcast_val, broadcastShape] at this ⊢
      split_ifs at this ⊢ <;> omega
    simp only [remJvpRight, mul, binop, neg, sign, floor, abs, div, ofReal]
    rw [h1, h2, Scalar.fin_div_fin _ (hy j)]
    rfl

/-! ## Correctness -/

/-- The JVP rules of `rem` compute the derivative of `rem`
where the divisor has no zeros and no quotient is a nonzero integer. -/
theorem remJvp_correct (h : Broadcastable sx sy) :
    IsJvp₂OfOn (remJvpLeft h) (remJvpRight h) (rem h)
      {p | ∀ j : Index (broadcastShape sx sy), p.2 (j.unbroadcast h.right) ≠ 0 ∧
        ∀ k : ℤ, k ≠ 0 →
          p.1 (j.unbroadcast (broadcastsTo_left sx sy)) ≠ k * p.2 (j.unbroadcast h.right)} := by
  intro x y hxy
  let ux : Index (broadcastShape sx sy) → Index sx :=
    fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  let c : Index (broadcastShape sx sy) → ℝ := fun j => Scalar.truncReal (x (ux j) / y (uy j))
  refine ⟨fun p j => p.1 (ux j) - Scalar.truncReal (p.1 (ux j) / p.2 (uy j)) * p.2 (uy j),
    ContinuousLinearMap.pi fun j =>
      (ContinuousLinearMap.proj (ux j) ∘L ContinuousLinearMap.fst ℝ _ _) -
      c j • (ContinuousLinearMap.proj (uy j) ∘L ContinuousLinearMap.snd ℝ _ _), ?_, ?_, ?_⟩
  · have hopen : IsOpen {p : (Index sx → ℝ) × (Index sy → ℝ) | ∀ j, p.2 (uy j) ≠ 0} := by
      simp only [Set.ofPred_forall]
      exact isOpen_iInter_of_finite fun j => isOpen_ne_fun (by fun_prop) continuous_const
    filter_upwards [hopen.mem_nhds fun j => (hxy j).1] with p hp
    funext j
    exact Scalar.rem_fin _ (hp j)
  · refine hasFDerivAt_pi.2 fun j => ?_
    -- Near `(x, y)` the rounded quotient is constant.
    have hq : ∀ k : ℤ, k ≠ 0 → x (ux j) / y (uy j) ≠ k := fun k hk heq =>
      (hxy j).2 k hk ((div_eq_iff (hxy j).1).1 heq)
    have hcont : ContinuousAt (fun p : (Index sx → ℝ) × (Index sy → ℝ) => p.1 (ux j) / p.2 (uy j))
        (x, y) := ContinuousAt.div (by fun_prop) (by fun_prop) (hxy j).1
    have hev : (fun p : (Index sx → ℝ) × (Index sy → ℝ) =>
        p.1 (ux j) - Scalar.truncReal (p.1 (ux j) / p.2 (uy j)) * p.2 (uy j)) =ᶠ[nhds (x, y)]
        fun p => p.1 (ux j) - c j * p.2 (uy j) := by
      filter_upwards [hcont.eventually (truncReal_eventuallyEq hq)] with p hp
      rw [hp]
    exact ((hasFDerivAt_fst_apply (κ := Index sy) (ux j) (x, y)).sub
      ((hasFDerivAt_snd_apply (ι := Index sx) (uy j) (x, y)).const_mul (c j))).congr_of_eventuallyEq
      hev
  · intro g k
    refine ⟨fun j => g (ux j), _, .of_eq (funext fun j => maybeBroadcast_apply _ _ j),
      remJvpRight_same h k x y fun j => (hxy j).1, ?_⟩
    funext j
    simp only [ContinuousLinearMap.pi_apply, sub_apply,
      ContinuousLinearMap.comp_apply, smul_apply,
      ContinuousLinearMap.proj_apply, ContinuousLinearMap.coe_fst', ContinuousLinearMap.coe_snd',
      smul_eq_mul, Pi.add_apply, c, ux, uy, truncReal_eq]
    ring

/-- The JVP rules of `rem` are not correct everywhere: `rem(x, 0)` is NaN.
No other rule would be correct there either, so this is a convention of JAX and not a bug.
The counterexample has shape `(1,)` and `x = y = 0`. -/
theorem remJvp_not_correct :
    ¬ IsJvp₂Of (remJvpLeft (.refl ![1])) (remJvpRight (.refl ![1])) (rem (.refl ![1])) := by
  intro hrem
  obtain ⟨F, L, hF, -, -⟩ := hrem 0 0 (Set.mem_univ _)
  have := congrFun hF.self_of_nhds fun _ => ⟨0, by simp [broadcastShape]⟩
  simp [rem, binop, ofReal, Scalar.rem] at this

end Jax.Lax
