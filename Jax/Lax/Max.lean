module

public import Jax.Lax.Primal
public import Mathlib.Analysis.Calculus.Deriv.Prod
public import Mathlib.Analysis.Calculus.Deriv.Mul

/-!
# `max`

Differentiation rules of the `max` primitive of `jax/_src/lax/lax.py`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {sx sy : Shape n} {w : FloatWidth}

/-- The JVP rule of `max` for its first argument.
`jax/_src/lax/lax.py:5644`: `lambda g, x, y: mul(g, _balanced_cmp(x, y))` -/
noncomputable def maxJvpLeft (h : Broadcastable sx sy) (g x : Tensor (.float w) (Index sx))
    (y : Tensor (.float w) (Index sy)) : Tensor (.float w) (Index (broadcastShape sx (broadcastShape sx sy))) :=
  mul (broadcastable_left sx sy) g (balancedCmp h x y)

/-- The JVP rule of `max` for its second argument.
`jax/_src/lax/lax.py:5645`: `lambda g, x, y: mul(g, _balanced_cmp(y, x))` -/
noncomputable def maxJvpRight (h : Broadcastable sx sy) (g : Tensor (.float w) (Index sy))
    (x : Tensor (.float w) (Index sx)) (y : Tensor (.float w) (Index sy)) :
    Tensor (.float w) (Index (broadcastShape sy (broadcastShape sy sx))) :=
  mul (broadcastable_left sy sx) g (balancedCmp h.symm y x)

/-- `_balanced_cmp` on real numbers. -/
private noncomputable def cmp (a b : ℝ) : ℝ := if a > b then 1 else if a = b then 1 / 2 else 0

private theorem cmp_fin (a b : ℝ) : balancedCmpScalar (.fin a : Scalar w) (.fin b) = .fin (cmp a b) := by
  unfold balancedCmpScalar cmp
  simp only [Scalar.gt_fin, Scalar.eq_fin]
  split_ifs <;> rfl

/-- The left rule of `max` on finite arrays. -/
private theorem maxJvpLeft_same (h : Broadcastable sx sy) (g x : Index sx → ℝ) (y : Index sy → ℝ) :
    (maxJvpLeft (w := w) h (ofReal g) (ofReal x) (ofReal y)).Same (ofReal (w := w) fun j =>
      g (j.unbroadcast (broadcastsTo_left sx sy)) *
        cmp (x (j.unbroadcast (broadcastsTo_left sx sy))) (y (j.unbroadcast h.right))) := by
  refine Tensor.same_of_shape_eq (fun d => ?_) fun i j hij => ?_
  · simp only [broadcastShape]; split_ifs <;> simp_all
  · have h1 : i.unbroadcast (broadcastsTo_left sx (broadcastShape sx sy)) =
        j.unbroadcast (broadcastsTo_left sx sy) := Index.unbroadcast_congr _ _ i j hij
    have h2 : i.unbroadcast (broadcastable_left sx sy).right = j :=
      Index.unbroadcast_eq_of_val _ i j hij
    simp only [maxJvpLeft, mul, balancedCmp, binop, ofReal]
    rw [h1, h2, cmp_fin]
    rfl

/-- The right rule of `max` on finite arrays. -/
private theorem maxJvpRight_same (h : Broadcastable sx sy) (g : Index sy → ℝ) (x : Index sx → ℝ)
    (y : Index sy → ℝ) :
    (maxJvpRight (w := w) h (ofReal g) (ofReal x) (ofReal y)).Same (ofReal (w := w) fun j =>
      g (j.unbroadcast h.right) *
        cmp (y (j.unbroadcast h.right)) (x (j.unbroadcast (broadcastsTo_left sx sy)))) := by
  have hs : ∀ d, broadcastShape sy sx d = broadcastShape sx sy d := fun d => by
    have := h d
    simp only [broadcastShape]; split_ifs <;> omega
  refine Tensor.same_of_shape_eq (fun d => ?_) fun i j hij => ?_
  · have := hs d
    simp only [broadcastShape] at this ⊢; split_ifs at this ⊢ <;> omega
  · have h1 : i.unbroadcast (broadcastsTo_left sy (broadcastShape sy sx)) =
        j.unbroadcast h.right := Index.unbroadcast_congr _ _ i j hij
    have h2 : (i.unbroadcast (broadcastable_left sy sx).right).unbroadcast
        (broadcastsTo_left sy sx) = j.unbroadcast h.right := by
      refine Index.ext_val fun d => ?_
      have := hs d; have := hij d; have := (j d).isLt
      simp only [Index.unbroadcast_val, broadcastShape] at this ⊢
      split_ifs at this ⊢ <;> omega
    have h3 : (i.unbroadcast (broadcastable_left sy sx).right).unbroadcast h.symm.right =
        j.unbroadcast (broadcastsTo_left sx sy) := by
      refine Index.ext_val fun d => ?_
      have := hs d; have := hij d; have := (j d).isLt; have := h d
      simp only [Index.unbroadcast_val, broadcastShape] at this ⊢
      split_ifs at this ⊢ <;> omega
    simp only [maxJvpRight, mul, balancedCmp, binop, ofReal]
    rw [h1, h2, h3, cmp_fin]
    rfl

/-- The JVP rules of `max` compute the derivative of `max` where the arguments differ everywhere. -/
theorem maxJvp_correct (h : Broadcastable sx sy) :
    IsJvp₂OfOn (maxJvpLeft h) (maxJvpRight h) (max (w := w) h)
      {p | ∀ j : Index (broadcastShape sx sy),
        p.1 (j.unbroadcast (broadcastsTo_left sx sy)) ≠ p.2 (j.unbroadcast h.right)} := by
  intro x y hxy
  let ux : Index (broadcastShape sx sy) → Index sx :=
    fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  refine ⟨fun p j => p.1 (ux j) ⊔ p.2 (uy j), ContinuousLinearMap.pi fun j =>
    cmp (x (ux j)) (y (uy j)) • (ContinuousLinearMap.proj (ux j) ∘L ContinuousLinearMap.fst ℝ _ _) +
    cmp (y (uy j)) (x (ux j)) • (ContinuousLinearMap.proj (uy j) ∘L ContinuousLinearMap.snd ℝ _ _),
    ?_, ?_, ?_⟩
  · exact .of_forall fun p => rfl
  · refine hasFDerivAt_pi.2 fun j => ?_
    have hne : x (ux j) ≠ y (uy j) := hxy j
    rcases lt_or_gt_of_ne hne with hlt | hlt
    · have hopen : IsOpen {p : (Index sx → ℝ) × (Index sy → ℝ) | p.1 (ux j) < p.2 (uy j)} :=
        isOpen_lt (by fun_prop) (by fun_prop)
      have hev : (fun p : (Index sx → ℝ) × (Index sy → ℝ) => p.1 (ux j) ⊔ p.2 (uy j))
          =ᶠ[nhds (x, y)] fun p => p.2 (uy j) := by
        filter_upwards [hopen.mem_nhds hlt] with p hp
        exact sup_eq_right.2 (le_of_lt hp)
      have hd := hasFDerivAt_snd_apply (ι := Index sx) (uy j) (x, y)
      convert hd.congr_of_eventuallyEq hev using 1
      simp [cmp, hlt, hlt.not_gt, hlt.ne]
    · have hopen : IsOpen {p : (Index sx → ℝ) × (Index sy → ℝ) | p.2 (uy j) < p.1 (ux j)} :=
        isOpen_lt (by fun_prop) (by fun_prop)
      have hev : (fun p : (Index sx → ℝ) × (Index sy → ℝ) => p.1 (ux j) ⊔ p.2 (uy j))
          =ᶠ[nhds (x, y)] fun p => p.1 (ux j) := by
        filter_upwards [hopen.mem_nhds hlt] with p hp
        exact sup_eq_left.2 (le_of_lt hp)
      have hd := hasFDerivAt_fst_apply (κ := Index sy) (ux j) (x, y)
      convert hd.congr_of_eventuallyEq hev using 1
      simp [cmp, hlt, hlt.not_gt, hlt.ne]
  · intro g k
    refine ⟨_, _, maxJvpLeft_same h g x y, maxJvpRight_same h k x y, ?_⟩
    ext j
    simp [ux, uy, mul_comm]

/-- The JVP rules of `max` are not correct everywhere: `max` is not differentiable at a tie.
No other rule would be correct there either, so this is a convention of JAX and not a bug.
The counterexample has shape `(1,)` and `x = y = 0`. -/
theorem maxJvp_not_correct :
    ¬ IsJvp₂Of (maxJvpLeft (.refl ![1])) (maxJvpRight (.refl ![1])) (max (w := w) (.refl ![1])) := by
  intro hmax
  obtain ⟨F, L, hF, hL, hrule⟩ := hmax 0 0 (Set.mem_univ _)
  let j₀ : Index (broadcastShape ![1] ![1]) := fun _ => ⟨0, by simp [broadcastShape]⟩
  -- The value of the rules for the tangents `(1, 0)`.
  obtain ⟨dx, dy, hdx, hdy, hsum⟩ := hrule 1 0
  have hdx₀ : dx j₀ = 1 / 2 := by
    rw [ofReal_injective (hdx.unique (maxJvpLeft_same _ 1 0 0))]
    simp [cmp]
  have hdy₀ : dy j₀ = 0 := by
    rw [ofReal_injective (hdy.unique (maxJvpRight_same _ 0 0 0))]
    simp
  -- `max` on finite arrays, and its restriction to the line `t ↦ (t, 0)`.
  let G : (Index ![1] → ℝ) × (Index ![1] → ℝ) → Index (broadcastShape ![1] ![1]) → ℝ :=
    fun p j => p.1 (j.unbroadcast (broadcastsTo_left _ _)) ⊔
      p.2 (j.unbroadcast (Broadcastable.refl ![1]).right)
  have hG : HasFDerivAt G L (0, 0) :=
    hL.congr_of_eventuallyEq (hF.mono fun p hp => ofReal_injective hp)
  let v : (Index ![1] → ℝ) × (Index ![1] → ℝ) := (1, 0)
  have hline : HasDerivAt (fun t : ℝ => t • v) v 0 := by
    simpa using (hasDerivAt_id (0 : ℝ)).smul_const v
  have hG' : HasFDerivAt G L ((0 : ℝ) • v) := by rw [zero_smul]; exact hG
  have hcomp := hasDerivAt_pi.1 (hG'.comp_hasDerivAt (0 : ℝ) hline) j₀
  have hval : L v j₀ = 1 / 2 := by
    rw [show L v = dx + dy from hsum]; simp [hdx₀, hdy₀]
  rw [hval] at hcomp
  -- On `t ≥ 0` the function is `t`, with derivative `1`.
  have hid : HasDerivWithinAt (fun t : ℝ => G (t • v) j₀) 1 (Set.Ici 0) 0 := by
    refine (hasDerivWithinAt_id (0 : ℝ) (Set.Ici 0)).congr (fun t ht => ?_) ?_
    · simpa [G, v] using ht
    · simp [G, v]
  have := (uniqueDiffOn_Ici (0 : ℝ) 0 (by simp)).eq_deriv _ hcomp.hasDerivWithinAt hid
  norm_num at this

end Jax.Lax
