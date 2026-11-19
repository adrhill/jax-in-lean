module

public import Jax.Lax.Primal
public import Mathlib.Analysis.Calculus.Deriv.Inv

/-!
# `div`

Differentiation rules of the `div` primitive of `jax/_src/lax/lax.py`.

`div` is linear in its first argument only.
The transpose rule is correct where the divisor has no zeros, also for operands that are broadcast
(`divTranspose_correct`).
For a zero divisor, `x ↦ x / 0` does not map real arrays to real arrays (`div_zero_not_isTransposeOf`).
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {sx sy : Shape n} {w : FloatWidth}

/-- The JVP rule of `div` for its first argument.
`jax/_src/lax/lax.py:5628`: `lambda g, x, y: div(g, y)` -/
noncomputable def divJvpLeft (h : Broadcastable sx sy) (g _x : Tensor (.float w) (Index sx))
    (y : Tensor (.float w) (Index sy)) : Tensor (.float w) (Index (broadcastShape sx sy)) :=
  div h g y

/-- The JVP rule of `div` for its second argument.
`jax/_src/lax/lax.py:5629`: `lambda g, x, y: mul(mul(neg(g), x), integer_pow(y, -2))` -/
noncomputable def divJvpRight (h : Broadcastable sx sy) (g : Tensor (.float w) (Index sy))
    (x : Tensor (.float w) (Index sx)) (y : Tensor (.float w) (Index sy)) :
    Tensor (.float w) (Index (broadcastShape (broadcastShape sy sx) sy)) :=
  mul (broadcastsTo_left sy sx).broadcastable.symm (mul h.symm (neg g) x) (integerPow (-2) y)

/-- The right rule of `div` on finite arrays where `y` has no zeros. -/
private theorem divJvpRight_same (h : Broadcastable sx sy) (k : Index sy → ℝ) (x : Index sx → ℝ)
    (y : Index sy → ℝ) (hy : ∀ j : Index (broadcastShape sx sy), y (j.unbroadcast h.right) ≠ 0) :
    (divJvpRight (w := w) h (ofReal k) (ofReal x) (ofReal y)).Same (ofReal (w := w) fun j =>
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
    IsJvp₂OfOn (divJvpLeft h) (divJvpRight h) (div (w := w) h)
      {p | ∀ j : Index (broadcastShape sx sy), p.2 (j.unbroadcast h.right) ≠ 0} := by
  intro x y hy
  let ux : Index (broadcastShape sx sy) → Index sx :=
    fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  refine ⟨fun p j => p.1 (ux j) / p.2 (uy j), ContinuousLinearMap.pi fun j =>
    x (ux j) • ((ContinuousLinearMap.toSpanSingleton ℝ (-(y (uy j) ^ 2)⁻¹)).comp
      (ContinuousLinearMap.proj (uy j) ∘L ContinuousLinearMap.snd ℝ _ _)) +
    (y (uy j))⁻¹ • (ContinuousLinearMap.proj (ux j) ∘L ContinuousLinearMap.fst ℝ _ _), ?_, ?_, ?_, ?_⟩
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
  · intro g
    refine .of_eq (funext fun j => ?_)
    refine (Scalar.fin_div_fin _ (hy j)).trans (congrArg Scalar.fin ?_)
    simp only [ContinuousLinearMap.pi_apply, add_apply, smul_apply, ux, uy,
      ContinuousLinearMap.comp_apply, ContinuousLinearMap.toSpanSingleton_apply,
      ContinuousLinearMap.proj_apply, ContinuousLinearMap.coe_snd', ContinuousLinearMap.coe_fst',
      smul_eq_mul, Pi.zero_apply]
    ring
  · intro k
    convert divJvpRight_same h k x y hy using 2
    funext j
    simp only [ContinuousLinearMap.pi_apply, add_apply, smul_apply, ux, uy,
      ContinuousLinearMap.comp_apply, ContinuousLinearMap.toSpanSingleton_apply,
      ContinuousLinearMap.proj_apply, ContinuousLinearMap.coe_snd', ContinuousLinearMap.coe_fst',
      smul_eq_mul, Pi.zero_apply, zpow_neg]
    have := hy j
    field_simp
    ring

/-! ## Transpose rule -/

theorem divTranspose_broadcastsTo (h : Broadcastable sx sy) :
    BroadcastsTo sx (broadcastShape (broadcastShape sx sy) sy) := fun d => by
  have := h d
  simp only [broadcastShape]
  split_ifs <;> omega

/-- The transpose rule of `div`, for its first argument.
The undefined primal `x` is replaced by its shape `sx`, and `y` is not a linear input:
the rule raises if `x` is not a linear input or `y` is one.
The cotangent `None` of `y` is dropped.
`jax/_src/lax/lax.py:5600`:
```
def _div_transpose_rule(cotangent, x, y):
  assert ad.is_undefined_primal(x)
  if ad.is_undefined_primal(y):
    raise RuntimeError("nonlinear div can't be transposed")
  if type(cotangent) is ad_util.Zero:
    return [ad_util.Zero(x.aval), None]
  else:
    return [_unbroadcast(x.aval, div(cotangent, y)), None]
``` -/
noncomputable def divTranspose (h : Broadcastable sx sy) (y : Tensor (.float w) (Index sy))
    (cotangent : Tan (.float w) (Index (broadcastShape sx sy))) : Tan (.float w) (Index sx) :=
  match cotangent with
  | .zero _ => .zero (.of (.float w) sx)
  | .arr cotangent => .arr (unbroadcast sx (divTranspose_broadcastsTo h) (div h.right.broadcastable.symm cotangent y))

/-- The transpose rule of `div` transposes `div` with a fixed divisor that has no zeros. -/
theorem divTranspose_correct (h : Broadcastable sx sy) (y : Index sy → ℝ)
    (hy : ∀ j : Index (broadcastShape sx sy), y (j.unbroadcast h.right) ≠ 0) :
    IsTanTransposeOf (divTranspose h (ofReal y)) (div (w := w) h · (ofReal y)) := by
  refine ⟨⟨rfl, rfl⟩, fun ct x => ?_⟩
  let ux : Index (broadcastShape sx sy) → Index sx := fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  refine ⟨fun j => x (ux j) / y (uy j), fun i => ∑ j with ux j = i, ct j / y (uy j),
    funext fun j => Scalar.fin_div_fin _ (hy j), .of_eq (funext fun i => ?_), ?_⟩
  · rw [unbroadcast_apply, ofReal, Scalar.fin_sum]
    refine sum_filter_index_congr (fun d => ?_) _ _ (fun j j' hjj => ?_) fun j j' hjj => ?_
    · have := h d
      simp only [broadcastShape]
      split_ifs <;> omega
    · rw [Index.unbroadcast_congr _ (broadcastsTo_left sx sy) j j' hjj]
    · simp only [div, binop, ofReal, uy]
      rw [Index.unbroadcast_eq_of_val _ j j' hjj, Index.unbroadcast_congr _ h.right j j' hjj]
      exact Scalar.fin_div_fin _ (hy j')
  · calc ∑ j, ct j * (x (ux j) / y (uy j))
        = ∑ i, ∑ j with ux j = i, ct j * (x (ux j) / y (uy j)) := (Finset.sum_fiberwise _ _ _).symm
      _ = ∑ i, (∑ j with ux j = i, ct j / y (uy j)) * x i := by
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun j hj => ?_
          rw [(Finset.mem_filter.1 hj).2]
          ring

/-- No transpose rule is correct for `x ↦ x / 0`: it does not map real arrays to real arrays,
because `0 / 0` is NaN. -/
theorem div_zero_not_isTransposeOf {s : Shape n} [Nonempty (Index s)]
    (tr : Tensor (.float w) (Index (broadcastShape s s)) → Tensor (.float w) (Index s)) :
    ¬ IsTransposeOf tr (div (.refl s) · (ofReal 0)) := by
  intro htr
  obtain ⟨y, -, hy, -⟩ := htr 0 0
  obtain ⟨i⟩ := ‹Nonempty (Index s)›
  have := congrFun hy fun d => ⟨i d, by simp [broadcastShape]⟩
  have h0 : (Scalar.fin 0 / Scalar.fin 0 : Scalar w) = .nan := by
    show (if (0 : ℝ) = 0 then Scalar.infWithSign 0 else _) = _
    simp [Scalar.infWithSign]
  simp only [div, binop, ofReal, Pi.zero_apply, h0] at this
  cases this

end Jax.Lax
