module

public import Jax.Lax.Primal
public import Mathlib.Analysis.Calculus.Deriv.Mul

/-!
# `polynomial`

Differentiation rules of the `polynomial` primitive of `jax/_src/lax/lax.py`.

Only the branch of the JVP rule for the tangent of `x` is modelled,
for coefficients that are constants and so have symbolic zero tangents.
The JVP rule is correct if the constant coefficient does not broadcast `x`,
and returns a tangent of the wrong shape for `x` of shape `(1,)` and a constant coefficient of shape `(3,)`.
-/

@[expose] public section

namespace Jax.Lax

variable {n K : ℕ} {sx : Shape n} {sc : Fin (K + 2) → Shape n}

theorem broadcastableAll_derivCoeffs (h : BroadcastableAll (sx :: List.ofFn sc)) :
    BroadcastableAll (sx :: List.ofFn fun k : Fin (K + 1) => sc k.succ) := by
  refine h.mono fun t ht => ?_
  rcases List.mem_cons.1 ht with rfl | ht
  · exact List.mem_cons_self
  · obtain ⟨k, rfl⟩ := List.mem_ofFn.1 ht
    exact List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨k.succ, rfl⟩)

/-- `deriv_coeffs` of `_polynomial_jvp`: the coefficients `k * a k` of the derivative, for `k ≥ 1`.
`jax/_src/lax/lax.py:5263`:
`deriv_coeffs = [mul(_const(c, k), c) for k, c in enumerate(coeffs[1:], 1)]` -/
noncomputable def polynomialDerivCoeffs (coeffs : (k : Fin (K + 2)) → Tensor (Index (sc k)))
    (k : Fin (K + 1)) : Tensor (Index (sc k.succ)) :=
  constMul (.fin ((k : ℝ) + 1)) (coeffs k.succ)

/-- The JVP rule of `polynomial` for the tangent `dotX` of `x`,
with constant coefficients whose tangents are symbolic zeros.
Then the second branch is skipped, and `ad.add_tangents(dot_y, t)` returns `t`
because `dot_y` is a symbolic zero, `jax/_src/interpreters/ad.py:996`.
`jax/_src/lax/lax.py:5255`:
```
def _polynomial_jvp(primals, tangents, *, unroll=None):
  x, *coeffs = primals
  dot_x, *dot_coeffs = tangents
  kwargs = {} if unroll is None else {'unroll': unroll}
  y = polynomial_p.bind(x, *coeffs, **kwargs)
  dot_y = ad.Zero(core.typeof(y).to_tangent_aval())

  if not isinstance(dot_x, ad.Zero):
    deriv_coeffs = [mul(_const(c, k), c) for k, c in enumerate(coeffs[1:], 1)]
    dy_dx = polynomial(x, deriv_coeffs, unroll=unroll)
    dot_y = ad.add_tangents(dot_y, mul(dot_x, dy_dx))
``` -/
noncomputable def polynomialJvp (h : BroadcastableAll (sx :: List.ofFn sc))
    (coeffs : (k : Fin (K + 2)) → Tensor (Index (sc k))) (dotX x : Tensor (Index sx)) :=
  mul (broadcastsTo_broadcastShapes (broadcastableAll_derivCoeffs h) List.mem_cons_self).broadcastable
    dotX (polynomial (broadcastableAll_derivCoeffs h) x (polynomialDerivCoeffs coeffs))

/-! ## Horner's rule on real numbers -/

private noncomputable def hornerR (x : ℝ) : {K : ℕ} → (Fin (K + 1) → ℝ) → ℝ
  | 0, a => a 0
  | _ + 1, a => hornerR x (Fin.tail a) * x + a 0

private theorem horner_fin (x : ℝ) : ∀ {K : ℕ} (a : Fin (K + 1) → ℝ),
    horner (.fin x) (fun k => .fin (a k)) = .fin (hornerR x a)
  | 0, _ => rfl
  | _ + 1, a => by
    show horner (.fin x) (fun k => .fin (Fin.tail a k)) * .fin x + .fin (a 0) = _
    rw [horner_fin]
    rfl

private theorem hornerR_add (x : ℝ) : ∀ {K : ℕ} (a b : Fin (K + 1) → ℝ),
    hornerR x (fun k => a k + b k) = hornerR x a + hornerR x b
  | 0, _, _ => rfl
  | _ + 1, a, b => by
    show hornerR x (fun k => Fin.tail a k + Fin.tail b k) * x + (a 0 + b 0) = _
    rw [hornerR_add]
    simp only [hornerR]
    ring

private theorem hasDerivAt_hornerR (x : ℝ) : ∀ {K : ℕ} (a : Fin (K + 2) → ℝ),
    HasDerivAt (fun t => hornerR t a) (hornerR x fun k : Fin (K + 1) => ((k : ℝ) + 1) * a k.succ) x
  | 0, a => by
    have : HasDerivAt (fun t => hornerR t a) (a (Fin.succ 0) * 1) x :=
      ((hasDerivAt_id x).const_mul (a (Fin.succ 0))).add_const (a 0)
    exact this.congr_deriv (by simp [hornerR])
  | K + 1, a => by
    have : HasDerivAt (fun t => hornerR t a)
        ((hornerR x fun k : Fin (K + 1) => ((k : ℝ) + 1) * a k.succ.succ) * x +
          (hornerR x (fun k : Fin (K + 1) => a k.succ.succ) * x + a (Fin.succ 0)) * 1) x :=
      ((hasDerivAt_hornerR x (Fin.tail a)).mul (hasDerivAt_id x)).add_const (a 0)
    refine this.congr_deriv ?_
    show _ = hornerR x (fun k : Fin (K + 1) => (((k.succ : Fin (K + 2)) : ℝ) + 1) * a k.succ.succ) * x +
      (((0 : Fin (K + 2)) : ℝ) + 1) * a (Fin.succ 0)
    have e : (fun k : Fin (K + 1) => (((k.succ : Fin (K + 2)) : ℝ) + 1) * a k.succ.succ) =
        fun k : Fin (K + 1) => ((k : ℝ) + 1) * a k.succ.succ + a k.succ.succ :=
      funext fun k => by simp only [Fin.val_succ, Nat.cast_add, Nat.cast_one]; ring
    rw [e, hornerR_add]
    simp only [Fin.val_zero, Nat.cast_zero]
    ring

/-! ## Correctness -/

/-- The JVP rule of `polynomial` computes the derivative of `polynomial` in `x`
if the constant coefficient broadcasts to the shape of `x` and the other coefficients. -/
theorem polynomialJvp_correct (h : BroadcastableAll (sx :: List.ofFn sc))
    (coeffs : (k : Fin (K + 2)) → Index (sc k) → ℝ)
    (h0 : BroadcastsTo (sc 0) (broadcastShapes (sx :: List.ofFn fun k : Fin (K + 1) => sc k.succ))) :
    IsJvpOf (polynomialJvp h fun k => ofReal (coeffs k)) (polynomial h · fun k => ofReal (coeffs k)) := by
  intro x _
  have hx := broadcastsTo_broadcastShapes h List.mem_cons_self
  have hc : ∀ k, BroadcastsTo (sc k) (broadcastShapes (sx :: List.ofFn sc)) := fun k =>
    broadcastsTo_broadcastShapes h (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨k, rfl⟩))
  let ux : Index (broadcastShapes (sx :: List.ofFn sc)) → Index sx := fun j => j.unbroadcast hx
  let a : Index (broadcastShapes (sx :: List.ofFn sc)) → Fin (K + 2) → ℝ :=
    fun j k => coeffs k (j.unbroadcast (hc k))
  refine ⟨fun y j => hornerR (y (ux j)) (a j), ContinuousLinearMap.pi fun j =>
    hornerR (x (ux j)) (fun k : Fin (K + 1) => ((k : ℝ) + 1) * a j k.succ) •
      ContinuousLinearMap.proj (ux j), ?_, ?_, fun g => ?_⟩
  · exact .of_forall fun y => funext fun j => horner_fin _ _
  · exact hasFDerivAt_pi.2 fun j => hasFDerivAt_elementwise x (ux j) (hasDerivAt_hornerR _ _)
  -- The shape `W` of `x` and the coefficients of positive degree.
  have hmem : ∀ s ∈ (sx :: List.ofFn fun k : Fin (K + 1) => sc k.succ), s ∈ sx :: List.ofFn sc := by
    intro s hs
    rcases List.mem_cons.1 hs with rfl | hs
    · exact List.mem_cons_self
    · obtain ⟨k, rfl⟩ := List.mem_ofFn.1 hs
      exact List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨k.succ, rfl⟩)
  have hW := h.mono hmem
  have hshape : ∀ d, broadcastShapes (sx :: List.ofFn sc) d =
      broadcastShapes (sx :: List.ofFn fun k : Fin (K + 1) => sc k.succ) d := fun d => by
    refine broadcastShapes_eq (fun s hs => ?_) ?_
    · rcases List.mem_cons.1 hs with rfl | hs
      · exact broadcastsTo_broadcastShapes hW List.mem_cons_self d
      · obtain ⟨k, rfl⟩ := List.mem_ofFn.1 hs
        cases k using Fin.cases with
        | zero => exact h0 d
        | succ k =>
          exact broadcastsTo_broadcastShapes hW
            (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨k, rfl⟩)) d
    · rcases broadcastShapes_cases (sx :: List.ofFn fun k : Fin (K + 1) => sc k.succ) d with
        h1 | ⟨s, hs, hsd⟩
      · exact Or.inl h1
      · exact Or.inr ⟨s, hmem s hs, hsd⟩
  -- The shape facts that relate the indices of the rule to those of `polynomial`.
  have hdy := broadcastsTo_broadcastShapes hW List.mem_cons_self
  have hdk : ∀ k : Fin (K + 1), BroadcastsTo (sc k.succ)
      (broadcastShapes (sx :: List.ofFn fun k : Fin (K + 1) => sc k.succ)) := fun k =>
    broadcastsTo_broadcastShapes hW (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨k, rfl⟩))
  refine Tensor.same_of_shape_eq (fun d => ?_) fun q j hqj => ?_
  · rw [hshape]
    have := broadcastsTo_broadcastShapes hW List.mem_cons_self d
    simp only [broadcastShape]
    split_ifs <;> omega
  · have e1 : q.unbroadcast (broadcastsTo_left sx _) = ux j := Index.unbroadcast_congr _ _ q j hqj
    have e2 : (q.unbroadcast hdy.broadcastable.right).unbroadcast hdy = ux j := by
      refine Index.ext_val fun d => ?_
      have := hdy d; have := hqj d; have := hx d
      simp only [ux, Index.unbroadcast_val, broadcastShape] at *
      split_ifs <;> omega
    have e3 : ∀ k : Fin (K + 1), (q.unbroadcast hdy.broadcastable.right).unbroadcast (hdk k) =
        j.unbroadcast (hc k.succ) := fun k => by
      refine Index.ext_val fun d => ?_
      have := hdy d; have := hdk k d; have := hqj d; have := hc k.succ d
      simp only [Index.unbroadcast_val, broadcastShape] at *
      split_ifs <;> omega
    simp only [polynomialJvp, mul, binop, polynomial, polynomialDerivCoeffs, constMul, ofReal, e1, e2,
      e3, Scalar.fin_mul_fin, horner_fin]
    simp only [ContinuousLinearMap.pi_apply, smul_apply,
      ContinuousLinearMap.proj_apply, smul_eq_mul, a]
    exact congrArg Scalar.fin (mul_comm _ _)

/-- The JVP rule of `polynomial` is wrong if the constant coefficient broadcasts `x`:
for `x` of shape `(1,)` and coefficients of shapes `(3,)` and `(1,)`,
it returns a tangent of shape `(1,)` instead of `(3,)`. -/
theorem polynomialJvp_not_correct
    (coeffs : (k : Fin 2) → Index ((![![3], ![1]] : Fin 2 → Shape 1) k) → ℝ) :
    ¬ IsJvpOf (polynomialJvp (sx := ![1]) (by unfold BroadcastableAll Broadcastable; decide)
        fun k => ofReal (coeffs k))
      (polynomial (by unfold BroadcastableAll Broadcastable; decide) · fun k => ofReal (coeffs k)) := by
  intro hj
  obtain ⟨_, _, _, _, hrule⟩ := hj 0 (Set.mem_univ _)
  obtain ⟨hn, hs, _⟩ := hrule 0
  have h0 := hs 0
  rw [show Fin.cast hn 0 = (0 : Fin 1) from rfl] at h0
  exact absurd h0 (by decide)

end Jax.Lax
