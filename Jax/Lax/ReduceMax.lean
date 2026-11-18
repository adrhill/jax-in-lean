module

public import Jax.Core
public import Jax.Lax.Primal

/-!
# `reduce_max`

Differentiation rule of the `reduce_max` primitive of `jax/_src/lax/lax.py`.

`_reduce_chooser_jvp_rule` divides the sum of the tangents at the positions of the maximum
by the number of these positions (`reduceMaxJvp_same`).
It is correct where every maximum is attained at one position (`reduceMaxJvp_correct`).
Over an empty axis there is no such position, and the rule returns `0 / 0`, which is NaN
(`reduceMaxJvp_of_isEmpty`), although `reduce_max` is the constant `-∞` there.
`Jax.Jaxpr.maxInitialJvp_not_correct` shows that this makes forward mode of `jnp.max(x, initial=0.)` wrong.
`reduce_min` shares the rule and is not modelled.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n} {w : FloatWidth}

/-- The shape `shape` of `_reduce_chooser_jvp_rule`: size 1 on the reduced axes. -/
def chooserShape (s : Shape n) (axes : Finset (Fin n)) : Shape n :=
  fun d => if d ∈ axes then 1 else s d

theorem prod_chooserShape (s : Shape n) (axes : Finset (Fin n)) :
    ∏ a, (s ∘ keptAxes axes) a = ∏ d, chooserShape s axes d := by
  refine Fintype.prod_of_injective _ (keptAxes_injective axes) _ _ (fun d hd => ?_) fun a => ?_
  · have hd' : d ∈ axes := by
      by_contra hc
      exact hd (keptAxes_surj hc)
    simp [chooserShape, hd']
  · simp [chooserShape, keptAxes_notMem]

theorem broadcastable_chooserShape (s : Shape n) (axes : Finset (Fin n)) :
    Broadcastable s (chooserShape s axes) := fun d => by
  unfold chooserShape
  split_ifs <;> simp

/-- Broadcasting `s` against a shape that has size 1 where `s` has size 1 gives `s`. -/
theorem broadcastShape_of_eq {s t : Shape n} (h : ∀ d, s d = 1 → t d = 1) (d : Fin n) :
    broadcastShape s t d = s d := by
  unfold broadcastShape
  split_ifs with h1
  · rw [h d h1, h1]
  · rfl

theorem broadcastShape_chooserShape (s : Shape n) (axes : Finset (Fin n)) (d : Fin n) :
    broadcastShape s (chooserShape s axes) d = s d :=
  broadcastShape_of_eq (fun d h => by unfold chooserShape; split_ifs <;> simp_all) d

theorem broadcastShape_broadcastShape_chooserShape (s : Shape n) (axes : Finset (Fin n)) (d : Fin n) :
    broadcastShape s (broadcastShape s (chooserShape s axes)) d = s d :=
  broadcastShape_of_eq (fun d h => by rw [broadcastShape_chooserShape, h]) d

/-- The shape of the result of `reduceMaxJvp`, as its code computes it. -/
abbrev reduceMaxJvpShape (s : Shape n) (axes : Finset (Fin n)) : Shape axesᶜ.card :=
  broadcastShape (broadcastShape s (broadcastShape s (chooserShape s axes)) ∘ keptAxes axes)
    (broadcastShape s (chooserShape s axes) ∘ keptAxes axes)

theorem broadcastable_reduceMaxJvp (s : Shape n) (axes : Finset (Fin n)) :
    Broadcastable (broadcastShape s (broadcastShape s (chooserShape s axes)) ∘ keptAxes axes)
      (broadcastShape s (chooserShape s axes) ∘ keptAxes axes) := fun a =>
  .inl (by simp [broadcastShape_broadcastShape_chooserShape, broadcastShape_chooserShape])

/-- The result of `reduceMaxJvp` has the shape of the result of `reduce_max`. -/
theorem reduceMaxJvpShape_eq (s : Shape n) (axes : Finset (Fin n)) (a : Fin axesᶜ.card) :
    reduceMaxJvpShape s axes a = (s ∘ keptAxes axes) a := by
  simp only [reduceMaxJvpShape]
  rw [broadcastShape_of_eq (fun a h => ?_) a]
  · exact broadcastShape_broadcastShape_chooserShape s axes _
  · simp only [Function.comp_apply] at h ⊢
    rw [broadcastShape_broadcastShape_chooserShape] at h
    rw [broadcastShape_chooserShape, h]

/-- The JVP rule of `reduce_max` and `reduce_min`.
`standard_jvp2` passes the primal result `ans` to the rule, `jax/_src/interpreters/ad.py:989`.
`jax/_src/lax/lax.py:9313`, registered for `reduce_max` at `jax/_src/lax/lax.py:9334`:
```
def _reduce_chooser_jvp_rule(g, ans, operand, *, axes, out_sharding):
  # TODO(mattjj): an alternative is to use variadic reduce to compute the chosen
  # locations in a single pass (rather than comparing equality) and use a
  # gather, and/or even push along the chosen elements of g (b/112040122)
  shape = [1 if i in axes else d for i, d in enumerate(operand.shape)]
  location_indicators = convert_element_type(
      _eq_meet(operand, reshape(ans, shape)), g.dtype)
  counts = reduce_sum(location_indicators, axes)
  return div(reduce_sum(mul(g, location_indicators), axes), counts)
```
`_eq_meet`, `jax/_src/lax/lax.py:10499`, is `eq` for arguments of the same dtype. -/
noncomputable def reduceMaxJvp (axes : Finset (Fin n)) (g : Tensor (.float w) (Index s))
    (ans : Tensor (.float w) (Index (s ∘ keptAxes axes))) (operand : Tensor (.float w) (Index s)) :
    Tensor (.float w) (Index (reduceMaxJvpShape s axes)) :=
  let locationIndicators := convertBool (eq (broadcastable_chooserShape s axes) operand
    (reshape (chooserShape s axes) (prod_chooserShape s axes) ans))
  let counts := reduceSum axes locationIndicators
  div (broadcastable_reduceMaxJvp s axes)
    (reduceSum axes (mul (broadcastable_left s _) g locationIndicators)) counts

/-! ## The rule entry by entry -/

/-- `reshape(ans, shape)` reads the entry of `ans` at the kept axes. -/
theorem reshape_chooserShape_apply (axes : Finset (Fin n)) (ans : Tensor (.float w) (Index (s ∘ keptAxes axes)))
    (k : Index (chooserShape s axes)) (j : Index (s ∘ keptAxes axes))
    (hkj : ∀ a, (k (keptAxes axes a) : ℕ) = j a) :
    reshape (chooserShape s axes) (prod_chooserShape s axes) ans k = ans j := by
  simp only [reshape]
  congr 1
  rw [Equiv.symm_apply_eq]
  refine Fin.ext ?_
  rw [Fin.val_cast, ← Index.rowMajor_proj (keptAxes_strictMono axes) (fun d hd => ?_) k]
  · exact Index.rowMajor_congr (fun a => by simp [chooserShape, keptAxes_notMem]) fun a => hkj a
  · have hd' : d ∈ axes := by
      by_contra hc
      exact hd (keptAxes_surj hc)
    simp [chooserShape, hd']

/-- Unbroadcasting to a shape of the same sizes keeps the values of an index. -/
private theorem unbroadcast_val_of_eq {s t : Shape n} (h : BroadcastsTo s t) (hst : ∀ d, s d = t d)
    (k : Index t) (d : Fin n) : (k.unbroadcast h d : ℕ) = k d := by
  rw [Index.unbroadcast_val]
  have := (k d).isLt
  have := hst d
  split_ifs <;> omega

/-- Projections of indices with the same values agree with indices with the same values. -/
private theorem proj_eq_iff {m : ℕ} {t t' : Shape n} {e : Fin m → Fin n} (k : Index t) (i : Index t')
    (hki : ∀ d, (k d : ℕ) = i d) (p : Index (t ∘ e)) (q : Index (t' ∘ e))
    (hpq : ∀ a, (p a : ℕ) = q a) : k.proj e = p ↔ i.proj e = q := by
  constructor <;> intro h <;> refine Index.ext_val fun a => ?_
  · have := congrArg (fun z => (z a : ℕ)) h
    simp only [Index.proj] at this ⊢
    rw [← hki, this, hpq]
  · have := congrArg (fun z => (z a : ℕ)) h
    simp only [Index.proj] at this ⊢
    rw [hki, this, hpq]

open scoped Classical in
/-- The JVP rule of `reduce_max` entry by entry:
the sum of the tangents at the positions where the operand equals `ans`,
divided by the number of these positions. -/
theorem reduceMaxJvp_same (axes : Finset (Fin n)) (g : Tensor (.float w) (Index s))
    (ans : Tensor (.float w) (Index (s ∘ keptAxes axes))) (operand : Tensor (.float w) (Index s)) :
    (reduceMaxJvp axes g ans operand).Same (d' := .float w) (fun j : Index (s ∘ keptAxes axes) =>
      (∑ i with i.proj (keptAxes axes) = j,
        g i * (if (operand i).eq (ans (i.proj (keptAxes axes))) then 1 else 0)) /
      ∑ i with i.proj (keptAxes axes) = j,
        (if (operand i).eq (ans (i.proj (keptAxes axes))) then 1 else 0)) := by
  have hs1 := broadcastShape_chooserShape s axes
  have hs2 := broadcastShape_broadcastShape_chooserShape s axes
  -- An entry of the location indicators.
  have hloc : ∀ (k : Index (broadcastShape s (chooserShape s axes))) (i : Index s),
      (∀ d, (k d : ℕ) = i d) →
      convertBool (w := w) (eq (broadcastable_chooserShape s axes) operand
        (reshape (chooserShape s axes) (prod_chooserShape s axes) ans)) k =
      if (operand i).eq (ans (i.proj (keptAxes axes))) then 1 else 0 := by
    intro k i hki
    have e1 : k.unbroadcast (broadcastsTo_left s _) = i := Index.unbroadcast_eq_of_val _ k i hki
    have e2 : reshape (chooserShape s axes) (prod_chooserShape s axes) ans
        (k.unbroadcast (broadcastable_chooserShape s axes).right) = ans (i.proj (keptAxes axes)) := by
      refine reshape_chooserShape_apply axes ans _ _ fun a => ?_
      rw [Index.unbroadcast_val]
      have := (k (keptAxes axes a)).isLt
      have := hki (keptAxes axes a)
      have := hs1 (keptAxes axes a)
      simp only [Index.proj, chooserShape, keptAxes_notMem, ↓reduceIte] at this ⊢
      split_ifs <;> omega
    simp only [convertBool, eq, binop, e1, e2, decide_eq_true_eq]
  refine Tensor.same_of_shape_eq (fun a => reduceMaxJvpShape_eq s axes a) fun i j hij => ?_
  simp only [reduceMaxJvp, div, binop, reduceSum]
  congr 1
  · refine sum_filter_index_congr hs2 _ _ (fun k i' hki' => proj_eq_iff k i' hki' _ _ fun a => ?_)
      fun k i' hki' => ?_
    · rw [unbroadcast_val_of_eq _ (fun a => (reduceMaxJvpShape_eq s axes a).trans (hs2 _).symm |>.symm),
        hij]
    · simp only [mul, binop]
      rw [Index.unbroadcast_eq_of_val _ k i' hki',
        hloc _ i' fun d => by rw [unbroadcast_val_of_eq _ (fun d => by rw [hs1, hs2]), hki']]
  · refine sum_filter_index_congr hs1 _ _ (fun k i' hki' => proj_eq_iff k i' hki' _ _ fun a => ?_)
      fun k i' hki' => hloc k i' hki'
    rw [unbroadcast_val_of_eq _ (fun a => (reduceMaxJvpShape_eq s axes a).trans (hs1 _).symm |>.symm),
      hij]

/-! ## Correctness -/

/-- The maximum of finitely many finite values. -/
theorem fold_max_fin {ι : Type} [DecidableEq ι] (S : Finset ι) (f : ι → ℝ) (hS : S.Nonempty) :
    S.fold (Scalar.max (w := w)) .negInf (fun i => .fin (f i)) = .fin (S.sup' hS f) := by
  induction S using Finset.induction_on with
  | empty => exact absurd hS Finset.not_nonempty_empty
  | insert b T hb ih =>
    rw [Finset.fold_insert hb]
    by_cases hT : T.Nonempty
    · rw [ih hT, Finset.sup'_insert hT]
      rfl
    · obtain rfl := Finset.not_nonempty_iff_eq_empty.1 hT
      rfl

open scoped Classical in
/-- The JVP rule of `reduce_max` computes the derivative of `reduce_max`
where every maximum is attained at one position. -/
theorem reduceMaxJvp_correct (axes : Finset (Fin n)) :
    IsJvpOfOn (fun g x => reduceMaxJvp axes g (reduceMax axes x) x) (reduceMax (s := s) (w := w) axes)
      {x | ∀ j : Index (s ∘ keptAxes axes), ∃ a, a.proj (keptAxes axes) = j ∧
        ∀ i, i.proj (keptAxes axes) = j → i ≠ a → x i < x a} := by
  intro x hx
  choose a ha hmax using hx
  -- Near `x`, every maximum is still attained only at `a j`.
  have hev : ∀ᶠ y in nhds x, ∀ j i, i.proj (keptAxes axes) = j → i ≠ a j → y i < y (a j) := by
    refine Filter.eventually_all.2 fun j => Filter.eventually_all.2 fun i => ?_
    by_cases hc : i.proj (keptAxes axes) = j ∧ i ≠ a j
    · filter_upwards [(isOpen_lt (continuous_apply i) (continuous_apply (a j))).mem_nhds
        (show x ∈ {b : Index s → ℝ | b i < b (a j)} from hmax j i hc.1 hc.2)] with y hy _ _ using hy
    · exact .of_forall fun y h1 h2 => absurd ⟨h1, h2⟩ hc
  have hval : ∀ y : Index s → ℝ, (∀ j i, i.proj (keptAxes axes) = j → i ≠ a j → y i < y (a j)) →
      ∀ j, reduceMax (w := w) axes (ofReal y) j = .fin (y (a j)) := by
    intro y hy j
    have hmem : a j ∈ ({i | i.proj (keptAxes axes) = j} : Finset (Index s)) := by simp [ha]
    refine (fold_max_fin (f := y) _ ⟨_, hmem⟩).trans ?_
    congr 1
    refine le_antisymm (Finset.sup'_le _ _ fun i hi => ?_) (Finset.le_sup' _ hmem)
    by_cases hia : i = a j
    · rw [hia]
    · exact (hy j i (Finset.mem_filter.1 hi).2 hia).le
  refine ⟨fun y j => y (a j), ContinuousLinearMap.pi fun j => ContinuousLinearMap.proj (a j), ?_,
    hasFDerivAt_pi.2 fun j => hasFDerivAt_apply (a j) x, fun g => ?_⟩
  · filter_upwards [hev] with y hy
    exact funext (hval y hy)
  · have hans := hval x hev.self_of_nhds
    have href := reduceMaxJvp_same (w := w) axes (ofReal g) (reduceMax axes (ofReal x)) (ofReal x)
    convert href using 1
    funext j
    have hind : ∀ i ∈ ({i | i.proj (keptAxes axes) = j} : Finset (Index s)),
        (if (ofReal (w := w) x i).eq (reduceMax (w := w) axes (ofReal x) (i.proj (keptAxes axes)))
          then (1 : Scalar w) else 0) = if i = a j then 1 else 0 := by
      intro i hi
      have hij := (Finset.mem_filter.1 hi).2
      rw [hij, hans]
      by_cases hia : i = a j
      · simp [hia, ofReal]
      · simp [hia, ofReal, (hmax j i hij hia).ne]
    have hmem : a j ∈ ({i | i.proj (keptAxes axes) = j} : Finset (Index s)) := by simp [ha]
    rw [Finset.sum_congr rfl fun i hi => by rw [hind i hi], Finset.sum_congr rfl hind,
      Finset.sum_eq_single_of_mem (a j) hmem fun i _ hia => by simp [hia, Scalar.zero_def, ofReal],
      Finset.sum_eq_single_of_mem (a j) hmem fun i _ hia => by simp [hia]]
    simp [ofReal, Scalar.one_def, Scalar.fin_div_fin]

/-! ## Over an empty axis -/

/-- If the operand has no entries, every entry of the rule is `0 / 0`, which is NaN. -/
theorem reduceMaxJvp_of_isEmpty (axes : Finset (Fin n)) (hs : ∃ d, s d = 0) (g : Tensor (.float w) (Index s))
    (ans : Tensor (.float w) (Index (s ∘ keptAxes axes))) (operand : Tensor (.float w) (Index s))
    (j : Index (reduceMaxJvpShape s axes)) : reduceMaxJvp axes g ans operand j = .nan := by
  obtain ⟨d, hd⟩ := hs
  have empty : ∀ {t : Shape n}, (∀ d, t d = s d) → IsEmpty (Index t) := fun ht =>
    ⟨fun i => by have := (i d).isLt; have := ht d; omega⟩
  have := empty (broadcastShape_chooserShape s axes)
  have := empty (broadcastShape_broadcastShape_chooserShape s axes)
  simp only [reduceMaxJvp, div, binop, reduceSum, Finset.univ_eq_empty, Finset.filter_empty,
    Finset.sum_empty]
  show (if (0 : ℝ) = 0 then Scalar.infWithSign 0 else _) = _
  simp [Scalar.infWithSign]

end Jax.Lax
