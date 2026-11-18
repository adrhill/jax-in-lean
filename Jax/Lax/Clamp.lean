module

public import Jax.Lax.Dynamic

/-!
# `clamp`

Differentiation rules of the `clamp` primitive of `jax/_src/lax/lax.py`,
for bounds `lo = min` and `hi = max` of the shape of `x`.

`clamp` lowers to StableHLO `minimum(maximum(x, lo), hi)`, which is `hi` wherever `lo > hi`.
The rule for `lo` guards against `lo > hi`, but the rule for `hi` does not:
where `x ≤ hi < lo` the result is `hi`, and every rule returns zero (`clampJvp_not_correct`).
Elsewhere the rules are correct wherever `clamp` is differentiable (`clampJvp_correct`).

For scalar bounds and an operand that is not a scalar, the rules for `min` and `max` raise:
`select` does not broadcast the scalar tangent `g` to the shape of `_zeros(operand)`
(`clampJvpScalar_not_defined`).

The batching rule raises if only `max` is batched and `min` and `x` are scalars
(`clampBatch_not_correct`).
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n} {w : FloatWidth}

theorem whichShape_self (s : Shape n) : WhichShape s s := .inr rfl

theorem whichShape_broadcastShape_broadcastShape (s : Shape n) :
    WhichShape (broadcastShape (broadcastShape s s) (broadcastShape s s)) s :=
  .inr (congrArg List.ofFn (funext fun d => by simp [broadcastShape]))

theorem whichShape_broadcastShape_self (s : Shape n) : WhichShape (broadcastShape s s) s :=
  .inr (congrArg List.ofFn (funext fun d => by simp [broadcastShape]))

/-- The JVP rule of `clamp` for `min`. `_zeros(operand)` has the shape of `x`.
`jax/_src/lax/lax.py:7775`:
```
lambda g, min, operand, max:
select(bitwise_and(gt(min, operand), lt(min, max)),
       g, _zeros(operand))
``` -/
noncomputable def clampJvpMin (g lo x hi : Tensor (.float w) (Index s)) : Tensor (.float w) (Index s) :=
  selectN rfl (whichShape_broadcastShape_broadcastShape s)
    (bitwiseAnd (.refl _) (gt (.refl s) lo x) (lt (.refl s) lo hi)) 0 g

/-- The JVP rule of `clamp` for `operand`.
`jax/_src/lax/lax.py:7778`:
```
lambda g, min, operand, max:
select(bitwise_and(gt(operand, min), lt(operand, max)),
       g, _zeros(operand))
``` -/
noncomputable def clampJvpOperand (g lo x hi : Tensor (.float w) (Index s)) : Tensor (.float w) (Index s) :=
  selectN rfl (whichShape_broadcastShape_broadcastShape s)
    (bitwiseAnd (.refl _) (gt (.refl s) x lo) (lt (.refl s) x hi)) 0 g

/-- The JVP rule of `clamp` for `max`.
`jax/_src/lax/lax.py:7781`:
```
lambda g, min, operand, max:
select(lt(max, operand), g, _zeros(operand))
``` -/
noncomputable def clampJvpMax (g _lo x hi : Tensor (.float w) (Index s)) : Tensor (.float w) (Index s) :=
  selectN rfl (whichShape_broadcastShape_self s) (lt (.refl s) hi x) 0 g

/-! ## Scalar bounds -/

section Scalar

variable {sl sh : Shape 0}

open scoped Classical in
/-- The JVP rule of `clamp` for `min`, for scalar bounds `min` and `max`.
`gt(min, operand)` has the shape of `operand`, and `lt(min, max)` is a scalar.
The proof `hc` is the shape check of `select` that its cases `g` and `_zeros(operand)` have the same shape.
`jax/_src/lax/lax.py:7775`:
```
lambda g, min, operand, max:
select(bitwise_and(gt(min, operand), lt(min, max)),
       g, _zeros(operand))
``` -/
noncomputable def clampJvpMinScalar (hc : List.ofFn sl = List.ofFn s) (g lo : Tensor (.float w) (Index sl))
    (x : Tensor (.float w) (Index s)) (hi : Tensor (.float w) (Index sh)) : Tensor (.float w) (Index s) :=
  selectN hc (whichShape_self s)
    (binopScalarRight (· && ·) (binopScalarLeft (fun a b => decide (a.gt b)) lo x)
      (lt (fun d => d.elim0) lo hi)) 0 g

open scoped Classical in
/-- The JVP rule of `clamp` for `max`, for scalar bounds `min` and `max`.
`lt(max, operand)` has the shape of `operand`.
The proof `hc` is the shape check of `select` that its cases `g` and `_zeros(operand)` have the same shape.
`jax/_src/lax/lax.py:7781`:
```
lambda g, min, operand, max:
select(lt(max, operand), g, _zeros(operand))
``` -/
noncomputable def clampJvpMaxScalar (hc : List.ofFn sh = List.ofFn s) (g : Tensor (.float w) (Index sh))
    (_lo : Tensor (.float w) (Index sl)) (x : Tensor (.float w) (Index s)) (hi : Tensor (.float w) (Index sh)) : Tensor (.float w) (Index s) :=
  selectN hc (whichShape_self s) (binopScalarLeft (fun a b => decide (a.lt b)) hi x) 0 g

/-- The shape check of the rules of `clamp` for scalar `min` and `max`
holds exactly for an operand of rank 0. -/
theorem clampJvpScalar_defined_iff : List.ofFn sl = List.ofFn s ↔ n = 0 :=
  ⟨fun h => (rank_eq_of_ofFn_eq h).symm, fun h => by subst h; simp⟩

/-- The JVP rules of `clamp` for `min` and `max` raise for scalar bounds and an operand that is not a scalar:
`clamp` accepts every operand with scalar bounds,
but the shape check of the rules does not hold for the operand shape `(3,)`. -/
theorem clampJvpScalar_not_defined :
    ¬ ∀ (n : ℕ) (s : Shape n) (sb : Shape 0), WhichShape sb s → List.ofFn sb = List.ofFn s :=
  fun h => absurd (clampJvpScalar_defined_iff.1 (h 1 ![3] ![] (.inl rfl))) (by decide)

end Scalar

/-! ## Proofs -/

private theorem ofWhich_val {sw : Shape n} (hw : WhichShape sw s) (j : Index s) (d : Fin n) :
    (Index.ofWhich hw j d : ℕ) = j d := by
  unfold Index.ofWhich
  split
  · next hk => exact (Fin.cast hk d).elim0
  · rfl

/-- Unbroadcasting an index of a shape with the sizes of `s` keeps its values. -/
private theorem unbroadcast_val_eq {t u : Shape n} (h : BroadcastsTo t u) (ht : ∀ d, t d = s d)
    (v : Index u) (j : Index s) (hv : ∀ d, (v d : ℕ) = j d) (d : Fin n) :
    (v.unbroadcast h d : ℕ) = j d := by
  rw [Index.unbroadcast_val]
  have := (j d).isLt; have := ht d; have := hv d
  split_ifs <;> omega

/-- A comparison of arrays of equal shape at an index with the values of `j`. -/
private theorem binop_refl_at {d : DType} (op : Scalar w → Scalar w → d.Elem) (a b : Tensor (.float w) (Index s))
    (v : Index (broadcastShape s s)) (j : Index s) (hv : ∀ d, (v d : ℕ) = j d) :
    binop op (.refl s) a b v = op (a j) (b j) := by
  unfold binop
  rw [Index.unbroadcast_eq_of_val (broadcastsTo_left s s) v j hv]

/-- The conjunction of two comparisons at an index with the values of `j`. -/
private theorem bitwiseAnd_at (p q : Index (broadcastShape s s) → Bool)
    (hp : ∀ v v' : Index (broadcastShape s s), (∀ d, (v d : ℕ) = v' d) → p v = p v')
    (hq : ∀ v v' : Index (broadcastShape s s), (∀ d, (v d : ℕ) = v' d) → q v = q v')
    (w : Index (broadcastShape (broadcastShape s s) (broadcastShape s s))) (v : Index (broadcastShape s s))
    (hw : ∀ d, (w d : ℕ) = v d) :
    bitwiseAnd (.refl _) p q w = (p v && q v) := by
  unfold bitwiseAnd
  rw [hp _ v fun d => unbroadcast_val_eq _ (fun _ => rfl) w v hw d,
    hq _ v fun d => unbroadcast_val_eq _ (fun _ => rfl) w v hw d]

/-- A comparison of arrays of equal shape depends only on the values of the index. -/
private theorem binop_refl_congr {d : DType} (op : Scalar w → Scalar w → d.Elem) (a b : Tensor (.float w) (Index s))
    (v v' : Index (broadcastShape s s)) (h : ∀ d, (v d : ℕ) = v' d) :
    binop op (.refl s) a b v = binop op (.refl s) a b v' := by
  rw [Index.ext_val h]

open scoped Classical in
/-- The predicate of the rules for `min` and `operand` at the index `j` of the result. -/
private theorem and_at (a b c e : Tensor (.float w) (Index s)) (j : Index s) :
    bitwiseAnd (.refl _) (gt (.refl s) a b) (lt (.refl s) c e)
      (Index.ofWhich (whichShape_broadcastShape_broadcastShape s) j) =
      (decide ((a j).gt (b j)) && decide ((c j).lt (e j))) := by
  let v : Index (broadcastShape s s) := fun d => ⟨j d, by simp [broadcastShape]⟩
  have hv : ∀ d, (v d : ℕ) = j d := fun _ => rfl
  rw [bitwiseAnd_at (gt (.refl s) a b) (lt (.refl s) c e) (binop_refl_congr _ _ _)
    (binop_refl_congr _ _ _) _ v
    fun d => by rw [ofWhich_val]]
  simp only [gt, lt, binop_refl_at _ _ _ v j hv]

open scoped Classical in
private theorem lt_at (a b : Tensor (.float w) (Index s)) (j : Index s) :
    lt (.refl s) a b (Index.ofWhich (whichShape_broadcastShape_self s) j) =
      decide ((a j).lt (b j)) :=
  binop_refl_at _ _ _ _ j (ofWhich_val _ j)

private theorem ofWhich_self (j : Index s) : Index.ofWhich (whichShape_self s) j = j :=
  Index.ext_val (ofWhich_val _ j)

/-- `clamp` on finite arrays. -/
theorem clamp_ofReal (lo x hi : Index s → ℝ) :
    clamp (w := w) (whichShape_self s) (whichShape_self s) (ofReal lo) (ofReal x) (ofReal hi) =
      ofReal fun j => (x j ⊔ lo j) ⊓ hi j := by
  funext j
  simp [clamp, ofReal, ofWhich_self]

/-- The rule for `min` on finite arrays. -/
theorem clampJvpMin_ofReal (g lo x hi : Index s → ℝ) :
    clampJvpMin (w := w) (ofReal g) (ofReal lo) (ofReal x) (ofReal hi) =
      ofReal fun j => if x j < lo j ∧ lo j < hi j then g j else 0 := by
  funext j
  simp only [clampJvpMin, selectN, and_at]
  simp only [ofReal, Scalar.gt_fin, Scalar.lt_fin]
  split_ifs <;> simp_all [Scalar.zero_def]

/-- The rule for `operand` on finite arrays. -/
theorem clampJvpOperand_ofReal (g lo x hi : Index s → ℝ) :
    clampJvpOperand (w := w) (ofReal g) (ofReal lo) (ofReal x) (ofReal hi) =
      ofReal fun j => if lo j < x j ∧ x j < hi j then g j else 0 := by
  funext j
  simp only [clampJvpOperand, selectN, and_at]
  simp only [ofReal, Scalar.gt_fin, Scalar.lt_fin]
  split_ifs <;> simp_all [Scalar.zero_def]

/-- The rule for `max` on finite arrays. -/
theorem clampJvpMax_ofReal (g lo x hi : Index s → ℝ) :
    clampJvpMax (w := w) (ofReal g) (ofReal lo) (ofReal x) (ofReal hi) =
      ofReal fun j => if hi j < x j then g j else 0 := by
  funext j
  simp only [clampJvpMax, selectN, lt_at]
  simp only [ofReal, Scalar.lt_fin]
  split_ifs <;> simp_all [Scalar.zero_def]

/-- The triples of finite arrays. -/
local notation "Triple" s => (Index s → ℝ) × (Index s → ℝ) × (Index s → ℝ)

private theorem hasFDerivAt_lo (j : Index s) (p : Triple s) :
    HasFDerivAt (fun q : Triple s => q.1 j)
      (ContinuousLinearMap.proj j ∘L ContinuousLinearMap.fst ℝ _ _) p :=
  (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Index s => ℝ) j ∘L
    ContinuousLinearMap.fst ℝ (Index s → ℝ) ((Index s → ℝ) × (Index s → ℝ))).hasFDerivAt

private theorem hasFDerivAt_x (j : Index s) (p : Triple s) :
    HasFDerivAt (fun q : Triple s => q.2.1 j)
      (ContinuousLinearMap.proj j ∘L ContinuousLinearMap.fst ℝ _ _ ∘L
        ContinuousLinearMap.snd ℝ _ _) p :=
  (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Index s => ℝ) j ∘L
    ContinuousLinearMap.fst ℝ (Index s → ℝ) (Index s → ℝ) ∘L
    ContinuousLinearMap.snd ℝ (Index s → ℝ) ((Index s → ℝ) × (Index s → ℝ))).hasFDerivAt

private theorem hasFDerivAt_hi (j : Index s) (p : Triple s) :
    HasFDerivAt (fun q : Triple s => q.2.2 j)
      (ContinuousLinearMap.proj j ∘L ContinuousLinearMap.snd ℝ _ _ ∘L
        ContinuousLinearMap.snd ℝ _ _) p :=
  (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Index s => ℝ) j ∘L
    ContinuousLinearMap.snd ℝ (Index s → ℝ) (Index s → ℝ) ∘L
    ContinuousLinearMap.snd ℝ (Index s → ℝ) ((Index s → ℝ) × (Index s → ℝ))).hasFDerivAt

/-! ## Correctness -/

/-- The JVP rules of `clamp` compute the derivative of `clamp` with bounds of the shape of `x`
where, in every entry, `x > hi`, or `lo < x < hi`, or `x < lo < hi`.
`clamp` is differentiable at every other point where `x ≤ hi < lo` in some entry,
and there the rules are wrong (`clampJvp_not_correct`). -/
theorem clampJvp_correct :
    IsJvp₃OfOn (clampJvpMin (s := s) (w := w)) clampJvpOperand clampJvpMax
      (clamp (w := w) (whichShape_self s) (whichShape_self s))
      {p | ∀ j, p.2.2 j < p.2.1 j ∨ (p.1 j < p.2.1 j ∧ p.2.1 j < p.2.2 j) ∨
        (p.2.1 j < p.1 j ∧ p.1 j < p.2.2 j)} := by
  intro lo x hi hdom
  let cl : Index s → ℝ := fun j => if x j < lo j ∧ lo j < hi j then 1 else 0
  let cx : Index s → ℝ := fun j => if lo j < x j ∧ x j < hi j then 1 else 0
  let ch : Index s → ℝ := fun j => if hi j < x j then 1 else 0
  refine ⟨fun p j => (p.2.1 j ⊔ p.1 j) ⊓ p.2.2 j, ContinuousLinearMap.pi fun j =>
    cl j • (ContinuousLinearMap.proj j ∘L ContinuousLinearMap.fst ℝ _ _) +
    cx j • (ContinuousLinearMap.proj j ∘L ContinuousLinearMap.fst ℝ _ _ ∘L
      ContinuousLinearMap.snd ℝ _ _) +
    ch j • (ContinuousLinearMap.proj j ∘L ContinuousLinearMap.snd ℝ _ _ ∘L
      ContinuousLinearMap.snd ℝ _ _), ?_, ?_, ?_⟩
  · exact .of_forall fun p => clamp_ofReal p.1 p.2.1 p.2.2
  · refine hasFDerivAt_pi.2 fun j => ?_
    rcases hdom j with h | ⟨h1, h2⟩ | ⟨h1, h2⟩
    · -- `hi < x`: the result is `hi`.
      have hopen : IsOpen {q : Triple s | q.2.2 j < q.2.1 j} :=
        isOpen_lt (by fun_prop) (by fun_prop)
      have hev : (fun q : Triple s => (q.2.1 j ⊔ q.1 j) ⊓ q.2.2 j) =ᶠ[nhds (lo, x, hi)]
          fun q => q.2.2 j := by
        filter_upwards [hopen.mem_nhds h] with q hq
        exact inf_eq_right.2 (hq.le.trans le_sup_left)
      have hl : ¬ (x j < lo j ∧ lo j < hi j) := fun ⟨a, b⟩ => by linarith
      have hx : ¬ (lo j < x j ∧ x j < hi j) := fun ⟨_, b⟩ => by linarith
      have hh : hi j < x j := h
      convert (hasFDerivAt_hi j (lo, x, hi)).congr_of_eventuallyEq hev using 1
      simp only [cl, cx, ch, hl, hx, hh, ↓reduceIte, zero_smul, one_smul, zero_add]
    · -- `lo < x < hi`: the result is `x`.
      have hopen : IsOpen {q : Triple s | q.1 j < q.2.1 j ∧ q.2.1 j < q.2.2 j} :=
        (isOpen_lt (f := fun q : Triple s => q.1 j) (g := fun q => q.2.1 j) (by fun_prop)
          (by fun_prop)).inter
        (isOpen_lt (f := fun q : Triple s => q.2.1 j) (g := fun q => q.2.2 j) (by fun_prop)
          (by fun_prop))
      have hev : (fun q : Triple s => (q.2.1 j ⊔ q.1 j) ⊓ q.2.2 j) =ᶠ[nhds (lo, x, hi)]
          fun q => q.2.1 j := by
        filter_upwards [hopen.mem_nhds ⟨h1, h2⟩] with q hq
        rw [sup_eq_left.2 hq.1.le, inf_eq_left.2 hq.2.le]
      have hl : ¬ (x j < lo j ∧ lo j < hi j) := fun ⟨a, _⟩ => by linarith
      have hh : ¬ hi j < x j := by linarith
      convert (hasFDerivAt_x j (lo, x, hi)).congr_of_eventuallyEq hev using 1
      simp only [cl, cx, ch, hl, h1, h2, hh, and_self, ↓reduceIte, zero_smul, one_smul,
        zero_add, add_zero]
    · -- `x < lo < hi`: the result is `lo`.
      have hopen : IsOpen {q : Triple s | q.2.1 j < q.1 j ∧ q.1 j < q.2.2 j} :=
        (isOpen_lt (f := fun q : Triple s => q.2.1 j) (g := fun q => q.1 j) (by fun_prop)
          (by fun_prop)).inter
        (isOpen_lt (f := fun q : Triple s => q.1 j) (g := fun q => q.2.2 j) (by fun_prop)
          (by fun_prop))
      have hev : (fun q : Triple s => (q.2.1 j ⊔ q.1 j) ⊓ q.2.2 j) =ᶠ[nhds (lo, x, hi)]
          fun q => q.1 j := by
        filter_upwards [hopen.mem_nhds ⟨h1, h2⟩] with q hq
        rw [sup_eq_right.2 hq.1.le, inf_eq_left.2 hq.2.le]
      have hx : ¬ (lo j < x j ∧ x j < hi j) := fun ⟨a, _⟩ => by linarith
      have hh : ¬ hi j < x j := by linarith
      convert (hasFDerivAt_lo j (lo, x, hi)).congr_of_eventuallyEq hev using 1
      simp only [cl, cx, ch, h1, h2, hx, hh, and_self, ↓reduceIte, zero_smul, one_smul,
        add_zero]
  · intro ga gb gc
    refine ⟨_, _, _, .of_eq (clampJvpMin_ofReal ga lo x hi),
      .of_eq (clampJvpOperand_ofReal gb lo x hi), .of_eq (clampJvpMax_ofReal gc lo x hi), ?_⟩
    ext j
    simp only [cl, cx, ch, ContinuousLinearMap.pi_apply, Pi.add_apply]
    split_ifs <;> simp

/-- Near a point where `x < hi < lo` in every entry, `clamp` is `hi`. -/
private theorem clamp_eventually_hi {lo x hi : Index s → ℝ}
    (h : ∀ j, x j < hi j ∧ hi j < lo j) :
    ∀ᶠ q in nhds (lo, x, hi), (fun j => (q.2.1 j ⊔ q.1 j) ⊓ q.2.2 j) =
      (ContinuousLinearMap.snd ℝ _ _ ∘L ContinuousLinearMap.snd ℝ _ _ : (Triple s) →L[ℝ] _) q := by
  have hopen : IsOpen {q : Triple s | ∀ j, q.2.1 j < q.2.2 j ∧ q.2.2 j < q.1 j} := by
    simp only [Set.ofPred_forall]
    exact isOpen_iInter_of_finite fun j =>
      (isOpen_lt (f := fun q : Triple s => q.2.1 j) (g := fun q => q.2.2 j) (by fun_prop)
        (by fun_prop)).inter
      (isOpen_lt (f := fun q : Triple s => q.2.2 j) (g := fun q => q.1 j) (by fun_prop)
        (by fun_prop))
  filter_upwards [hopen.mem_nhds h] with q hq
  funext j
  simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.coe_snd']
  rw [sup_eq_right.2 ((hq j).1.trans (hq j).2).le, inf_eq_right.2 (hq j).2.le]

/-- The JVP rules of `clamp` are not correct everywhere `clamp` is differentiable.
Near a point where `x < hi < lo`, `clamp` is `hi`, so its derivative is the tangent of `hi`,
but the rule for `max` returns zero there because `hi < x` is false,
and the rule for `min` returns zero because `lo < hi` is false.
The counterexample has shape `(1,)`, `lo = 2`, `x = -1` and `hi = 0`. -/
theorem clampJvp_not_correct :
    ¬ IsJvp₃Of (clampJvpMin (s := ![1]) (w := w)) clampJvpOperand clampJvpMax
      (clamp (w := w) (whichShape_self ![1]) (whichShape_self ![1])) := by
  intro hc
  have hpt : ∀ j : Index ![1], (fun _ => (-1 : ℝ)) j < (fun _ => (0 : ℝ)) j ∧
      (fun _ => (0 : ℝ)) j < (fun _ => (2 : ℝ)) j := fun _ => by norm_num
  obtain ⟨F, L, hF, hL, hrule⟩ := hc (fun _ => 2) (fun _ => -1) (fun _ => 0) (Set.mem_univ _)
  -- Near the point, `F` is the projection to `hi`, so `L` is too.
  have hev : F =ᶠ[nhds ((fun _ => 2), (fun _ => -1), (fun _ => 0))]
      (ContinuousLinearMap.snd ℝ _ _ ∘L ContinuousLinearMap.snd ℝ _ _ : (Triple ![1]) →L[ℝ] _) := by
    filter_upwards [hF, clamp_eventually_hi hpt] with q hq hq'
    rw [← hq', ← ofReal_injective.eq_iff, ← hq, clamp_ofReal]
  have hL' := hL.unique ((ContinuousLinearMap.hasFDerivAt _).congr_of_eventuallyEq hev)
  -- The rules return zero for the tangent `1` of `hi`.
  obtain ⟨da, db, dc, hda, hdb, hdc, hsum⟩ := hrule 0 0 1
  let j₀ : Index ![1] := fun _ => ⟨0, by simp⟩
  have ha := congrFun (ofReal_injective ((Tensor.Same.of_eq (clampJvpMin_ofReal _ _ _ _)).unique hda)) j₀
  have hb := congrFun (ofReal_injective ((Tensor.Same.of_eq (clampJvpOperand_ofReal _ _ _ _)).unique hdb)) j₀
  have hc' := congrFun (ofReal_injective ((Tensor.Same.of_eq (clampJvpMax_ofReal _ _ _ _)).unique hdc)) j₀
  have := congrFun hsum j₀
  rw [hL'] at this
  norm_num [← ha, ← hb, ← hc'] at this

/-! ## Batching rule -/

/-- The general case of `_clamp_batch_rule`, which moves the batch axes to the front
and broadcasts the arguments against each other.
`jax/_src/lax/lax.py:7751`:
```
  min = batching.bdim_at_front(min, min_bdim, size) if np.shape(min) else min
  max = batching.bdim_at_front(max, max_bdim, size) if np.shape(max) else max
  x = batching.bdim_at_front(x, x_bdim, size) if np.shape(x) else x
  if np.ndim(min) == 0 and np.ndim(x) > 0:
    min = broadcast(min, x.shape)
  if np.ndim(max) == 0 and np.ndim(x) > 0:
    max = broadcast(max, x.shape)
  if 0 < np.ndim(min) < np.ndim(x):
    assert np.ndim(min) == 1, np.ndim(min)
    min = broadcast_in_dim(min, x.shape, [0])
  if 0 < np.ndim(max) < np.ndim(x):
    assert np.ndim(max) == 1, np.ndim(max)
    max = broadcast_in_dim(max, x.shape, [0])
  if np.ndim(min) > np.ndim(x):
    assert np.ndim(x) == 0, np.ndim(x)
    x = broadcast(x, min.shape)
  return clamp_p.bind(min, x, max), 0
``` -/
noncomputable def clampBatchGeneral (size : ℕ) (lo x hi : Batched) : Option Batched := do
  let lo ← if lo.val.rank ≠ 0 then lo.val.bdimAtFront lo.bdim size else some lo.val
  let hi ← if hi.val.rank ≠ 0 then hi.val.bdimAtFront hi.bdim size else some hi.val
  let x ← if x.val.rank ≠ 0 then x.val.bdimAtFront x.bdim size else some x.val
  let lo ← if lo.rank = 0 ∧ x.rank > 0 then lo.broadcast x.shapeList else some lo
  let hi ← if hi.rank = 0 ∧ x.rank > 0 then hi.broadcast x.shapeList else some hi
  let lo ← if 0 < lo.rank ∧ lo.rank < x.rank then
    (if lo.rank = 1 then lo.broadcastInDim x.shapeList [0] else none) else some lo
  let hi ← if 0 < hi.rank ∧ hi.rank < x.rank then
    (if hi.rank = 1 then hi.broadcastInDim x.shapeList [0] else none) else some hi
  let x ← if lo.rank > x.rank then
    (if x.rank = 0 then x.broadcast lo.shapeList else none) else some x
  let out ← Arr.clamp lo x hi
  some ⟨out, some 0⟩

/-- The batching rule of `clamp`.
`jax/_src/lax/lax.py:7733`:
```
def _clamp_batch_rule(batched_args, batch_dims, **params):
  min, x, max = batched_args
  min_bdim, x_bdim, max_bdim = batch_dims
  size = next(x.shape[i] for x, i in zip(batched_args, batch_dims)
              if i is not None)

  # avoid transposes and some broadcasts in special cases
  if min_bdim == x_bdim == max_bdim:
    if np.shape(min) == np.shape(x) == np.shape(max):
      return clamp_p.bind(min, x, max), x_bdim
    elif np.ndim(min) == np.ndim(max) == 0:
      return clamp_p.bind(min, x, max), x_bdim
    elif np.ndim(min) == np.ndim(max) == 1:
      min = broadcast_in_dim(min, x.shape, [min_bdim])
      max = broadcast_in_dim(max, x.shape, [max_bdim])
      return clamp_p.bind(min, x, max), x_bdim
  elif np.ndim(min) == 0 and np.ndim(max) == 0 and x_bdim is not None:
    return clamp_p.bind(min, x, max), x_bdim
  ...
``` -/
noncomputable def clampBatch (lo x hi : Batched) : Option Batched := do
  let size ← batchSize [lo, x, hi]
  let direct : Option Batched := (Arr.clamp lo.val x.val hi.val).map (⟨·, x.bdim⟩)
  if lo.bdim = x.bdim ∧ x.bdim = hi.bdim then
    if lo.val.shapeList = x.val.shapeList ∧ x.val.shapeList = hi.val.shapeList then direct
    else if lo.val.rank = 0 ∧ hi.val.rank = 0 then direct
    else if lo.val.rank = 1 ∧ hi.val.rank = 1 then do
      let lo' ← lo.val.broadcastInDim x.val.shapeList lo.bdim.toList
      let hi' ← hi.val.broadcastInDim x.val.shapeList hi.bdim.toList
      let out ← Arr.clamp lo' x.val hi'
      some ⟨out, x.bdim⟩
    else clampBatchGeneral size lo x hi
  else if lo.val.rank = 0 ∧ hi.val.rank = 0 ∧ x.bdim.isSome then direct
  else clampBatchGeneral size lo x hi

/-- The batching rule of `clamp` is not correct: it raises if only `max` is batched
and `min` and `x` are scalars.
The general case broadcasts a scalar `x` to the shape of a batched `min`, but not of a batched `max`,
so `clamp` gets a scalar operand and a `max` of shape `(size,)`.
The counterexample has a batch of size 1. -/
theorem clampBatch_not_correct : ¬ IsBatchOf₃ clampBatch Arr.clamp := by
  intro h
  let scalar : Arr := ⟨(![] : Shape 0), (0 : Tensor .f32 _)⟩
  let batch : Arr := ⟨(![1] : Shape 1), (0 : Tensor .f32 _)⟩
  obtain ⟨y, hy, -⟩ := h 1 ⟨scalar, none⟩ ⟨scalar, none⟩ ⟨batch, some 0⟩ Nat.one_pos
    (by simp [Batched.HasSize]) (by simp [Batched.HasSize])
    (by simp [Batched.HasSize, batch]) (by simp) (fun i hi => by
      obtain rfl : i = 0 := by omega
      exact ⟨_, _, _, rfl, rfl, rfl, by decide⟩)
  have hnone : (clampBatch ⟨scalar, none⟩ ⟨scalar, none⟩ ⟨batch, some 0⟩).isSome = false := by
    decide
  rw [hy] at hnone
  exact absurd hnone (by simp)

end Jax.Lax
