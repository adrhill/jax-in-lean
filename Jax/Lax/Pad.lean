module

public import Jax.Lax.Primal

/-!
# `pad`

Differentiation rules of the `pad` primitive of `jax/_src/lax/lax.py`.

`pad` is linear in the pair of `operand` and `padding_value`.
Its transpose rule is correct where both are linear inputs (`padTranspose_correct`),
also for negative and interior padding.

Where only `padding_value` is a linear input, `_pad_transpose` calls `reduce_sum(None, …)` and raises
(`padTranspose_not_correct`).
See `bugs/pad-transpose-padding-value.json`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n} {sv : Shape 0} {w : FloatWidth}

/-- The JVP rule of `pad`, which applies the primitive to the tangents.
The tangents come first, then the primals.
`jax/_src/lax/lax.py:8260`: `ad.deflinear2(pad_p, _pad_transpose)` and
`jax/_src/interpreters/ad.py:950`: `return val_out, primitive.bind(*tangents, **params)` -/
noncomputable def padJvp (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1))
    (operandDot : Tensor (.float w) (Index s)) (paddingValueDot : Tensor (.float w) (Index sv))
    (_operand : Tensor (.float w) (Index s)) (_paddingValue : Tensor (.float w) (Index sv)) :
    Tensor (.float w) (Index (padShape s lo hi interior)) :=
  pad lo hi interior h operandDot paddingValueDot

/-- The shape check of the `pad` in `_pad_transpose`, which undoes the edge padding. -/
theorem padTranspose_unpad (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1)) (d : Fin n) :
    0 ≤ (-lo) d + (-hi) d + dilateDim (padShape s lo hi interior d) ((0 : Fin n → ℕ) d + 1) := by
  have := h d
  simp only [padShape, dilateDim, Pi.neg_apply, Pi.zero_apply] at this ⊢
  omega

/-- Two shapes of the rank of a sum over all axes can be broadcast against each other. -/
theorem broadcastable_total (a b : Shape (Finset.univ : Finset (Fin n))ᶜ.card) :
    Broadcastable a b :=
  fun d => (Fin.cast (by simp) d : Fin 0).elim0

/-- The transpose rule of `pad` for `operand`:
it undoes the edge padding with the negated configuration, and the interior padding with a strided slice.
The undefined primal `operand` is replaced by its shape `s`.
`jax/_src/lax/lax.py:8217`:
```
def _pad_transpose(t, operand, padding_value, *, padding_config):
  if type(t) is ad_util.Zero:
    ...
  else:
    lo, hi, interior = util.unzip3(padding_config)
    total = lambda x: reduce_sum(x, list(range(t.ndim)))

    def t_op():
      unpad_config = safe_zip(np.negative(lo), np.negative(hi),
                              np.zeros_like(interior))
      unpadded = pad(t, np.array(0., t.dtype), unpad_config)
      return slicing.slice(unpadded, list(np.zeros_like(lo)), unpadded.shape,
                           list(np.add(interior, 1)))

    t_operand = t_op() if ad.is_undefined_primal(operand) else None
    t_padv = sub(total(t), total(t_operand)) if ad.is_undefined_primal(padding_value) else None
  return [t_operand, t_padv]
``` -/
noncomputable def padTransposeOperand (s : Shape n) (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1))
    (t : Tensor (.float w) (Index (padShape s lo hi interior))) :
    Tensor (.float w) (Index (sliceShape 0 (padShape (padShape s lo hi interior) (-lo) (-hi) 0)
      (interior + 1))) :=
  slice 0 _ (interior + 1) (fun _ => le_rfl) (fun _ => Nat.zero_le _)
    (fun d => Nat.succ_pos (interior d))
    (pad (sv := ![]) (-lo) (-hi) 0 (padTranspose_unpad lo hi interior h) t 0)

/-- The transpose rule of `pad` for `padding_value`, where `operand` is a linear input too:
the sum of the cotangent minus the sum of the cotangent of `operand`.
`jax/_src/lax/lax.py:8223`:
```
    total = lambda x: reduce_sum(x, list(range(t.ndim)))
    ...
    t_padv = sub(total(t), total(t_operand)) if ad.is_undefined_primal(padding_value) else None
``` -/
noncomputable def padTransposePaddingValue (s : Shape n) (lo hi : Fin n → ℤ)
    (interior : Fin n → ℕ) (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1))
    (t : Tensor (.float w) (Index (padShape s lo hi interior))) :
    Tensor (.float w) (Index (broadcastShape (padShape s lo hi interior ∘ keptAxes Finset.univ)
      (sliceShape 0 (padShape (padShape s lo hi interior) (-lo) (-hi) 0) (interior + 1) ∘
        keptAxes Finset.univ))) :=
  sub (broadcastable_total _ _) (reduceSum Finset.univ t)
    (reduceSum Finset.univ (padTransposeOperand s lo hi interior h t))

/-- The transpose rule of `pad`.
An argument that is a linear input is `none`, as in `IsPartialTranspose₂Of`,
and the result `none` is a raised exception:
`total(t_operand)` calls `reduce_sum(None, ...)` if `operand` is not a linear input.
`jax/_src/lax/lax.py:8232`:
```
    t_operand = t_op() if ad.is_undefined_primal(operand) else None
    t_padv = sub(total(t), total(t_operand)) if ad.is_undefined_primal(padding_value) else None
  return [t_operand, t_padv]
``` -/
noncomputable def padTranspose (s : Shape n) (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1))
    (t : Tensor (.float w) (Index (padShape s lo hi interior))) (operand : Option (Tensor (.float w) (Index s)))
    (paddingValue : Option (Tensor (.float w) (Index sv))) :
    Option (Option (Tensor (.float w) (Index (sliceShape 0 (padShape (padShape s lo hi interior) (-lo) (-hi) 0)
        (interior + 1)))) ×
      Option (Tensor (.float w) (Index (broadcastShape (padShape s lo hi interior ∘ keptAxes Finset.univ)
        (sliceShape 0 (padShape (padShape s lo hi interior) (-lo) (-hi) 0) (interior + 1) ∘
          keptAxes Finset.univ))))) :=
  let tOperand := if operand.isNone then some (padTransposeOperand s lo hi interior h t) else none
  if paddingValue.isNone then
    tOperand.map fun tOp => (tOperand,
      some (sub (broadcastable_total _ _) (reduceSum Finset.univ t) (reduceSum Finset.univ tOp)))
  else some (tOperand, none)

/-! ## Proofs -/

/-- Entry `i` of the operand is entry `j` of the result of `pad`. -/
private abbrev Hits (lo : Fin n → ℤ) (interior : Fin n → ℕ) {s t : Shape n} (i : Index s)
    (j : Index t) : Prop :=
  ∀ d, ((j d : ℕ) : ℤ) = lo d + (i d : ℕ) * ((interior d : ℤ) + 1)

private theorem Hits.left_unique {lo : Fin n → ℤ} {interior : Fin n → ℕ} {t : Shape n}
    {i i' : Index s} {j : Index t} (h : Hits lo interior i j) (h' : Hits lo interior i' j) :
    i = i' := by
  refine Index.ext_val fun d => ?_
  have hk : ((interior d : ℤ) + 1) ≠ 0 := by omega
  have := h d
  have := h' d
  exact_mod_cast mul_right_cancel₀ hk (by linarith :
    ((i d : ℕ) : ℤ) * ((interior d : ℤ) + 1) = ((i' d : ℕ) : ℤ) * ((interior d : ℤ) + 1))

private theorem Hits.right_unique {lo : Fin n → ℤ} {interior : Fin n → ℕ} {t : Shape n}
    {i : Index s} {j j' : Index t} (h : Hits lo interior i j) (h' : Hits lo interior i j') :
    j = j' := by
  refine Index.ext_val fun d => ?_
  have := h d
  have := h' d
  omega

/-- A choice among the elements with a property that only `a` has is `a`. -/
private theorem dite_choose {α β : Type} {p : α → Prop} [Decidable (∃ a, p a)] (f : α → β) (b : β) {a : α} (ha : p a)
    (huniq : ∀ a', p a' → a' = a) : (if h : ∃ a, p a then f h.choose else b) = f a := by
  split_ifs with h
  · exact congrArg f (huniq _ h.choose_spec)
  · exact absurd ⟨a, ha⟩ h

/-- `pad` on finite arrays, a linear map of the pair of `operand` and `padding_value`. -/
private noncomputable def padReal (lo hi : Fin n → ℤ) (interior : Fin n → ℕ) :
    (Index s → ℝ) × (Index sv → ℝ) →ₗ[ℝ] Index (padShape s lo hi interior) → ℝ where
  toFun p j := if h : ∃ i : Index s, Hits lo interior i j then p.1 h.choose
    else p.2 fun d => d.elim0
  map_add' _ _ := by
    ext j
    simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply]
    split_ifs <;> rfl
  map_smul' _ _ := by
    ext j
    simp only [Prod.smul_fst, Prod.smul_snd, Pi.smul_apply, RingHom.id_apply]
    split_ifs <;> rfl

private theorem pad_ofReal (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1)) (x : Index s → ℝ)
    (v : Index sv → ℝ) :
    pad (d := .float w) lo hi interior h (ofReal x) (ofReal v) = ofReal (padReal lo hi interior (x, v)) := by
  funext j
  simp only [pad, padReal, ofReal, Hits, LinearMap.coe_mk, AddHom.coe_mk]
  exact (apply_dite Scalar.fin _ _ _).symm

private theorem pad_of_hits {lo hi : Fin n → ℤ} {interior : Fin n → ℕ}
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1)) (x : Tensor (.float w) (Index s))
    (v : Tensor (.float w) (Index sv)) {i : Index s} {j : Index (padShape s lo hi interior)}
    (hij : Hits lo interior i j) : pad lo hi interior h x v j = x i :=
  dite_choose (p := fun i => Hits lo interior i j) x _ hij fun _ hi' => Hits.left_unique hi' hij

private theorem pad_of_not_hits {lo hi : Fin n → ℤ} {interior : Fin n → ℕ}
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1)) (x : Tensor (.float w) (Index s))
    (v : Tensor (.float w) (Index sv)) {j : Index (padShape s lo hi interior)}
    (hj : ¬ ∃ i : Index s, Hits lo interior i j) :
    pad lo hi interior h x v j = v fun d => d.elim0 := by
  simp only [pad, hj, ↓reduceDIte]

/-- `pad` reads entry `i` of `operand` at the position that `i` is padded to. -/
theorem pad_apply_of_eq {lo hi : Fin n → ℤ} {interior : Fin n → ℕ}
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1)) (x : Tensor (.float w) (Index s))
    (v : Tensor (.float w) (Index sv)) {i : Index s} {j : Index (padShape s lo hi interior)}
    (hij : ∀ d, ((j d : ℕ) : ℤ) = lo d + (i d : ℕ) * ((interior d : ℤ) + 1)) :
    pad lo hi interior h x v j = x i :=
  pad_of_hits h x v hij

/-- `pad` reads `padding_value` at the positions that no entry of `operand` is padded to. -/
theorem pad_apply_of_not_exists {lo hi : Fin n → ℤ} {interior : Fin n → ℕ}
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1)) (x : Tensor (.float w) (Index s))
    (v : Tensor (.float w) (Index sv)) {j : Index (padShape s lo hi interior)}
    (hj : ¬ ∃ i : Index s, ∀ d, ((j d : ℕ) : ℤ) = lo d + (i d : ℕ) * ((interior d : ℤ) + 1)) :
    pad lo hi interior h x v j = v fun d => d.elim0 :=
  pad_of_not_hits h x v hj

/-- The cotangent of `operand` on finite arrays: the entry of the cotangent that `i` is padded to,
and zero if `i` is removed. -/
private noncomputable def padTransposeReal (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (ct : Index (padShape s lo hi interior) → ℝ) (i : Index s) : ℝ :=
  if h : ∃ j : Index (padShape s lo hi interior), Hits lo interior i j then ct h.choose else 0

private theorem dilateDim_one (x : ℕ) : dilateDim x 1 = x := by
  unfold dilateDim
  omega

private theorem strideDim_dilateDim (m k : ℕ) : strideDim (dilateDim m (k + 1)) 1 (k + 1) = m := by
  unfold strideDim dilateDim
  have hK : (0 : ℤ) < ((k + 1 : ℕ) : ℤ) := by positivity
  generalize ((k + 1 : ℕ) : ℤ) = K at hK ⊢
  cases m with
  | zero =>
    have e1 : (1 + K * (((0 : ℕ) : ℤ) - 1)).toNat = 0 := by
      have : K * (((0 : ℕ) : ℤ) - 1) = -K := by push_cast; ring
      omega
    rw [e1]
    have hq : (((0 : ℕ) : ℤ) - ((1 : ℕ) : ℤ)) / K = -1 := by
      have h1 : (((0 : ℕ) : ℤ) - ((1 : ℕ) : ℤ)) / K < 0 :=
        Int.ediv_neg_of_neg_of_pos (by norm_num) hK
      have h2 : -1 ≤ (((0 : ℕ) : ℤ) - ((1 : ℕ) : ℤ)) / K :=
        (Int.le_ediv_iff_mul_le hK).2 (by push_cast; omega)
      omega
    rw [hq]
    rfl
  | succ m =>
    have h0 : 0 ≤ K * (m : ℤ) := by positivity
    have e1 : ((1 + K * (((m + 1 : ℕ) : ℤ) - 1)).toNat : ℤ) = 1 + K * m := by
      have : K * (((m + 1 : ℕ) : ℤ) - 1) = K * m := by push_cast; ring
      omega
    rw [e1, show 1 + K * (m : ℤ) - ((1 : ℕ) : ℤ) = K * m by push_cast; ring,
      Int.mul_ediv_cancel_left _ hK.ne']
    omega

/-- The rule for `operand` returns an array of the shape of `operand`. -/
private theorem padTransposeOperand_shape (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1)) (d : Fin n) :
    sliceShape 0 (padShape (padShape s lo hi interior) (-lo) (-hi) 0) (interior + 1) d = s d := by
  have hd := h d
  have hU : padShape (padShape s lo hi interior) (-lo) (-hi) 0 d =
      dilateDim (s d) (interior d + 1) := by
    simp only [padShape, Pi.neg_apply, Pi.zero_apply, zero_add, dilateDim_one]
    omega
  simp only [sliceShape, hU, Pi.zero_apply, Nat.sub_zero, Pi.add_apply, Pi.one_apply]
  exact strideDim_dilateDim _ _

/-- The entry of the unpadded cotangent that the strided slice reads for the entry `i` of `operand`. -/
private theorem unpad_apply (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1))
    (ct : Index (padShape s lo hi interior) → ℝ)
    (k : Index (padShape (padShape s lo hi interior) (-lo) (-hi) 0)) (i : Index s)
    (hk : ∀ d, (k d : ℕ) = i d * (interior d + 1)) :
    pad (d := .float w) (sv := ![]) (-lo) (-hi) 0 (padTranspose_unpad lo hi interior h) (ofReal ct) 0 k =
      .fin (padTransposeReal lo hi interior ct i) := by
  have hiff : ∀ j : Index (padShape s lo hi interior),
      Hits (-lo) 0 j k ↔ Hits lo interior i j := fun j => forall_congr' fun d => by
    have := hk d
    simp only [Pi.neg_apply, Pi.zero_apply, this]
    push_cast
    constructor <;> intro h' <;> linarith
  by_cases hex : ∃ j : Index (padShape s lo hi interior), Hits lo interior i j
  · obtain ⟨j, hj⟩ := hex
    rw [pad_of_hits _ _ _ ((hiff j).2 hj), padTransposeReal,
      dite_choose (p := fun j => Hits lo interior i j) ct 0 hj fun _ hj' =>
        Hits.right_unique hj' hj]
    rfl
  · rw [pad_of_not_hits _ _ _ fun ⟨j, hj⟩ => hex ⟨j, (hiff j).1 hj⟩]
    simp only [padTransposeReal, hex, ↓reduceDIte]
    rfl

/-- The rule for `operand` on finite arrays. -/
private theorem padTransposeOperand_same (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1))
    (ct : Index (padShape s lo hi interior) → ℝ) :
    (padTransposeOperand (w := w) s lo hi interior h (ofReal ct)).Same
      (ofReal (w := w) (padTransposeReal lo hi interior ct)) := by
  refine Tensor.same_of_shape_eq (padTransposeOperand_shape lo hi interior h) fun i' i hii => ?_
  simp only [padTransposeOperand, slice]
  exact unpad_apply lo hi interior h ct _ i fun d => by simp [hii d]

/-- The sum over all axes of an array with the finite entries `y`. -/
private theorem total_same {t t' : Shape n} {a : Tensor (.float w) (Index t')} {y : Index t → ℝ}
    (h : a.Same (ofReal (w := w) y)) (k : Index (t' ∘ keptAxes Finset.univ)) :
    reduceSum Finset.univ a k = .fin (∑ i, y i) := by
  obtain ⟨_, hs, hv⟩ := h.apply_eq
  rw [Scalar.fin_sum]
  simp only [reduceSum]
  rw [sum_filter_index_congr (s := t') (t := t) (fun d => by simpa using hs d)
    (P := fun i => i.proj (keptAxes Finset.univ) = k) (Q := fun _ => True) (fun i => a i)
    (fun j => Scalar.fin (y j))
    (fun i j _ => ⟨fun _ => trivial, fun _ => funext fun d => (Fin.cast (by simp) d : Fin 0).elim0⟩)
    (fun i j hij => hv i j (by simpa using hij))]
  simp

/-- The rule for `padding_value` on finite arrays. -/
private theorem padTransposePaddingValue_same (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1))
    (ct : Index (padShape s lo hi interior) → ℝ) :
    (padTransposePaddingValue (w := w) s lo hi interior h (ofReal ct)).Same
      (ofReal (w := w) fun _ : Index sv => ∑ j, ct j - ∑ i, padTransposeReal lo hi interior ct i) := by
  have hr : (Finset.univ : Finset (Fin n))ᶜ.card = 0 := by simp
  refine ⟨rfl, hr, fun d => (Fin.cast hr d).elim0, fun a b _ => ?_⟩
  simp only [padTransposePaddingValue, sub, binop, ofReal]
  rw [total_same (.of_eq rfl), total_same (padTransposeOperand_same lo hi interior h ct)]
  simp [sub_eq_add_neg]

/-- A sum over the pairs of a relation that matches every element with at most one partner
can be taken over either side. -/
private theorem sum_match {α β : Type} [Fintype α] [Fintype β] (M : α → β → Prop)
    [∀ j, Decidable (∃ i, M i j)] [∀ i, Decidable (∃ j, M i j)]
    (hα : ∀ i i' j, M i j → M i' j → i = i') (hβ : ∀ i j j', M i j → M i j' → j = j')
    (f : α → ℝ) (g : β → ℝ) :
    ∑ j, g j * (if h : ∃ i, M i j then f h.choose else 0) =
      ∑ i, (if h : ∃ j, M i j then g h.choose else 0) * f i := by
  classical
  have h1 : ∀ j, g j * (if h : ∃ i, M i j then f h.choose else 0) =
      ∑ i, if M i j then g j * f i else 0 := fun j => by
    split_ifs with h
    · rw [Finset.sum_eq_single h.choose (fun i _ hi => by
        have hn : ¬ M i j := fun hm => hi (hα _ _ _ hm h.choose_spec)
        simp only [hn, ↓reduceIte])
        (fun hn => absurd (Finset.mem_univ _) hn)]
      simp only [h.choose_spec, ↓reduceIte]
    · rw [mul_zero]
      refine (Finset.sum_eq_zero fun i _ => ?_).symm
      have hn : ¬ M i j := fun hm => h ⟨i, hm⟩
      simp only [hn, ↓reduceIte]
  have h2 : ∀ i, (if h : ∃ j, M i j then g h.choose else 0) * f i =
      ∑ j, if M i j then g j * f i else 0 := fun i => by
    split_ifs with h
    · rw [Finset.sum_eq_single h.choose (fun j _ hj => by
        have hn : ¬ M i j := fun hm => hj (hβ _ _ _ hm h.choose_spec)
        simp only [hn, ↓reduceIte])
        (fun hn => absurd (Finset.mem_univ _) hn)]
      simp only [h.choose_spec, ↓reduceIte]
    · rw [zero_mul]
      refine (Finset.sum_eq_zero fun j _ => ?_).symm
      have hn : ¬ M i j := fun hm => h ⟨j, hm⟩
      simp only [hn, ↓reduceIte]
  simp only [h1, h2]
  exact Finset.sum_comm

/-! ## Correctness -/

/-- The JVP rule of `pad` computes the derivative of `pad`. -/
theorem padJvp_correct (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1)) :
    IsJointJvp₂Of (padJvp (sv := sv) lo hi interior h) (pad (d := .float w) lo hi interior h) := by
  intro x v _
  let F := padReal (s := s) (sv := sv) lo hi interior
  let L := LinearMap.toContinuousLinearMap F
  exact ⟨F, L, .of_forall fun p => pad_ofReal lo hi interior h p.1 p.2, L.hasFDerivAt,
    fun g k => .of_eq (pad_ofReal lo hi interior h g k)⟩

/-- The transpose rules of `pad` transpose `pad`,
where `operand` and `padding_value` are both linear inputs. -/
theorem padTranspose_correct (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1)) :
    IsTranspose₂Of (padTransposeOperand s lo hi interior h)
      (padTransposePaddingValue s lo hi interior h) (pad (d := .float w) (sv := sv) lo hi interior h) := by
  classical
  intro ct x v
  refine ⟨padReal lo hi interior (x, v), padTransposeReal lo hi interior ct,
    fun _ => ∑ j, ct j - ∑ i, padTransposeReal lo hi interior ct i, pad_ofReal lo hi interior h x v,
    padTransposeOperand_same lo hi interior h ct, padTransposePaddingValue_same lo hi interior h ct,
    ?_⟩
  obtain ⟨v0, hv0⟩ : ∃ v0, v0 = v fun d => d.elim0 := ⟨_, rfl⟩
  have hv : ∀ k, v k = v0 := fun k => hv0 ▸ congrArg v (Subsingleton.elim _ _)
  have hz : ∀ j, padReal lo hi interior (x, v) j =
      (if h : ∃ i : Index s, Hits lo interior i j then x h.choose - v0 else 0) + v0 := fun j => by
    simp only [padReal, LinearMap.coe_mk, AddHom.coe_mk, ← hv0]
    split_ifs <;> ring
  have key := sum_match (fun (i : Index s) (j : Index (padShape s lo hi interior)) =>
    Hits lo interior i j) (fun _ _ _ h1 h2 => Hits.left_unique h1 h2)
    (fun _ _ _ h1 h2 => Hits.right_unique h1 h2) (fun i => x i - v0) ct
  simp only [hz, hv, mul_add, Finset.sum_add_distrib, key, padTransposeReal, mul_sub,
    Finset.sum_sub_distrib, ← Finset.sum_mul, Finset.univ_unique, Finset.sum_singleton]
  ring

/-- Where both arguments are linear inputs,
`padTranspose` returns the two cotangents of `padTranspose_correct`. -/
theorem padTranspose_none_none (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1))
    (t : Tensor (.float w) (Index (padShape s lo hi interior))) :
    padTranspose (sv := sv) s lo hi interior h t none none =
      some (some (padTransposeOperand s lo hi interior h t),
        some (padTransposePaddingValue s lo hi interior h t)) :=
  rfl

/-- The transpose rule of `pad` is not correct for any padding configuration:
it raises where `padding_value` is a linear input and `operand` is not. -/
theorem padTranspose_not_correct (lo hi : Fin n → ℤ) (interior : Fin n → ℕ)
    (h : ∀ d, 0 ≤ lo d + hi d + dilateDim (s d) (interior d + 1)) :
    ¬ IsPartialTranspose₂Of (padTranspose (sv := sv) s lo hi interior h)
      (pad (d := .float w) lo hi interior h) := by
  rintro ⟨_, _, _, hrule⟩
  obtain ⟨_, _, hsome, _⟩ := hrule 0 (some 0) none (.inr rfl)
  simp [padTranspose] at hsome

end Jax.Lax
