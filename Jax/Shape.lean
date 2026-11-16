module

public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Data.Finset.Sort
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Logic.Equiv.Fin.Basic
public import Mathlib.Tactic.Ring

/-!
# Shapes and indices

A shape of rank `n` gives the size of each of its `n` axes.
The JAX shape `(2, 3)` is `![2, 3] : Shape 2`.
-/

@[expose] public section

namespace Jax

/-- An array shape of rank `n`. -/
abbrev Shape (n : ℕ) := Fin n → ℕ

/-- A position in an array of shape `s`: one index per axis. -/
abbrev Index {n : ℕ} (s : Shape n) := (d : Fin n) → Fin (s d)

variable {m n : ℕ}

/-- The index that keeps axis `e d` of `i` as its axis `d`. -/
def Index.proj {s : Shape n} (e : Fin m → Fin n) (i : Index s) : Index (s ∘ e) := fun d => i (e d)

/-! ## Broadcasting -/

/-- An array of shape `s` can be broadcast to shape `t`: every axis has size 1 or the size in `t`. -/
def BroadcastsTo (s t : Shape n) : Prop := ∀ d, s d = 1 ∨ s d = t d

/-- The entry of an array of shape `s` that is read at position `j` once it is broadcast to shape `t`.
[StableHLO `broadcast_in_dim`](https://openxla.org/stablehlo/spec#broadcast_in_dim):
`operand_index[d] = 0` if `dim(operand, d) = 1`, and `result_index[d]` otherwise. -/
def Index.unbroadcast {s t : Shape n} (h : BroadcastsTo s t) (j : Index t) : Index s :=
  fun d => if h1 : s d = 1 then ⟨0, by omega⟩ else ⟨j d, by have := h d; have := (j d).isLt; omega⟩

/-- Two shapes can be broadcast against each other: on every axis the sizes are equal or one is 1. -/
def Broadcastable (s t : Shape n) : Prop := ∀ d, s d = t d ∨ s d = 1 ∨ t d = 1

/-- The shape of the broadcast of arrays of shapes `s` and `t`.
`jax/_src/lax/lax.py:121`: `# if all dims are equal (or 1), the result is the non-1 size` -/
def broadcastShape (s t : Shape n) : Shape n := fun d => if s d = 1 then t d else s d

/-- Arrays of shapes `ss` can be broadcast against each other. -/
def BroadcastableAll (ss : List (Shape n)) : Prop := ∀ s ∈ ss, ∀ t ∈ ss, Broadcastable s t

/-- The shape of the broadcast of arrays of shapes `ss`: on every axis the first size other than 1, or 1.
`jax/_src/lax/lax.py:121`: `# if all dims are equal (or 1), the result is the non-1 size` -/
def broadcastShapes (ss : List (Shape n)) : Shape n :=
  fun d => ((ss.map (· d)).find? (· ≠ 1)).getD 1

/-- The axes that remain when `axes` are removed, in increasing order.
This is `np.delete(np.arange(n), axes)`. -/
def keptAxes (axes : Finset (Fin n)) : Fin axesᶜ.card → Fin n := axesᶜ.orderEmbOfFin rfl

/-! ## Row-major order -/

/-- The position of an index among the indices of shape `s` in row-major order,
in which the first axis varies slowest and the last axis fastest.
`Index.rowMajor_val` gives the position as a sum. -/
def Index.rowMajor : {n : ℕ} → {s : Shape n} → Index s ≃ Fin (∏ d, s d)
  | 0, s => (Equiv.ofUnique (Index s) (Fin 1)).trans (finCongr (Fin.prod_univ_zero s).symm)
  | _ + 1, s => (Fin.consEquiv fun d => Fin (s d)).symm.trans <|
      (Equiv.prodCongr (Equiv.refl _) Index.rowMajor).trans <|
        finProdFinEquiv.trans (finCongr (Fin.prod_univ_succ s).symm)

/-! ## Lemmas -/

theorem BroadcastsTo.refl (s : Shape n) : BroadcastsTo s s := fun _ => Or.inr rfl

theorem Broadcastable.refl (s : Shape n) : Broadcastable s s := fun _ => Or.inl rfl

theorem Broadcastable.symm {s t : Shape n} (h : Broadcastable s t) : Broadcastable t s :=
  fun d => by have := h d; omega

theorem broadcastsTo_left (s t : Shape n) : BroadcastsTo s (broadcastShape s t) :=
  fun d => by unfold broadcastShape; split <;> simp_all

theorem Broadcastable.right {s t : Shape n} (h : Broadcastable s t) :
    BroadcastsTo t (broadcastShape s t) :=
  fun d => by unfold broadcastShape; have := h d; split <;> omega

theorem broadcastable_left (s t : Shape n) : Broadcastable s (broadcastShape s t) :=
  fun d => by unfold broadcastShape; split <;> simp_all

theorem keptAxes_notMem (axes : Finset (Fin n)) (a : Fin axesᶜ.card) : keptAxes axes a ∉ axes :=
  Finset.mem_compl.1 (Finset.orderEmbOfFin_mem _ _ a)

theorem keptAxes_strictMono (axes : Finset (Fin n)) : StrictMono (keptAxes axes) :=
  (axesᶜ.orderEmbOfFin rfl).strictMono

theorem keptAxes_injective (axes : Finset (Fin n)) : Function.Injective (keptAxes axes) :=
  (keptAxes_strictMono axes).injective

/-- `keptAxes` is the only increasing enumeration of the axes not in `axes`. -/
theorem keptAxes_eq_of_strictMono {axes : Finset (Fin n)} {k : ℕ} (hk : axesᶜ.card = k)
    (f : Fin k → Fin n) (hf : ∀ i, f i ∉ axes) (hmono : StrictMono f) (a : Fin axesᶜ.card) :
    keptAxes axes a = f (Fin.cast hk a) := by
  subst hk
  have := Finset.orderEmbOfFin_unique (s := axesᶜ) rfl (fun i => Finset.mem_compl.2 (hf i)) hmono
  rw [keptAxes, ← this]
  rfl

/-- Every axis not in `axes` is kept. -/
theorem keptAxes_orderIsoOfFin_symm (axes : Finset (Fin m)) (p : Fin m) (hp : p ∈ axesᶜ) :
    keptAxes axes ((axesᶜ.orderIsoOfFin rfl).symm ⟨p, hp⟩) = p := by
  show ((axesᶜ.orderIsoOfFin rfl ((axesᶜ.orderIsoOfFin rfl).symm ⟨p, hp⟩) : Fin m)) = p
  rw [OrderIso.apply_symm_apply]

theorem orderIsoOfFin_symm_keptAxes (axes : Finset (Fin m)) (a : Fin axesᶜ.card)
    (hp : keptAxes axes a ∈ axesᶜ) : (axesᶜ.orderIsoOfFin rfl).symm ⟨keptAxes axes a, hp⟩ = a := by
  rw [OrderIso.symm_apply_eq]
  rfl

theorem keptAxes_surj {axes : Finset (Fin n)} {p : Fin n} (hp : p ∉ axes) :
    ∃ a, keptAxes axes a = p :=
  ⟨_, keptAxes_orderIsoOfFin_symm axes p (Finset.mem_compl.2 hp)⟩

/-- Removing no axes keeps all of them in place. -/
theorem keptAxes_val_of_eq_empty {axes : Finset (Fin n)} (h : axes = ∅) (a : Fin axesᶜ.card) :
    (keptAxes axes a : ℕ) = a := by
  subst h
  have hk : (∅ : Finset (Fin n))ᶜ.card = n := by simp
  rw [keptAxes_eq_of_strictMono hk id (fun _ => Finset.notMem_empty _) strictMono_id]
  rfl

/-- An axis of `broadcastShapes ss` has size `v` if every shape in `ss` has size 1 or `v` there,
and one of them has size `v` unless `v = 1`. -/
theorem broadcastShapes_eq {ss : List (Shape n)} {d : Fin n} {v : ℕ}
    (hall : ∀ s ∈ ss, s d = 1 ∨ s d = v) (hv : v = 1 ∨ ∃ s ∈ ss, s d = v) :
    broadcastShapes ss d = v := by
  unfold broadcastShapes
  cases hf : (ss.map (· d)).find? (· ≠ 1) with
  | none =>
    rw [Option.getD_none]
    rcases hv with hv | ⟨s, hs, hsv⟩
    · exact hv.symm
    · have := (List.find?_eq_none.1 hf) (s d) (List.mem_map_of_mem hs)
      have := hall s hs
      simp at *
      omega
  | some u =>
    rw [Option.getD_some]
    have hu := List.find?_some hf
    obtain ⟨s, hs, rfl⟩ := List.mem_map.1 (List.mem_of_find?_eq_some hf)
    have := hall s hs
    simp at hu
    omega

theorem broadcastsTo_broadcastShapes {ss : List (Shape n)} (h : BroadcastableAll ss) {s : Shape n}
    (hs : s ∈ ss) : BroadcastsTo s (broadcastShapes ss) := fun d => by
  by_cases h1 : s d = 1
  · exact Or.inl h1
  · refine Or.inr (broadcastShapes_eq (fun t ht => ?_) (Or.inr ⟨s, hs, rfl⟩)).symm
    have := h t ht s hs d
    omega

/-- An axis of `broadcastShapes ss` has size 1 or the size of one of `ss`. -/
theorem broadcastShapes_cases (ss : List (Shape n)) (d : Fin n) :
    broadcastShapes ss d = 1 ∨ ∃ s ∈ ss, s d = broadcastShapes ss d := by
  unfold broadcastShapes
  cases hf : (ss.map (· d)).find? (· ≠ 1) with
  | none => exact Or.inl rfl
  | some u =>
    obtain ⟨s, hs, rfl⟩ := List.mem_map.1 (List.mem_of_find?_eq_some hf)
    exact Or.inr ⟨s, hs, rfl⟩

theorem BroadcastableAll.mono {ss ts : List (Shape n)} (h : BroadcastableAll ss)
    (hts : ∀ t ∈ ts, t ∈ ss) : BroadcastableAll ts :=
  fun s hs t ht => h s (hts s hs) t (hts t ht)

theorem BroadcastsTo.broadcastable {s t : Shape n} (h : BroadcastsTo s t) : Broadcastable s t :=
  fun d => by have := h d; omega

/-- The row-major position of `i` is `∑ d, i d * ∏ e > d, s e`. -/
theorem Index.rowMajor_val : ∀ {n : ℕ} {s : Shape n} (i : Index s),
    (Index.rowMajor i : ℕ) = ∑ d, (i d : ℕ) * ∏ e ∈ Finset.Ioi d, s e
  | 0, _, _ => by simp [Index.rowMajor]
  | n + 1, s, i => by
    have ih := Index.rowMajor_val (s := fun d => s d.succ) (Fin.tail i)
    simp only [Index.rowMajor, Equiv.trans_apply, finCongr_apply, Fin.val_cast,
      finProdFinEquiv_apply_val, Equiv.prodCongr_apply, Equiv.coe_refl]
    show (Index.rowMajor (Fin.tail i) : ℕ) + (∏ d : Fin n, s d.succ) * i 0 = _
    rw [ih, Fin.sum_univ_succ, Fin.prod_Ioi_zero]
    simp only [Fin.prod_Ioi_succ, Fin.tail]
    ring

/-- The row-major position does not change if axes of size 1 are removed. -/
theorem Index.rowMajor_proj {k : ℕ} {s : Shape n} {e : Fin k → Fin n} (he : StrictMono e)
    (hs : ∀ d ∉ Set.range e, s d = 1) (j : Index s) :
    (Index.rowMajor (j.proj e) : ℕ) = Index.rowMajor j := by
  rw [Index.rowMajor_val, Index.rowMajor_val]
  refine Fintype.sum_of_injective e he.injective _ _ (fun d hd => ?_) fun a => ?_
  · have h1 : (j d : ℕ) < 1 := hs d hd ▸ (j d).isLt
    simp [Nat.lt_one_iff.1 h1]
  · show (j (e a) : ℕ) * ∏ b ∈ Finset.Ioi a, s (e b) = _
    congr 1
    let em : Fin k ↪ Fin n := ⟨e, he.injective⟩
    rw [show ∏ b ∈ Finset.Ioi a, s (e b) = ∏ d ∈ (Finset.Ioi a).map em, s d from
      (Finset.prod_map _ em s).symm]
    refine Finset.prod_subset (fun d hd => ?_) fun d hd hdm => hs d ?_
    · obtain ⟨b, hb, rfl⟩ := Finset.mem_map.1 hd
      exact Finset.mem_Ioi.2 (he (Finset.mem_Ioi.1 hb))
    · rintro ⟨b, rfl⟩
      exact hdm (Finset.mem_map.2 ⟨b, Finset.mem_Ioi.2 (he.lt_iff_lt.1 (Finset.mem_Ioi.1 hd)), rfl⟩)

/-- The row-major position depends only on the values of the index and of the shape. -/
theorem Index.rowMajor_congr {s t : Shape n} (hst : ∀ d, s d = t d) {i : Index s} {j : Index t}
    (hij : ∀ d, (i d : ℕ) = j d) : (Index.rowMajor i : ℕ) = Index.rowMajor j := by
  rw [Index.rowMajor_val, Index.rowMajor_val]
  simp only [hij, hst]

theorem Index.ext_val {s : Shape n} {i j : Index s} (h : ∀ d, (i d : ℕ) = j d) : i = j :=
  funext fun d => Fin.ext (h d)

theorem Index.unbroadcast_val {s t : Shape n} (h : BroadcastsTo s t) (j : Index t) (d : Fin n) :
    (j.unbroadcast h d : ℕ) = if s d = 1 then 0 else (j d : ℕ) := by
  unfold Index.unbroadcast
  split <;> rfl

theorem Index.unbroadcast_eq_of_val {s t : Shape n} (h : BroadcastsTo s t) (j : Index t)
    (i : Index s) (hij : ∀ d, (j d : ℕ) = i d) : j.unbroadcast h = i := by
  refine Index.ext_val fun d => ?_
  rw [Index.unbroadcast_val]
  have := (i d).isLt
  split <;> have := hij d <;> omega

theorem Index.unbroadcast_refl {s : Shape n} (j : Index s) :
    j.unbroadcast (BroadcastsTo.refl s) = j :=
  Index.unbroadcast_eq_of_val _ j j fun _ => rfl

theorem Index.unbroadcast_congr {s t t' : Shape n} (h : BroadcastsTo s t) (h' : BroadcastsTo s t')
    (j : Index t) (j' : Index t') (hjj : ∀ d, (j d : ℕ) = j' d) :
    j.unbroadcast h = j'.unbroadcast h' := by
  refine Index.ext_val fun d => ?_
  rw [Index.unbroadcast_val, Index.unbroadcast_val, hjj d]

end Jax
