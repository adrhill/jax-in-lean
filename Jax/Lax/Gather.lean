module

public import Jax.Lax.Primal

/-!
# `gather`

Differentiation rules of the `gather` primitive of `jax/_src/lax/slicing.py`.

`gather` is linear in the operand if no window is filled with a nonzero value.
Its JVP rule gathers the tangent with the fill value 0 (`gatherJvp_correct`).
With the default fill value NaN of the mode `FILL_OR_DROP`, the result is not a real array
where a window does not lie inside the operand (`gather_not_isJvpOf`).

Its transpose rule adds the cotangent to zeros with `scatter_add`.
It is correct in the modes `CLIP` and `FILL_OR_DROP` (`gatherTranspose_correct`).
In the mode `PROMISE_IN_BOUNDS`, `gather` clamps a window that does not lie inside the operand
and `scatter_add` drops it (`gatherTranspose_promiseInBounds_not_correct`).
The docstring of `GatherScatterMode` says that gradients are not correct in this case.

The shape rule of `gather` accepts slices larger than the operand if the indices are empty,
and the shape rule of `scatter_add` rejects their cotangent, so the transpose rule raises
(`gatherTranspose_not_defined`).
See `bugs/gather-transpose-empty-indices.json`.

In the modes `CLIP` and `PROMISE_IN_BOUNDS`, every position of the result reads the operand
(`gatherIndex_isSome`).

The batching rule `_gather_batching_rule` is not modelled yet.
-/

@[expose] public section

namespace Jax.Lax

variable {n ni nu : ℕ} {w : FloatWidth} {wi : IntWidth}

/-! ## Dimension numbers -/

theorem length_batchDims {r : ℕ} {l : List ℕ} (hnd : l.Nodup) (hr : ∀ d ∈ l, d < r) :
    (batchDims r l).length + l.length = r := by
  have hperm : ((List.range r).filter (· ∈ l)).Perm l :=
    (List.perm_ext_iff_of_nodup (List.nodup_range.filter _) hnd).2 fun d => by
      simp only [List.mem_filter, List.mem_range, decide_eq_true_eq]
      exact ⟨fun h => h.2, fun h => ⟨hr d h, h⟩⟩
  have hsplit := List.length_eq_length_filter_add (l := List.range r) (fun d => decide (d ∈ l))
  have hneg : (List.range r).filter (fun d => !decide (d ∈ l)) = batchDims r l :=
    List.filter_congr fun d _ => by simp
  rw [hneg, hperm.length_eq, List.length_range] at hsplit
  omega

theorem windowDims_eq (n : ℕ) (dn : GatherDimensionNumbers) :
    windowDims n dn = batchDims n (dn.collapsedSliceDims ++ dn.operandBatchingDims) :=
  List.filter_congr fun d _ => by simp

theorem nodup_batchDims (r : ℕ) (l : List ℕ) : (batchDims r l).Nodup := List.nodup_range.filter _

theorem nodup_windowDims (n : ℕ) (dn : GatherDimensionNumbers) : (windowDims n dn).Nodup :=
  List.nodup_range.filter _

theorem mem_batchDims {r : ℕ} {l : List ℕ} {d : ℕ} : d ∈ batchDims r l ↔ d < r ∧ d ∉ l := by
  simp [batchDims]

theorem mem_windowDims {dn : GatherDimensionNumbers} {d : ℕ} :
    d ∈ windowDims n dn ↔ d < n ∧ d ∉ dn.collapsedSliceDims ∧ d ∉ dn.operandBatchingDims := by
  simp [windowDims]

theorem getD_eq_getElem {l : List ℕ} {q : ℕ} (h : q < l.length) : l.getD q 0 = l[q] := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

theorem getD_idxOf {l : List ℕ} {d : ℕ} (h : d ∈ l) : l.getD (l.idxOf d) 0 = d := by
  rw [getD_eq_getElem (List.idxOf_lt_length_of_mem h), List.getElem_idxOf]

theorem idxOf_getD {l : List ℕ} (hnd : l.Nodup) {q : ℕ} (h : q < l.length) :
    l.idxOf (l.getD q 0) = q := by
  rw [getD_eq_getElem h, hnd.idxOf_getElem]

theorem getD_mem {l : List ℕ} {q : ℕ} (h : q < l.length) : l.getD q 0 ∈ l := by
  rw [getD_eq_getElem h]
  exact List.getElem_mem h

theorem Shape.sizeD_of_lt (s : Shape n) {d : ℕ} (h : d < n) : s.sizeD d = s ⟨d, h⟩ := by
  simp [Shape.sizeD, h]

theorem Index.val_of_lt {s : Shape n} (i : Index s) {d : ℕ} (h : d < n) : i.val d = i ⟨d, h⟩ := by
  simp [Index.val, h]

/-- The window axes of a valid gather correspond to its offset axes. -/
theorem GatherValid.length_windowDims {so : Shape n} {si : Shape ni} {dn : GatherDimensionNumbers}
    {sliceSizes : List ℕ} (h : GatherValid so si dn sliceSizes) :
    (windowDims n dn).length = dn.offsetDims.length := by
  obtain ⟨-, -, -, -, -, -, hc, hcn, hb, hbn, hcb, -, -, -, -, -, hn, -⟩ := h
  have hnd : (dn.collapsedSliceDims ++ dn.operandBatchingDims).Nodup :=
    List.nodup_append.2 ⟨hc.imp ne_of_lt, hb.imp ne_of_lt, fun a ha b hb' hab => hcb a ha (hab ▸ hb')⟩
  have := length_batchDims hnd fun d hd => (List.mem_append.1 hd).elim (hcn d) (hbn d)
  rw [windowDims_eq]
  rw [List.length_append] at this
  omega

/-- The windows of the scatter that transposes a gather have the slice sizes of the gather. -/
theorem scatterSliceSizes_gatherShape {so : Shape n} {si : Shape ni} {dn : GatherDimensionNumbers}
    {sliceSizes : List ℕ} (h : GatherValid so si dn sliceSizes)
    (hl : GatherLowers so si dn sliceSizes) (hne : ∀ d, gatherShape si dn sliceSizes d ≠ 0) :
    scatterSliceSizes n (gatherShape si dn sliceSizes) dn.toScatter = sliceSizes := by
  have hW := h.length_windowDims
  obtain ⟨-, ho, hor, -, -, -, -, -, -, -, -, -, -, -, -, hlen, -, -, hc1, -⟩ := h
  refine List.ext_getElem (by simp [scatterSliceSizes, hlen]) fun d hd _ => ?_
  have hdn : d < n := by simpa [scatterSliceSizes] using hd
  simp only [scatterSliceSizes, List.getElem_map, List.getElem_range]
  show (if d ∈ windowDims n dn then
    (gatherShape si dn sliceSizes).sizeD (dn.offsetDims.getD ((windowDims n dn).idxOf d) 0) else 1) = _
  rw [← getD_eq_getElem (l := sliceSizes) (by omega)]
  split_ifs with hw
  · have hq : (windowDims n dn).idxOf d < dn.offsetDims.length :=
      hW ▸ List.idxOf_lt_length_of_mem hw
    have hmem := getD_mem hq
    rw [Shape.sizeD_of_lt _ (hor _ hmem)]
    simp only [gatherShape, hmem, ↓reduceIte, idxOf_getD (ho.imp ne_of_lt) hq, hlen, getD_idxOf hw]
  · have hd' : d ∈ dn.collapsedSliceDims ∨ d ∈ dn.operandBatchingDims := by
      by_contra hc
      exact hw (mem_windowDims.2 ⟨hdn, fun h1 => hc (.inl h1), fun h2 => hc (.inr h2)⟩)
    rcases hd' with hd' | hd'
    · exact (hc1 d hd').symm
    · exact ((hl hne).2 d hd').symm

/-! ## Clamped windows -/

/-- A clamped window lies inside the operand if no slice is larger than the operand,
the batching index is a position of the operand, and the offset is a position of the slice. -/
theorem gatherIndex_isSome_of {so : Shape n} {si : Shape ni} {dn : GatherDimensionNumbers}
    {sliceSizes : List ℕ} (indices : Tensor (.int wi) (Index si)) (j : ℕ → ℕ)
    (hss : ∀ d : Fin n, sliceSizes.getD d 0 ≤ so d)
    (hsb : ∀ d ∈ dn.startIndexMap, d ∉ dn.operandBatchingDims)
    (hb : ∀ d : Fin n, (d : ℕ) ∈ dn.operandBatchingDims →
      j ((batchDims (dn.offsetDims.length + (ni - 1)) dn.offsetDims).getD
        (dn.startIndicesBatchingDims.getD (dn.operandBatchingDims.idxOf (d : ℕ)) 0) 0) < so d)
    (hc : ∀ d : Fin n, (d : ℕ) ∉ dn.operandBatchingDims → (d : ℕ) ∉ windowDims n dn →
      1 ≤ sliceSizes.getD d 0)
    (hw : ∀ d ∈ windowDims n dn,
      j (dn.offsetDims.getD ((windowDims n dn).idxOf d) 0) < sliceSizes.getD d 0) :
    (gatherIndex so si dn sliceSizes true indices j).isSome := by
  have hclamp : ∀ (v : ℤ) (a b : ℕ), b ≤ a →
      0 ≤ (v ⊔ 0) ⊓ ((a : ℤ) - b) ∧ (v ⊔ 0) ⊓ ((a : ℤ) - b) + b ≤ a := fun v a b hab =>
    ⟨le_inf le_sup_right (by omega), by have := inf_le_right (a := v ⊔ 0) (b := (a : ℤ) - b); omega⟩
  simp only [gatherIndex]
  rw [dite_eq_left_of_eq_true]
  · rfl
  refine eq_true fun d => ?_
  have h1 := hss d
  simp only [Shape.sizeD_of_lt so d.isLt, Fin.eta, ↓reduceIte]
  by_cases hs : (d : ℕ) ∈ dn.startIndexMap
  · have hbd := hsb _ hs
    simp only [hs, hbd, ↓reduceIte]
    generalize readD indices _ = v
    obtain ⟨c1, c2⟩ := hclamp v _ _ h1
    by_cases hwd : (d : ℕ) ∈ windowDims n dn
    · have := hw _ hwd
      simp only [hwd, ↓reduceIte]
      omega
    · have := hc d hbd hwd
      simp only [hwd, ↓reduceIte]
      omega
  · simp only [hs, ↓reduceIte]
    by_cases hbd : (d : ℕ) ∈ dn.operandBatchingDims
    · have hwd : (d : ℕ) ∉ windowDims n dn := fun hwd => (mem_windowDims.1 hwd).2.2 hbd
      have := hb d hbd
      simp only [hbd, hwd, ↓reduceIte]
      omega
    · by_cases hwd : (d : ℕ) ∈ windowDims n dn
      · have := hw _ hwd
        simp only [hbd, hwd, ↓reduceIte]
        omega
      · have := hc d hbd hwd
        simp only [hbd, hwd, ↓reduceIte]
        omega

/-- In the modes `CLIP` and `PROMISE_IN_BOUNDS`, every position of the result of a `gather` reads the operand:
the clamped windows lie inside the operand, and `gather` does not use its fill value. -/
theorem gatherIndex_isSome {so : Shape n} {si : Shape ni} {dn : GatherDimensionNumbers}
    {sliceSizes : List ℕ} (h : GatherValid so si dn sliceSizes)
    (hl : GatherLowers so si dn sliceSizes) (indices : Tensor (.int wi) (Index si))
    (j : Index (gatherShape si dn sliceSizes)) :
    (gatherIndex so si dn sliceSizes true indices j.val).isSome := by
  have hW := h.length_windowDims
  obtain ⟨hss, hb1⟩ := hl fun d => (j d).pos.ne'
  obtain ⟨-, ho, hor, -, -, -, -, -, -, -, -, hsb, hsnd, hsr, hmap, hlen, -, -, hc1, -⟩ := h
  have hcount := length_batchDims (ho.imp ne_of_lt) hor
  refine gatherIndex_isSome_of indices j.val hss hsb (fun d hd => ?_) (fun d hbd hwd => ?_)
    fun d hd => ?_
  · -- A batching axis: the batch index is a position of the indices, which have the size of the operand.
    have hq := List.idxOf_lt_length_of_mem hd
    have hlenmap := congrArg List.length hmap
    simp only [List.length_map] at hlenmap
    have ha := hsr _ (getD_mem (l := dn.startIndicesBatchingDims) (q := dn.operandBatchingDims.idxOf (d : ℕ))
      (by omega))
    have hmem := getD_mem (l := batchDims (dn.offsetDims.length + (ni - 1)) dn.offsetDims)
      (q := dn.startIndicesBatchingDims.getD (dn.operandBatchingDims.idxOf (d : ℕ)) 0) (by omega)
    have hr := (mem_batchDims.1 hmem).1
    have hsize : so.sizeD d =
        si.sizeD (dn.startIndicesBatchingDims.getD (dn.operandBatchingDims.idxOf (d : ℕ)) 0) := by
      have := congrArg (fun l => l.getD (dn.operandBatchingDims.idxOf (d : ℕ)) 0) hmap
      simpa [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hq,
        List.getElem?_eq_getElem (show dn.operandBatchingDims.idxOf (d : ℕ) <
          dn.startIndicesBatchingDims.length by omega)] using this
    rw [Index.val_of_lt _ hr]
    refine lt_of_lt_of_eq (j ⟨_, hr⟩).isLt ?_
    rw [Shape.sizeD_of_lt so d.isLt] at hsize
    have hidx := idxOf_getD (nodup_batchDims (dn.offsetDims.length + (ni - 1)) dn.offsetDims)
      (q := dn.startIndicesBatchingDims.getD (dn.operandBatchingDims.idxOf (d : ℕ)) 0) (by omega)
    simp only [gatherShape, (mem_batchDims.1 hmem).2, ↓reduceIte, hidx]
    exact hsize.symm
  · -- A collapsed axis has slice size 1.
    have : (d : ℕ) ∈ dn.collapsedSliceDims := by
      by_contra hcd
      exact hwd (mem_windowDims.2 ⟨d.isLt, hcd, hbd⟩)
    exact (hc1 _ this).ge
  · -- A window axis: the offset is a position of the result, which has the slice size.
    have hq : (windowDims n dn).idxOf d < dn.offsetDims.length := hW ▸ List.idxOf_lt_length_of_mem hd
    have hmem := getD_mem hq
    rw [Index.val_of_lt _ (hor _ hmem)]
    refine lt_of_lt_of_eq (j ⟨_, hor _ hmem⟩).isLt ?_
    simp only [gatherShape, hmem, ↓reduceIte, idxOf_getD (ho.imp ne_of_lt) hq, hlen, getD_idxOf hd]

/-! ## JVP rule -/

/-- The JVP rule of `gather`, which gathers the tangent with the fill value 0.
`jax/_src/lax/slicing.py:2199`:
```
def _gather_jvp_rule(g, operand, indices, *, dimension_numbers,
                     slice_sizes, unique_indices, indices_are_sorted, mode,
                     fill_value):
  return gather(g, indices, dimension_numbers, slice_sizes,
                unique_indices=unique_indices,
                indices_are_sorted=indices_are_sorted, mode=mode,
                fill_value=0)
``` -/
def gatherJvp {so : Shape n} {si : Shape ni} (dimensionNumbers : GatherDimensionNumbers)
    (sliceSizes : List ℕ) (mode : GatherScatterMode)
    (h : GatherValid so si dimensionNumbers sliceSizes)
    (hl : GatherLowers so si dimensionNumbers sliceSizes)
    (g _operand : Tensor (.float w) (Index so)) (indices : Tensor (.int wi) (Index si)) :
    Tensor (.float w) (Index (gatherShape si dimensionNumbers sliceSizes)) :=
  gather dimensionNumbers sliceSizes mode 0 h hl g indices

/-- The JVP rule of `gather` computes the derivative of `gather` in the operand,
if the fill value is finite or every window lies inside the operand. -/
theorem gatherJvp_correct {so : Shape n} {si : Shape ni} (dn : GatherDimensionNumbers)
    (sliceSizes : List ℕ) (mode : GatherScatterMode) (fillValue : Scalar w)
    (h : GatherValid so si dn sliceSizes) (hl : GatherLowers so si dn sliceSizes)
    (indices : Tensor (.int wi) (Index si))
    (hfill : ∀ j : Index (gatherShape si dn sliceSizes),
      gatherIndex so si dn sliceSizes (mode != .fillOrDrop) indices j.val = none →
        ∃ r : ℝ, fillValue = .fin r) :
    IsJvpOf (fun g x => gatherJvp dn sliceSizes mode h hl g x indices)
      (fun x => gather (w := w) dn sliceSizes mode fillValue h hl x indices) := by
  refine isJvpOfOn_of_apply fun x _ g j => ?_
  cases hp : gatherIndex so si dn sliceSizes (mode != .fillOrDrop) indices j.val with
  | some i =>
    exact ⟨fun y => y i, ContinuousLinearMap.proj i, .of_forall fun y => by simp [gather, hp, ofReal],
      hasFDerivAt_apply i x, by simp [gatherJvp, gather, hp, ofReal]⟩
  | none =>
    obtain ⟨r, rfl⟩ := hfill j hp
    exact ⟨fun _ => r, 0, .of_forall fun y => by simp [gather, hp], hasFDerivAt_const r x,
      by simp [gatherJvp, gather, hp, Scalar.zero_def]⟩

/-- No JVP rule is correct for a `gather` that fills a window with a value that is not finite,
such as the default fill value NaN: the result is not a real array. -/
theorem gather_not_isJvpOf {so : Shape n} {si : Shape ni} {dn : GatherDimensionNumbers}
    {sliceSizes : List ℕ} {mode : GatherScatterMode} {fillValue : Scalar w}
    (h : GatherValid so si dn sliceSizes) (hl : GatherLowers so si dn sliceSizes)
    (indices : Tensor (.int wi) (Index si)) (j : Index (gatherShape si dn sliceSizes))
    (hj : gatherIndex so si dn sliceSizes (mode != .fillOrDrop) indices j.val = none)
    (hfill : ∀ r : ℝ, fillValue ≠ .fin r) {n' : ℕ} {t' : Shape n'}
    (jvp : Tensor (.float w) (Index so) → Tensor (.float w) (Index so) → Tensor (.float w) (Index t')) :
    ¬ IsJvpOf jvp (fun x => gather dn sliceSizes mode fillValue h hl x indices) := by
  intro hjvp
  obtain ⟨F, -, hF, -⟩ := hjvp 0 (Set.mem_univ _)
  have := congrFun hF.self_of_nhds j
  simp only [gather, hj, ofReal] at this
  exact hfill _ this

/-! ## Transpose rule -/

/-- The transpose rule of `gather`, which adds the cotangent to zeros with `scatter_add`.
Only the branch for a `ValAccum` is modelled, not the in-place updates of a `RefAccum`.
The proof `hs` is the shape check of `scatter_add`,
which does not follow from the shape check of `gather` (`gatherTranspose_not_defined`).
`jax/_src/lax/slicing.py:2262`:
```
    zeros = lax.full(op_shape, 0, typeof(out_ct).dtype,
                     sharding=operand_aval.sharding)
    zeros = core.pvary(zeros, tuple(operand_aval.mat.varying))
    operand.accum(scatter_add(zeros, indices, out_ct, scatter_dnums,
                              unique_indices=unique_indices,
                              indices_are_sorted=indices_are_sorted,
                              mode=mode))
``` -/
def gatherTranspose (so : Shape n) {si : Shape ni} (dimensionNumbers : GatherDimensionNumbers)
    (sliceSizes : List ℕ) (mode : GatherScatterMode)
    (hs : ScatterValid so si (gatherShape si dimensionNumbers sliceSizes) dimensionNumbers.toScatter)
    (outCt : Tensor (.float w) (Index (gatherShape si dimensionNumbers sliceSizes))) (indices : Tensor (.int wi) (Index si)) :
    Tensor (.float w) (Index so) :=
  scatterAdd dimensionNumbers.toScatter mode hs 0 indices outCt

/-- The transpose rule of `gather` raises for some gathers:
the shape rule of `gather` accepts slices that are larger than the operand if the indices are empty,
and the shape rule of `scatter_add` rejects the cotangent of such a gather. -/
theorem gatherTranspose_not_defined :
    ¬ ∀ (so : Shape 1) (si : Shape 2) (dn : GatherDimensionNumbers) (sliceSizes : List ℕ),
      GatherValid so si dn sliceSizes → GatherLowers so si dn sliceSizes →
        ScatterValid so si (gatherShape si dn sliceSizes) dn.toScatter :=
  fun h => absurd (h ![2] ![0, 1] ⟨[1], [], [0], [], []⟩ [3] (by decide) (by decide)) (by decide)

/-- The transpose rule of `gather` transposes `gather` with the fill value 0,
in the modes `CLIP` and `FILL_OR_DROP`. -/
theorem gatherTranspose_correct {so : Shape n} {si : Shape ni} (dn : GatherDimensionNumbers)
    (sliceSizes : List ℕ) (mode : GatherScatterMode) (h : GatherValid so si dn sliceSizes)
    (hl : GatherLowers so si dn sliceSizes)
    (hs : ScatterValid so si (gatherShape si dn sliceSizes) dn.toScatter)
    (indices : Tensor (.int wi) (Index si)) (hm : mode ≠ .promiseInBounds) :
    IsTransposeOf (fun ct => gatherTranspose so dn sliceSizes mode hs ct indices)
      (fun x => gather (w := w) dn sliceSizes mode 0 h hl x indices) := by
  intro ct x
  let P : Index (gatherShape si dn sliceSizes) → Option (Index so) :=
    fun j => gatherIndex so si dn sliceSizes (mode != .fillOrDrop) indices j.val
  refine ⟨fun j => (P j).elim 0 x, fun k => ∑ j with P j = some k, ct j, funext fun j => ?_,
    .of_eq (funext fun k => ?_), ?_⟩
  · show (match P j with | some i => ofReal x i | none => 0) = Scalar.fin ((P j).elim 0 x)
    cases P j <;> rfl
  · rcases isEmpty_or_nonempty (Index (gatherShape si dn sliceSizes)) with hE | ⟨⟨j0⟩⟩
    · simp [gatherTranspose, scatterAdd, ofReal, Finset.univ_eq_empty, Scalar.zero_def]
    · have hne : ∀ d, gatherShape si dn sliceSizes d ≠ 0 := fun d => (j0 d).pos.ne'
      have hflag : (mode == GatherScatterMode.clip) = (mode != .fillOrDrop) := by
        cases mode <;> first | rfl | exact absurd rfl hm
      simp only [gatherTranspose, scatterAdd, scatterSliceSizes_gatherShape h hl hne, hflag, ofReal,
        Pi.zero_apply, zero_add, Scalar.fin_sum]
      rfl
  · have hrow : ∀ j, ct j * (P j).elim 0 x = ∑ k, if P j = some k then ct j * x k else 0 := fun j => by
      cases hp : P j with
      | none => simp
      | some i => simp [Finset.sum_ite_eq]
    simp only [hrow, Finset.sum_mul, Finset.sum_filter, ite_mul, zero_mul]
    exact Finset.sum_comm

/-- The transpose rule of `gather` is not correct in the mode `PROMISE_IN_BOUNDS`
if a window does not lie inside the operand: `gather` clamps the window, and `scatter_add` drops it.
Here the operand has shape `(2,)`, and the index 5 reads its last entry.
The docstring of `GatherScatterMode` says that gradients are not correct in this case. -/
theorem gatherTranspose_promiseInBounds_not_correct :
    ¬ IsTransposeOf
      (fun ct => gatherTranspose ![2] (si := ![1, 1]) ⟨[], [0], [0], [], []⟩ [1] .promiseInBounds
        (by decide) ct (wi := wi) fun _ => 5)
      (fun x => gather (w := w) (wi := wi) (so := ![2]) (si := ![1, 1]) ⟨[], [0], [0], [], []⟩ [1]
        .promiseInBounds 0 (by decide) (by decide) x fun _ => 5) := by
  intro htr
  obtain ⟨y, u, hy, hu, hsum⟩ := htr (fun _ => 1) (fun _ => 1)
  have hread : ∀ j : Index (gatherShape ![1, 1] ⟨[], [0], [0], [], []⟩ [1]),
      (gatherIndex ![2] ![1, 1] ⟨[], [0], [0], [], []⟩ [1] true (fun _ => 5) j.val).isSome := by
    decide
  have hdrop : ∀ j : Index (gatherShape ![1, 1] ⟨[], [0], [0], [], []⟩ [1]),
      gatherIndex ![2] ![1, 1] ⟨[], [0], [0], [], []⟩ [1] false (fun _ => 5) j.val = none := by
    decide
  have hy1 : ∀ j, y j = 1 := fun j => by
    have := congrFun hy j
    simp only [gather, ofReal] at this
    obtain ⟨i, hi⟩ := Option.isSome_iff_exists.1 (hread j)
    rw [show (GatherScatterMode.promiseInBounds != .fillOrDrop) = true from rfl, hi] at this
    exact (Scalar.fin.inj this).symm
  have hu0 : u = 0 := by
    refine ofReal_injective (hu.unique (.of_eq (funext fun k => ?_)))
    simp only [gatherTranspose, scatterAdd]
    rw [Finset.sum_eq_zero fun j hj => absurd (Finset.mem_filter.1 hj).2 ?_]
    · exact add_zero _
    · rw [show (GatherScatterMode.promiseInBounds == .clip) = false from rfl]
      exact fun hc => absurd ((hdrop j).symm.trans hc) (by simp)
  simp [hy1, hu0] at hsum
  exact absurd hsum (by decide)

end Jax.Lax
