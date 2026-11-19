module

public import Jax.Lax.Gather

/-!
# `scatter_add`

Differentiation rules of the `scatter_add` primitive of `jax/_src/lax/slicing.py`.

`scatter_add` is linear in the pair of `operand` and `updates`.
Its transpose rule reads the cotangent of `updates` with `gather`.
It is correct in the modes `CLIP` and `FILL_OR_DROP`, where that `gather` passes its shape rule
(`scatterAddTranspose_correct`).

The `gather` has slice size 1 on the inserted and batching axes, so its shape rule fails if the operand
has size 0 on such an axis and the indices are not empty.
Then `scatter_add` drops every update, and the rule raises (`scatterAddTranspose_not_correct`).
See `bugs/scatter-add-transpose-empty-operand.json`.

In the mode `PROMISE_IN_BOUNDS`, `scatter_add` drops a window that does not lie inside the operand
and the `gather` of the rule clamps it (`scatterAddTranspose_promiseInBounds_not_correct`).
The docstring of `GatherScatterMode` says that gradients are not correct in this case.

In the mode `CLIP`, `scatter_add` drops no update unless an inserted axis of the operand is empty
(`gatherIndex_scatter_isSome`).

The batching rule `_scatter_batching_rule` is not modelled yet,
and neither are `scatter`, `scatter_mul`, `scatter_min` and `scatter_max`.
`scatter_sub` is in `Jax/Lax/ScatterSub.lean`.
-/

@[expose] public section

namespace Jax.Lax

variable {n ni nu : ℕ} {so : Shape n} {si : Shape ni} {su : Shape nu} {w : FloatWidth} {wi : IntWidth}

/-! ## Rules -/

/-- The JVP rule of `scatter_add`, which applies the primitive to the tangents.
The tangents come first, then the primals.
If both tangents are symbolic zeros, the result is the symbolic zero of the output.
`jax/_src/lax/slicing.py:2847`:
```
def _scatter_addsub_jvp(
    prim, primals, tangents, *, update_jaxpr, update_consts, dimension_numbers,
    indices_are_sorted, unique_indices, mode):
  operand, indices, updates = primals
  g_operand, g_indices, g_updates = tangents
  del g_indices  # ignored
  val_out = prim.bind(...)
  if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
    tangent_out = ad_util.p2tz(val_out)
  else:
    g_operand = ad.instantiate_zeros(g_operand)
    g_updates = ad.instantiate_zeros(g_updates)
    tangent_out = prim.bind(
        g_operand, indices, g_updates, update_jaxpr=update_jaxpr,
        update_consts=update_consts, dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
        mode=mode)
  return val_out, tangent_out
``` -/
def scatterAddJvp (dimensionNumbers : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (h : ScatterValid so si su dimensionNumbers)
    (gOperand : Tan (.float w) (Index so)) (gUpdates : Tan (.float w) (Index su))
    (_operand : Tensor (.float w) (Index so)) (indices : Tensor (.int wi) (Index si)) (_updates : Tensor (.float w) (Index su)) :
    Tan (.float w) (Index so) :=
  match gOperand, gUpdates with
  | .zero _, .zero _ => .zero (.of (.float w) so)
  | gOperand, gUpdates => .arr (scatterAdd dimensionNumbers mode h gOperand.inst indices gUpdates.inst)

/-- The transpose rule of `scatter_add`.
The cotangent of `operand` is the cotangent `t`,
and the cotangent of `updates` gathers the windows of `t` with the fill value 0.
An argument that is a linear input is `none`, as in `IsPartialTranspose₂Of`,
and the result `none` is a raised exception: the `gather` raises if its checks fail.
For a symbolic zero cotangent, the cotangent of a linear input is the symbolic zero of its aval.
`jax/_src/lax/slicing.py:2871`:
```
def _scatter_addsub_transpose_rule(
    prim, t, operand, indices, updates, *, update_jaxpr, update_consts,
    dimension_numbers, indices_are_sorted, unique_indices, mode):
  assert not ad.is_undefined_primal(indices)
  if ad.is_undefined_primal(updates):
    updates_shape = updates.aval.shape
  else:
    updates_shape = updates.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
  else:
    operand_t = update_t = None
    if ad.is_undefined_primal(operand):
      operand_t = t

    if ad.is_undefined_primal(updates):
      gather_dnums = GatherDimensionNumbers(...)
      slice_sizes = []
      pos = 0
      for i in range(len(t.shape)):
        if (
            i in dimension_numbers.inserted_window_dims
            or i in dimension_numbers.operand_batching_dims
        ):
          slice_sizes.append(1)
        else:
          slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
          pos += 1
      update_t = gather(t, indices, dimension_numbers=gather_dnums,
                        slice_sizes=slice_sizes, mode=mode, fill_value=0)
      if prim is scatter_sub_p:
        update_t = lax.neg(update_t)
  return [operand_t, None, update_t]
``` -/
def scatterAddTranspose (dimensionNumbers : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (t : Tan (.float w) (Index so)) (operand : Option (Tensor (.float w) (Index so))) (indices : Tensor (.int wi) (Index si))
    (updates : Option (Tensor (.float w) (Index su))) :
    Option (Option (Tan (.float w) (Index so)) × Option (Tan (.float w) (Index
      (gatherShape si dimensionNumbers.toGather (scatterSliceSizes n su dimensionNumbers))))) :=
  match t with
  | .zero _ => some (if operand.isNone then some (.zero (.of (.float w) so)) else none,
      if updates.isNone then some (.zero (.of (.float w) su)) else none)
  | .arr t =>
  let operandT := if operand.isNone then some (.arr t) else none
  if updates.isNone then
    if h : GatherValid so si dimensionNumbers.toGather (scatterSliceSizes n su dimensionNumbers) ∧
        GatherLowers so si dimensionNumbers.toGather (scatterSliceSizes n su dimensionNumbers) then
      some (operandT, some (.arr (gather dimensionNumbers.toGather
        (scatterSliceSizes n su dimensionNumbers) mode 0 h.1 h.2 t indices)))
    else none
  else some (operandT, none)

/-! ## Proofs -/

/-- The window axes of a valid scatter correspond to its update window axes. -/
theorem ScatterValid.length_windowDims {dn : ScatterDimensionNumbers} (h : ScatterValid so si su dn) :
    (windowDims n dn.toGather).length = dn.updateWindowDims.length := by
  obtain ⟨-, -, -, -, hc, hcn, hb, hbn, hcb, -, -, -, -, -, hn, -⟩ := h
  have hnd : (dn.insertedWindowDims ++ dn.operandBatchingDims).Nodup :=
    List.nodup_append.2 ⟨hc.imp ne_of_lt, hb.imp ne_of_lt, fun a ha b hb' hab => hcb a ha (hab ▸ hb')⟩
  have := length_batchDims hnd fun d hd => (List.mem_append.1 hd).elim (hcn d) (hbn d)
  rw [windowDims_eq]
  rw [List.length_append] at this
  show (batchDims n (dn.insertedWindowDims ++ dn.operandBatchingDims)).length = _
  omega

/-- The gather that reads the windows of a scatter returns an array of the shape of the updates. -/
theorem gatherShape_scatterSliceSizes {dn : ScatterDimensionNumbers} (h : ScatterValid so si su dn)
    (d : Fin (dn.updateWindowDims.length + (ni - 1))) :
    gatherShape si dn.toGather (scatterSliceSizes n su dn) d = su.sizeD d := by
  have hW := h.length_windowDims
  obtain ⟨-, hnu, ho, hor, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hbatch⟩ := h
  have hlen : (scatterSliceSizes n su dn).length = n := by simp [scatterSliceSizes]
  have hB : batchDims nu dn.updateWindowDims =
      batchDims (dn.updateWindowDims.length + (ni - 1)) dn.updateWindowDims := by
    rw [hnu, Nat.add_comm]
  show (if (d : ℕ) ∈ dn.updateWindowDims then
    (scatterSliceSizes n su dn).getD ((windowDims (scatterSliceSizes n su dn).length dn.toGather).getD
      (dn.updateWindowDims.idxOf (d : ℕ)) 0) 0
    else si.sizeD ((batchDims (dn.updateWindowDims.length + (ni - 1)) dn.updateWindowDims).idxOf
      (d : ℕ))) = _
  split_ifs with hd
  · have hq : dn.updateWindowDims.idxOf (d : ℕ) < (windowDims n dn.toGather).length :=
      by rw [hW]; exact List.idxOf_lt_length_of_mem hd
    have hw := getD_mem hq
    have hwn : (windowDims n dn.toGather).getD (dn.updateWindowDims.idxOf (d : ℕ)) 0 < n :=
      (mem_windowDims.1 hw).1
    rw [hlen, getD_eq_getElem (by omega)]
    simp only [scatterSliceSizes, List.getElem_map, List.getElem_range, hw, ↓reduceIte,
      idxOf_getD (nodup_windowDims n dn.toGather) hq, getD_idxOf hd]
  · have hmem : (d : ℕ) ∈ batchDims (dn.updateWindowDims.length + (ni - 1)) dn.updateWindowDims :=
      mem_batchDims.2 ⟨d.isLt, hd⟩
    have hcount := length_batchDims (r := dn.updateWindowDims.length + (ni - 1)) (ho.imp ne_of_lt)
      fun e he => by have := hor e he; omega
    have ha := List.idxOf_lt_length_of_mem hmem
    rw [← hbatch _ (by omega), hB, getD_idxOf hmem]

theorem scatterSliceSizes_getD (dn : ScatterDimensionNumbers) (d : Fin n) :
    (scatterSliceSizes n su dn).getD d 0 = if (d : ℕ) ∈ windowDims n dn.toGather then
      su.sizeD (dn.updateWindowDims.getD ((windowDims n dn.toGather).idxOf (d : ℕ)) 0) else 1 := by
  rw [getD_eq_getElem (by simp [scatterSliceSizes])]
  simp [scatterSliceSizes]

/-- In the mode `CLIP`, `scatter_add` drops no update if no inserted axis of the operand is empty:
the clamped windows lie inside the operand.
An update is dropped if an inserted axis is empty, as in `bugs/scatter-add-transpose-empty-operand.json`. -/
theorem gatherIndex_scatter_isSome {dn : ScatterDimensionNumbers} (h : ScatterValid so si su dn)
    (hins : ∀ d ∈ dn.insertedWindowDims, so.sizeD d ≠ 0) (indices : Tensor (.int wi) (Index si)) (u : Index su) :
    (gatherIndex so si dn.toGather (scatterSliceSizes n su dn) true indices u.val).isSome := by
  have hW := h.length_windowDims
  obtain ⟨-, hnu, ho, hor, -, -, -, -, -, hsb, -, hsr, -, hmap, -, -, -, -, hwin, -⟩ := h
  have hB : batchDims (dn.updateWindowDims.length + (ni - 1)) dn.updateWindowDims =
      batchDims nu dn.updateWindowDims := by rw [hnu, Nat.add_comm]
  have hcount := length_batchDims (ho.imp ne_of_lt) hor
  have hpos : ∀ b, b < nu → u.val b < su.sizeD b := fun b hb => by
    rw [Index.val_of_lt _ hb, Shape.sizeD_of_lt _ hb]
    exact (u ⟨b, hb⟩).isLt
  -- A batching axis: the batch index is a position of the updates, which have the size of the operand.
  have hbat : ∀ d : Fin n, (d : ℕ) ∈ dn.operandBatchingDims →
      u.val ((batchDims nu dn.updateWindowDims).getD
        (dn.scatterIndicesBatchingDims.getD (dn.operandBatchingDims.idxOf (d : ℕ)) 0) 0) < so d :=
    fun d hd => by
      have hq := List.idxOf_lt_length_of_mem hd
      have hlenmap := congrArg List.length hmap
      simp only [List.length_map] at hlenmap
      have ha := hsr _ (getD_mem (l := dn.scatterIndicesBatchingDims)
        (q := dn.operandBatchingDims.idxOf (d : ℕ)) (by omega))
      have hmem := getD_mem (l := batchDims nu dn.updateWindowDims)
        (q := dn.scatterIndicesBatchingDims.getD (dn.operandBatchingDims.idxOf (d : ℕ)) 0) (by omega)
      have hsize : so.sizeD d = su.sizeD ((batchDims nu dn.updateWindowDims).getD
          (dn.scatterIndicesBatchingDims.getD (dn.operandBatchingDims.idxOf (d : ℕ)) 0) 0) := by
        have := congrArg (fun l => l.getD (dn.operandBatchingDims.idxOf (d : ℕ)) 0) hmap
        simpa [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hq,
          List.getElem?_eq_getElem (show dn.operandBatchingDims.idxOf (d : ℕ) <
            dn.scatterIndicesBatchingDims.length by omega)] using this
      rw [Shape.sizeD_of_lt so d.isLt] at hsize
      exact lt_of_lt_of_eq (hpos _ (mem_batchDims.1 hmem).1) hsize.symm
  refine gatherIndex_isSome_of (dn := dn.toGather) indices u.val (fun d => ?_) hsb
    (fun d hd => by rw [show dn.toGather.offsetDims = dn.updateWindowDims from rfl, hB]; exact hbat d hd)
    (fun d _ hwd => by simp only [scatterSliceSizes_getD, hwd, ↓reduceIte, le_refl]) fun d hd => ?_
  · rw [scatterSliceSizes_getD]
    split_ifs with hwd
    · have hq : (windowDims n dn.toGather).idxOf (d : ℕ) < dn.updateWindowDims.length :=
        hW ▸ List.idxOf_lt_length_of_mem hwd
      have := hwin _ hq
      rwa [getD_idxOf hwd, Shape.sizeD_of_lt so d.isLt] at this
    · by_cases hbd : (d : ℕ) ∈ dn.operandBatchingDims
      · have := hbat d hbd
        omega
      · have hid : (d : ℕ) ∈ dn.insertedWindowDims := by
          by_contra hid
          exact hwd (mem_windowDims.2 ⟨d.isLt, hid, hbd⟩)
        have := hins _ hid
        rw [Shape.sizeD_of_lt so d.isLt] at this
        exact Nat.one_le_iff_ne_zero.2 this
  · have hdn : d < n := (mem_windowDims.1 hd).1
    have hq : (windowDims n dn.toGather).idxOf d < dn.updateWindowDims.length :=
      hW ▸ List.idxOf_lt_length_of_mem hd
    have := scatterSliceSizes_getD (su := su) dn ⟨d, hdn⟩
    simp only [hd, ↓reduceIte] at this
    rw [this]
    exact hpos _ (hor _ (getD_mem hq))

/-- The positions of a gather only depend on the values of the index of the result. -/
theorem Index.val_eq_of_same {m m' : ℕ} {s : Shape m} {t : Shape m'} (hm : m = m') {i : Index s}
    {j : Index t} (hij : ∀ d, (i d : ℕ) = j (Fin.cast hm d)) : i.val = j.val := by
  subst hm
  funext d
  by_cases hd : d < m
  · simpa [Index.val, hd] using hij ⟨d, hd⟩
  · simp [Index.val, hd]

/-- The real function that `scatter_add` computes. -/
noncomputable def scatterAddReal (dn : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (indices : Tensor (.int wi) (Index si)) : (Index so → ℝ) × (Index su → ℝ) →ₗ[ℝ] Index so → ℝ where
  toFun p k := p.1 k + ∑ u with gatherIndex so si dn.toGather (scatterSliceSizes n su dn)
    (mode == .clip) indices u.val = some k, p.2 u
  map_add' p q := by
    funext k
    simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, Finset.sum_add_distrib]
    ring
  map_smul' c p := by
    funext k
    simp only [Prod.smul_fst, Prod.smul_snd, Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
    rw [mul_add, Finset.mul_sum]

theorem scatterAdd_ofReal (dn : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (h : ScatterValid so si su dn) (indices : Tensor (.int wi) (Index si)) (x : Index so → ℝ) (y : Index su → ℝ) :
    scatterAdd (w := w) dn mode h (ofReal x) indices (ofReal y) =
      ofReal (scatterAddReal dn mode indices (x, y)) := by
  funext k
  simp only [scatterAdd, ofReal, ← Scalar.fin_sum]
  rfl

/-! ## Correctness -/

/-- The JVP rule of `scatter_add` computes the derivative of `scatter_add`. -/
theorem scatterAddJvp_correct (dn : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (h : ScatterValid so si su dn) (indices : Tensor (.int wi) (Index si)) :
    IsJointJvp₂Of (fun g k x y => scatterAddJvp dn mode h g k x indices y)
      (fun x y => scatterAdd (w := w) dn mode h x indices y) := by
  intro x y _
  let L := LinearMap.toContinuousLinearMap (scatterAddReal (so := so) (su := su) dn mode indices)
  refine ⟨scatterAddReal dn mode indices, L,
    .of_forall fun p => scatterAdd_ofReal dn mode h indices p.1 p.2, L.hasFDerivAt,
    fun zg zh g k hz => ?_⟩
  cases zg <;> cases zh
  · exact .of_eq (scatterAdd_ofReal dn mode h indices g k)
  · exact .of_eq (scatterAdd_ofReal dn mode h indices g 0)
  · exact .of_eq (scatterAdd_ofReal dn mode h indices 0 k)
  · simp at hz

/-- In the modes `CLIP` and `FILL_OR_DROP`, `scatter_add` is transposed by the identity for `operand`
and by the `gather` of its windows with the fill value 0 for `updates`. -/
theorem scatterAdd_isTranspose₂Of (dn : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (h : ScatterValid so si su dn)
    (hg : GatherValid so si dn.toGather (scatterSliceSizes n su dn))
    (hgl : GatherLowers so si dn.toGather (scatterSliceSizes n su dn))
    (indices : Tensor (.int wi) (Index si)) (hm : mode ≠ .promiseInBounds) :
    IsTranspose₂Of id
      (fun t => gather (w := w) dn.toGather (scatterSliceSizes n su dn) mode 0 hg hgl t indices)
      (fun x y => scatterAdd (w := w) dn mode h x indices y) := by
  intro ct x y
  · let Q : Index su → Option (Index so) := fun u =>
      gatherIndex so si dn.toGather (scatterSliceSizes n su dn) (mode == .clip) indices u.val
    have hflag : (mode != GatherScatterMode.fillOrDrop) = (mode == .clip) := by
      cases mode <;> first | rfl | exact absurd rfl hm
    have hrank : dn.updateWindowDims.length + (ni - 1) = nu := by have := h.2.1; omega
    refine ⟨scatterAddReal dn mode indices (x, y), ct, fun u => (Q u).elim 0 ct,
      scatterAdd_ofReal dn mode h indices x y, .of_eq rfl, ⟨rfl, hrank, fun d => ?_, fun i j hij => ?_⟩, ?_⟩
    · rw [gatherShape_scatterSliceSizes h d]
      exact Shape.sizeD_of_lt su (by omega)
    · show (match gatherIndex so si dn.toGather (scatterSliceSizes n su dn)
        (mode != .fillOrDrop) indices i.val with | some k => ofReal ct k | none => 0) =
          Scalar.fin ((Q j).elim 0 ct)
      have hval : i.val = j.val := Index.val_eq_of_same hrank hij
      rw [hflag, hval]
      show (match Q j with | some k => ofReal ct k | none => 0) = _
      cases Q j <;> rfl
    · have hrow : ∀ u, (Q u).elim 0 ct * y u = ∑ k, if Q u = some k then ct k * y u else 0 :=
        fun u => by
          cases hp : Q u with
          | none => simp
          | some i => simp [Finset.sum_ite_eq]
      show ∑ k, ct k * (x k + ∑ u with Q u = some k, y u) = _
      simp only [hrow, mul_add, Finset.sum_add_distrib, Finset.mul_sum, Finset.sum_filter, mul_ite,
        mul_zero]
      rw [Finset.sum_comm (f := fun u k => if Q u = some k then ct k * y u else 0)]

/-- The transpose rule of `scatter_add` transposes `scatter_add` in the modes `CLIP` and `FILL_OR_DROP`,
where the `gather` of the rule passes its checks, whichever arguments are linear inputs. -/
theorem scatterAddTranspose_correct (dn : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (h : ScatterValid so si su dn)
    (hg : GatherValid so si dn.toGather (scatterSliceSizes n su dn))
    (hgl : GatherLowers so si dn.toGather (scatterSliceSizes n su dn))
    (indices : Tensor (.int wi) (Index si)) (hm : mode ≠ .promiseInBounds) :
    IsPartialTranspose₂Of (fun t x y => scatterAddTranspose dn mode t x indices y)
      (fun x y => scatterAdd (w := w) dn mode h x indices y) := by
  refine ⟨id, fun t => gather dn.toGather (scatterSliceSizes n su dn) mode 0 hg hgl t indices,
    scatterAdd_isTranspose₂Of dn mode h hg hgl indices hm, fun x y hxy => ⟨fun ct => ?_, ?_⟩⟩
  · refine ⟨if x.isNone then some (.arr (ofReal ct)) else none,
      if y.isNone then some (.arr (gather dn.toGather (scatterSliceSizes n su dn) mode 0 hg hgl
        (ofReal ct) indices)) else none, ?_, fun hx => by simp [hx], fun hy => by simp [hy]⟩
    cases y <;> simp [scatterAddTranspose, hg, hgl]
  · exact ⟨_, _, rfl, fun hx => by simp [hx, Tan.Same], fun hy => by simp [hy, Tan.Same]⟩

/-- The transpose rule of `scatter_add` raises for an operand of size 0 on an inserted axis:
here the operand has shape `(0,)`, as in `jnp.zeros(0).at[jnp.array([0])].add(updates)`.
`scatter_add` accepts it and drops the update, but the `gather` of the rule has slice size 1
on the axis of size 0, which its shape rule rejects. -/
theorem scatterAddTranspose_not_correct (mode : GatherScatterMode)
    (indices : Tensor (.int wi) (Index ![1, 1])) :
    ¬ IsPartialTranspose₂Of
      (fun t x y => scatterAddTranspose (so := ![0]) (su := ![1]) ⟨[], [0], [0], [], []⟩ mode t x
        indices y)
      (fun x y => scatterAdd (w := w) (so := ![0]) (su := ![1]) ⟨[], [0], [0], [], []⟩ mode
        (by decide) x indices y) := by
  rintro ⟨_, _, _, hrule⟩
  obtain ⟨_, _, hsome, _⟩ := (hrule (some 0) none (.inr rfl)).1 0
  have hno : ¬ (GatherValid ![0] ![1, 1] (ScatterDimensionNumbers.toGather ⟨[], [0], [0], [], []⟩)
      (scatterSliceSizes 1 ![1] ⟨[], [0], [0], [], []⟩) ∧
      GatherLowers ![0] ![1, 1] (ScatterDimensionNumbers.toGather ⟨[], [0], [0], [], []⟩)
        (scatterSliceSizes 1 ![1] ⟨[], [0], [0], [], []⟩)) := by decide
  simp [scatterAddTranspose, hno] at hsome

/-- The transpose rule of `scatter_add` is not correct in the mode `PROMISE_IN_BOUNDS`
if a window does not lie inside the operand: `scatter_add` drops the window,
and the `gather` of the rule clamps it.
Here the operand has shape `(2,)`, and the index 5 reads the cotangent of its last entry. -/
theorem scatterAddTranspose_promiseInBounds_not_correct :
    ¬ IsPartialTranspose₂Of
      (fun t x y => scatterAddTranspose (wi := wi) (so := ![2]) (su := ![1]) ⟨[], [0], [0], [], []⟩
        .promiseInBounds t x (fun _ : Index ![1, 1] => 5) y)
      (fun x y => scatterAdd (w := w) (wi := wi) (so := ![2]) (su := ![1]) ⟨[], [0], [0], [], []⟩
        .promiseInBounds (by decide) x (fun _ : Index ![1, 1] => 5) y) := by
  rintro ⟨trX, trY, htr, hrule⟩
  have hread : ∀ j : Index (gatherShape ![1, 1]
      (ScatterDimensionNumbers.toGather ⟨[], [0], [0], [], []⟩)
      (scatterSliceSizes 1 ![1] ⟨[], [0], [0], [], []⟩)),
      (gatherIndex ![2] ![1, 1] (ScatterDimensionNumbers.toGather ⟨[], [0], [0], [], []⟩)
        (scatterSliceSizes 1 ![1] ⟨[], [0], [0], [], []⟩) true (fun _ => 5) j.val).isSome := by
    decide
  have hdrop : ∀ u : Index ![1],
      gatherIndex ![2] ![1, 1] (ScatterDimensionNumbers.toGather ⟨[], [0], [0], [], []⟩)
        (scatterSliceSizes 1 ![1] ⟨[], [0], [0], [], []⟩) false (fun _ => 5) u.val = none := by
    decide
  have hyes : GatherValid ![2] ![1, 1] (ScatterDimensionNumbers.toGather ⟨[], [0], [0], [], []⟩)
      (scatterSliceSizes 1 ![1] ⟨[], [0], [0], [], []⟩) ∧
      GatherLowers ![2] ![1, 1] (ScatterDimensionNumbers.toGather ⟨[], [0], [0], [], []⟩)
        (scatterSliceSizes 1 ![1] ⟨[], [0], [0], [], []⟩) := by decide
  obtain ⟨u, v, hsome, -, hv⟩ := (hrule (some 0) none (.inr rfl)).1 fun _ => 1
  obtain ⟨z, a, b, hz, -, hb, hsum⟩ := htr (fun _ => 1) 0 (fun _ => 1)
  -- The rule returns the `gather` of the cotangent, which reads the cotangent 1 of the last entry.
  have hgather : (trY (ofReal fun _ => 1)).Same (ofReal (w := w) fun _ : Index ![1] => (1 : ℝ)) := by
    simp only [scatterAddTranspose, Option.isNone_none, ↓reduceIte, hyes, and_self, ↓reduceDIte,
      Option.some.injEq, Prod.mk.injEq] at hsome
    have hrw := Tan.arr.inj (Option.some.inj ((hv rfl).symm.trans hsome.2.symm))
    rw [hrw]
    refine ⟨rfl, rfl, by decide, fun i j _ => ?_⟩
    obtain ⟨k, hk⟩ := Option.isSome_iff_exists.1 (hread i)
    simp only [gather]
    rw [show (GatherScatterMode.promiseInBounds != .fillOrDrop) = true from rfl, hk]
    rfl
  have hb1 : b = fun _ => 1 := ofReal_injective (hb.unique hgather)
  -- `scatter_add` drops the update.
  have hz0 : z = 0 := by
    refine ofReal_injective (hz.symm.trans (funext fun k => ?_))
    simp only [scatterAdd]
    rw [Finset.sum_eq_zero fun j hj => absurd (Finset.mem_filter.1 hj).2 ?_]
    · exact add_zero _
    · rw [show (GatherScatterMode.promiseInBounds == .clip) = false from rfl]
      exact fun hc => absurd ((hdrop j).symm.trans hc) (by simp)
  simp [hz0, hb1] at hsum

end Jax.Lax
