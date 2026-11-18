module

public import Jax.Lax.Primal

/-!
# `broadcast_in_dim`

Differentiation rules and the forwarding rule of the `broadcast_in_dim` primitive of `jax/_src/lax/lax.py`.

The transpose rule is correct for increasing `broadcast_dimensions`,
and returns an array of the wrong shape for `broadcast_dimensions = (1, 0)`.
The forwarding rule is correct for increasing `broadcast_dimensions`,
and forwards the operand instead of its transpose for `broadcast_dimensions = (1, 0)` and equal shapes.
-/

@[expose] public section

namespace Jax.Lax

variable {m n : ℕ} {so : Shape m} {w : FloatWidth} {d : DType}

/-- The JVP rule of `broadcast_in_dim`, which applies the primitive to the tangent.
`jax/_src/lax/lax.py:7609`:
```
    y_dot = broadcast_in_dim_p.bind(operand_dot, shape=shape,
                                    broadcast_dimensions=broadcast_dimensions,
                                    sharding=sharding)
``` -/
def broadcastInDimJvp (shape : Shape n) (broadcastDimensions : Fin m → Fin n)
    (h : BroadcastsTo so (shape ∘ broadcastDimensions)) (hd : Function.Injective broadcastDimensions)
    (operandDot _operand : Tensor (.float w) (Index so)) : Tensor (.float w) (Index shape) :=
  broadcastInDim shape broadcastDimensions h hd operandDot

/-- The JVP rule of `broadcast_in_dim` computes the derivative of `broadcast_in_dim`. -/
theorem broadcastInDimJvp_correct (shape : Shape n) (broadcastDimensions : Fin m → Fin n)
    (h : BroadcastsTo so (shape ∘ broadcastDimensions)) (hd : Function.Injective broadcastDimensions) :
    IsJvpOf (broadcastInDimJvp shape broadcastDimensions h hd)
      (broadcastInDim (d := .float w) shape broadcastDimensions h hd) :=
  isJvpOf_of_linear
    { toFun := fun x j => x ((j.proj broadcastDimensions).unbroadcast h)
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl }
    fun _ => rfl

/-! ## Transpose rule -/

/-- `unit_dims` of `_broadcast_in_dim_transpose_rule`: the axes of the operand of size 1.
Sharding is not modelled, so every `spec` is `None`. -/
def unitDims (so : Shape m) : Finset (Fin m) := {i | so i = 1}

/-- `axes` of `_broadcast_in_dim_transpose_rule`: the axes of the result that are not in
`bdims = tuple_delete(broadcast_dimensions, unit_dims)`.
`tuple_delete(tup, idx)`, `jax/_src/lax/lax.py:6340`, keeps the entries of `tup` at the positions not in `idx`. -/
def broadcastInDimTransposeAxes (so : Shape m) (broadcastDimensions : Fin m → Fin n) :
    Finset (Fin n) :=
  (Finset.univ.image (broadcastDimensions ∘ keptAxes (unitDims so)))ᶜ

theorem card_compl_broadcastInDimTransposeAxes {broadcastDimensions : Fin m → Fin n}
    (hd : Function.Injective broadcastDimensions) :
    (broadcastInDimTransposeAxes so broadcastDimensions)ᶜ.card = (unitDims so)ᶜ.card := by
  rw [broadcastInDimTransposeAxes, compl_compl,
    Finset.card_image_of_injective _ (hd.comp (keptAxes_injective _)), Finset.card_univ,
    Fintype.card_fin]

/-- For distinct `broadcast_dimensions`, the result of the transpose rule has the rank of the operand,
so `canonicalize_axis` accepts `unit_dims`. -/
theorem broadcastInDimTranspose_rank {broadcastDimensions : Fin m → Fin n}
    (hd : Function.Injective broadcastDimensions) :
    (broadcastInDimTransposeAxes so broadcastDimensions)ᶜ.card +
      ((unitDims so).map Fin.valEmbedding).card = m := by
  rw [card_compl_broadcastInDimTransposeAxes hd, Finset.card_map, Finset.card_compl_add_card,
    Fintype.card_fin]

theorem broadcastInDimTranspose_canonicalize {broadcastDimensions : Fin m → Fin n}
    (hd : Function.Injective broadcastDimensions) :
    ∀ i ∈ (unitDims so).map Fin.valEmbedding,
      i < (broadcastInDimTransposeAxes so broadcastDimensions)ᶜ.card +
        ((unitDims so).map Fin.valEmbedding).card := by
  intro i hi
  rw [broadcastInDimTranspose_rank hd]
  obtain ⟨d, _, rfl⟩ := Finset.mem_map.1 hi
  exact d.isLt

/-- The transpose rule of `broadcast_in_dim`.
It sums the cotangent over the axes that do not come from an operand axis of size other than 1,
and inserts the axes of size 1 of the operand.
The undefined primal `operand` is replaced by its shape `so`.
The proof `hd` is the check for duplicates in `_broadcast_in_dim_shape_rule`, `jax/_src/lax/lax.py:7485`.
`jax/_src/lax/lax.py:7543`:
```
def _broadcast_in_dim_transpose_rule(ct, operand,
                                     shape, broadcast_dimensions, sharding):
  if not isinstance(operand, ad.UndefinedPrimal):
    return [None]  # transpose wrt literal
  if type(ct) is ad_util.Zero:
    return [ad_util.Zero(operand.aval)]
  ct_s = operand.aval.sharding
  unit_dims = [
      i for i, (sh, spec) in enumerate(zip(operand.aval.shape, ct_s.spec.partitions))
      if core.definitely_equal(sh, 1) and spec is None
  ]
  bdims = tuple_delete(broadcast_dimensions, unit_dims)
  axes = tuple_delete(tuple(range(len(shape))), bdims)
  ct_s = ct_s.update(spec=ct_s.spec.update(
      partitions=tuple_delete(ct_s.spec.partitions, unit_dims)))
  return [expand_dims(reduce_sum(ct, axes, out_sharding=ct_s), unit_dims)]
``` -/
noncomputable def broadcastInDimTranspose (so : Shape m) (shape : Shape n)
    (broadcastDimensions : Fin m → Fin n) (hd : Function.Injective broadcastDimensions)
    (ct : Tensor (.float w) (Index shape)) :=
  expandDims (s := shape ∘ keptAxes (broadcastInDimTransposeAxes so broadcastDimensions))
    ((unitDims so).map Fin.valEmbedding) (broadcastInDimTranspose_canonicalize hd)
    (reduceSum (broadcastInDimTransposeAxes so broadcastDimensions) ct)

/-- For increasing `broadcast_dimensions`, `reduce_sum` keeps the axes `bdims` in their order. -/
private theorem keptAxes_broadcastInDimTransposeAxes {broadcastDimensions : Fin m → Fin n}
    (hd : StrictMono broadcastDimensions) (a) :
    keptAxes (broadcastInDimTransposeAxes so broadcastDimensions) a =
      broadcastDimensions (keptAxes (unitDims so)
        (Fin.cast (card_compl_broadcastInDimTransposeAxes hd.injective) a)) :=
  keptAxes_eq_of_strictMono _ (broadcastDimensions ∘ keptAxes (unitDims so))
    (fun i => by simp [broadcastInDimTransposeAxes]) (hd.comp (keptAxes_strictMono _)) a

/-- `expand_dims` puts the axes of the operand of size other than 1 back in their place. -/
private theorem keptAxes_dimsSet (U : Finset (Fin m)) {N : ℕ} (hN : N = m)
    (hk : (dimsSet N (U.map Fin.valEmbedding))ᶜ.card = Uᶜ.card) (b) :
    (keptAxes (dimsSet N (U.map Fin.valEmbedding)) b : ℕ) = keptAxes U (Fin.cast hk b) := by
  subst hN
  rw [keptAxes_eq_of_strictMono hk (keptAxes U) (fun i => ?_) (keptAxes_strictMono U) b]
  simp only [dimsSet, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_map,
    Fin.valEmbedding_apply, not_exists, not_and]
  exact fun x hx hxe => keptAxes_notMem U i (Fin.ext hxe ▸ hx)

private theorem unitDims_notMem {d : Fin m} : d ∉ unitDims so ↔ so d ≠ 1 := by
  simp [unitDims]

/-- The transpose rule of `broadcast_in_dim` transposes `broadcast_in_dim`
if `broadcast_dimensions` is strictly increasing. -/
theorem broadcastInDimTranspose_correct (shape : Shape n) (broadcastDimensions : Fin m → Fin n)
    (h : BroadcastsTo so (shape ∘ broadcastDimensions)) (hd : StrictMono broadcastDimensions) :
    IsTransposeOf (broadcastInDimTranspose so shape broadcastDimensions hd.injective)
      (broadcastInDim (d := .float w) shape broadcastDimensions h hd.injective) := by
  intro ct x
  have hAc := card_compl_broadcastInDimTransposeAxes (so := so) hd.injective
  have hN := broadcastInDimTranspose_rank (so := so) hd.injective
  have hk := card_compl_dimsSet (broadcastInDimTranspose_canonicalize (so := so) hd.injective)
  have hkU := hk.trans hAc
  -- An axis of the result that is not a unit dimension is an operand axis of size other than 1.
  have hcast : ∀ b, Fin.cast hN (keptAxes (dimsSet _ ((unitDims so).map Fin.valEmbedding)) b) =
      keptAxes (unitDims so) (Fin.cast hkU b) :=
    fun b => Fin.ext (keptAxes_dimsSet _ hN hkU b)
  have hshape : ∀ c, shape (broadcastDimensions (keptAxes (unitDims so) c)) =
      so (keptAxes (unitDims so) c) := fun c =>
    ((h _).resolve_left (unitDims_notMem.1 (keptAxes_notMem _ c))).symm
  refine ⟨fun j => x ((j.proj broadcastDimensions).unbroadcast h),
    fun i => ∑ j with (j.proj broadcastDimensions).unbroadcast h = i, ct j, rfl,
    ⟨rfl, hN, fun p => ?_, fun q i hq => ?_⟩, ?_⟩
  · -- The shape of the result is the shape of the operand.
    unfold expandDimsShape
    split_ifs with hp
    · obtain ⟨d, hd1, hdp⟩ : ∃ d, so d = 1 ∧ (d : ℕ) = p := by
        rw [dimsSet, Finset.mem_filter] at hp
        simpa [unitDims] using hp.2
      rw [show Fin.cast hN p = d from Fin.ext hdp.symm, hd1]
    · obtain ⟨b, rfl⟩ := keptAxes_surj hp
      rw [keptAxesIdx_keptAxes, Function.comp_apply, keptAxes_broadcastInDimTransposeAxes hd,
        hcast, hshape]
      rfl
  · -- The entries of the result sum the cotangent over the entries that are broadcast from `i`.
    unfold broadcastInDimTranspose expandDims broadcastInDim
    generalize hq' : Index.unbroadcast _ (Index.proj _ q) = q'
    unfold reduceSum ofReal
    rw [← Scalar.fin_sum]
    refine congrArg Scalar.fin ?_
    have hval : ∀ a, (q' a : ℕ) = i (keptAxes (unitDims so) (Fin.cast hAc a)) := fun a => by
      have hne : (shape ∘ keptAxes (broadcastInDimTransposeAxes so broadcastDimensions)) a ≠ 1 := by
        rw [Function.comp_apply, keptAxes_broadcastInDimTransposeAxes hd, hshape]
        exact unitDims_notMem.1 (keptAxes_notMem _ _)
      rw [← hq', Index.unbroadcast_val]
      simp only [hne, ↓reduceIte]
      exact (hq _).trans (congrArg (fun d => (i d : ℕ)) (hcast _))
    refine Finset.sum_congr (Finset.filter_congr fun j _ => ⟨fun hj => ?_, fun hj => ?_⟩) fun _ _ => rfl
    · refine Index.ext_val fun d => ?_
      rw [Index.unbroadcast_val]
      split_ifs with h1
      · have := (i d).isLt; omega
      · obtain ⟨c, rfl⟩ := keptAxes_surj (unitDims_notMem.2 h1)
        have := congrArg (fun q => (q (Fin.cast hAc.symm c) : ℕ)) hj
        simp only [Index.proj, hval] at this
        exact (congrArg (fun z => (j z : ℕ))
          (keptAxes_broadcastInDimTransposeAxes hd (Fin.cast hAc.symm c))).symm.trans this
    · refine Index.ext_val fun a => ?_
      have := congrArg (fun i => (i (keptAxes (unitDims so) (Fin.cast hAc a)) : ℕ)) hj
      simp only [Index.unbroadcast_val, unitDims_notMem.1 (keptAxes_notMem _ _), ↓reduceIte] at this
      rw [hval, ← this]
      exact congrArg (fun z => (j z : ℕ)) (keptAxes_broadcastInDimTransposeAxes hd a)
  · -- Pairing the cotangent with the broadcast operand is summing over the fibres.
    let φ := fun j : Index shape => (j.proj broadcastDimensions).unbroadcast h
    calc ∑ j, ct j * x (φ j)
        = ∑ i, ∑ j with φ j = i, ct j * x (φ j) := (Finset.sum_fiberwise _ _ _).symm
      _ = ∑ i, (∑ j with φ j = i, ct j) * x i := by
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun j hj => ?_
          rw [(Finset.mem_filter.1 hj).2]

/-- The transpose rule of `broadcast_in_dim` is wrong for `broadcast_dimensions = (1, 0)`:
for an operand of shape `(3, 2)` and a result of shape `(2, 3)`,
it returns a cotangent of shape `(2, 3)` instead of `(3, 2)`. -/
theorem broadcastInDimTranspose_not_correct :
    ¬ IsTransposeOf (broadcastInDimTranspose ![3, 2] ![2, 3] ![1, 0] (by decide))
      (broadcastInDim (d := .float w) (so := ![3, 2]) ![2, 3] ![1, 0] (by unfold BroadcastsTo; decide)
        (by decide)) := by
  intro htr
  obtain ⟨_, _, _, ⟨_, hN, hs, _⟩, _⟩ := htr 0 0
  -- Axis 0 of the result of the rule has size 2, and axis 0 of the operand has size 3.
  have h0 := hs ⟨0, by decide⟩
  rw [show Fin.cast hN ⟨0, _⟩ = (0 : Fin 2) from rfl] at h0
  exact absurd h0 (by decide)

/-! ## Forwarding rule -/

/-- The forwarding rule of `broadcast_in_dim`.
It forwards the operand of shape `so` if `shape` equals `so`, whatever `broadcast_dimensions` is.
Sharding is not modelled, so `eqn.params['sharding']` is `None`.
`jax/_src/lax/lax.py:7583`:
```
def _broadcast_in_dim_fwd_rule(eqn):
  v, = eqn.invars
  if (core.definitely_equal_shape(eqn.params['shape'], v.aval.shape)
      and (eqn.params['sharding'] is None or
           eqn.params['sharding'] == v.aval.sharding)):
    return [0], None
  else:
    return [None], eqn
``` -/
def broadcastInDimFwd (so : Shape m) (shape : Shape n) (_broadcastDimensions : Fin m → Fin n) :
    Bool :=
  definitelyEqualShape shape so

/-- The forwarding rule of `broadcast_in_dim` is correct
if `broadcast_dimensions` is strictly increasing. -/
theorem broadcastInDimFwd_correct (shape : Shape n) (broadcastDimensions : Fin m → Fin n)
    (h : BroadcastsTo so (shape ∘ broadcastDimensions)) (hd : StrictMono broadcastDimensions) :
    IsFwdOf (broadcastInDimFwd so shape broadcastDimensions)
      (broadcastInDim (d := d) shape broadcastDimensions h hd.injective) := by
  intro hf x
  have hf' : List.ofFn shape = List.ofFn so := beq_iff_eq.1 hf
  obtain rfl : n = m := by simpa using congrArg List.length hf'
  obtain rfl := List.ofFn_injective hf'
  obtain rfl := hd.eq_id
  refine Tensor.Same.of_eq (funext fun j => congrArg x (Index.ext_val fun d => ?_))
  rw [Index.unbroadcast_val]
  split_ifs with h1
  · have := (j d).isLt
    omega
  · rfl

/-- The forwarding rule of `broadcast_in_dim` is wrong for `broadcast_dimensions = (1, 0)`:
for an operand of shape `(2, 2)`, the result is its transpose, and the rule forwards the operand. -/
theorem broadcastInDimFwd_not_correct :
    ¬ IsFwdOf (broadcastInDimFwd ![2, 2] ![2, 2] ![1, 0])
      (broadcastInDim (d := .float w) (so := ![2, 2]) ![2, 2] ![1, 0] (by unfold BroadcastsTo; decide)
        (by decide)) := by
  intro hf
  -- The array `[[0, 1], [0, 1]]`, whose entry `(0, 1)` differs from the entry `(1, 0)`.
  obtain ⟨_, _, _, he⟩ := hf (by decide) (ofReal fun i => ((i 1 : ℕ) : ℝ))
  let i : Index (![2, 2] : Shape 2) := fun d => ⟨d, by fin_cases d <;> decide⟩
  have := he i i fun _ => rfl
  simp [i, ofReal, broadcastInDim, Index.proj, Index.unbroadcast] at this

end Jax.Lax
