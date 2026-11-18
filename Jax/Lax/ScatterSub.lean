module

public import Jax.Lax.ScatterAdd

/-!
# `scatter_sub`

Differentiation rules of the `scatter_sub` primitive of `jax/_src/lax/slicing.py`.
It shares its rules with `scatter_add`: the JVP rule applies the primitive to the tangents,
and the transpose rule negates the `gather` of the cotangent.

The transpose rule is correct in the modes `CLIP` and `FILL_OR_DROP`, where that `gather` passes its shape rule
(`scatterSubTranspose_correct`).
It raises if the operand has size 0 on an inserted or batching axis and the indices are not empty
(`scatterSubTranspose_not_correct`), as for `scatter_add`.
See `bugs/scatter-add-transpose-empty-operand.json`.

In the mode `PROMISE_IN_BOUNDS` the rule is not correct for a window that does not lie inside the operand,
as `scatterAddTranspose_promiseInBounds_not_correct` shows for `scatter_add`.
-/

@[expose] public section

namespace Jax.Lax

variable {n ni nu : ℕ} {so : Shape n} {si : Shape ni} {su : Shape nu} {w : FloatWidth} {wi : IntWidth}

/-! ## Rules -/

/-- The JVP rule of `scatter_sub`, which applies the primitive to the tangents.
It is `_scatter_addsub_jvp` with `prim = scatter_sub_p`, quoted at `scatterAddJvp`.
`jax/_src/lax/slicing.py:3040`:
```
ad.primitive_jvps[scatter_sub_p] = partial(_scatter_addsub_jvp, scatter_sub_p)
``` -/
def scatterSubJvp (dimensionNumbers : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (h : ScatterValid so si su dimensionNumbers)
    (gOperand : Tensor (.float w) (Index so)) (gUpdates : Tensor (.float w) (Index su))
    (_operand : Tensor (.float w) (Index so)) (indices : Tensor (.int wi) (Index si)) (_updates : Tensor (.float w) (Index su)) :
    Tensor (.float w) (Index so) :=
  scatterSub dimensionNumbers mode h gOperand indices gUpdates

/-- The transpose rule of `scatter_sub`.
It is `_scatter_addsub_transpose_rule` with `prim = scatter_sub_p`, quoted at `scatterAddTranspose`,
which negates the cotangent of `updates`, `jax/_src/lax/slicing.py:2908`:
```
      if prim is scatter_sub_p:
        update_t = lax.neg(update_t)
``` -/
def scatterSubTranspose (dimensionNumbers : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (t : Tensor (.float w) (Index so)) (operand : Option (Tensor (.float w) (Index so))) (indices : Tensor (.int wi) (Index si))
    (updates : Option (Tensor (.float w) (Index su))) :
    Option (Option (Tensor (.float w) (Index so)) × Option (Tensor (.float w) (Index
      (gatherShape si dimensionNumbers.toGather (scatterSliceSizes n su dimensionNumbers))))) :=
  let operandT := if operand.isNone then some t else none
  if updates.isNone then
    if h : GatherValid so si dimensionNumbers.toGather (scatterSliceSizes n su dimensionNumbers) ∧
        GatherLowers so si dimensionNumbers.toGather (scatterSliceSizes n su dimensionNumbers) then
      some (operandT, some (neg (gather dimensionNumbers.toGather
        (scatterSliceSizes n su dimensionNumbers) mode 0 h.1 h.2 t indices)))
    else none
  else some (operandT, none)

/-! ## Proofs -/

/-- `scatter_sub` adds the negated updates. -/
theorem scatterSub_eq_scatterAdd (dn : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (h : ScatterValid so si su dn) (operand : Tensor (.float w) (Index so)) (indices : Tensor (.int wi) (Index si))
    (updates : Tensor (.float w) (Index su)) :
    scatterSub dn mode h operand indices updates = scatterAdd dn mode h operand indices (neg updates) :=
  rfl

/-- The real function that `scatter_sub` computes. -/
noncomputable def scatterSubReal (dn : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (indices : Tensor (.int wi) (Index si)) : (Index so → ℝ) × (Index su → ℝ) →ₗ[ℝ] Index so → ℝ :=
  (scatterAddReal dn mode indices).comp (LinearMap.prodMap LinearMap.id (-LinearMap.id))

theorem scatterSub_ofReal (dn : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (h : ScatterValid so si su dn) (indices : Tensor (.int wi) (Index si)) (x : Index so → ℝ) (y : Index su → ℝ) :
    scatterSub (w := w) dn mode h (ofReal x) indices (ofReal y) =
      ofReal (scatterSubReal dn mode indices (x, y)) := by
  rw [scatterSub_eq_scatterAdd, show neg (ofReal (w := w) y) = ofReal (-y) from rfl, scatterAdd_ofReal]
  rfl

/-- The JVP rule of `scatter_sub` computes the derivative of `scatter_sub`. -/
theorem scatterSubJvp_correct (dn : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (h : ScatterValid so si su dn) (indices : Tensor (.int wi) (Index si)) :
    IsJointJvp₂Of (fun g k x y => scatterSubJvp dn mode h g k x indices y)
      (fun x y => scatterSub (w := w) dn mode h x indices y) := by
  intro x y _
  let L := LinearMap.toContinuousLinearMap (scatterSubReal (so := so) (su := su) dn mode indices)
  exact ⟨scatterSubReal dn mode indices, L,
    .of_forall fun p => scatterSub_ofReal dn mode h indices p.1 p.2, L.hasFDerivAt,
    fun g k => .of_eq (scatterSub_ofReal dn mode h indices g k)⟩

/-- The transpose rule of `scatter_sub` transposes `scatter_sub` in the modes `CLIP` and `FILL_OR_DROP`,
where the `gather` of the rule passes its checks, whichever arguments are linear inputs. -/
theorem scatterSubTranspose_correct (dn : ScatterDimensionNumbers) (mode : GatherScatterMode)
    (h : ScatterValid so si su dn)
    (hg : GatherValid so si dn.toGather (scatterSliceSizes n su dn))
    (hgl : GatherLowers so si dn.toGather (scatterSliceSizes n su dn))
    (indices : Tensor (.int wi) (Index si)) (hm : mode ≠ .promiseInBounds) :
    IsPartialTranspose₂Of (fun t x y => scatterSubTranspose dn mode t x indices y)
      (fun x y => scatterSub (w := w) dn mode h x indices y) := by
  refine ⟨id,
    fun t => neg (gather dn.toGather (scatterSliceSizes n su dn) mode 0 hg hgl t indices),
    fun ct x y => ?_, fun ct x y hxy => ?_⟩
  · obtain ⟨z, u, v, hz, hu, ⟨_, hr, hs, he⟩, hsum⟩ :=
      scatterAdd_isTranspose₂Of dn mode h hg hgl indices hm ct x (-y)
    refine ⟨z, u, -v, hz, hu, ⟨rfl, hr, hs, fun i j hij => ?_⟩, ?_⟩
    · show -(gather dn.toGather (scatterSliceSizes n su dn) mode 0 hg hgl (ofReal ct) indices i) = _
      exact (congrArg (-·) (he i j hij)).trans rfl
    · simpa using hsum
  · refine ⟨if x.isNone then some (ofReal ct) else none,
      if y.isNone then some (neg (gather dn.toGather (scatterSliceSizes n su dn) mode 0 hg hgl
        (ofReal ct) indices)) else none, ?_, fun hx => by simp [hx], fun hy => by simp [hy]⟩
    cases y <;> simp [scatterSubTranspose, hg, hgl]

/-- The transpose rule of `scatter_sub` raises for an operand of size 0 on an inserted axis:
here the operand has shape `(0,)`, as in `jnp.zeros(0).at[jnp.array([0])].subtract(updates)`.
`scatter_sub` accepts it and drops the update, but the `gather` of the rule has slice size 1
on the axis of size 0, which its shape rule rejects. -/
theorem scatterSubTranspose_not_correct (mode : GatherScatterMode)
    (indices : Tensor (.int wi) (Index ![1, 1])) :
    ¬ IsPartialTranspose₂Of
      (fun t x y => scatterSubTranspose (so := ![0]) (su := ![1]) ⟨[], [0], [0], [], []⟩ mode t x
        indices y)
      (fun x y => scatterSub (w := w) (so := ![0]) (su := ![1]) ⟨[], [0], [0], [], []⟩ mode
        (by decide) x indices y) := by
  rintro ⟨_, _, _, hrule⟩
  obtain ⟨_, _, hsome, _⟩ := hrule 0 (some 0) none (.inr rfl)
  have hno : ¬ (GatherValid ![0] ![1, 1] (ScatterDimensionNumbers.toGather ⟨[], [0], [0], [], []⟩)
      (scatterSliceSizes 1 ![1] ⟨[], [0], [0], [], []⟩) ∧
      GatherLowers ![0] ![1, 1] (ScatterDimensionNumbers.toGather ⟨[], [0], [0], [], []⟩)
        (scatterSliceSizes 1 ![1] ⟨[], [0], [0], [], []⟩)) := by decide
  simp [scatterSubTranspose, hno] at hsome

end Jax.Lax
