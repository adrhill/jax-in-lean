module

public import Jax.Lax.Primal

/-!
# `reduce`

The JVP rule of the `reduce` primitive of `jax/_src/lax/lax.py`, with one operand,
and a refutation: the rule assumes that the reduction is commutative,
which the `lax.reduce` docstring does not require.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ}

/-- `_reduce_tree` on `n` values: while more than one value is left,
combine the first half with the second half, padded with `initValue` to the same length.
`jax/_src/lax/lax.py:9122`:
```
  def _reduce_tree(*xs, axis=0):
    """Reduce by repeatedly splitting the array and multiplying."""
    while xs[0].shape[axis] > 1:
      n = xs[0].shape[axis]
      n1 = (n + 1) // 2
      n2 = n - n1
      xs1 = [slicing.slice_in_dim(x, 0, n1) for x in xs]
      xs2 = [slicing.slice_in_dim(x, n1, None) for x in xs]
      if n2 != n1:
        paddings = [(0, 0, 0)] * len(xs[0].shape)
        paddings[axis] = (0, 1, 0)
        xs2 = [pad(x2, i, paddings) for x2, i in zip(xs2, init_values)]
      xs = reducer(*(xs1 + xs2))
    if xs[0].shape[axis] == 0:
      return [full(tuple(input_shape[non_axes]), i) for i in init_values]
    return tuple(squeeze(x, (axis,)) for x in xs)
``` -/
def reduceTree {α : Type} (reducer : α → α → α) (initValue : α) : (n : ℕ) → (Fin n → α) → α
  | 0, _ => initValue
  | 1, xs => xs 0
  | n + 2, xs =>
    reduceTree reducer initValue ((n + 3) / 2) fun i =>
      reducer (xs ⟨i, by have := i.isLt; omega⟩)
        (if h : (n + 3) / 2 + i < n + 2 then xs ⟨(n + 3) / 2 + i, h⟩ else initValue)
decreasing_by omega

/-- The JVP rule of `reduce`, `_reduce_jvp_rule`, which differentiates `_reduce_tree` with `api.jvp`.
The tangent `operandDot` comes first, as in the other rules.
`_reduce_jvp` moves the axes `dimensions` to the front in the order in which they are listed,
flattens them in row-major order and maps `_reduce_tree` over the other axes:
here each entry of the result reduces `reductionSlice dimensions`.
`api.jvp` pairs each primal with its tangent, `jvp` differentiates `reducer`,
and the padding value `initValue` and `full` are constants with the tangent `0`.
The rule raises `NotImplementedError` for a nonzero tangent of `init_values`,
so `initValue` has none.
`jax/_src/lax/lax.py:9108`:
```
def _reduce_jvp(reducer, init_values, primals, tangents, axes):
  input_shape = np.array(primals[0].shape, dtype=int)

  n = np.prod(input_shape[list(axes)])
  non_axes = np.delete(np.arange(len(input_shape)), axes)

  # Move the reduced axes to the front, and flatten them to 1D.
  permutation = axes + tuple(non_axes)
  new_shape = (n,) + tuple(input_shape[non_axes])
  primals = tuple(reshape(x, new_shape, permutation) for x in primals)
  tangents = tuple(reshape(t, new_shape, permutation) for t in tangents)

  for d in range(len(non_axes) + 1):
    reducer = api.vmap(reducer)
  def _reduce_tree(*xs, axis=0):
    ...
  return api.jvp(_reduce_tree, primals, tangents)

def _reduce_jvp_rule(primals, tangents, *, computation, jaxpr, dimensions):
  primal_xs, init_values = split_list(primals, [len(primals) // 2])
  tangent_xs, tangent_init = split_list(tangents, [len(tangents) // 2])
  # This test may be too strict, if a value is actually zero but we cannot prove
  # it is symbolically zero.
  if any(type(t) is not ad_util.Zero for t in tangent_init):
    raise NotImplementedError(
      "Gradient of general lax.reduce with non-zero tangents for "
      "initial values to reduction not implemented")
  reducer = core.jaxpr_as_fun(jaxpr)
  return _reduce_jvp(reducer, init_values, primal_xs, tangent_xs, dimensions)
``` -/
def reduceJvp {s : Shape n} (computation : Computation) (initValue : Scalar)
    (dimensions : List (Fin n)) (_hd : dimensions.Nodup) (operandDot operand : Tensor (Index s)) :
    Tensor (Index (s ∘ keptAxes dimensions.toFinset)) :=
  fun j => (reduceTree (fun a b => (computation.op a.1 b.1, computation.jvp a.1 b.1 a.2 b.2))
    (initValue, 0) _ fun p =>
      (reductionSlice dimensions (fun _ => List.mem_toFinset.symm) operand j p,
        reductionSlice dimensions (fun _ => List.mem_toFinset.symm) operandDot j p)).2

/-! ## A reduction that is not commutative -/

open scoped Classical in
/-- The computation `lambda x, y: lax.select(y == 1., x, y)`, which keeps the last argument other than `1`.
`select(p, a, b)` is `select_n(p, b, a)`, and its JVP selects the tangents,
`jax/_src/lax/lax.py:8972`:
```
def _select_jvp(primals, tangents):
  which, *case_primals = primals
  case_tangents = tangents[1:]
  out = select_n(which, *case_primals)
  ...
    out_dot = select_n(which, *case_tangents)
  return out, out_dot
```
The comparison `eq` has no tangent. -/
noncomputable def lastNotOne : Computation where
  op x y := if y.eq 1 then x else y
  jvp _ y dx dy := if y.eq 1 then dx else dy

/-- `lastNotOne` is associative. -/
theorem lastNotOne_assoc (x y z : Scalar) :
    lastNotOne.op (lastNotOne.op x y) z = lastNotOne.op x (lastNotOne.op y z) := by
  simp only [lastNotOne]
  split_ifs <;> simp_all

/-- `1` is a left identity of `lastNotOne`. -/
theorem lastNotOne_one_left (x : Scalar) : lastNotOne.op 1 x = x := by
  simp only [lastNotOne]
  split_ifs with h
  exacts [h.1.symm, rfl]

/-- `1` is a right identity of `lastNotOne`. -/
theorem lastNotOne_one_right (x : Scalar) : lastNotOne.op x 1 = x := by
  simp [lastNotOne, Scalar.eq, Scalar.one_def]

/-- `lastNotOne` is not commutative. -/
theorem lastNotOne_not_comm : lastNotOne.op (.fin 2) (.fin 3) ≠ lastNotOne.op (.fin 3) (.fin 2) := by
  simp only [lastNotOne, Scalar.one_def, Scalar.eq_fin]
  norm_num

/-- The JVP of `lastNotOne` is correct where its second argument is not `1`:
near `(x, y)` it computes `(x, y) ↦ y`, and its JVP applies the derivative of that map. -/
theorem lastNotOne_jvp_correct {x y : ℝ} (hy : y ≠ 1) :
    (∀ᶠ p : ℝ × ℝ in nhds (x, y), lastNotOne.op (.fin p.1) (.fin p.2) = .fin p.2) ∧
    HasFDerivAt (fun p : ℝ × ℝ => p.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) (x, y) ∧
    ∀ dx dy, lastNotOne.jvp (.fin x) (.fin y) (.fin dx) (.fin dy) =
      .fin (ContinuousLinearMap.snd ℝ ℝ ℝ (dx, dy)) := by
  refine ⟨?_, hasFDerivAt_snd, fun dx dy => by simp [lastNotOne, Scalar.one_def, hy]⟩
  filter_upwards [(isOpen_ne_fun continuous_snd continuous_const).mem_nhds hy] with p hp
  simp [lastNotOne, Scalar.one_def, hp]

/-! ## Refutation -/

private theorem rowMajor_symm_rank_one {t : Shape 1} (p : Fin (∏ d, t d)) :
    ((Index.rowMajor.symm p) 0 : ℕ) = p := by
  have := Index.rowMajor_val (Index.rowMajor.symm p)
  simp only [Equiv.apply_symm_apply, Finset.univ_unique, Fin.default_eq_zero,
    Finset.sum_singleton] at this
  rw [this]
  simp

/-- The slice of an operand of shape `(3,)` along its only axis is its list of entries. -/
private theorem reductionSlice_eq {D : Finset (Fin 1)} (hax : ∀ d, d ∈ [0] ↔ d ∈ D)
    (x : Tensor (Index ![3])) (j : Index (![3] ∘ keptAxes D)) (p : Fin (∏ k, (![3] ∘ [0].get) k)) :
    reductionSlice [(0 : Fin 1)] hax x j p = x fun _ => ⟨p, by simpa using p.2⟩ := by
  refine congrArg x (Index.ext_val fun d => ?_)
  rw [Subsingleton.elim d 0, Index.merge_val_of_mem _ _ _ _ (List.mem_singleton_self 0)]
  exact rowMajor_symm_rank_one (t := ![3] ∘ [(0 : Fin 1)].get) p

theorem reduceTree_congr {α : Type} (reducer : α → α → α) (initValue : α) {m k : ℕ} (h : m = k)
    (xs : Fin m → α) :
    reduceTree reducer initValue m xs = reduceTree reducer initValue k fun i => xs (Fin.cast h.symm i) := by
  subst h; rfl

/-- `_reduce_tree` on three values reduces `[op(x₀, x₂), op(x₁, init)]`. -/
theorem reduceTree_three {α : Type} (reducer : α → α → α) (initValue : α) (xs : Fin 3 → α) :
    reduceTree reducer initValue 3 xs = reducer (reducer (xs 0) (xs 2)) (reducer (xs 1) initValue) := by
  rw [reduceTree, reduceTree, reduceTree]
  rfl

/-- The last index of an operand of shape `(3,)`. -/
private def lastIdx : Index ![3] := fun _ => ⟨2, by simp⟩

/-- `reduce` with `lastNotOne` over an operand of shape `(3,)` returns the last entry if it is not `1`. -/
private theorem reduce_lastNotOne (y : Index ![3] → ℝ) (hy : y lastIdx ≠ 1) :
    reduce lastNotOne 1 [0] (List.nodup_singleton _) (ofReal y) = ofReal fun _ => y lastIdx := by
  funext j
  have key : ∀ ax : List (Fin 1), ax = [0] → ∀ hax : ∀ d, d ∈ ax ↔ d ∈ ([0] : List (Fin 1)).toFinset,
      (List.ofFn (reductionSlice ax hax (ofReal y) j)).foldl lastNotOne.op 1 = .fin (y lastIdx) := by
    rintro ax rfl hax
    rw [List.ofFn_congr (by simp : ∏ k, (![3] ∘ [(0 : Fin 1)].get) k = 3)]
    simp only [List.ofFn_succ, List.ofFn_zero, List.foldl_cons, List.foldl_nil, reductionSlice_eq]
    have hl : (fun _ => ⟨2, by simp⟩ : Index ![3]) = lastIdx := rfl
    simp [ofReal, lastNotOne, Scalar.one_def, hl, hy]
  exact key _ (by decide) _

/-- The JVP rule of `reduce` is wrong for the associative, non-commutative `lastNotOne` with identity `1`,
at a point where all its entries differ from `1`, so that `lastNotOne` and its JVP are correct
at every step of the reduction (`lastNotOne_jvp_correct`).
For `x = [2, 3, 4]` the reduction returns its last entry `4`, but the tree of `_reduce_tree`
reduces `[op(2, 4), op(3, 1)] = [4, 3]` and returns `3`, together with the tangent of the entry `3`. -/
theorem reduceJvp_not_correct :
    ¬ IsJvpOfOn (reduceJvp (s := ![3]) lastNotOne 1 [0] (List.nodup_singleton _))
      (reduce lastNotOne 1 [0] (List.nodup_singleton _)) {x | ∀ i, x i ≠ 1} := by
  intro hj
  let x₀ : Index ![3] → ℝ := fun i => ((i 0 : ℕ) : ℝ) + 2
  have hx₀ : x₀ ∈ {x | ∀ i, x i ≠ 1} := fun i => by
    have : (0 : ℝ) ≤ ((i 0 : ℕ) : ℝ) := Nat.cast_nonneg _
    simp only [x₀]; intro h; linarith
  obtain ⟨F, L, hF, hL, hrule⟩ := hj x₀ hx₀
  -- Near `x₀` the reduction is the last entry, a linear map.
  let L' : (Index ![3] → ℝ) →L[ℝ] Index (![3] ∘ keptAxes ([0] : List (Fin 1)).toFinset) → ℝ :=
    ContinuousLinearMap.pi fun _ => ContinuousLinearMap.proj lastIdx
  have hlast : ∀ᶠ y in nhds x₀, y lastIdx ≠ 1 :=
    (isOpen_ne_fun (continuous_apply lastIdx) continuous_const).mem_nhds (hx₀ lastIdx)
  have hFL : F =ᶠ[nhds x₀] L' := by
    filter_upwards [hF, hlast] with y hy hy1
    exact ofReal_injective (hy.symm.trans (reduce_lastNotOne y hy1))
  have hLL : L = L' := hL.unique (L'.hasFDerivAt.congr_of_eventuallyEq hFL)
  -- The rule returns the tangent of the middle entry instead of the last one.
  let g : Index ![3] → ℝ := fun i => if i = lastIdx then 1 else 0
  obtain ⟨_, _, hv⟩ := hrule g
  have hc : (([0] : List (Fin 1)).toFinset)ᶜ.card = 0 := by decide
  let e : Index (![3] ∘ keptAxes ([0] : List (Fin 1)).toFinset) := fun d => (Fin.cast hc d).elim0
  have := hv e e fun _ => rfl
  rw [hLL] at this
  simp only [reduceJvp] at this
  rw [reduceTree_congr _ _ (by simp : ∏ k, (![3] ∘ [(0 : Fin 1)].get) k = 3)] at this
  rw [reduceTree_three] at this
  simp only [reductionSlice_eq] at this
  simp [ofReal, x₀, g, L', lastIdx, lastNotOne, Scalar.one_def, funext_iff, Fin.ext_iff] at this

end Jax.Lax
