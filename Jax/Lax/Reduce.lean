module

public import Jax.Lax.Primal

/-!
# `reduce`

The JVP rule of the `reduce` primitive of `jax/_src/lax/lax.py`, with one operand.
The rule is correct for a commutative monoid (`reduceJvp_correct`).
It is wrong for a monoid that is not commutative (`reduceJvp_not_correct`),
which the `lax.reduce` docstring allows.
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

/-! ## Commutative reductions -/

theorem reduceTree_congr {α : Type} (reducer : α → α → α) (initValue : α) {m k : ℕ} (h : m = k)
    (xs : Fin m → α) :
    reduceTree reducer initValue m xs = reduceTree reducer initValue k fun i => xs (Fin.cast h.symm i) := by
  subst h; rfl

/-- `_reduce_tree` commutes with maps that commute with the reducer. -/
theorem reduceTree_map {α β : Type} {r : α → α → α} {init : α} {r' : β → β → β} {init' : β}
    (h : α → β) (hr : ∀ a b, h (r a b) = r' (h a) (h b)) (hinit : h init = init') :
    ∀ (m : ℕ) (xs : Fin m → α), h (reduceTree r init m xs) = reduceTree r' init' m (h ∘ xs)
  | 0, _ => by simp [reduceTree, hinit]
  | 1, _ => by simp [reduceTree]
  | n + 2, xs => by
    rw [reduceTree, reduceTree, reduceTree_map h hr hinit]
    congr 1
    funext i
    simp only [Function.comp_apply, hr]
    split_ifs <;> simp [hinit]
termination_by m => m
decreasing_by omega

/-- A property `P` of the values holds for the result of `_reduce_tree` on at least one value
if the reducer preserves it for values that are related by `R`,
and the reducer preserves `R`, which holds for every two values at different positions. -/
theorem reduceTree_induction {α : Type} (r : α → α → α) (init : α) (P : α → Prop)
    (R : α → α → Prop) (hr : ∀ a b, P a → P b → R a b → P (r a b))
    (hpad : ∀ a, P a → P (r a init)) (hR : ∀ a b c, R a c → R b c → R (r a b) c)
    (hRpad : ∀ a c, R a c → R (r a init) c) (hsymm : ∀ a b, R a b → R b a) :
    ∀ (m : ℕ) (xs : Fin m → α), 0 < m → (∀ i, P (xs i)) → (∀ i k, i ≠ k → R (xs i) (xs k)) →
      P (reduceTree r init m xs)
  | 0, _, h, _, _ => absurd h (lt_irrefl 0)
  | 1, xs, _, hP, _ => by rw [reduceTree]; exact hP 0
  | n + 2, xs, _, hP, hxs => by
    rw [reduceTree]
    -- The value at position `i` of the next level, and the values it combines.
    have hy : ∀ (i : Fin ((n + 3) / 2)) (c : α), R (xs ⟨i, by omega⟩) c →
        (∀ h : (n + 3) / 2 + i < n + 2, R (xs ⟨_, h⟩) c) →
        R (r (xs ⟨i, by omega⟩)
          (if h : (n + 3) / 2 + i < n + 2 then xs ⟨(n + 3) / 2 + i, h⟩ else init)) c := by
      intro i c h1 h2
      split_ifs with h
      · exact hR _ _ _ h1 (h2 h)
      · exact hRpad _ _ h1
    have hne : ∀ (a b : ℕ) (ha : a < n + 2) (hb : b < n + 2), a ≠ b →
        R (xs ⟨a, ha⟩) (xs ⟨b, hb⟩) := fun a b ha hb hab => hxs _ _ (by simpa [Fin.ext_iff] using hab)
    refine reduceTree_induction r init P R hr hpad hR hRpad hsymm _ _ (by omega) (fun i => ?_)
      fun i k hik => ?_
    · split_ifs with h
      · exact hr _ _ (hP _) (hP _) (hne _ _ _ _ (by omega))
      · exact hpad _ (hP _)
    · have hik' : (i : ℕ) ≠ k := fun h => hik (Fin.ext h)
      refine hy i _ (hsymm _ _ (hy k _ (hne _ _ _ _ (by omega)) fun _ => hne _ _ _ _ (by omega)))
        fun h => hsymm _ _ (hy k _ (hne _ _ _ _ (by omega)) fun _ => hne _ _ _ _ (by omega))
termination_by m => m
decreasing_by omega

section CommMonoid

variable {α : Type} {op : α → α → α} {e : α}

/-- The commutative monoid of an associative and commutative `op` with identity `e`. -/
abbrev IsMonoidOp.toCommMonoid (h : IsMonoidOp op e) (hc : ∀ a b, op a b = op b a) :
    CommMonoid α where
  mul := op
  one := e
  mul_assoc := h.assoc
  one_mul := h.left_id
  mul_one := h.right_id
  mul_comm := hc

private theorem foldl_ofFn_mul {M : Type} [CommMonoid M] {k : ℕ} (xs : Fin k → M) :
    (List.ofFn xs).foldl (· * ·) 1 = ∏ i, xs i := by
  rw [← List.prod_eq_foldl, List.prod_ofFn]

private theorem reduceOn_mul {M : Type} [CommMonoid M] {k : ℕ} (xs : Fin k → M)
    (A : Finset (Fin k)) : reduceOn (· * ·) 1 xs A = ∏ p ∈ A, xs p := by
  rw [reduceOn, foldl_ofFn_mul, Finset.prod_ite_mem, Finset.univ_inter]

/-- In a commutative monoid, `_reduce_tree` computes the product. -/
private theorem reduceTree_mul {M : Type} [CommMonoid M] :
    ∀ (m : ℕ) (xs : Fin m → M), reduceTree (· * ·) 1 m xs = ∏ i, xs i
  | 0, _ => by simp [reduceTree]
  | 1, xs => by simp [reduceTree]
  | n + 2, xs => by
    rw [reduceTree, reduceTree_mul]
    let f : ℕ → M := fun k => if h : k < n + 2 then xs ⟨k, h⟩ else 1
    have h1 : ∏ i, xs i = ∏ k ∈ Finset.range ((n + 3) / 2 + (n + 3) / 2), f k := by
      have : ∏ i, xs i = ∏ i : Fin (n + 2), f i := Finset.prod_congr rfl fun i _ => by simp [f]
      rw [this, Fin.prod_univ_eq_prod_range f]
      refine Finset.prod_subset (Finset.range_subset_range.2 (by omega)) fun k _ hk => ?_
      simp only [Finset.mem_range] at hk
      simp [f, hk]
    rw [h1, Finset.prod_range_add, ← Finset.prod_mul_distrib,
      ← Fin.prod_univ_eq_prod_range (fun k => f k * f ((n + 3) / 2 + k))]
    refine Finset.prod_congr rfl fun i _ => ?_
    have hi : (i : ℕ) < n + 2 := by omega
    simp [f, hi]
termination_by m => m
decreasing_by omega

/-- For a commutative monoid, `_reduce_tree` computes the left fold. -/
theorem IsMonoidOp.reduceTree_eq_foldl (h : IsMonoidOp op e) (hc : ∀ a b, op a b = op b a)
    (m : ℕ) (xs : Fin m → α) : reduceTree op e m xs = (List.ofFn xs).foldl op e := by
  let := h.toCommMonoid hc
  exact (reduceTree_mul m xs).trans (foldl_ofFn_mul xs).symm

/-- For a commutative monoid, the left fold does not depend on the order. -/
theorem IsMonoidOp.foldl_ofFn_equiv (h : IsMonoidOp op e) (hc : ∀ a b, op a b = op b a)
    {k k' : ℕ} (σ : Fin k ≃ Fin k') (xs : Fin k' → α) :
    (List.ofFn (xs ∘ σ)).foldl op e = (List.ofFn xs).foldl op e := by
  let := h.toCommMonoid hc
  exact (foldl_ofFn_mul _).trans
    ((Fintype.prod_equiv σ _ _ fun _ => rfl).trans (foldl_ofFn_mul xs).symm)

theorem IsMonoidOp.reduceOn_union (h : IsMonoidOp op e) (hc : ∀ a b, op a b = op b a) {k : ℕ}
    (xs : Fin k → α) {A B : Finset (Fin k)} (hAB : Disjoint A B) :
    reduceOn op e xs (A ∪ B) = op (reduceOn op e xs A) (reduceOn op e xs B) := by
  let := h.toCommMonoid hc
  have h1 := reduceOn_mul xs (A ∪ B)
  rw [Finset.prod_union hAB, ← reduceOn_mul, ← reduceOn_mul] at h1
  exact h1

theorem IsMonoidOp.reduceOn_singleton (h : IsMonoidOp op e) (hc : ∀ a b, op a b = op b a)
    {k : ℕ} (xs : Fin k → α) (p : Fin k) : reduceOn op e xs {p} = xs p := by
  let := h.toCommMonoid hc
  exact (reduceOn_mul xs {p}).trans (Finset.prod_singleton _ _)

theorem IsMonoidOp.reduceOn_comp_equiv (h : IsMonoidOp op e) (hc : ∀ a b, op a b = op b a)
    {k k' : ℕ} (σ : Fin k ≃ Fin k') (xs : Fin k' → α) (A : Finset (Fin k)) :
    reduceOn op e (xs ∘ σ) A = reduceOn op e xs (A.map σ.toEmbedding) := by
  let := h.toCommMonoid hc
  exact (reduceOn_mul _ A).trans ((Finset.prod_map A σ.toEmbedding xs).symm.trans
    (reduceOn_mul xs _).symm)

end CommMonoid

section Derivative

variable {E : Type} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Near `x`, `φ` computes a real function that is differentiable at `x`,
and `t` is its derivative applied to `g`. -/
private def HasJvpAt (φ : E → Scalar) (x g : E) (t : Scalar) : Prop :=
  ∃ (Φ : E → ℝ) (Φ' : E →L[ℝ] ℝ), (∀ᶠ y in nhds x, φ y = .fin (Φ y)) ∧ HasFDerivAt Φ Φ' x ∧
    t = .fin (Φ' g)

/-- The chain rule for a computation whose JVP is correct at the values of its arguments. -/
private theorem HasJvpAt.op {c : Computation} {φ₁ φ₂ : E → Scalar} {x g : E} {t₁ t₂ : Scalar}
    (h₁ : HasJvpAt φ₁ x g t₁) (h₂ : HasJvpAt φ₂ x g t₂) {a b : ℝ} (ha : φ₁ x = .fin a)
    (hb : φ₂ x = .fin b) (hc : c.IsJvpAt a b) :
    HasJvpAt (fun y => c.op (φ₁ y) (φ₂ y)) x g (c.jvp (φ₁ x) (φ₂ x) t₁ t₂) := by
  obtain ⟨Φ₁, Φ₁', hev₁, hd₁, rfl⟩ := h₁
  obtain ⟨Φ₂, Φ₂', hev₂, hd₂, rfl⟩ := h₂
  obtain ⟨F, L, hevF, hdF, hjvp⟩ := hc
  have ha' : Φ₁ x = a := Scalar.fin.inj (hev₁.self_of_nhds.symm.trans ha)
  have hb' : Φ₂ x = b := Scalar.fin.inj (hev₂.self_of_nhds.symm.trans hb)
  have hd : HasFDerivAt (fun y => (Φ₁ y, Φ₂ y)) (Φ₁'.prod Φ₂') x := hd₁.prodMk hd₂
  have hT : Filter.Tendsto (fun y => (Φ₁ y, Φ₂ y)) (nhds x) (nhds (a, b)) := by
    rw [← ha', ← hb']
    exact hd.continuousAt.tendsto
  refine ⟨fun y => F (Φ₁ y, Φ₂ y), L ∘L (Φ₁'.prod Φ₂'), ?_, ?_, ?_⟩
  · filter_upwards [hev₁, hev₂, hT.eventually hevF] with y h1 h2 h3
    rw [h1, h2]
    exact h3
  · rw [← ha', ← hb'] at hdF
    exact hdF.comp x hd
  · rw [ha, hb, hjvp]
    rfl

/-- Combining with the identity in the second argument, with the tangent `0`. -/
private theorem HasJvpAt.pad {c : Computation} {init : Scalar} (hright : ∀ a, c.op a init = a)
    (hpad : ∀ a da : ℝ, c.jvp (.fin a) init (.fin da) 0 = .fin da) {φ : E → Scalar} {x g : E}
    {t : Scalar} (h : HasJvpAt φ x g t) :
    HasJvpAt (fun y => c.op (φ y) init) x g (c.jvp (φ x) init t 0) := by
  obtain ⟨Φ, Φ', hev, hd, rfl⟩ := h
  refine ⟨Φ, Φ', ?_, hd, ?_⟩
  · simpa only [hright] using hev
  · rw [hev.self_of_nhds, hpad]

end Derivative

/-- The arrays at which the JVP of `computation` is correct at every pair of reductions
of two disjoint, nonempty sets of entries of the same slice.
These are the pairs that an evaluation tree of a commutative reduction may combine. -/
def reduceJvpDomain {s : Shape n} (computation : Computation) (initValue : Scalar)
    (dimensions : List (Fin n)) : Set (Index s → ℝ) :=
  {x | ∀ (j : Index (s ∘ keptAxes dimensions.toFinset)) A B, Disjoint A B → A.Nonempty →
    B.Nonempty → ∃ a b,
      reduceOn computation.op initValue
        (reductionSlice (sortedAxes _) (fun _ => mem_sortedAxes) (ofReal x) j) A = .fin a ∧
      reduceOn computation.op initValue
        (reductionSlice (sortedAxes _) (fun _ => mem_sortedAxes) (ofReal x) j) B = .fin b ∧
      computation.IsJvpAt a b}

/-- The JVP rule of `reduce` is correct for a commutative monoid,
where the JVP of the computation is correct at every pair of partial reductions that may be combined.
Besides the monoid conditions of the `lax.reduce` docstring and commutativity, it needs
that the JVP of the computation passes the tangent through the identity in the second argument,
where `_reduce_tree` pads, and that the initial value is finite if a reduced axis is empty.
`reduceJvp_not_correct` shows that commutativity cannot be dropped. -/
theorem reduceJvp_correct {s : Shape n} (computation : Computation) (initValue : Scalar)
    (dimensions : List (Fin n)) (hd : dimensions.Nodup)
    (hmon : IsMonoidOp computation.op initValue)
    (hcomm : ∀ a b, computation.op a b = computation.op b a)
    (hpad : ∀ a da : ℝ, computation.jvp (.fin a) initValue (.fin da) 0 = .fin da)
    (hinit : (∃ c : ℝ, initValue = .fin c) ∨ ∀ d ∈ dimensions, s d ≠ 0) :
    IsJvpOfOn (reduceJvp (s := s) computation initValue dimensions hd)
      (reduce computation initValue dimensions hd) (reduceJvpDomain computation initValue dimensions) := by
  refine isJvpOfOn_of_apply fun x hx g j => ?_
  rw [reduceJvpDomain, Set.mem_ofPred_eq] at hx
  let K := ∏ k, (s ∘ dimensions.get) k
  -- The slice in the order of `dimensions`, and in increasing order of the axes.
  let slice₁ : (Index s → ℝ) → Fin K → Scalar :=
    fun y => reductionSlice dimensions (fun _ => List.mem_toFinset.symm) (ofReal y) j
  let slice₂ : (Index s → ℝ) → Fin (∏ k, (s ∘ (sortedAxes dimensions.toFinset).get) k) → Scalar :=
    fun y => reductionSlice (sortedAxes _) (fun _ => mem_sortedAxes) (ofReal y) j
  let σ : Fin K ≃ Fin (∏ k, (s ∘ (sortedAxes dimensions.toFinset).get) k) :=
    Index.rowMajor.symm.trans ((Index.reorderEquiv hd (nodup_sortedAxes _)
      fun d => by rw [mem_sortedAxes, List.mem_toFinset]).trans Index.rowMajor)
  have hσ : ∀ y, slice₁ y = slice₂ y ∘ σ := fun y => funext fun p => by
    simp only [slice₁, slice₂, reductionSlice, σ, Function.comp_apply, Equiv.trans_apply,
      Equiv.symm_apply_apply, Index.reorderEquiv, Equiv.coe_fn_mk]
    rw [Index.merge_reorder]
  have hprimal : ∀ y, reduce computation initValue dimensions hd (ofReal y) j =
      (List.ofFn (slice₁ y)).foldl computation.op initValue := fun y => by
    rw [hσ, hmon.foldl_ofFn_equiv hcomm]
    rfl
  by_cases hK : K = 0
  · -- An empty slice: the reduction is the initial value.
    obtain ⟨c, hc⟩ : ∃ c : ℝ, initValue = .fin c := hinit.resolve_right fun h => by
      obtain ⟨k, -, hk⟩ := Finset.prod_eq_zero_iff.1 hK
      exact h _ (List.get_mem _ _) hk
    refine ⟨fun _ => c, 0, .of_forall fun y => ?_, hasFDerivAt_const _ _, ?_⟩
    · rw [hprimal, List.ofFn_eq_nil_iff.2 hK, List.foldl_nil, hc]
    · simp only [reduceJvp]
      rw [reduceTree_congr _ _ hK, reduceTree.eq_1]
      simp [Scalar.zero_def]
  -- Run `_reduce_tree` on functions of the array, tangents and the sets of positions they reduce.
  let T := ((Index s → ℝ) → Scalar) × Scalar × Finset (Fin K)
  let r : T → T → T := fun a b => (fun y => computation.op (a.1 y) (b.1 y),
    computation.jvp (a.1 x) (b.1 x) a.2.1 b.2.1, a.2.2 ∪ b.2.2)
  let i₀ : T := (fun _ => initValue, 0, ∅)
  let leaf : Fin K → T := fun p =>
    (fun y => slice₁ y p, reductionSlice dimensions (fun _ => List.mem_toFinset.symm) (ofReal g) j p,
      {p})
  let P : T → Prop := fun a => a.2.2.Nonempty ∧
    (∀ y, a.1 y = reduceOn computation.op initValue (slice₁ y) a.2.2) ∧ HasJvpAt a.1 x g a.2.1
  have hr : ∀ a b, P a → P b → Disjoint a.2.2 b.2.2 → P (r a b) := by
    rintro a b ⟨hna, hfa, hja⟩ ⟨hnb, hfb, hjb⟩ hab
    refine ⟨hna.mono Finset.subset_union_left, fun y => ?_, ?_⟩
    · show computation.op (a.1 y) (b.1 y) = _
      rw [hfa, hfb, hmon.reduceOn_union hcomm _ hab]
    · obtain ⟨u, v, hu, hv, huv⟩ := hx j (a.2.2.map σ.toEmbedding) (b.2.2.map σ.toEmbedding)
        ((Finset.disjoint_map _).2 hab) (hna.map) (hnb.map)
      refine hja.op hjb ?_ ?_ huv
      · rw [hfa, hσ, hmon.reduceOn_comp_equiv hcomm]
        exact hu
      · rw [hfb, hσ, hmon.reduceOn_comp_equiv hcomm]
        exact hv
  have hpad' : ∀ a, P a → P (r a i₀) := by
    rintro a ⟨hna, hfa, hja⟩
    refine ⟨show (a.2.2 ∪ ∅).Nonempty by simpa using hna, fun y => ?_, hja.pad hmon.right_id hpad⟩
    show computation.op (a.1 y) initValue = reduceOn _ _ _ (a.2.2 ∪ ∅)
    rw [Finset.union_empty, hmon.right_id, hfa]
  have hleaf : ∀ p, P (leaf p) := by
    intro p
    refine ⟨Finset.singleton_nonempty p, fun y => (hmon.reduceOn_singleton hcomm (slice₁ y) p).symm, ?_⟩
    let i := Index.merge dimensions (fun _ => List.mem_toFinset.symm) (Index.rowMajor.symm p) j
    exact ⟨fun y => y i, ContinuousLinearMap.proj i, .of_forall fun y => rfl,
      hasFDerivAt_apply i x, rfl⟩
  obtain ⟨-, -, Φ, Φ', hev, hdΦ, ht⟩ := reduceTree_induction r i₀ P
    (fun a b => Disjoint a.2.2 b.2.2) hr hpad'
    (fun _ _ _ hac hbc => Finset.disjoint_union_left.2 ⟨hac, hbc⟩)
    (fun a c hac => show Disjoint (a.2.2 ∪ ∅) c.2.2 by simpa using hac) (fun _ _ h => h.symm) K leaf
    (Nat.pos_of_ne_zero hK) hleaf fun p q hpq => Finset.disjoint_singleton.2 hpq
  refine ⟨Φ, Φ', ?_, hdΦ, ?_⟩
  · filter_upwards [hev] with y hy
    rw [← hy, hprimal, ← hmon.reduceTree_eq_foldl hcomm]
    exact (reduceTree_map (r := r) (init := i₀) (r' := computation.op) (init' := initValue) (fun a => a.1 y)
      (fun _ _ => rfl) rfl K leaf).symm
  · rw [← ht]
    exact congrArg Prod.snd (reduceTree_map (r := r) (init := i₀) (r' := fun a b => (computation.op a.1 b.1,
      computation.jvp a.1 b.1 a.2 b.2)) (init' := (initValue, 0)) (fun a : T => (a.1 x, a.2.1))
      (fun _ _ => rfl) rfl K leaf).symm

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
theorem lastNotOne_jvp_correct {x y : ℝ} (hy : y ≠ 1) : lastNotOne.IsJvpAt x y := by
  refine ⟨fun p => p.2, ContinuousLinearMap.snd ℝ ℝ ℝ, ?_, hasFDerivAt_snd,
    fun dx dy => by simp [lastNotOne, Scalar.one_def, hy]⟩
  filter_upwards [(isOpen_ne_fun continuous_snd continuous_const).mem_nhds hy] with p hp
  simp [lastNotOne, Scalar.one_def, hp]

/-! ## `lastNotOne` meets every hypothesis of `reduceJvp_correct` except commutativity -/

/-- `lastNotOne` with `1` is a monoid. -/
theorem lastNotOne_isMonoidOp : IsMonoidOp lastNotOne.op 1 :=
  ⟨lastNotOne_assoc, lastNotOne_one_left, lastNotOne_one_right⟩

/-- The JVP of `lastNotOne` passes the tangent through the identity `1` in the second argument. -/
theorem lastNotOne_jvp_one (a da : ℝ) : lastNotOne.jvp (.fin a) 1 (.fin da) 0 = .fin da := by
  simp [lastNotOne, Scalar.eq, Scalar.one_def]

/-- A left fold of `lastNotOne` over values that are `1` or finite and not `1`,
not all of them `1`, is finite and not `1`. -/
private theorem foldl_lastNotOne : ∀ (l : List Scalar) (acc : Scalar),
    (∀ v ∈ acc :: l, v = 1 ∨ ∃ r ≠ 1, v = .fin r) → (∃ v ∈ acc :: l, v ≠ 1) →
      ∃ r ≠ 1, l.foldl lastNotOne.op acc = .fin r
  | [], acc, hl, ⟨v, hv, hv1⟩ => by
    obtain rfl : v = acc := by simpa using hv
    obtain h | h := hl v (by simp)
    exacts [absurd h hv1, h]
  | v :: l, acc, hl, hne => by
    rw [List.foldl_cons]
    obtain rfl | ⟨r, hr, rfl⟩ := hl v (by simp)
    · have : lastNotOne.op acc 1 = acc := lastNotOne_one_right acc
      rw [this]
      refine foldl_lastNotOne l acc (fun w hw => hl w ?_) ?_
      · simp only [List.mem_cons] at hw ⊢
        tauto
      · obtain ⟨w, hw, hw1⟩ := hne
        simp only [List.mem_cons] at hw
        refine ⟨w, ?_, hw1⟩
        rcases hw with rfl | rfl | hw
        · simp
        · exact absurd rfl hw1
        · simp [hw]
    · have : lastNotOne.op acc (.fin r) = .fin r := by
        simp [lastNotOne, Scalar.one_def, hr]
      rw [this]
      exact foldl_lastNotOne l (.fin r) (fun w hw => (List.mem_cons.1 hw).elim
        (fun h => .inr ⟨r, hr, h⟩) fun h => hl w (by simp [h])) ⟨_, List.mem_cons_self, by
          simp [Scalar.one_def, hr]⟩

/-- Arrays whose entries all differ from `1` are in the domain of `reduceJvp_correct` for `lastNotOne`. -/
theorem lastNotOne_mem_reduceJvpDomain {s : Shape n} (dimensions : List (Fin n))
    {x : Index s → ℝ} (hx : ∀ i, x i ≠ 1) : x ∈ reduceJvpDomain lastNotOne 1 dimensions := by
  have key : ∀ (j : Index (s ∘ keptAxes dimensions.toFinset)) (A : Finset _), A.Nonempty →
      ∃ r ≠ 1, reduceOn lastNotOne.op 1
        (reductionSlice (sortedAxes _) (fun _ => mem_sortedAxes) (ofReal x) j) A = .fin r := by
    rintro j A ⟨p, hp⟩
    refine foldl_lastNotOne _ 1 (fun v hv => ?_) ⟨_, List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨p, rfl⟩), ?_⟩
    · rcases List.mem_cons.1 hv with rfl | hv
      · exact .inl rfl
      · obtain ⟨q, rfl⟩ := List.mem_ofFn.1 hv
        split_ifs
        exacts [.inr ⟨_, hx _, rfl⟩, .inl rfl]
    · simp [hp, reductionSlice, ofReal, Scalar.one_def, hx]
  intro j A B _ hA hB
  obtain ⟨a, -, ha⟩ := key j A hA
  obtain ⟨b, hb, hb'⟩ := key j B hB
  exact ⟨a, b, ha, hb', lastNotOne_jvp_correct hb⟩

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
reduces `[op(2, 4), op(3, 1)] = [4, 3]` and returns `3`, together with the tangent of the entry `3`.

`lastNotOne` meets every hypothesis of `reduceJvp_correct` except commutativity:
it is a monoid (`lastNotOne_isMonoidOp`), its JVP passes tangents through the identity
(`lastNotOne_jvp_one`), `1` is finite, and the domain here lies in the domain of `reduceJvp_correct`
(`lastNotOne_mem_reduceJvpDomain`). So commutativity is exactly what this refutation breaks,
and `reduceJvp_correct_needs_comm` states that it cannot be dropped. -/
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

/-- `reduceJvp_correct` is false without commutativity, already for operands of shape `(3,)`. -/
theorem reduceJvp_correct_needs_comm :
    ¬ ∀ (computation : Computation) (initValue : Scalar) (dimensions : List (Fin 1))
      (hd : dimensions.Nodup), IsMonoidOp computation.op initValue →
      (∀ a da : ℝ, computation.jvp (.fin a) initValue (.fin da) 0 = .fin da) →
      ((∃ c : ℝ, initValue = .fin c) ∨ ∀ d ∈ dimensions, (![3] : Shape 1) d ≠ 0) →
      IsJvpOfOn (reduceJvp (s := ![3]) computation initValue dimensions hd)
        (reduce computation initValue dimensions hd)
        (reduceJvpDomain computation initValue dimensions) := fun h =>
  reduceJvp_not_correct fun x hx =>
    h lastNotOne 1 [0] _ lastNotOne_isMonoidOp lastNotOne_jvp_one (.inl ⟨1, rfl⟩) x
      (lastNotOne_mem_reduceJvpDomain _ hx)

end Jax.Lax
