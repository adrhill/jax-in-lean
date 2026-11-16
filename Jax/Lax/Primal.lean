module

public import Jax.Core

/-!
# Primal semantics

What each `lax` primitive computes, and the helper functions that differentiation rules call.
The semantics are those of the `lax` docstrings and the [StableHLO spec](https://openxla.org/stablehlo/spec).
They live in one file because the rules of one primitive call other primitives.
-/

@[expose] public section

namespace Jax.Lax

variable {ι : Type} {m n : ℕ}

/-! ## Elementwise unary primitives -/

/-- `lax.neg`, elementwise negation. `jax/_src/lax/lax.py:248` -/
def neg (x : Tensor ι) : Tensor ι := fun i => -x i

/-- `lax.sin`, elementwise sine. `jax/_src/lax/lax.py:735`; `accuracy` is dropped -/
noncomputable def sin (x : Tensor ι) : Tensor ι := fun i => (x i).sin

/-- `lax.cos`, elementwise cosine. `jax/_src/lax/lax.py:768`; `accuracy` is dropped -/
noncomputable def cos (x : Tensor ι) : Tensor ι := fun i => (x i).cos

/-- `lax.log`, elementwise natural logarithm. `jax/_src/lax/lax.py:570`; `accuracy` is dropped -/
noncomputable def log (x : Tensor ι) : Tensor ι := fun i => (x i).log

/-- The loop of `_integer_pow`, which multiplies `acc` by `x ^ y` by repeated squaring.
`none` is the empty product. -/
noncomputable def integerPowLoop (acc : Option Scalar) (x : Scalar) (y : ℕ) : Option Scalar :=
  if y = 0 then acc else
    integerPowLoop (if y % 2 = 1 then some (match acc with | none => x | some a => a * x) else acc)
      (if y / 2 > 0 then x * x else x) (y / 2)
termination_by y
decreasing_by omega

/-- `_integer_pow` on one entry. The assertion `acc is not None` holds because `y ≠ 0`. -/
noncomputable def integerPowScalar (y : ℤ) (x : Scalar) : Scalar :=
  if y = 0 then 1 else
    let acc := (integerPowLoop none x y.natAbs).getD 0
    if y < 0 then 1 / acc else acc

/-- `lax.integer_pow`, the elementwise power `x ^ y` for an integer `y`. `jax/_src/lax/lax.py:968`.
XLA computes it as `_integer_pow` does, `jax/_src/lax/lax.py:5393`:
```
def _integer_pow(x, *, y):
  # This should be kept in sync with the jax2tf translation rule.
  if y == 0:
    return full_like(x, 1)
  is_reciprocal = y < 0
  if is_reciprocal:
    y = -y
  acc: Array | None = None
  while y > 0:
    if y & 1:
      acc = x if acc is None else mul(acc, x)
    y >>= 1
    if y > 0:
      # We don't call square because it calls integer_pow.
      x = mul(x, x)
  assert acc is not None
  return div(full_like(acc, 1), acc) if is_reciprocal else acc
```
`_integer_pow_lowering` emits `x`, `x * x`, `x * x * x` and `1 / x` for `y = 1, 2, 3, -1`,
which agree with `_integer_pow` because multiplication is commutative. -/
noncomputable def integerPow (y : ℤ) (x : Tensor ι) : Tensor ι := fun i => integerPowScalar y (x i)

/-- `mul(_const(x, c), x)`, the product with the rank-0 constant `c`.
`broadcasting_shape_rule` ignores operands of rank 0, `jax/_src/lax/lax.py:4508`,
so the result has the shape of `x`. -/
noncomputable def constMul (c : Scalar) (x : Tensor ι) : Tensor ι := fun i => c * x i

/-! ## Elementwise binary primitives -/

/-- An elementwise binary primitive, which broadcasts its arguments against each other.
The proof `h` is the shape check of `broadcasting_shape_rule`, `jax/_src/lax/lax.py:4503`. -/
def binop {sx sy : Shape n} (op : Scalar → Scalar → Scalar) (h : Broadcastable sx sy)
    (x : Tensor (Index sx)) (y : Tensor (Index sy)) : Tensor (Index (broadcastShape sx sy)) :=
  fun j => op (x (j.unbroadcast (broadcastsTo_left sx sy))) (y (j.unbroadcast h.right))

variable {sx sy : Shape n}

/-- `lax.mul`, elementwise product. `jax/_src/lax/lax.py:1275`; `out_dtype` is dropped -/
noncomputable def mul (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  binop (· * ·) h x y

/-- `lax.div`, elementwise quotient. `jax/_src/lax/lax.py:1327` -/
noncomputable def div (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  binop (· / ·) h x y

/-- `lax.max`, elementwise maximum. `jax/_src/lax/lax.py:1385` -/
noncomputable def max (h : Broadcastable sx sy) (x : Tensor (Index sx)) (y : Tensor (Index sy)) :
    Tensor (Index (broadcastShape sx sy)) :=
  binop Scalar.max h x y

/-- `lax.reshape` without `dimensions`: the entries of the operand in row-major order,
laid out in the shape `newSizes`.
`jax/_src/lax/lax.py:3053`; `out_sharding` is dropped, and `dimensions` other than `None` are not modelled.
The proof `h` is the constraint of [StableHLO `reshape`](https://openxla.org/stablehlo/spec#reshape)
that the sizes agree. `_reshape_shape_rule`, `jax/_src/lax/lax.py:8360`, does not check it. -/
def reshape {s : Shape m} (newSizes : Shape n) (h : ∏ d, s d = ∏ d, newSizes d)
    (operand : Tensor (Index s)) : Tensor (Index newSizes) :=
  fun j => operand (Index.rowMajor.symm (Fin.cast h.symm (Index.rowMajor j)))

/-! ## Polynomials -/

/-- Horner's rule for the coefficients `a 0, …, a K` at `x`, in the order of `_polynomial_lower`:
`acc` starts at the coefficient of highest degree, and each step computes `add(mul(acc, x), c)`.
`jax/_src/lax/lax.py:5297`:
```
  if len(coeffs) <= unroll_threshold:
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
      acc = add(mul(acc, x), c)
    return acc
``` -/
noncomputable def horner (x : Scalar) : {K : ℕ} → (Fin (K + 1) → Scalar) → Scalar
  | 0, a => a 0
  | _ + 1, a => horner x (Fin.tail a) * x + a 0

/-- `lax.polynomial` with at least one coefficient, `a 0 + x * (a 1 + … + x * a K)`,
which broadcasts `x` and all coefficients against each other.
With two or more coefficients this is the primitive `polynomial_p`.
The proof `h` is the shape check of `broadcasting_shape_rule`, `jax/_src/lax/lax.py:4503`.
`unroll` is dropped, and the semantics are those of the unrolled loop in `_polynomial_lower`.
`jax/_src/lax/lax.py:4104`:
```
  if len(c_args) == 1:
    out_shape = broadcasting_shape_rule('polynomial', typeof(x), typeof(c_args[0]))
    out_sharding = broadcasting_sharding_rule('polynomial', typeof(x), typeof(c_args[0]))
    return _maybe_broadcast(out_shape, c_args[0], out_sharding)
  kwargs = {} if unroll is None else {'unroll': unroll}
  return polynomial_p.bind(x, *c_args, **kwargs)
``` -/
noncomputable def polynomial {K : ℕ} {sx : Shape n} {sc : Fin (K + 1) → Shape n}
    (h : BroadcastableAll (sx :: List.ofFn sc)) (x : Tensor (Index sx))
    (coeffs : (k : Fin (K + 1)) → Tensor (Index (sc k))) :
    Tensor (Index (broadcastShapes (sx :: List.ofFn sc))) :=
  fun j => horner (x (j.unbroadcast (broadcastsTo_broadcastShapes h List.mem_cons_self)))
    fun k => coeffs k (j.unbroadcast (broadcastsTo_broadcastShapes h
      (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨k, rfl⟩))))

/-! ## Structural primitives -/

/-- `lax.broadcast_in_dim`: axis `d` of the operand becomes axis `broadcastDimensions d` of the result.
`jax/_src/lax/lax.py:2946`; `out_sharding` is dropped.
The proof `h` is the shape check of `_broadcast_in_dim_shape_rule`, `jax/_src/lax/lax.py:7474`.
The rule also rejects duplicates in `broadcast_dimensions`, which the semantics do not need. -/
def broadcastInDim {so : Shape m} (shape : Shape n) (broadcastDimensions : Fin m → Fin n)
    (h : BroadcastsTo so (shape ∘ broadcastDimensions)) (operand : Tensor (Index so)) :
    Tensor (Index shape) :=
  fun j => operand ((j.proj broadcastDimensions).unbroadcast h)

/-- `lax.reduce_sum`, the sum over `axes`. The result keeps the other axes in order.
`jax/_src/lax/lax.py:3412`; `out_sharding` is dropped -/
def reduceSum {s : Shape n} (axes : Finset (Fin n)) (operand : Tensor (Index s)) :
    Tensor (Index (s ∘ keptAxes axes)) :=
  fun j => ∑ i with i.proj (keptAxes axes) = j, operand i

/-! ## Helpers of differentiation rules -/

/-- The axes of rank `n` that are listed in `dimensions`. -/
def dimsSet (n : ℕ) (dimensions : Finset ℕ) : Finset (Fin n) := {p | (p : ℕ) ∈ dimensions}

theorem card_dimsSet {N : ℕ} {dimensions : Finset ℕ} (h : ∀ i ∈ dimensions, i < N) :
    (dimsSet N dimensions).card = dimensions.card := by
  rw [← Finset.card_map Fin.valEmbedding]
  congr 1
  ext i
  simp only [Finset.mem_map, dimsSet, Finset.mem_filter, Finset.mem_univ, true_and,
    Fin.valEmbedding_apply]
  exact ⟨fun ⟨p, hp, hpi⟩ => hpi ▸ hp, fun hi => ⟨⟨i, h i hi⟩, hi, rfl⟩⟩

theorem card_compl_dimsSet {k : ℕ} {dimensions : Finset ℕ}
    (h : ∀ i ∈ dimensions, i < k + dimensions.card) :
    (dimsSet (k + dimensions.card) dimensions)ᶜ.card = k := by
  rw [Finset.card_compl, Fintype.card_fin, card_dimsSet h, Nat.add_sub_cancel]

/-- The shape `result_shape` of `expand_dims`: size 1 on the axes in `dims`,
and the axes of `s` in order on the others.
`jax/_src/lax/lax.py:3911`:
```
  result_shape = list(np.shape(array))
  for i in sorted(dims_set):
    result_shape.insert(i, 1)
``` -/
noncomputable def expandDimsShape {k n : ℕ} (s : Shape k) (dims : Finset (Fin n))
    (h : dimsᶜ.card = k) : Shape n :=
  fun p => if hp : p ∈ dims then 1 else
    s (Fin.cast h ((dimsᶜ.orderIsoOfFin rfl).symm ⟨p, Finset.mem_compl.2 hp⟩))

theorem expandDimsShape_broadcastsTo {k n : ℕ} (s : Shape k) (dims : Finset (Fin n))
    (h : dimsᶜ.card = k) :
    BroadcastsTo s (expandDimsShape s dims h ∘ (keptAxes dims ∘ Fin.cast h.symm)) :=
  fun d => Or.inr <| by
    simp only [Function.comp_apply, expandDimsShape, keptAxes_notMem, ↓reduceDIte,
      orderIsoOfFin_symm_keptAxes, Fin.cast_cast, Fin.cast_eq_self]

/-- `lax.expand_dims`, which inserts axes of size 1 at the positions `dimensions` of the result.
Negative axes are not modelled.
The proof `h` is the bounds check of `canonicalize_axis`, `jax/_src/util.py:458`.
`jax/_src/lax/lax.py:3904`:
```
def expand_dims(array: ArrayLike, dimensions: Sequence[int]) -> Array:
  """Insert any number of size 1 dimensions into an array."""
  if len(set(dimensions)) != len(dimensions):
    raise ValueError(f'repeated axis in lax.expand_dims: {dimensions}')
  ndim_out = np.ndim(array) + len(dimensions)
  dims = [canonicalize_axis(i, ndim_out) for i in dimensions]
  if len(set(dims)) != len(dims):  # check again after canonicalizing
    raise ValueError(f'repeated axis in lax.expand_dims: {dims}')
  dims_set = frozenset(dims)
  result_shape = list(np.shape(array))
  for i in sorted(dims_set):
    result_shape.insert(i, 1)
  broadcast_dims = [i for i in range(ndim_out) if i not in dims_set]
  return broadcast_in_dim(array, result_shape, broadcast_dims)
``` -/
noncomputable def expandDims {k : ℕ} {s : Shape k} (dimensions : Finset ℕ)
    (h : ∀ i ∈ dimensions, i < k + dimensions.card) (array : Tensor (Index s)) :
    Tensor (Index (expandDimsShape s (dimsSet _ dimensions) (card_compl_dimsSet h))) :=
  broadcastInDim _ (keptAxes (dimsSet _ dimensions) ∘ Fin.cast (card_compl_dimsSet h).symm)
    (expandDimsShape_broadcastsTo _ _ _) array

open scoped Classical in
/-- One entry of `_balanced_cmp`: `1` if `a > b`, `1/2` if `a == b`, `0` otherwise. -/
noncomputable def balancedCmpScalar (a b : Scalar) : Scalar :=
  if a.gt b then 1 else if a.eq b then .fin (1 / 2) else 0

/-- Helper of the JVP rules of `max` and `min`.
The boolean masks and `select` are folded into an `if`.
`jax/_src/lax/lax.py:10489`:
```
def _balanced_cmp(x, y):
  # 1.0 if x > y, 0.5 if x == y, 0.0 if x < y or NaN
  gt_mask = gt(x, y)
  eq_mask = eq(x, y)
  ones = full_like(gt_mask, 1, dtype=x.dtype)
  zeros = full_like(gt_mask, 0, dtype=x.dtype)
  half = full_like(gt_mask, 0.5, dtype=x.dtype)
  return select(gt_mask, ones, select(eq_mask, half, zeros))
``` -/
noncomputable def balancedCmp (h : Broadcastable sx sy) (x : Tensor (Index sx))
    (y : Tensor (Index sy)) : Tensor (Index (broadcastShape sx sy)) :=
  binop balancedCmpScalar h x y

theorem prod_unbroadcast {sa sx : Shape n} (h : BroadcastsTo sa sx) :
    ∏ a, (sx ∘ keptAxes {d | sx d ≠ sa d}) a = ∏ d, sa d := by
  refine Fintype.prod_of_injective _ (keptAxes_injective _) _ _ (fun d hd => ?_) fun a => ?_
  · obtain hd' : sx d ≠ sa d := by
      by_contra hc
      exact hd (keptAxes_surj (by simpa using hc))
    have := h d
    omega
  · have := keptAxes_notMem {d | sx d ≠ sa d} a
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_not] at this
    exact this

/-- `_unbroadcast`: the sum of `x` over the axes on which its shape differs from the shape `sa` of `aval`,
reshaped to `sa`. It transposes the implicit broadcast of an array of shape `sa` to the shape of `x`.
Only arrays of equal rank are modelled, and sharding is dropped.
If no axes differ, `_unbroadcast` skips `reduce_sum` or returns `x`, which computes the same array.
The proof `h` is the assertion under `enable_checks` that `aval` has size 1 on the axes in `dims`.
`jax/_src/lax/lax.py:4637`:
```
def _unbroadcast(aval, x):
  if not isinstance(aval, ShapedArray):
    raise TypeError(
        'transpose with implicit broadcasting of unshaped values. Got'
        f' {type(aval)}')
  x_shape = np.shape(x)
  if (core.definitely_equal_shape(aval.shape, x_shape) and
      aval.sharding == typeof(x).sharding):
    return x
  assert not aval.shape or len(x_shape) == len(aval.shape)
  if not aval.shape:
    return reduce_sum(x, list(range(len(x_shape))))
  else:
    dims = [i for i, (a, b) in enumerate(zip(x_shape, aval.shape))
            if not core.definitely_equal(a, b)]
    if config.enable_checks.value:
      assert all(aval.shape[i] == 1 for i in dims)
    x = reduce_sum(x, dims) if dims else x
    return reshape(x, aval.shape, out_sharding=aval.sharding)
``` -/
def unbroadcast {sx : Shape n} (sa : Shape n) (h : BroadcastsTo sa sx) (x : Tensor (Index sx)) :
    Tensor (Index sa) :=
  reshape sa (prod_unbroadcast h) (reduceSum {d | sx d ≠ sa d} x)

/-! ## Lemmas -/

/-- A binary primitive on arrays of equal shape acts entry by entry. -/
theorem binop_refl_same {s : Shape n} (op : Scalar → Scalar → Scalar) (x y : Tensor (Index s))
    {c : Tensor (Index s)} (hc : ∀ i, op (x i) (y i) = c i) :
    (binop op (Broadcastable.refl s) x y).Same c :=
  Tensor.same_of_shape_eq (fun d => by simp [broadcastShape]) fun i j hij => by
    rw [← hc j, binop, Index.unbroadcast_eq_of_val _ i j hij]

/-- The loop of `_integer_pow` on finite values multiplies `acc` by `x ^ y`. -/
theorem integerPowLoop_fin (x : ℝ) (y : ℕ) : ∀ acc : Option ℝ, (acc.isSome ∨ 0 < y) →
    integerPowLoop (acc.map .fin) (.fin x) y = some (.fin (acc.getD 1 * x ^ y)) := by
  induction y using Nat.strong_induction_on generalizing x with
  | _ y ih =>
  intro acc hacc
  rw [integerPowLoop.eq_def]
  by_cases hy : y = 0
  · subst hy
    cases acc <;> simp_all
  simp only [hy, ↓reduceIte]
  have key := ih (y / 2) (by omega) (if y / 2 > 0 then x * x else x)
    (if y % 2 = 1 then some (acc.getD 1 * x) else acc)
    (by split_ifs <;> simp_all; omega)
  have hacc' : (if y % 2 = 1 then some (match acc.map Scalar.fin with
      | none => Scalar.fin x | some a => a * Scalar.fin x) else acc.map .fin) =
      (if y % 2 = 1 then some (acc.getD 1 * x) else acc).map .fin := by
    cases acc <;> split_ifs <;> simp
  have hx : (if y / 2 > 0 then Scalar.fin x * .fin x else .fin x) =
      .fin (if y / 2 > 0 then x * x else x) := by
    split_ifs <;> rfl
  rw [hacc', hx, key]
  congr 2
  obtain ⟨k, hk⟩ : ∃ k, y / 2 = k := ⟨_, rfl⟩
  have hy' : y = 2 * k + y % 2 := by omega
  rw [hk]
  conv_rhs => rw [hy']
  rcases Nat.mod_two_eq_zero_or_one y with h2 | h2
  · have : k ≠ 0 := by omega
    simp only [h2, zero_ne_one, ↓reduceIte, Nat.pos_of_ne_zero this, add_zero]
    ring
  · simp only [h2, ↓reduceIte, Option.getD_some]
    split_ifs with hk0
    · ring
    · obtain rfl : k = 0 := by omega
      ring

/-- `integer_pow` of a finite value is finite where the power is defined. -/
theorem integerPowScalar_fin (y : ℤ) (x : ℝ) (h : 0 ≤ y ∨ x ≠ 0) :
    integerPowScalar y (.fin x) = .fin (x ^ y) := by
  have hloop := integerPowLoop_fin x y.natAbs none
  simp only [Option.map_none, Option.isSome_none, Bool.false_eq_true, false_or,
    Option.getD_none, one_mul] at hloop
  unfold integerPowScalar
  split_ifs with h0 hneg
  · subst h0
    rfl
  · have hx : x ≠ 0 := h.resolve_left (by omega)
    rw [hloop (by omega), Option.getD_some, Scalar.one_def,
      Scalar.fin_div_fin _ (pow_ne_zero _ hx)]
    congr 1
    rw [show y = -(y.natAbs : ℤ) by omega, zpow_neg, zpow_natCast, Int.natAbs_neg, Int.natAbs_natCast,
      one_div]
  · rw [hloop (by omega), Option.getD_some, show y = (y.natAbs : ℤ) by omega, zpow_natCast,
      Int.natAbs_natCast]

/-- `_unbroadcast` sums `x` over the indices that are broadcast from `i`. -/
theorem unbroadcast_apply {sa sx : Shape n} (h : BroadcastsTo sa sx) (x : Tensor (Index sx))
    (i : Index sa) : unbroadcast sa h x i = ∑ j with j.unbroadcast h = i, x j := by
  let D : Finset (Fin n) := {d | sx d ≠ sa d}
  have hkept : ∀ a, sx (keptAxes D a) = sa (keptAxes D a) := fun a => by
    have := keptAxes_notMem D a
    simpa [D] using this
  -- The index of the sum that `reshape` reads is `i` without the axes in `D`.
  let c : Index (sx ∘ keptAxes D) := fun a => ⟨i (keptAxes D a), by
    rw [Function.comp_apply, hkept]; exact (i _).isLt⟩
  have hc : Index.rowMajor.symm (Fin.cast (prod_unbroadcast h).symm (Index.rowMajor i)) = c := by
    rw [Equiv.symm_apply_eq]
    refine Fin.ext ?_
    rw [Fin.val_cast, Index.rowMajor_congr (t := sa ∘ keptAxes D) (fun a => hkept a)
      (j := i.proj (keptAxes D)) fun _ => rfl]
    refine (Index.rowMajor_proj (keptAxes_strictMono D) (fun d hd => ?_) i).symm
    have hd' : sx d ≠ sa d := by
      by_contra hc
      exact hd (keptAxes_surj (by simpa [D] using hc))
    have := h d
    omega
  simp only [unbroadcast, reshape, reduceSum]
  rw [hc]
  refine Finset.sum_congr (Finset.filter_congr fun j _ => ⟨fun hj => ?_, fun hj => ?_⟩) fun _ _ => rfl
  · refine Index.ext_val fun d => ?_
    rw [Index.unbroadcast_val]
    split_ifs with h1
    · have := (i d).isLt; omega
    · have hd : d ∉ D := by have := h d; simp [D]; omega
      obtain ⟨a, rfl⟩ := keptAxes_surj hd
      exact congrArg (fun k => ((k a : Fin _) : ℕ)) hj
  · refine Index.ext_val fun a => ?_
    have hja := congrArg (fun k => (k (keptAxes D a) : ℕ)) hj
    have h2 := hkept a
    have := (j (keptAxes D a)).isLt
    have := (i (keptAxes D a)).isLt
    simp only [Index.unbroadcast_val] at hja
    show (j (keptAxes D a) : ℕ) = i (keptAxes D a)
    split_ifs at hja <;> omega

end Jax.Lax
