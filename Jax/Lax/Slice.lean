module

public import Jax.Lax.Pad

/-!
# `slice`

Differentiation rules of the `slice` primitive of `jax/_src/lax/slicing.py`.

The transpose rule pads the cotangent back to the shape of the operand.
It is correct also for strides other than 1 (`sliceTranspose_correct`).
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n} {w : FloatWidth}

/-- The JVP rule of `slice`, which applies the primitive to the tangent.
`jax/_src/lax/slicing.py:1419`:
```
def _slice_jvp_rule(primals, tangents, *, start_indices, limit_indices, strides):
  (p,), (t,) = primals, tangents
  primal_out = slice_p.bind(p, start_indices=start_indices,
                            limit_indices=limit_indices, strides=strides)
  if type(t) is ad.Zero:
    return primal_out, ad.p2tz(primal_out)
  else:
    tangent_out = slice_p.bind(t, start_indices=start_indices,
                               limit_indices=limit_indices, strides=strides)
    return primal_out, tangent_out
``` -/
def sliceJvp (startIndices limitIndices strides : Fin n → ℕ) (hl : ∀ d, limitIndices d ≤ s d)
    (hs : ∀ d, startIndices d ≤ limitIndices d) (hst : ∀ d, 0 < strides d)
    (operandDot _operand : Tensor (.float w) (Index s)) :
    Tensor (.float w) (Index (sliceShape startIndices limitIndices strides)) :=
  slice startIndices limitIndices strides hl hs hst operandDot

/-- `real_limits` of the transpose rule of `slice`: one more than the last position that the slice reads,
and `start_indices` if it reads none.
`jax/_src/lax/slicing.py:1444`:
```
      real_limits = np.add(
        start_indices,
        np.where(np.array(out_ct.shape) == 0, 0,
                 np.add(1, np.multiply(np.subtract(out_ct.shape, 1), strides))))
``` -/
def sliceRealLimits (startIndices limitIndices strides : Fin n → ℕ) : Fin n → ℕ :=
  fun d => startIndices d + if sliceShape startIndices limitIndices strides d = 0 then 0
    else 1 + (sliceShape startIndices limitIndices strides d - 1) * strides d

/-- The padding of the transpose rule fills the cotangent up to the size of the operand. -/
theorem slice_pad_eq (s : Shape n) (startIndices limitIndices strides : Fin n → ℕ)
    (hst : ∀ d, 0 < strides d) (d : Fin n) :
    (startIndices d : ℤ) + ((s d : ℤ) - sliceRealLimits startIndices limitIndices strides d) +
      dilateDim (sliceShape startIndices limitIndices strides d) (strides d - 1 + 1) = s d := by
  have hk : strides d - 1 + 1 = strides d := Nat.sub_add_cancel (hst d)
  simp only [sliceRealLimits, dilateDim, hk]
  rcases Nat.eq_zero_or_pos (sliceShape startIndices limitIndices strides d) with h0 | hpos
  · have := hst d
    simp only [h0, ↓reduceIte, Nat.cast_zero, zero_sub, mul_neg, mul_one]
    omega
  · obtain ⟨m, hm⟩ : ∃ m, sliceShape startIndices limitIndices strides d = m + 1 :=
      ⟨_, (Nat.sub_add_cancel hpos).symm⟩
    have h0 : (0 : ℤ) ≤ (strides d : ℤ) * m := by positivity
    simp only [hm, Nat.add_eq_zero_iff, one_ne_zero, and_false, ↓reduceIte, Nat.add_sub_cancel]
    push_cast
    rw [show (1 : ℤ) + (strides d : ℤ) * ((m : ℤ) + 1 - 1) = 1 + (strides d : ℤ) * m by ring,
      Int.toNat_of_nonneg (by omega)]
    ring

/-- The shape check of the `pad` in the transpose rule of `slice`. -/
theorem sliceTranspose_pad (s : Shape n) (startIndices limitIndices strides : Fin n → ℕ)
    (hst : ∀ d, 0 < strides d) (d : Fin n) :
    0 ≤ (startIndices d : ℤ) + ((s d : ℤ) - sliceRealLimits startIndices limitIndices strides d) +
      dilateDim (sliceShape startIndices limitIndices strides d) (strides d - 1 + 1) := by
  rw [slice_pad_eq s startIndices limitIndices strides hst d]
  exact Int.natCast_nonneg _

/-- The transpose rule of `slice`, which pads the cotangent with zeros to the shape of the operand:
`start_indices` zeros before, `strides - 1` zeros between the entries, and the rest after.
The accumulator `operand` is replaced by its shape `s`, and a `RefAccum` is not modelled.
For strides 1, `real_limits` is `limit_indices`, and both branches compute the same padding.
`jax/_src/lax/slicing.py:1430`:
```
def _slice_transpose_fancy(out_ct, operand, *, start_indices, limit_indices, strides):
  ...
    operand_aval, = lax_utils.ensure_shaped(operand.aval)
    if strides is None or np.all(np.equal(strides, 1)):
      pads = zip(start_indices, np.subtract(operand_aval.shape, limit_indices),
                 (0,) * len(start_indices))
    else:
      real_limits = np.add(
        start_indices,
        np.where(np.array(out_ct.shape) == 0, 0,
                 np.add(1, np.multiply(np.subtract(out_ct.shape, 1), strides))))
      pads = zip(start_indices, np.subtract(operand_aval.shape, real_limits),
                 np.subtract(strides, 1))
    operand.accum(lax.pad(out_ct, lax._const(out_ct, 0), pads))
``` -/
noncomputable def sliceTranspose (s : Shape n) (startIndices limitIndices strides : Fin n → ℕ)
    (hst : ∀ d, 0 < strides d)
    (outCt : Tensor (.float w) (Index (sliceShape startIndices limitIndices strides))) :
    Tensor (.float w) (Index (padShape (sliceShape startIndices limitIndices strides)
      (fun d => (startIndices d : ℤ))
      (fun d => (s d : ℤ) - sliceRealLimits startIndices limitIndices strides d)
      (fun d => strides d - 1))) :=
  pad (sv := ![]) _ _ _ (sliceTranspose_pad s startIndices limitIndices strides hst) outCt 0

/-! ## Proofs -/

/-- The position of the operand that `slice` reads for the entry `j` of the result. -/
private def source (startIndices limitIndices strides : Fin n → ℕ) (hl : ∀ d, limitIndices d ≤ s d)
    (hst : ∀ d, 0 < strides d) (j : Index (sliceShape startIndices limitIndices strides)) :
    Index s :=
  fun d => ⟨startIndices d + j d * strides d, slice_index_lt (hl d) (hst d) (j d).isLt⟩

private theorem source_injective (startIndices limitIndices strides : Fin n → ℕ)
    (hl : ∀ d, limitIndices d ≤ s d) (hst : ∀ d, 0 < strides d) :
    Function.Injective (source startIndices limitIndices strides hl hst) := fun j j' hjj =>
  Index.ext_val fun d => by
    have := congrArg (fun i : Index s => (i d : ℕ)) hjj
    simp only [source] at this
    exact Nat.eq_of_mul_eq_mul_right (hst d) (by omega)

/-- `slice` on finite arrays. -/
private def sliceReal (startIndices limitIndices strides : Fin n → ℕ)
    (hl : ∀ d, limitIndices d ≤ s d) (hst : ∀ d, 0 < strides d) :
    (Index s → ℝ) →ₗ[ℝ] Index (sliceShape startIndices limitIndices strides) → ℝ where
  toFun x j := x (source startIndices limitIndices strides hl hst j)
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

private theorem slice_ofReal (startIndices limitIndices strides : Fin n → ℕ)
    (hl : ∀ d, limitIndices d ≤ s d) (hs : ∀ d, startIndices d ≤ limitIndices d)
    (hst : ∀ d, 0 < strides d) (x : Index s → ℝ) :
    slice (d := .float w) startIndices limitIndices strides hl hs hst (ofReal x) =
      ofReal (sliceReal startIndices limitIndices strides hl hst x) :=
  rfl

/-- The transpose rule on finite arrays: the cotangent at the positions that `slice` reads,
and zero elsewhere. -/
private theorem sliceTranspose_same (startIndices limitIndices strides : Fin n → ℕ)
    (hl : ∀ d, limitIndices d ≤ s d) (hst : ∀ d, 0 < strides d)
    (ct : Index (sliceShape startIndices limitIndices strides) → ℝ) :
    (sliceTranspose (w := w) s startIndices limitIndices strides hst (ofReal ct)).Same
      (ofReal (w := w) (Function.extend (source startIndices limitIndices strides hl hst) ct 0)) := by
  refine Tensor.same_of_shape_eq (fun d => ?_) fun i' i hii => ?_
  · simp only [padShape, slice_pad_eq s startIndices limitIndices strides hst d, Int.toNat_natCast]
  · have hiff : ∀ j : Index (sliceShape startIndices limitIndices strides),
        (∀ d, ((i' d : ℕ) : ℤ) =
          (startIndices d : ℤ) + (j d : ℕ) * (((strides d - 1 : ℕ) : ℤ) + 1)) ↔
        source startIndices limitIndices strides hl hst j = i := fun j => by
      have hk : ∀ d, ((strides d - 1 : ℕ) : ℤ) + 1 = strides d := fun d => by
        have := hst d
        omega
      simp only [hk, hii]
      refine ⟨fun hj => Index.ext_val fun d => ?_, fun hj d => ?_⟩
      · exact_mod_cast (hj d).symm
      · rw [← hj]
        simp [source]
    simp only [sliceTranspose, ofReal]
    by_cases hex : ∃ j, source startIndices limitIndices strides hl hst j = i
    · obtain ⟨j, rfl⟩ := hex
      rw [pad_apply_of_eq _ _ _ ((hiff j).2 rfl),
        (source_injective startIndices limitIndices strides hl hst).extend_apply]
      rfl
    · rw [pad_apply_of_not_exists _ _ _ fun ⟨j, hj⟩ => hex ⟨j, (hiff j).1 hj⟩,
        Function.extend_apply' _ _ _ hex]
      rfl

/-! ## Correctness -/

/-- The JVP rule of `slice` computes the derivative of `slice`. -/
theorem sliceJvp_correct (startIndices limitIndices strides : Fin n → ℕ)
    (hl : ∀ d, limitIndices d ≤ s d) (hs : ∀ d, startIndices d ≤ limitIndices d)
    (hst : ∀ d, 0 < strides d) :
    IsJvpOf (sliceJvp startIndices limitIndices strides hl hs hst)
      (slice (d := .float w) startIndices limitIndices strides hl hs hst) :=
  isJvpOf_of_linear (sliceReal startIndices limitIndices strides hl hst)
    (slice_ofReal startIndices limitIndices strides hl hs hst)

/-- The transpose rule of `slice` transposes `slice`. -/
theorem sliceTranspose_correct (startIndices limitIndices strides : Fin n → ℕ)
    (hl : ∀ d, limitIndices d ≤ s d) (hs : ∀ d, startIndices d ≤ limitIndices d)
    (hst : ∀ d, 0 < strides d) :
    IsTransposeOf (sliceTranspose s startIndices limitIndices strides hst)
      (slice (d := .float w) startIndices limitIndices strides hl hs hst) := by
  intro ct x
  refine ⟨_, _, slice_ofReal startIndices limitIndices strides hl hs hst x,
    sliceTranspose_same startIndices limitIndices strides hl hst ct, ?_⟩
  refine Fintype.sum_of_injective _ (source_injective startIndices limitIndices strides hl hst) _ _
    (fun i hi => ?_) fun j => ?_
  · rw [Function.extend_apply' _ _ _ hi]
    simp
  · rw [(source_injective startIndices limitIndices strides hl hst).extend_apply]
    rfl

end Jax.Lax
