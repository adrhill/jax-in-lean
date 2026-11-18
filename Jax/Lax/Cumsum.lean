module

public import Jax.Lax.Dynamic

/-!
# `cumsum`

Differentiation rules of the `cumsum` primitive of `jax/_src/lax/control_flow/loops.py`.

`cumsum` is linear, and its transpose rule flips `reverse` (`cumsumTranspose_correct`).
Its batching rule shifts `axis` past the batch axis (`cumsumBatch_correct`).
The other cumulative reductions, `cumprod`, `cummax`, `cummin` and `cumlogsumexp`,
are differentiated through `associative_scan` and are not modelled.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n} {w : FloatWidth}

/-- The JVP rule of `cumsum`, which applies the primitive to the tangent.
`jax/_src/lax/control_flow/loops.py:3251`: `ad.deflinear2(cumsum_p, _cumsum_transpose_rule)` and
`jax/_src/interpreters/ad.py:950`: `return val_out, primitive.bind(*tangents, **params)` -/
def cumsumJvp (axis : Fin n) (reverse : Bool) (operandDot _operand : Tensor (.float w) (Index s)) :
    Tensor (.float w) (Index s) :=
  cumsum axis reverse operandDot

/-- The transpose rule of `cumsum`, the cumulative sum in the other direction.
`jax/_src/lax/control_flow/loops.py:3196`:
```
def _cumsum_transpose_rule(t, operand, *, axis: int, reverse: bool):
  return [cumsum(t, axis=axis, reverse=not reverse)]
``` -/
def cumsumTranspose (axis : Fin n) (reverse : Bool) (t : Tensor (.float w) (Index s)) : Tensor (.float w) (Index s) :=
  cumsum axis (!reverse) t

/-- `cumsum` on finite arrays. -/
private def cumsumReal (axis : Fin n) (reverse : Bool) : (Index s → ℝ) →ₗ[ℝ] Index s → ℝ where
  toFun x j := ∑ i with (∀ d, d ≠ axis → i d = j d) ∧
    (if reverse then j axis ≤ i axis else i axis ≤ j axis), x i
  map_add' _ _ := by ext; simp [Finset.sum_add_distrib]
  map_smul' _ _ := by ext; simp [Finset.mul_sum]

private theorem cumsum_ofReal (axis : Fin n) (reverse : Bool) (x : Index s → ℝ) :
    cumsum (w := w) axis reverse (ofReal x) = ofReal (cumsumReal axis reverse x) := by
  funext j
  simp [cumsum, ofReal, cumsumReal, Scalar.fin_sum]

/-- The JVP rule of `cumsum` computes the derivative of `cumsum`. -/
theorem cumsumJvp_correct (axis : Fin n) (reverse : Bool) :
    IsJvpOf (cumsumJvp axis reverse) (cumsum (s := s) (w := w) axis reverse) :=
  isJvpOf_of_linear (cumsumReal axis reverse) (cumsum_ofReal axis reverse)

/-- The transpose rule of `cumsum` transposes `cumsum`. -/
theorem cumsumTranspose_correct (axis : Fin n) (reverse : Bool) :
    IsTransposeOf (cumsumTranspose axis reverse) (cumsum (s := s) (w := w) axis reverse) := by
  intro ct x
  refine ⟨cumsumReal axis reverse x, cumsumReal axis (!reverse) ct, cumsum_ofReal axis reverse x,
    .of_eq (cumsum_ofReal axis (!reverse) ct), ?_⟩
  simp only [cumsumReal, LinearMap.coe_mk, AddHom.coe_mk, Finset.mul_sum, Finset.sum_mul,
    Finset.sum_filter]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  -- Entry `i` is summed into entry `j` exactly if `j` is summed into `i` in the other direction.
  have hiff : ((∀ d, d ≠ axis → i d = j d) ∧ (if reverse then j axis ≤ i axis else i axis ≤ j axis)) ↔
      ((∀ d, d ≠ axis → j d = i d) ∧
        (if (!reverse) = true then i axis ≤ j axis else j axis ≤ i axis)) := by
    cases reverse <;> simp [eq_comm]
  simp only [hiff]
  split_ifs <;> simp

/-! ## Batching rule -/

/-- The batching rule of `cumsum`, which keeps the batch axis in place.
The rule is only called for a batched operand: `axis < None` raises.
`jax/_src/lax/control_flow/loops.py:3213`:
```
def _cumred_batch_rule(prim, batched_args, batch_dims, *, axis: int,
                       reverse: bool):
  operand, = batched_args
  bdim, = batch_dims
  axis = axis if axis < bdim else axis + 1
  return prim.bind(operand, axis=axis, reverse=reverse), bdim
``` -/
def cumsumBatch (axis : ℕ) (reverse : Bool) (operand : Batched) : Option Batched := do
  let bdim ← operand.bdim
  let out ← operand.val.cumsum (if axis < bdim then axis else axis + 1) reverse
  some ⟨out, some bdim⟩

private theorem insertNth_succAbove {s : Shape (n + 1)} (p : Fin (n + 1)) (i : Fin (s p))
    (k : Index (Fin.removeNth p s)) (e : Fin n) :
    Fin.insertNth (α := fun d => Fin (s d)) p i k (p.succAbove e) = k e :=
  Fin.insertNth_apply_succAbove (α := fun d => Fin (s d)) p i k e

private theorem insertNth_same {s : Shape (n + 1)} (p : Fin (n + 1)) (i : Fin (s p))
    (k : Index (Fin.removeNth p s)) : Fin.insertNth (α := fun d => Fin (s d)) p i k p = i :=
  Fin.insertNth_apply_same (α := fun d => Fin (s d)) p i k

/-- `cumsum` along an axis other than `p` acts on every entry along `p` separately. -/
private theorem cumsum_insertNth {s : Shape (n + 1)} (p : Fin (n + 1)) (axis : Fin n)
    (reverse : Bool) (x : Tensor (.float w) (Index s)) (i : Fin (s p)) (j : Index (Fin.removeNth p s)) :
    cumsum (p.succAbove axis) reverse x (Fin.insertNth (α := fun d => Fin (s d)) p i j) =
      cumsum (s := Fin.removeNth p s) axis reverse
        (fun k => x (Fin.insertNth (α := fun d => Fin (s d)) p i k)) j := by
  simp only [cumsum]
  symm
  refine Finset.sum_nbij' (fun k => Fin.insertNth (α := fun d => Fin (s d)) p i k)
    (fun k e => k (p.succAbove e)) (fun k hk => ?_) (fun k hk => ?_)
    (fun k _ => ?_) (fun k hk => ?_) fun _ _ => rfl
  · obtain ⟨h1, h2⟩ := (Finset.mem_filter.1 hk).2
    refine Finset.mem_filter.2 ⟨Finset.mem_univ _, fun e he => ?_, ?_⟩
    · rcases Fin.eq_self_or_eq_succAbove p e with rfl | ⟨e', rfl⟩
      · rw [insertNth_same, insertNth_same]
      · rw [insertNth_succAbove, insertNth_succAbove]
        exact h1 e' fun hc => he (hc ▸ rfl)
    · rw [insertNth_succAbove, insertNth_succAbove]
      exact h2
  · obtain ⟨h1, h2⟩ := (Finset.mem_filter.1 hk).2
    refine Finset.mem_filter.2 ⟨Finset.mem_univ _, fun e he => ?_, ?_⟩
    · have := h1 (p.succAbove e) fun hc => he (Fin.succAbove_right_injective hc)
      rwa [insertNth_succAbove] at this
    · rwa [insertNth_succAbove] at h2
  · exact funext fun e => insertNth_succAbove p i k e
  · obtain ⟨h1, -⟩ := (Finset.mem_filter.1 hk).2
    have hp := h1 p (Fin.succAbove_ne p axis).symm
    rw [insertNth_same] at hp
    rw [← hp]
    exact Fin.insertNth_self_removeNth (α := fun d => Fin (s d)) p k

/-- The batching rule of `cumsum` is correct. -/
theorem cumsumBatch_correct (axis : ℕ) (reverse : Bool) :
    IsBatchOf₁ (cumsumBatch axis reverse) (Arr.cumsum · axis reverse) := by
  rintro size ⟨⟨shape, val⟩, bdim⟩ hsize hHas hb hf
  obtain ⟨d, rfl⟩ := Option.isSome_iff_exists.1 hb
  rename_i rank dt
  obtain ⟨hd, hshape⟩ : ∃ h : d < rank, shape ⟨d, h⟩ = size := hHas d rfl
  obtain ⟨m, rfl⟩ : ∃ m, rank = m + 1 := ⟨rank - 1, by omega⟩
  let p : Fin (m + 1) := ⟨d, hd⟩
  have hget : ∀ i (hi : i < size), Batched.get ⟨⟨shape, val⟩, some d⟩ i = some ⟨Fin.removeNth p shape,
      fun j => val (Fin.insertNth (α := fun e => Fin (shape e)) p ⟨i, hshape ▸ hi⟩ j)⟩ :=
    fun i hi => Arr.unbatch_mk shape val p (hshape ▸ hi)
  -- `cumsum` accepts the entries of the batch, so they are floating-point arrays.
  obtain ⟨w, rfl⟩ : ∃ w, dt = .float w := by
    obtain ⟨a0, ha0, hsome⟩ := hf 0 hsize
    rw [hget 0 hsize] at ha0
    obtain rfl := Option.some.inj ha0
    cases dt <;> simp [Arr.cumsum] at hsome ⊢
  -- `cumsum` accepts the entries of the batch, so `axis` is one of their axes.
  have haxis : axis < m := by
    obtain ⟨a0, ha0, hsome⟩ := hf 0 hsize
    rw [hget 0 hsize] at ha0
    obtain rfl := Option.some.inj ha0
    by_contra hc
    simp [Arr.cumsum, hc] at hsome
  have haxis' : (if axis < d then axis else axis + 1) < m + 1 := by split_ifs <;> omega
  have hp : (⟨if axis < d then axis else axis + 1, haxis'⟩ : Fin (m + 1)) =
      p.succAbove ⟨axis, haxis⟩ := by
    refine Fin.ext ?_
    simp only [Fin.succAbove, Fin.lt_def, Fin.val_castSucc, p]
    split_ifs <;> rfl
  refine ⟨⟨⟨shape, cumsum ⟨_, haxis'⟩ reverse val⟩, some d⟩, ?_, fun i hi => ?_⟩
  · simp [cumsumBatch, Arr.cumsum, Nat.lt_succ_iff.1 haxis']
  · let xi : Tensor (.float w) (Index (Fin.removeNth p shape)) :=
      fun j => val (Fin.insertNth (α := fun e => Fin (shape e)) p ⟨i, hshape ▸ hi⟩ j)
    refine ⟨⟨_, xi⟩, ⟨_, cumsum ⟨axis, haxis⟩ reverse xi⟩, _, hget i hi, ?_,
      Arr.unbatch_mk shape _ p (hshape ▸ hi), .of_eq (funext fun j => ?_)⟩
    · simp [Arr.cumsum, haxis]
    · rw [hp]
      exact cumsum_insertNth p ⟨axis, haxis⟩ reverse val _ j

end Jax.Lax
