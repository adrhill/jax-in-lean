module

public import Jax.Lax.Primal

/-!
# `cumsum`

Differentiation rules of the `cumsum` primitive of `jax/_src/lax/control_flow/loops.py`.

`cumsum` is linear, and its transpose rule flips `reverse` (`cumsumTranspose_correct`).
The other cumulative reductions, `cumprod`, `cummax`, `cummin` and `cumlogsumexp`,
are differentiated through `associative_scan` and are not modelled.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n}

/-- The JVP rule of `cumsum`, which applies the primitive to the tangent.
`jax/_src/lax/control_flow/loops.py:3251`: `ad.deflinear2(cumsum_p, _cumsum_transpose_rule)` and
`jax/_src/interpreters/ad.py:950`: `return val_out, primitive.bind(*tangents, **params)` -/
def cumsumJvp (axis : Fin n) (reverse : Bool) (operandDot _operand : Tensor (Index s)) :
    Tensor (Index s) :=
  cumsum axis reverse operandDot

/-- The transpose rule of `cumsum`, the cumulative sum in the other direction.
`jax/_src/lax/control_flow/loops.py:3196`:
```
def _cumsum_transpose_rule(t, operand, *, axis: int, reverse: bool):
  return [cumsum(t, axis=axis, reverse=not reverse)]
``` -/
def cumsumTranspose (axis : Fin n) (reverse : Bool) (t : Tensor (Index s)) : Tensor (Index s) :=
  cumsum axis (!reverse) t

/-- `cumsum` on finite arrays. -/
private def cumsumReal (axis : Fin n) (reverse : Bool) : (Index s → ℝ) →ₗ[ℝ] Index s → ℝ where
  toFun x j := ∑ i with (∀ d, d ≠ axis → i d = j d) ∧
    (if reverse then j axis ≤ i axis else i axis ≤ j axis), x i
  map_add' _ _ := by ext; simp [Finset.sum_add_distrib]
  map_smul' _ _ := by ext; simp [Finset.mul_sum]

private theorem cumsum_ofReal (axis : Fin n) (reverse : Bool) (x : Index s → ℝ) :
    cumsum axis reverse (ofReal x) = ofReal (cumsumReal axis reverse x) := by
  funext j
  simp [cumsum, ofReal, cumsumReal, Scalar.fin_sum]

/-- The JVP rule of `cumsum` computes the derivative of `cumsum`. -/
theorem cumsumJvp_correct (axis : Fin n) (reverse : Bool) :
    IsJvpOf (cumsumJvp axis reverse) (cumsum (s := s) axis reverse) :=
  isJvpOf_of_linear (cumsumReal axis reverse) (cumsum_ofReal axis reverse)

/-- The transpose rule of `cumsum` transposes `cumsum`. -/
theorem cumsumTranspose_correct (axis : Fin n) (reverse : Bool) :
    IsTransposeOf (cumsumTranspose axis reverse) (cumsum (s := s) axis reverse) := by
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

end Jax.Lax
