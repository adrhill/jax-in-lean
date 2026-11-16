module

public import Jax.Lax.Primal

/-!
# `reduce_sum`

Differentiation rules of the `reduce_sum` primitive of `jax/_src/lax/lax.py`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n}

/-- The JVP rule of `reduce_sum`, which applies the primitive to the tangent.
`jax/_src/lax/lax.py:9291`: `ad.deflinear2(reduce_sum_p, _reduce_sum_transpose_rule)` and
`jax/_src/interpreters/ad.py:950`: `return val_out, primitive.bind(*tangents, **params)` -/
def reduceSumJvp (axes : Finset (Fin n)) (operandDot _operand : Tensor (Index s)) :
    Tensor (Index (s ∘ keptAxes axes)) :=
  reduceSum axes operandDot

/-- `reduce_sum` on finite arrays. -/
private def reduceSumReal (axes : Finset (Fin n)) :
    (Index s → ℝ) →ₗ[ℝ] Index (s ∘ keptAxes axes) → ℝ where
  toFun x j := ∑ i with i.proj (keptAxes axes) = j, x i
  map_add' _ _ := by ext; simp [Finset.sum_add_distrib]
  map_smul' _ _ := by ext; simp [Finset.mul_sum]

private theorem reduceSum_ofReal (axes : Finset (Fin n)) (x : Index s → ℝ) :
    reduceSum axes (ofReal x) = ofReal (reduceSumReal axes x) := by
  funext j
  simp [reduceSum, ofReal, reduceSumReal, Scalar.fin_sum]

/-- The JVP rule of `reduce_sum` computes the derivative of `reduce_sum`. -/
theorem reduceSumJvp_correct (axes : Finset (Fin n)) :
    IsJvpOf (reduceSumJvp axes) (reduceSum (s := s) axes) :=
  isJvpOf_of_linear (reduceSumReal axes) (reduceSum_ofReal axes)

/-- The transpose rule of `reduce_sum`, which broadcasts the cotangent back over the summed axes.
The undefined primal `operand` is dropped: only its shape `s` is used.
`jax/_src/lax/lax.py:9204`:
```
def _reduce_sum_transpose_rule(cotangent, operand, *, axes, out_sharding):
  assert ad.is_undefined_primal(operand)
  input_shape = operand.aval.shape
  broadcast_dimensions = tuple(np.delete(np.arange(len(input_shape)), axes))
  result = broadcast_in_dim(
      cotangent, input_shape, broadcast_dimensions,
      out_sharding=operand.aval.sharding)
  assert result.shape == input_shape
  return [result]
``` -/
def reduceSumTranspose (axes : Finset (Fin n)) (cotangent : Tensor (Index (s ∘ keptAxes axes))) :
    Tensor (Index s) :=
  broadcastInDim s (keptAxes axes) (.refl _) (keptAxes_injective axes) cotangent

/-- The transpose rule of `reduce_sum` transposes `reduce_sum`. -/
theorem reduceSumTranspose_correct (axes : Finset (Fin n)) :
    IsTransposeOf (reduceSumTranspose axes) (reduceSum (s := s) axes) := by
  intro ct x
  refine ⟨reduceSumReal axes x, fun i => ct (i.proj (keptAxes axes)), reduceSum_ofReal axes x,
    .of_eq ?_, ?_⟩
  · funext i
    simp [reduceSumTranspose, broadcastInDim, ofReal, Index.unbroadcast_refl]
  · calc ∑ j, ct j * ∑ i with i.proj (keptAxes axes) = j, x i
        = ∑ j, ∑ i with i.proj (keptAxes axes) = j, ct (i.proj (keptAxes axes)) * x i := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun i hi => ?_
          rw [(Finset.mem_filter.1 hi).2]
      _ = ∑ i, ct (i.proj (keptAxes axes)) * x i := Finset.sum_fiberwise _ _ _

end Jax.Lax
