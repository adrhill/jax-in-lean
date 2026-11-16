module

public import Jax.Lax.Primal

/-!
# `broadcast_in_dim`

Differentiation rules of the `broadcast_in_dim` primitive of `jax/_src/lax/lax.py`.

The transpose rule `_broadcast_in_dim_transpose_rule` is not modelled yet.
-/

@[expose] public section

namespace Jax.Lax

variable {m n : ℕ} {so : Shape m}

/-- The JVP rule of `broadcast_in_dim`, which applies the primitive to the tangent.
`jax/_src/lax/lax.py:7609`:
```
    y_dot = broadcast_in_dim_p.bind(operand_dot, shape=shape,
                                    broadcast_dimensions=broadcast_dimensions,
                                    sharding=sharding)
``` -/
def broadcastInDimJvp (shape : Shape n) (broadcastDimensions : Fin m → Fin n)
    (h : BroadcastsTo so (shape ∘ broadcastDimensions)) (operandDot _operand : Tensor (Index so)) :
    Tensor (Index shape) :=
  broadcastInDim shape broadcastDimensions h operandDot

/-- The JVP rule of `broadcast_in_dim` computes the derivative of `broadcast_in_dim`. -/
theorem broadcastInDimJvp_correct (shape : Shape n) (broadcastDimensions : Fin m → Fin n)
    (h : BroadcastsTo so (shape ∘ broadcastDimensions)) :
    IsJvpOf (broadcastInDimJvp shape broadcastDimensions h)
      (broadcastInDim shape broadcastDimensions h) :=
  isJvpOf_of_linear
    { toFun := fun x j => x ((j.proj broadcastDimensions).unbroadcast h)
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl }
    fun _ => rfl

end Jax.Lax
