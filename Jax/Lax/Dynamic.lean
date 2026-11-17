module

public import Jax.Lax.Primal

/-!
# Primitives on arrays of any shape

The primitives of `Jax/Lax/Primal.lean` take their shape checks as proof arguments.
Python code that branches on shapes, such as a batching rule, calls them on arrays of any shape,
and the call raises if a check fails.
Here every primitive is a function of `Arr` that returns `none` if a shape check fails.
-/

@[expose] public section

namespace Jax

variable {m n : ℕ}

/-- The shape with the sizes `l`. -/
abbrev Shape.ofList (l : List ℕ) : Shape l.length := fun d => l.get d

instance {s t : Shape n} : Decidable (BroadcastsTo s t) :=
  inferInstanceAs (Decidable (∀ d, s d = 1 ∨ s d = t d))

instance {sw : Shape m} {s : Shape n} : Decidable (Lax.WhichShape sw s) :=
  inferInstanceAs (Decidable (m = 0 ∨ List.ofFn sw = List.ofFn s))

namespace Arr

/-- `lax.clamp(min, x, max)`. -/
noncomputable def clamp (lo x hi : Arr) : Option Arr :=
  if h : Lax.WhichShape lo.shape x.shape ∧ Lax.WhichShape hi.shape x.shape then
    some ⟨x.shape, Lax.clamp h.1 h.2 lo.val x.val hi.val⟩
  else none

/-- `lax.broadcast_in_dim(operand, shape, broadcast_dimensions)`.
`_broadcast_in_dim_shape_rule`, `jax/_src/lax/lax.py:7456`, also checks
that `broadcast_dimensions` has one entry per axis of the operand, each an axis of `shape`. -/
def broadcastInDim (operand : Arr) (shape broadcastDimensions : List ℕ) : Option Arr :=
  if h : broadcastDimensions.length = operand.rank ∧ ∀ i ∈ broadcastDimensions, i < shape.length then
    let bd : Fin operand.rank → Fin shape.length :=
      fun d => ⟨broadcastDimensions.get (Fin.cast h.1.symm d), h.2 _ (List.get_mem _ _)⟩
    if h' : BroadcastsTo operand.shape (Shape.ofList shape ∘ bd) ∧ Function.Injective bd then
      some ⟨Shape.ofList shape, Lax.broadcastInDim _ bd h'.1 h'.2 operand.val⟩
    else none
  else none

/-- `lax.transpose(operand, permutation)`.
`_transpose_shape_rule`, `jax/_src/lax/lax.py:8799`, checks that `permutation` is a permutation
of the axes of the operand. -/
noncomputable def transpose (operand : Arr) (permutation : List ℕ) : Option Arr :=
  if h : permutation.length = operand.rank ∧ ∀ i ∈ permutation, i < operand.rank then
    let p : Fin operand.rank → Fin operand.rank :=
      fun d => ⟨permutation.get (Fin.cast h.1.symm d), h.2 _ (List.get_mem _ _)⟩
    if hp : Function.Injective p then some ⟨operand.shape ∘ p, Lax.transpose p hp operand.val⟩
    else none
  else none

/-- `lax.broadcast(operand, sizes)`, which adds leading axes of the sizes `sizes`.
`jax/_src/lax/lax.py:2942`:
```
  dims = tuple(range(len(sizes), len(sizes) + np.ndim(operand)))
  return broadcast_in_dim(operand, tuple(sizes) + np.shape(operand), dims,
                          out_sharding=out_sharding)
``` -/
def broadcast (operand : Arr) (sizes : List ℕ) : Option Arr :=
  broadcastInDim operand (sizes ++ operand.shapeList) (List.range' sizes.length operand.rank)

/-- `lax.cumsum(operand, axis, reverse)`.
`_cumred_shape_rule`, `jax/_src/lax/control_flow/loops.py:3181`, checks that `axis` is an axis of the operand.
Negative axes, which it rejects, are not representable. -/
def cumsum (operand : Arr) (axis : ℕ) (reverse : Bool) : Option Arr :=
  if h : axis < operand.rank then some ⟨operand.shape, Lax.cumsum ⟨axis, h⟩ reverse operand.val⟩
  else none

/-! ## Helpers of batching rules -/

/-- `moveaxis(x, src, dst)` for one axis. Negative axes are not modelled.
`jax/_src/util.py:475`:
```
def moveaxis(x: Array, src: int | Sequence[int], dst: int | Sequence[int]) -> Array:
  if src == dst:
    return x
  ...
  perm = [i for i in range(np.ndim(x)) if i not in src]
  for d, s in sorted(zip(dst, src)):
    perm.insert(d, s)
  return x.transpose(perm)
``` -/
noncomputable def moveaxis (x : Arr) (src dst : ℕ) : Option Arr :=
  if src = dst then some x
  else transpose x (((List.range x.rank).filter (· ≠ src)).insertIdx dst src)

/-- `batching.broadcast(x, sz, axis, mesh_axis)`, which adds the axis `axis` of size `sz`.
Sharding is dropped.
`jax/_src/interpreters/batching.py:716`:
```
  shape = list(np.shape(x))
  shape.insert(axis, sz)
  broadcast_dims = tuple(np.delete(np.arange(len(shape)), axis))
  ...
    x, = spmd_names_insert_pvary(lax.broadcast_in_dim(
        x, shape, broadcast_dims, out_sharding=sharding))
``` -/
def batchBroadcast (x : Arr) (sz axis : ℕ) : Option Arr :=
  broadcastInDim x (x.shapeList.insertIdx axis sz) ((List.range (x.rank + 1)).eraseIdx axis)

/-- `batching.bdim_at_front(x, bdim, size)`. `jax/_src/interpreters/batching.py:776`:
```
def bdim_at_front(x, bdim, size, mesh_axis=None):
  if bdim is None:
    return broadcast(x, size, 0, mesh_axis=mesh_axis)
  else:
    return moveaxis(x, bdim, 0)
``` -/
noncomputable def bdimAtFront (x : Arr) (bdim : Option ℕ) (size : ℕ) : Option Arr :=
  match bdim with
  | none => batchBroadcast x size 0
  | some d => moveaxis x d 0

end Arr

/-- The size of the batch axis of the first batched argument:
`next(x.shape[i] for x, i in zip(batched_args, batch_dims) if i is not None)`. -/
def batchSize (args : List Batched) : Option ℕ :=
  args.findSome? fun x => x.bdim.bind fun d => x.val.shapeList[d]?

end Jax
