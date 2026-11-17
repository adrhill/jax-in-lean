module

public import Jax.Lax.Max
public import Jax.Lax.ReduceMax

/-!
# `jnp.max(x, initial=0.)` over an empty axis

`jnp.max(x, axis=axes, initial=0.)` traces to
```
b = reduce_max[axes=axes] x
c = max 0.0 b
```
If a reduced axis of `x` has size 0, `b` is `-∞` everywhere and `c` is the constant `0`,
so the derivative is `0` (`maxInitial_hasZeroJvp`).
Forward mode returns NaN instead (`maxInitialJvp_of_empty`):
the JVP rule of `reduce_max` divides by the number of positions of the maximum, which is `0`
(`Jax.Lax.reduceMaxJvp_of_isEmpty`), and the rule of `max` multiplies the NaN by `0`.
Reverse mode is right, because the cotangent of an empty `x` has no entries.
`bugs/reduce-max-jvp-empty-axis.json` has the reproducer, where the NaN reaches the Jacobian of a non-empty array.

The model compares with an array of zeros of the shape of `b` instead of the rank-0 constant `0.0`,
which broadcasts to the same array.
-/

@[expose] public section

namespace Jax.Jaxpr

open Jax.Lax

variable {n : ℕ} {s : Shape n}

/-- `jnp.max(x, axis=axes, initial=0.)`, as `max(0, reduce_max(x, axes))`. -/
noncomputable def maxInitial (axes : Finset (Fin n)) (x : Tensor (Index s)) :
    Tensor (Index (broadcastShape (s ∘ keptAxes axes) (s ∘ keptAxes axes))) :=
  Lax.max (.refl _) 0 (reduceMax axes x)

theorem reduceMaxJvpShape_ofFn (s : Shape n) (axes : Finset (Fin n)) :
    List.ofFn (reduceMaxJvpShape s axes) = List.ofFn (s ∘ keptAxes axes) :=
  congrArg List.ofFn (funext (reduceMaxJvpShape_eq s axes))

/-- Forward mode: the JVP rule of `reduce_max`, then the rule of `max` for its second argument.
The constant `0.0` has a symbolic zero tangent, so the rule of `max` for its first argument is not evaluated.
The tangent of `reduce_max` has the shape of `b` (`reduceMaxJvpShape_eq`) and is read as an array of that shape. -/
noncomputable def maxInitialJvp (axes : Finset (Fin n)) (g x : Tensor (Index s)) :=
  maxJvpRight (.refl _)
    (fun j => reduceMaxJvp axes g (reduceMax axes x) x (Index.castShape (reduceMaxJvpShape_ofFn s axes) j))
    0 (reduceMax axes x)

/-- An axis of size 0 leaves no index. -/
private theorem isEmpty_index (hs : ∃ d, s d = 0) : IsEmpty (Index s) := by
  obtain ⟨d, hd⟩ := hs
  exact ⟨fun i => by have := (i d).isLt; omega⟩

/-- Over an empty axis, `reduce_max` is `-∞` and `jnp.max(x, initial=0.)` is `0`. -/
theorem maxInitial_of_empty (axes : Finset (Fin n)) (hs : ∃ d, s d = 0) (x : Tensor (Index s)) :
    maxInitial axes x = 0 := by
  have := isEmpty_index hs
  funext j
  simp only [maxInitial, Lax.max, binop, reduceMax, Finset.univ_eq_empty, Finset.filter_empty,
    Finset.fold_empty]
  rfl

/-- Over an empty axis, `jnp.max(x, initial=0.)` is constant, so its derivative is `0`. -/
theorem maxInitial_hasZeroJvp (axes : Finset (Fin n)) (hs : ∃ d, s d = 0) :
    IsJvpOf (fun _ _ => (0 : Tensor (Index (broadcastShape (s ∘ keptAxes axes) (s ∘ keptAxes axes)))))
      (maxInitial (s := s) axes) := by
  intro x _
  refine ⟨fun _ => 0, 0, .of_forall fun y => ?_, hasFDerivAt_const _ _, fun g => .of_eq ?_⟩
  · rw [maxInitial_of_empty axes hs]
    rfl
  · rfl

/-- Over an empty axis, forward mode returns NaN: `(0 / 0) * 0`. -/
theorem maxInitialJvp_of_empty (axes : Finset (Fin n)) (hs : ∃ d, s d = 0) (g x : Tensor (Index s))
    (j : Index (broadcastShape (s ∘ keptAxes axes) (broadcastShape (s ∘ keptAxes axes) (s ∘ keptAxes axes)))) :
    maxInitialJvp axes g x j = .nan := by
  simp only [maxInitialJvp, maxJvpRight, mul, binop, reduceMaxJvp_of_isEmpty axes hs]
  rfl

/-- Forward mode of `jnp.max(x, initial=0.)` is wrong wherever a reduced axis has size 0
and the result is not empty. -/
theorem maxInitialJvp_not_correct (axes : Finset (Fin n)) (hs : ∃ d, s d = 0)
    (j₀ : Index (s ∘ keptAxes axes)) :
    ¬ IsJvpOf (maxInitialJvp axes) (maxInitial (s := s) axes) := by
  intro hj
  have := isEmpty_index hs
  obtain ⟨_, L, -, -, hrule⟩ := hj isEmptyElim (Set.mem_univ _)
  obtain ⟨hr, -, hv⟩ := hrule isEmptyElim
  let j : Index (broadcastShape (s ∘ keptAxes axes)
      (broadcastShape (s ∘ keptAxes axes) (s ∘ keptAxes axes))) :=
    fun d => ⟨j₀ d, by simp [broadcastShape]⟩
  have := hv j (fun d => ⟨j₀ d, by simp [broadcastShape]⟩) fun d => rfl
  rw [maxInitialJvp_of_empty axes hs] at this
  cases this

/-- The case of the reproducer: `jnp.max(x, initial=0.)` for `x` of shape `(0,)`. -/
theorem maxInitialJvp_not_correct_vector :
    ¬ IsJvpOf (maxInitialJvp (s := ![0]) {0}) (maxInitial (s := ![0]) {0}) :=
  maxInitialJvp_not_correct _ ⟨0, rfl⟩ fun d => (Fin.cast (by decide) d).elim0

end Jax.Jaxpr
