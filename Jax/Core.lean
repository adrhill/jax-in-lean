module

public import Mathlib.Analysis.Calculus.FDeriv.Pi

/-!
# Core definitions

Arrays are functions from a finite index type to the reals.
A JAX shape `(m, n)` corresponds to the index type `Fin m × Fin n`.
-/

@[expose] public section

namespace Jax

/-- A real-valued array with index type `ι`. -/
abbrev Tensor (ι : Type) := ι → ℝ

/-- `jvp` is a correct JVP rule for `f`:
at every point, `f` is differentiable and `jvp g x` is its derivative at `x` applied to the tangent `g`.

The tangent comes first, as in the rules passed to `ad.defjvp`. -/
def IsJvpOf {ι κ : Type} [Fintype ι] [Fintype κ]
    (jvp : Tensor ι → Tensor ι → Tensor κ) (f : Tensor ι → Tensor κ) : Prop :=
  ∀ x, ∃ L : Tensor ι →L[ℝ] Tensor κ, HasFDerivAt f L x ∧ ∀ g, L g = jvp g x

end Jax
