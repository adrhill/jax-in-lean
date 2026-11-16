module

public import Jax.Core
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# `sin`

Mirrors the `sin` primitive of `jax/_src/lax/lax.py`.
-/

@[expose] public section

namespace Jax.Lax

variable {ι : Type}

/-- `lax.sin`, elementwise sine. `jax/_src/lax/lax.py:735` -/
noncomputable def sin (x : Tensor ι) : Tensor ι := fun i => Real.sin (x i)

/-- `lax.cos`, elementwise cosine. `jax/_src/lax/lax.py:768` -/
noncomputable def cos (x : Tensor ι) : Tensor ι := fun i => Real.cos (x i)

/-- `lax.mul`, elementwise product of arrays of equal shape. `jax/_src/lax/lax.py:5581`;
`out_dtype` is dropped -/
def mul (x y : Tensor ι) : Tensor ι := fun i => x i * y i

/-- The JVP rule of `sin`.
`jax/_src/lax/lax.py:4928`: `ad.defjvp(sin_p, lambda g, x, accuracy: mul(g, cos(x, accuracy=accuracy)))` -/
noncomputable def sinJvp (g x : Tensor ι) : Tensor ι := mul g (cos x)

/-- The JVP rule of `sin` computes the derivative of `sin`. -/
theorem sinJvp_correct [Fintype ι] : IsJvpOf sinJvp (sin (ι := ι)) := by
  intro x
  refine ⟨ContinuousLinearMap.pi fun i => Real.cos (x i) • ContinuousLinearMap.proj i, ?_, ?_⟩
  · rw [show sin = fun (x : Tensor ι) i => Real.sin (x i) from rfl, hasFDerivAt_pi]
    intro i
    have hproj : HasFDerivAt (fun y : Tensor ι => y i) (ContinuousLinearMap.proj (R := ℝ) i) x :=
      hasFDerivAt_apply i x
    exact (Real.hasDerivAt_sin (x i)).comp_hasFDerivAt x hproj
  · intro g
    ext i
    simp [sinJvp, mul, cos, mul_comm]

end Jax.Lax
