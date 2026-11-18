module

public import Jax.Lax.Primal
public import Mathlib.Analysis.SpecialFunctions.Pow.Deriv

/-!
# `integer_pow`

Differentiation rules of the `integer_pow` primitive of `jax/_src/lax/lax.py`.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n} {w : FloatWidth}

/-- The shape of the result of `_integer_pow_jvp`:
the shape of `g` for the exponents 0 and 1, where it returns `_zeros(g)` or `g`,
and the shape of `mul(g, …)` otherwise. -/
def integerPowJvpShape (s : Shape n) : ℤ → Shape n
  | .ofNat 0 | .ofNat 1 => s
  | .ofNat (_ + 2) | .negSucc _ => broadcastShape s s

/-- The JVP rule of `integer_pow` for the exponent `y`.
`jax/_src/lax/lax.py:5378`:
```
def _integer_pow_jvp(g, x, *, y):
  if y == 0:
    return _zeros(g)
  if y == 1:
    return g
  if y == 2:
    return mul(g, mul(_const(x, y), x))
  return mul(g, mul(_const(x, y), integer_pow(x, y - 1)))
``` -/
noncomputable def integerPowJvp :
    (y : ℤ) → (g x : Tensor (.float w) (Index s)) → Tensor (.float w) (Index (integerPowJvpShape s y))
  | .ofNat 0, _, _ => fun _ => 0
  | .ofNat 1, g, _ => g
  | .ofNat 2, g, x => mul (.refl s) g (constMul (.fin 2) x)
  | y@(.ofNat (_ + 3)), g, x | y@(.negSucc _), g, x =>
    mul (.refl s) g (constMul (.fin y) (integerPow (y - 1) x))

/-- The JVP rule of `integer_pow` computes the derivative of `integer_pow` for a nonnegative exponent,
and at arrays without zeros for a negative exponent. -/
theorem integerPowJvp_correct (y : ℤ) :
    IsJvpOfOn (integerPowJvp (s := s) (w := w) y) (integerPow y) {x | 0 ≤ y ∨ ∀ i, x i ≠ 0} := by
  intro x hx
  have hx' : ∀ i, 0 ≤ y ∨ x i ≠ 0 := fun i => hx.imp id (· i)
  refine ⟨fun z i => z i ^ y,
    ContinuousLinearMap.pi fun i => ((y : ℝ) * x i ^ (y - 1)) • ContinuousLinearMap.proj i,
    ?_, ?_, fun g => ?_⟩
  · have hopen : IsOpen {z : Index s → ℝ | 0 ≤ y ∨ ∀ i, z i ≠ 0} := by
      by_cases hy : 0 ≤ y
      · simp [hy]
      · simp only [hy, false_or, Set.ofPred_forall]
        exact isOpen_iInter_of_finite fun i => isOpen_ne_fun (continuous_apply i) continuous_const
    filter_upwards [hopen.mem_nhds hx] with z hz
    funext i
    exact integerPowScalar_fin y (z i) (hz.imp id (· i))
  · exact hasFDerivAt_pi.2 fun i =>
      hasFDerivAt_elementwise x i (hasDerivAt_zpow y (x i) (hx' i).symm)
  · have hL : ∀ i, (ContinuousLinearMap.pi fun i => ((y : ℝ) * x i ^ (y - 1)) •
        ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Index s => ℝ) i) g i =
        (y : ℝ) * x i ^ (y - 1) * g i := fun _ => rfl
    rcases y with (_ | _ | _ | k) | k
    · exact .of_eq (funext fun i => by simp [integerPowJvp, ofReal, Scalar.zero_def])
    · exact .of_eq (funext fun i => by simp [integerPowJvp, ofReal])
    · refine binop_refl_same _ _ _ fun i => ?_
      simp only [ofReal, constMul, Scalar.fin_mul_fin, hL]
      congr 1
      norm_num
      ring
    · refine binop_refl_same _ _ _ fun i => ?_
      simp only [ofReal, constMul, integerPow]
      rw [integerPowScalar_fin _ _ (Or.inl (by simp only [Int.ofNat_eq_natCast]; omega)),
        Scalar.fin_mul_fin, Scalar.fin_mul_fin, hL]
      congr 1
      ring
    · refine binop_refl_same _ _ _ fun i => ?_
      simp only [ofReal, constMul, integerPow,
        integerPowScalar_fin _ _ (Or.inr ((hx' i).resolve_left (by omega))), Scalar.fin_mul_fin, hL]
      congr 1
      ring

end Jax.Lax
