module

public import Jax.Lax.Mul
public import Jax.Lax.SelectN

/-!
# `select` with a NaN in the unused case, jax-ml/jax#20601

`jax.grad` of `β ↦ select(False, β * x, 1.0)` returns NaN for `x = NaN`, although the function is
the constant `1`. Forward mode returns `0`. For finite `x`, reverse mode returns `0`.
The jaxpr and its transpose are
```
c = mul β x
d = select_n False 1.0 c
ct_c = select_n False 0.0 ct
ct_β = mul ct_c x
```
This is not a bug of a rule. The transpose of `select_n` is correct everywhere
(`selectNTranspose_correct`) and returns the cotangent `0` for the unused case.
The transpose of `mul` multiplies it by `x = NaN`, and `0 * NaN = NaN` (`selectNanVjp_nan`).
The transpose rule of `mul` is correct for finite `x` (`mulTransposeLeft_correct`),
where reverse mode returns `0` (`selectNanVjp_finite`).
For `x = NaN`, `β ↦ β * x` is not a function of real arrays to real arrays,
so no transpose rule can be correct for it (`mul_nan_not_isTransposeOf`).
This is the behaviour of `jnp.where` that the JAX FAQ documents.
-/

@[expose] public section

namespace Jax.Jaxpr

open Jax.Lax

variable {n : ℕ} {s : Shape n} {w : FloatWidth}

/-- The predicate `False`, a scalar. -/
def falseWhich : Tensor .bool (Index (![] : Shape 0)) := fun _ => false

theorem whichShape_scalar (s : Shape n) : WhichShape (![] : Shape 0) s := .inl rfl

/-- The function of jax-ml/jax#20601, `select(False, β * x, 1.0)`, which is `select_n(False, 1.0, β * x)`. -/
noncomputable def selectNan (β x : Tensor (.float w) (Index s)) : Tensor (.float w) (Index (broadcastShape s s)) :=
  selectN rfl (whichShape_scalar _) falseWhich 1 (mul (.refl s) β x)

/-- Reverse mode in `β`: the transpose rules of `select_n` for the case `β * x`
and of `mul` for its first argument applied in turn to the cotangent `ct`. -/
noncomputable def selectNanVjp (x : Tensor (.float w) (Index s)) (ct : Tensor (.float w) (Index (broadcastShape s s))) :
    Tensor (.float w) (Index s) :=
  mulTransposeLeft (.refl s) x (selectNTranspose1 (whichShape_scalar _) falseWhich ct)

/-- The function is the constant `1`, also if `x` is NaN. -/
theorem selectNan_eq_one (β x : Tensor (.float w) (Index s)) : selectNan β x = 1 := by
  funext j
  simp [selectNan, selectN, falseWhich]

/-- Reverse mode sums the products of the cotangent `0` with entries of `x`
over the nonempty set of entries that are broadcast from `i`. -/
private theorem selectNanVjp_apply (x : Tensor (.float w) (Index s)) (ct : Tensor (.float w) (Index (broadcastShape s s)))
    (i : Index s) : ∃ S : Finset (Index (broadcastShape (broadcastShape s s) s)), S.Nonempty ∧
      ∃ f : Index (broadcastShape (broadcastShape s s) s) → Index s,
        selectNanVjp x ct i = ∑ j ∈ S, 0 * x (f j) := by
  rw [selectNanVjp, mulTransposeLeft, unbroadcast_apply]
  refine ⟨_, ?_, fun j => j.unbroadcast (mulTransposeLeft_broadcastsTo (.refl s)),
    Finset.sum_congr rfl fun j _ => ?_⟩
  · exact ⟨fun d => ⟨i d, by simp [broadcastShape]⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _,
      Index.unbroadcast_eq_of_val _ _ _ fun _ => rfl⟩⟩
  · simp only [mul, binop, selectNTranspose1, selectN, falseWhich, Bool.false_eq_true, ↓reduceIte,
      Pi.zero_apply]

private theorem nan_add (a : Scalar w) : Scalar.nan + a = .nan := by
  cases a <;> rfl

/-- For `x = NaN`, reverse mode returns NaN: the cotangent `0` of the unused case times NaN. -/
theorem selectNanVjp_nan (ct : Index (broadcastShape s s) → ℝ) :
    selectNanVjp (w := w) (fun _ => .nan) (ofReal ct) = fun _ => .nan := by
  funext i
  obtain ⟨S, ⟨a, ha⟩, f, h⟩ := selectNanVjp_apply (fun _ => .nan) (ofReal ct) i
  rw [h, ← Finset.add_sum_erase S _ ha]
  exact nan_add _

/-- For finite `x`, reverse mode returns `0`, the transpose of the zero derivative. -/
theorem selectNanVjp_finite (x : Index s → ℝ) (ct : Index (broadcastShape s s) → ℝ) :
    selectNanVjp (w := w) (ofReal x) (ofReal ct) = ofReal 0 := by
  funext i
  obtain ⟨S, -, f, h⟩ := selectNanVjp_apply (ofReal x) (ofReal ct) i
  rw [h, Finset.sum_eq_zero fun j _ => ?_]
  · rfl
  · simp [ofReal, Scalar.zero_def]

/-- No transpose rule is correct for `β ↦ β * NaN`: it does not map real arrays to real arrays. -/
theorem mul_nan_not_isTransposeOf [Nonempty (Index s)]
    (tr : Tensor (.float w) (Index (broadcastShape s s)) → Tensor (.float w) (Index s)) :
    ¬ IsTransposeOf tr (mul (.refl s) · fun _ => .nan) := by
  intro htr
  obtain ⟨y, -, hy, -⟩ := htr 0 0
  obtain ⟨i⟩ := ‹Nonempty (Index s)›
  have := congrFun hy fun d => ⟨i d, by simp [broadcastShape]⟩
  cases this

end Jax.Jaxpr
