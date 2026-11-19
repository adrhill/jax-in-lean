module

public import Jax.Lax.Compare
public import Jax.Lax.Mul
public import Jax.Lax.SelectN
public import Jax.Lax.Sqrt

/-!
# `sqrt(relu(x))`, jax-ml/jax#6636

`jax.jacfwd` of `x ↦ sqrt(relu(x))` returns NaN at `x = -10`, and `jax.jacrev` returns `0`.
Near a negative `x` the function is the constant `0`, so its derivative is `0`:
forward mode is wrong there, and reverse mode is right.

This is not a bug of a rule. `jax.nn.relu` differentiates like `select(x > 0, x, 0)`,
and the jaxpr of the function and of its JVP is, for `jax.grad` of
`lambda x: lax.sqrt(lax.select(x > 0., x, jnp.zeros_like(x)))`:
```
b = gt x 0.0
c = select_n b 0.0 x
d = sqrt c
e = div 0.5 d
f = mul ct e
g = select_n b 0.0 f
```
At a negative `x` the rules of `gt` and `select_n` are correct (`gtJvp_correct`, `selectNJvp_correct`),
but `sqrt` is evaluated at `c = 0` (`relu_of_neg`), where no rule of `sqrt` is correct
(`sqrt_not_isJvpOfOn`): the chain rule needs the derivative of `sqrt` at `0`, which does not exist.
Forward mode multiplies the zero tangent of `c` by `0.5 / 0 = ∞`, which gives `0 * ∞ = NaN`
(`sqrtReluJvp_of_neg`). Reverse mode multiplies the cotangent by `∞`,
and the transpose of `select_n` discards the result where `b` is false (`sqrtReluVjp_of_neg`).

The model compares with an array of zeros of the shape of `x` instead of the rank-0 constant `0.0`,
which broadcasts to the same array.
-/

@[expose] public section

namespace Jax.Jaxpr

open Jax.Lax

variable {n : ℕ} {s : Shape n} {w : FloatWidth}

theorem whichShape_broadcastShape (s : Shape n) : WhichShape (broadcastShape s s) s :=
  .inr (congrArg List.ofFn (funext fun d => by simp [broadcastShape]))

/-- `relu` as `select_n(x > 0, 0, x)`. -/
noncomputable def relu (x : Tensor (.float w) (Index s)) : Tensor (.float w) (Index s) :=
  selectN rfl (whichShape_broadcastShape s) (gt (.refl s) x 0) 0 x

/-- The function of jax-ml/jax#6636, `sqrt(relu(x))`. -/
noncomputable def sqrtRelu (x : Tensor (.float w) (Index s)) : Tensor (.float w) (Index s) := sqrt (relu x)

/-- Forward mode, `jax.jvp`: the JVP rules of `select_n` and `sqrt` applied in turn,
and the tangent of the result instantiated.
`gt` has a boolean result, whose tangent is a symbolic zero, so `select_n` has no tangent for `which`,
and the constant case `0` has a symbolic zero tangent. -/
noncomputable def sqrtReluJvp (g x : Tensor (.float w) (Index s)) : Tensor (.float w) (Index (broadcastShape s s)) :=
  (standardJvp₁ (fun g c => sqrtJvp g (sqrt c) c) sqrt
    (selectNJvp rfl (whichShape_broadcastShape s) (gt (.refl s) x 0) (.zero (.of (.float w) s)) (.arr g) 0 x)
    (relu x)).inst

/-- Reverse mode: the transpose rules of `mul`, for the tangent `mul(g, div(0.5, ans))` of `sqrt`,
and of `select_n` applied in turn to the cotangent `ct`. -/
noncomputable def sqrtReluVjp (ct : Tensor (.float w) (Index (broadcastShape s s))) (x : Tensor (.float w) (Index s)) :
    Tensor (.float w) (Index s) :=
  selectNTranspose1 (whichShape_broadcastShape s) (gt (.refl s) x 0)
    (mulTransposeLeft (.refl s) (constDiv (.fin (1 / 2)) (sqrtRelu x)) ct)

/-! ## At a negative array -/

private theorem gt_of_neg {x : Index s → ℝ} (hx : ∀ i, x i < 0) (j : Index (broadcastShape s s)) :
    gt (w := w) (.refl s) (ofReal x) 0 j = false := by
  simp [gt, binop, ofReal, Scalar.zero_def, (hx _).le]

/-- At a negative array, `relu` is `0`, where `sqrt` has no correct JVP rule. -/
theorem relu_of_neg {x : Index s → ℝ} (hx : ∀ i, x i < 0) : relu (w := w) (ofReal x) = ofReal 0 := by
  funext j
  simp [relu, selectN, gt_of_neg hx, ofReal, Scalar.zero_def]

/-- Near a negative array, `sqrt(relu(x))` is the constant `0`, so its derivative is `0`. -/
theorem sqrtRelu_hasZeroJvp :
    IsJvpOfOn (fun _ _ => (0 : Tensor (.float w) (Index s))) (sqrtRelu (s := s) (w := w)) {x | ∀ i, x i < 0} := by
  intro x hx
  have hopen : IsOpen {y : Index s → ℝ | ∀ i, y i < 0} := by
    simp only [Set.ofPred_forall]
    exact isOpen_iInter_of_finite fun i => isOpen_lt (continuous_apply i) continuous_const
  refine ⟨fun _ => 0, 0, ?_, hasFDerivAt_const _ _, fun g => .of_eq (funext fun _ => rfl)⟩
  filter_upwards [hopen.mem_nhds hx] with y hy
  funext j
  simp [sqrtRelu, sqrt, relu_of_neg hy, ofReal, Scalar.sqrt_fin le_rfl]

/-- At a negative array, forward mode returns NaN: `0 * (0.5 / sqrt 0) = 0 * ∞`. -/
theorem sqrtReluJvp_of_neg (g : Index s → ℝ) {x : Index s → ℝ} (hx : ∀ i, x i < 0)
    (j : Index (broadcastShape s s)) : sqrtReluJvp (w := w) (ofReal g) (ofReal x) j = .nan := by
  have h1 : (Scalar.fin (w := w) 2⁻¹) / Scalar.fin 0 = .posInf := by
    show (if (0 : ℝ) = 0 then Scalar.infWithSign 2⁻¹ else _) = _
    simp [Scalar.infWithSign]
  have h2 : Scalar.fin (w := w) 0 * .posInf = .nan := by
    show Scalar.infWithSign 0 = _
    simp [Scalar.infWithSign]
  simp [sqrtReluJvp, standardJvp₁, Tan.inst, sqrtJvp, selectNJvp, selectN, gt_of_neg hx, mul, binop,
    constDiv, sqrt, relu_of_neg hx, ofReal, Scalar.zero_def, Scalar.sqrt_fin le_rfl, h1, h2]

/-- Forward mode is wrong at every negative array, where the function is differentiable. -/
theorem sqrtReluJvp_not_correct [Nonempty (Index s)] :
    ¬ IsJvpOfOn sqrtReluJvp (sqrtRelu (s := s) (w := w)) {x | ∀ i, x i < 0} := by
  intro hj
  obtain ⟨_, L, -, -, hrule⟩ := hj (fun _ => -1) fun _ => by norm_num
  obtain ⟨_, hr, -, hv⟩ := hrule 0
  obtain ⟨i⟩ := ‹Nonempty (Index s)›
  let j : Index (broadcastShape s s) := fun d => ⟨i d, by simp [broadcastShape]⟩
  have := hv j i fun d => by simp [j]
  rw [sqrtReluJvp_of_neg 0 (fun _ => by norm_num) j] at this
  cases this

/-- At a negative array, reverse mode returns `0`, the transpose of the zero derivative. -/
theorem sqrtReluVjp_of_neg (ct : Index (broadcastShape s s) → ℝ) {x : Index s → ℝ}
    (hx : ∀ i, x i < 0) : sqrtReluVjp (w := w) (ofReal ct) (ofReal x) = ofReal 0 := by
  funext j
  simp [sqrtReluVjp, selectNTranspose1, selectN, gt_of_neg hx, ofReal, Scalar.zero_def]

end Jax.Jaxpr
