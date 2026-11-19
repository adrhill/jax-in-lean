module

public import Jax.Jaxpr.SqrtRelu
public import Jax.Lax.Exp

/-!
# `logsumexp` with a zero scale factor, jax-ml/jax#22398

`jax.jacfwd` and `jax.jacrev` of `b ↦ logsumexp(a, b=b)` return `0` for every entry where `b` is `0`.
The function is `log(Σ_i b_i exp(a_i))`, whose partial derivative in `b_i` is `exp(a_i) / Σ_j b_j exp(a_j)`,
which is not `0`. For `a = (0, 0)` and `b = (1, 0)`, both return `(1, 0)` instead of `(1, 1)`.

`logsumexp` replaces `a` by `-∞` where `b` is `0` before it sums the terms `exp(a) * b`,
`jax/_src/ops/special.py:81`:
```
    a_arr = jnp.where(b_arr != 0, a_arr, -np.inf)
```
The jaxpr of the terms and of their JVP in `b` is
```
c = ne b 0.0
d = select_n c -inf a
e = exp d
f = mul e b
ḟ = mul e ḃ
```
The terms are `exp(a) * b` also where `b` is `0` (`lseTerms_same`),
but `e` is `0` there, so their tangent is `0` too (`lseTermsJvp_same`).
Forward mode is correct where no entry of `b` is `0` (`lseTermsJvp_correct`),
and wrong at every `b` with an entry `0` (`lseTermsJvp_not_correct`).
This is not a bug of a rule: `ne` is not locally constant at `b = 0`,
where its zero tangent is wrong (`Jax.Lax.neJvp_not_correct`).
`log` and `reduce_sum`, which `logsumexp` applies to the terms, are differentiable where the sum is positive,
so the wrong tangent of a term reaches the Jacobian of `logsumexp`.

The model omits the shift of `logsumexp` by the maximum `amax` of `d`:
it computes `exp(d - amax) * b` and adds `amax` back after the `log`,
and `amax` is under `stop_gradient`, so the shift changes neither the result nor its tangent.
-/

@[expose] public section

namespace Jax.Jaxpr

open Jax.Lax

variable {n : ℕ} {s : Shape n} {w : FloatWidth}

/-- `where(b != 0, a, -∞)`, as `select_n(b != 0, -∞, a)`. -/
noncomputable def lseMask (a b : Tensor (.float w) (Index s)) : Tensor (.float w) (Index s) :=
  selectN rfl (whichShape_broadcastShape s) (ne (.refl s) b 0) (fun _ => .negInf) a

/-- The terms of `logsumexp(a, b=b)` as a function of `b`, `exp(where(b != 0, a, -∞)) * b`. -/
noncomputable def lseTerms (a b : Tensor (.float w) (Index s)) : Tensor (.float w) (Index (broadcastShape s s)) :=
  mul (.refl s) (exp (lseMask a b)) b

/-- Forward mode in `b`, `jax.jvp`: `standard_jvp` of `mul`, and the tangent of the result instantiated.
`ne` has a boolean result, whose tangent is a symbolic zero,
and the cases `-∞` and `a` of `select_n` are constants with symbolic zero tangents.
So every input of `select_n` and then of `exp` has a symbolic zero tangent,
`JVPTrace.process_primitive` calls no rule for them, and the tangent of `e` is a symbolic zero. -/
noncomputable def lseTermsJvp (a g b : Tensor (.float w) (Index s)) : Tensor (.float w) (Index (broadcastShape s s)) :=
  (standardJvp₂ (mulJvpLeft (.refl s)) (mulJvpRight (.refl s)) (mul (.refl s))
    (.zero (.of (.float w) s)) (.arr g) (exp (lseMask a b)) b).inst

/-- The index of `s` that an index of `broadcastShape s s` reads. -/
private abbrev unb (j : Index (broadcastShape s s)) : Index s := j.unbroadcast (broadcastsTo_left s s)

private theorem ofWhich_unbroadcast (i : Index s) :
    (Index.ofWhich (whichShape_broadcastShape s) i).unbroadcast (broadcastsTo_left s s) = i := by
  refine Index.unbroadcast_eq_of_val _ _ i fun d => ?_
  unfold Index.ofWhich
  split
  · subst_vars
    exact d.elim0
  · rfl

private theorem lseMask_ofReal (a b : Index s → ℝ) (i : Index s) :
    lseMask (w := w) (ofReal a) (ofReal b) i = if b i = 0 then .negInf else .fin (a i) := by
  by_cases hb : b i = 0 <;>
    simp [lseMask, selectN, ne, binop, ofWhich_unbroadcast, ofReal, Scalar.ne, Scalar.eq, Scalar.zero_def, hb]

private theorem lseTerms_ofReal (a b : Index s → ℝ) :
    lseTerms (w := w) (ofReal a) (ofReal b) = ofReal fun j => Real.exp (a (unb j)) * b (unb j) := by
  funext j
  by_cases hb : b (unb j) = 0 <;>
    simp [lseTerms, mul, binop, exp, Scalar.exp, lseMask_ofReal, ofReal, hb]

private theorem lseTermsJvp_ofReal (a g b : Index s → ℝ) :
    lseTermsJvp (w := w) (ofReal a) (ofReal g) (ofReal b) =
      ofReal fun j => (if b (unb j) = 0 then 0 else Real.exp (a (unb j))) * g (unb j) := by
  funext j
  by_cases hb : b (unb j) = 0 <;>
    simp [lseTermsJvp, standardJvp₂, Tan.inst, mulJvpRight, mul, binop, exp, Scalar.exp, lseMask_ofReal,
      ofReal, hb]

private theorem same_of_unb {f : Index s → ℝ} :
    (ofReal (w := w) fun j : Index (broadcastShape s s) => f (unb j)).Same (ofReal (w := w) f) :=
  Tensor.same_of_shape_eq (fun d => by simp [broadcastShape]) fun j i hij => by
    show Scalar.fin (f (unb j)) = Scalar.fin (f i)
    rw [show unb j = i from Index.unbroadcast_eq_of_val _ j i hij]

/-- On finite arrays, the terms are `exp(a) * b`, also where `b` is `0`. -/
theorem lseTerms_same (a b : Index s → ℝ) :
    (lseTerms (w := w) (ofReal a) (ofReal b)).Same (ofReal (w := w) fun i => Real.exp (a i) * b i) := by
  rw [lseTerms_ofReal]
  exact same_of_unb (f := fun i => Real.exp (a i) * b i)

/-- On finite arrays, forward mode multiplies the tangent by `exp(a)` where `b` is not `0`, and by `0` where it is. -/
theorem lseTermsJvp_same (a g b : Index s → ℝ) :
    (lseTermsJvp (w := w) (ofReal a) (ofReal g) (ofReal b)).Same
      (ofReal (w := w) fun i => (if b i = 0 then 0 else Real.exp (a i)) * g i) := by
  rw [lseTermsJvp_ofReal]
  exact same_of_unb (f := fun i => (if b i = 0 then 0 else Real.exp (a i)) * g i)

/-- The derivative of the terms in `b`, the product with `exp(a)`. -/
private noncomputable def lseTermsDeriv (a : Index s → ℝ) : (Index s → ℝ) →L[ℝ] Index (broadcastShape s s) → ℝ :=
  ContinuousLinearMap.pi fun j => Real.exp (a (unb j)) • ContinuousLinearMap.proj (unb j)

/-- Every correct JVP rule for the terms returns the product of the tangent with `exp(a)`. -/
private theorem lseTerms_deriv {a b : Index s → ℝ}
    {jvp : Tensor (.float w) (Index s) → Tensor (.float w) (Index s) → Tensor (.float w) (Index (broadcastShape s s))}
    (hj : IsJvpOfOn jvp (lseTerms (w := w) (ofReal a)) {b}) (g : Index s → ℝ) :
    (jvp (ofReal g) (ofReal b)).Same (ofReal (w := w) (lseTermsDeriv a g)) := by
  obtain ⟨F, L, hF, hL, hrule⟩ := hj b rfl
  have hev : F =ᶠ[nhds b] lseTermsDeriv a := by
    filter_upwards [hF] with y hy
    rw [← ofReal_injective.eq_iff, ← hy, lseTerms_ofReal]
    rfl
  rw [← hL.unique ((lseTermsDeriv a).hasFDerivAt.congr_of_eventuallyEq hev)]
  exact hrule g

/-- Forward mode is correct where no entry of `b` is `0`. -/
theorem lseTermsJvp_correct (a : Index s → ℝ) :
    IsJvpOfOn (lseTermsJvp (ofReal a)) (lseTerms (w := w) (ofReal a)) {b | ∀ i, b i ≠ 0} := by
  intro b hb
  refine ⟨lseTermsDeriv a, lseTermsDeriv a, .of_forall fun y => lseTerms_ofReal a y,
    (lseTermsDeriv a).hasFDerivAt, fun g => .of_eq ?_⟩
  rw [lseTermsJvp_ofReal]
  funext j
  have hb' : b (unb j) ≠ 0 := hb (unb j)
  simp [lseTermsDeriv, ofReal, hb']

/-- Forward mode is wrong at every `b` with an entry `0`, where the terms are differentiable. -/
theorem lseTermsJvp_not_correct (a : Index s → ℝ) {b : Index s → ℝ} {i : Index s} (hb : b i = 0) :
    ¬ IsJvpOfOn (lseTermsJvp (ofReal a)) (lseTerms (w := w) (ofReal a)) {b} := by
  intro hj
  have h := ofReal_injective ((Tensor.Same.of_eq (lseTermsJvp_ofReal a 1 b)).unique (lseTerms_deriv hj 1))
  let j : Index (broadcastShape s s) := fun d => ⟨i d, by simp [broadcastShape]⟩
  have hj : unb j = i := Index.unbroadcast_eq_of_val _ j i fun _ => rfl
  have := congrFun h j
  simp [lseTermsDeriv, hj, hb, (Real.exp_pos (a i)).ne] at this

/-- The case of the reproducer: `a = (0, 0)` and `b = (1, 0)`. -/
theorem lseTermsJvp_not_correct_reproducer :
    ¬ IsJvpOf (lseTermsJvp (ofReal (ι := Index ![2]) 0)) (lseTerms (w := w) (ofReal 0)) := fun hj =>
  lseTermsJvp_not_correct (b := fun i => if (i 0 : ℕ) = 0 then 1 else 0) (i := fun _ => ⟨1, by simp⟩) 0 rfl
    fun x hx => hj x (Set.mem_univ x)

end Jax.Jaxpr
