module

public import Jax.Scalar
public import Jax.Shape
public import Mathlib.Analysis.Calculus.FDeriv.Pi
public import Mathlib.Analysis.Calculus.FDeriv.Prod
public import Mathlib.Analysis.Calculus.FDeriv.Linear
public import Mathlib.Analysis.Calculus.FDeriv.Mul
public import Mathlib.Topology.Algebra.Module.FiniteDimension
public import Mathlib.Analysis.Calculus.Deriv.Comp

/-!
# Core definitions

Arrays are functions from an index type to scalars.
An array of shape `s` has the index type `Index s`.

Differentiation is a statement about finite arrays.
The correctness notions below embed real arrays with `ofReal`
and require the rules to return finite arrays of the right shape.
-/

@[expose] public section

namespace Jax

/-- An array with index type `ι`. -/
abbrev Tensor (ι : Type) := ι → Scalar

/-- The array with the finite entries `x`. -/
def ofReal {ι : Type} (x : ι → ℝ) : Tensor ι := fun i => .fin (x i)

variable {m m' n n' k k' : ℕ}

/-- `a` and `b` are the same array: they have the same rank, the same shape and the same entries.
This compares arrays whose shapes are computed in different ways. -/
def Tensor.Same {s : Shape m} {t : Shape n} (a : Tensor (Index s)) (b : Tensor (Index t)) : Prop :=
  ∃ h : m = n, (∀ d, s d = t (Fin.cast h d)) ∧
    ∀ i j, (∀ d, (i d : ℕ) = j (Fin.cast h d)) → a i = b j

/-! ## JVP rules of unary primitives -/

/-- `jvp` is a correct JVP rule for `f` at every finite array `x` in `dom`:
near `x`, `f` computes a real function `F`, which is differentiable at `x`,
and `jvp g x` is the derivative of `F` at `x` applied to the finite tangent `g`.

The tangent comes first, as in the rules passed to `ad.defjvp`. -/
def IsJvpOfOn {s : Shape m} {t : Shape n} {t' : Shape n'}
    (jvp : Tensor (Index s) → Tensor (Index s) → Tensor (Index t'))
    (f : Tensor (Index s) → Tensor (Index t)) (dom : Set (Index s → ℝ)) : Prop :=
  ∀ x ∈ dom, ∃ (F : (Index s → ℝ) → Index t → ℝ) (L : (Index s → ℝ) →L[ℝ] Index t → ℝ),
    (∀ᶠ y in nhds x, f (ofReal y) = ofReal (F y)) ∧ HasFDerivAt F L x ∧
    ∀ g, (jvp (ofReal g) (ofReal x)).Same (ofReal (L g))

/-- `jvp` is a correct JVP rule for `f` at every finite array. -/
def IsJvpOf {s : Shape m} {t : Shape n} {t' : Shape n'}
    (jvp : Tensor (Index s) → Tensor (Index s) → Tensor (Index t'))
    (f : Tensor (Index s) → Tensor (Index t)) : Prop :=
  IsJvpOfOn jvp f Set.univ

/-! ## JVP rules of binary primitives -/

/-- `jvpX` and `jvpY` are correct JVP rules for the binary `f` at every pair of finite arrays in `dom`:
near `(x, y)`, `f` computes a real function `F`, which is differentiable at `(x, y)`,
both rules return finite arrays of the output shape,
and their sum is the derivative of `F` at `(x, y)` applied to the finite tangents `(g, h)`.

`ad.defjvp` takes one rule per argument and adds their results.
`jax/_src/interpreters/ad.py:978`:
```
def standard_jvp(jvprules, primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  tangents_out = [rule(t, *primals, **params) for rule, t in zip(jvprules, tangents)
                  if rule is not None and type(t) is not Zero]
  return val_out, functools.reduce(add_tangents, tangents_out, p2tz(val_out))
```
Symbolic zero tangents are not modelled: both rules are always evaluated. -/
def IsJvp₂OfOn {sx : Shape m} {sy : Shape m'} {t : Shape n} {tx : Shape k} {ty : Shape k'}
    (jvpX : Tensor (Index sx) → Tensor (Index sx) → Tensor (Index sy) → Tensor (Index tx))
    (jvpY : Tensor (Index sy) → Tensor (Index sx) → Tensor (Index sy) → Tensor (Index ty))
    (f : Tensor (Index sx) → Tensor (Index sy) → Tensor (Index t))
    (dom : Set ((Index sx → ℝ) × (Index sy → ℝ))) : Prop :=
  ∀ x y, (x, y) ∈ dom →
    ∃ (F : (Index sx → ℝ) × (Index sy → ℝ) → Index t → ℝ)
      (L : (Index sx → ℝ) × (Index sy → ℝ) →L[ℝ] Index t → ℝ),
    (∀ᶠ p in nhds (x, y), f (ofReal p.1) (ofReal p.2) = ofReal (F p)) ∧ HasFDerivAt F L (x, y) ∧
    ∀ g h, ∃ dx dy : Index t → ℝ,
      (jvpX (ofReal g) (ofReal x) (ofReal y)).Same (ofReal dx) ∧
      (jvpY (ofReal h) (ofReal x) (ofReal y)).Same (ofReal dy) ∧ L (g, h) = dx + dy

/-- `jvpX` and `jvpY` are correct JVP rules for the binary `f` at every pair of finite arrays. -/
def IsJvp₂Of {sx : Shape m} {sy : Shape m'} {t : Shape n} {tx : Shape k} {ty : Shape k'}
    (jvpX : Tensor (Index sx) → Tensor (Index sx) → Tensor (Index sy) → Tensor (Index tx))
    (jvpY : Tensor (Index sy) → Tensor (Index sx) → Tensor (Index sy) → Tensor (Index ty))
    (f : Tensor (Index sx) → Tensor (Index sy) → Tensor (Index t)) : Prop :=
  IsJvp₂OfOn jvpX jvpY f Set.univ

/-! ## Transpose rules -/

/-- `tr` is a correct transpose rule for `f`:
on finite arrays, `f` returns a finite array, `tr` returns a finite array of the shape of the input,
and pairing a cotangent `ct` with `f x` equals pairing `tr ct` with `x`.

The cotangent comes first, as in the rules passed to `ad.deflinear2`. -/
def IsTransposeOf {s : Shape m} {s' : Shape m'} {t : Shape n}
    (tr : Tensor (Index t) → Tensor (Index s')) (f : Tensor (Index s) → Tensor (Index t)) : Prop :=
  ∀ (ct : Index t → ℝ) (x : Index s → ℝ), ∃ (y : Index t → ℝ) (u : Index s → ℝ),
    f (ofReal x) = ofReal y ∧ (tr (ofReal ct)).Same (ofReal u) ∧
    ∑ j, ct j * y j = ∑ i, u i * x i

/-! ## Lemmas -/

theorem ofReal_injective {ι : Type} : Function.Injective (ofReal (ι := ι)) :=
  fun _ _ h => funext fun i => Scalar.fin.inj (congrFun h i)

theorem Tensor.Same.of_eq {s : Shape n} {a b : Tensor (Index s)} (h : a = b) : a.Same b :=
  ⟨rfl, fun _ => rfl, fun i j hij => by rw [h, Index.ext_val (j := j) hij]⟩

/-- Arrays of equal rank are the same if their shapes and entries agree. -/
theorem Tensor.same_of_shape_eq {s t : Shape n} {a : Tensor (Index s)} {b : Tensor (Index t)}
    (hs : ∀ d, s d = t d) (hv : ∀ i j, (∀ d, (i d : ℕ) = j d) → a i = b j) : a.Same b :=
  ⟨rfl, hs, hv⟩

/-- An array is the same as at most one array of a given shape. -/
theorem Tensor.Same.unique {s : Shape m} {t : Shape n} {a : Tensor (Index s)}
    {b c : Tensor (Index t)} (hb : a.Same b) (hc : a.Same c) : b = c := by
  obtain ⟨rfl, hs, hv⟩ := hb
  obtain ⟨_, _, hv'⟩ := hc
  funext j
  let i : Index s := fun d => ⟨j d, by rw [hs d]; exact (j d).isLt⟩
  exact (hv i j fun _ => rfl).symm.trans (hv' i j fun _ => rfl)

/-- Sums over the indices of two shapes with the same sizes agree
if the summands agree at indices with the same values. -/
theorem sum_filter_index_congr {s t : Shape n} (hst : ∀ d, s d = t d) {P : Index s → Prop}
    {Q : Index t → Prop} [DecidablePred P] [DecidablePred Q] {M : Type} [AddCommMonoid M]
    (f : Index s → M) (g : Index t → M)
    (hPQ : ∀ i j, (∀ d, (i d : ℕ) = j d) → (P i ↔ Q j))
    (hfg : ∀ i j, (∀ d, (i d : ℕ) = j d) → f i = g j) :
    ∑ i with P i, f i = ∑ j with Q j, g j := by
  rw [Finset.sum_filter, Finset.sum_filter]
  exact Fintype.sum_equiv (Equiv.piCongrRight fun d => finCongr (hst d)) _ _ fun i =>
    if_congr (hPQ i _ fun _ => rfl) (hfg i _ fun _ => rfl) rfl

/-- A linear primitive is its own JVP rule.
`ad.deflinear2` registers `linear_jvp`, `jax/_src/interpreters/ad.py:942`,
which applies the primitive to the tangent: `primitive.bind(*tangents, **params)`. -/
theorem isJvpOf_of_linear {s : Shape m} {t : Shape n} {f : Tensor (Index s) → Tensor (Index t)}
    (F : (Index s → ℝ) →ₗ[ℝ] Index t → ℝ) (hF : ∀ x, f (ofReal x) = ofReal (F x)) :
    IsJvpOf (fun g _ => f g) f := by
  intro x _
  let L : (Index s → ℝ) →L[ℝ] Index t → ℝ := LinearMap.toContinuousLinearMap F
  exact ⟨F, L, .of_forall hF, L.hasFDerivAt, fun g => .of_eq (hF g)⟩

/-- The derivative of an entry of an elementwise primitive. -/
theorem hasFDerivAt_elementwise {ι : Type} [Fintype ι] {f : ℝ → ℝ} {f' : ℝ} (x : ι → ℝ) (i : ι)
    (hf : HasDerivAt f f' (x i)) :
    HasFDerivAt (fun y : ι → ℝ => f (y i)) (f' • ContinuousLinearMap.proj (R := ℝ) i) x := by
  have hproj : HasFDerivAt (fun y : ι → ℝ => y i) (ContinuousLinearMap.proj (R := ℝ) i) x :=
    hasFDerivAt_apply i x
  exact hf.comp_hasFDerivAt x hproj

/-- The derivative of an entry of the first argument of a binary primitive. -/
theorem hasFDerivAt_fst_apply {ι κ : Type} [Fintype ι] [Fintype κ] (i : ι) (p : (ι → ℝ) × (κ → ℝ)) :
    HasFDerivAt (fun q : (ι → ℝ) × (κ → ℝ) => q.1 i)
      (ContinuousLinearMap.proj i ∘L ContinuousLinearMap.fst ℝ (ι → ℝ) (κ → ℝ)) p :=
  HasFDerivAt.comp (g := fun f : ι → ℝ => f i) p (hasFDerivAt_apply i p.1) hasFDerivAt_fst

/-- The derivative of an entry of the second argument of a binary primitive. -/
theorem hasFDerivAt_snd_apply {ι κ : Type} [Fintype ι] [Fintype κ] (j : κ) (p : (ι → ℝ) × (κ → ℝ)) :
    HasFDerivAt (fun q : (ι → ℝ) × (κ → ℝ) => q.2 j)
      (ContinuousLinearMap.proj j ∘L ContinuousLinearMap.snd ℝ (ι → ℝ) (κ → ℝ)) p :=
  HasFDerivAt.comp (g := fun f : κ → ℝ => f j) p (hasFDerivAt_apply j p.2) hasFDerivAt_snd

end Jax
