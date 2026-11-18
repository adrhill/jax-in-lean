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

An array of dtype `d` is a function from an index type to the entries `d.Elem`.
An array of shape `s` has the index type `Index s`.
The entries of a float array, `Scalar w`, carry the width `w`,
so arrays of different widths have different types.

Differentiation is a statement about finite arrays.
The correctness notions below embed real arrays with `ofReal`
and require the rules to return finite arrays of the right shape and dtype.
-/

@[expose] public section

namespace Jax

/-- An array of dtype `d` with index type `ι`. -/
abbrev Tensor (d : DType) (ι : Type) := ι → d.Elem

/-- The floating-point array of width `w` with the finite entries `x`. -/
def ofReal {w : FloatWidth} {ι : Type} (x : ι → ℝ) : Tensor (.float w) ι := fun i => .fin (x i)

variable {m m' n n' k k' : ℕ} {w w' wx wy : FloatWidth} {d d' d₁ d₂ d₃ : DType}

/-- `a` and `b` are the same array: they have the same dtype, the same rank, the same shape
and the same entries.
This compares arrays whose dtypes and shapes are computed in different ways. -/
def Tensor.Same {d d' : DType} {s : Shape m} {t : Shape n} (a : Tensor d (Index s))
    (b : Tensor d' (Index t)) : Prop :=
  ∃ hd : d = d', ∃ h : m = n, (∀ e, s e = t (Fin.cast h e)) ∧
    ∀ i j, (∀ e, (i e : ℕ) = j (Fin.cast h e)) → cast (congrArg DType.Elem hd) (a i) = b j

/-! ## JVP rules of unary primitives -/

/-- `jvp` is a correct JVP rule for `f` at every finite array `x` in `dom`:
near `x`, `f` computes a real function `F`, which is differentiable at `x`,
and `jvp g x` is the derivative of `F` at `x` applied to the finite tangent `g`.
The input of `f` has dtype `float w` and its output `float w'`, so their tangents have the same dtypes
(`DType.tangent`), and the rule must return an array of dtype `float w'`.

The tangent comes first, as in the rules passed to `ad.defjvp`. -/
def IsJvpOfOn {s : Shape m} {t : Shape n} {t' : Shape n'}
    (jvp : Tensor (.float w) (Index s) → Tensor (.float w) (Index s) → Tensor d (Index t'))
    (f : Tensor (.float w) (Index s) → Tensor (.float w') (Index t)) (dom : Set (Index s → ℝ)) :
    Prop :=
  ∀ x ∈ dom, ∃ (F : (Index s → ℝ) → Index t → ℝ) (L : (Index s → ℝ) →L[ℝ] Index t → ℝ),
    (∀ᶠ y in nhds x, f (ofReal y) = ofReal (F y)) ∧ HasFDerivAt F L x ∧
    ∀ g, (jvp (ofReal g) (ofReal x)).Same (ofReal (w := w') (L g))

/-- `jvp` is a correct JVP rule for `f` at every finite array. -/
def IsJvpOf {s : Shape m} {t : Shape n} {t' : Shape n'}
    (jvp : Tensor (.float w) (Index s) → Tensor (.float w) (Index s) → Tensor d (Index t'))
    (f : Tensor (.float w) (Index s) → Tensor (.float w') (Index t)) : Prop :=
  IsJvpOfOn jvp f Set.univ

/-! ## JVP rules of binary primitives -/

/-- `jvpX` and `jvpY` are correct JVP rules for the binary `f` at every pair of finite arrays in `dom`:
near `(x, y)`, `f` computes a real function `F`, which is differentiable at `(x, y)`,
both rules return finite arrays of the output shape,
and their sum is the derivative of `F` at `(x, y)` applied to the finite tangents `(g, h)`.
Both rules must return arrays of the dtype of the output of `f`, the dtype of its tangents.

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
    (jvpX : Tensor (.float wx) (Index sx) → Tensor (.float wx) (Index sx) →
      Tensor (.float wy) (Index sy) → Tensor d₁ (Index tx))
    (jvpY : Tensor (.float wy) (Index sy) → Tensor (.float wx) (Index sx) →
      Tensor (.float wy) (Index sy) → Tensor d₂ (Index ty))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t))
    (dom : Set ((Index sx → ℝ) × (Index sy → ℝ))) : Prop :=
  ∀ x y, (x, y) ∈ dom →
    ∃ (F : (Index sx → ℝ) × (Index sy → ℝ) → Index t → ℝ)
      (L : (Index sx → ℝ) × (Index sy → ℝ) →L[ℝ] Index t → ℝ),
    (∀ᶠ p in nhds (x, y), f (ofReal p.1) (ofReal p.2) = ofReal (F p)) ∧ HasFDerivAt F L (x, y) ∧
    ∀ g h, ∃ dx dy : Index t → ℝ,
      (jvpX (ofReal g) (ofReal x) (ofReal y)).Same (ofReal (w := w) dx) ∧
      (jvpY (ofReal h) (ofReal x) (ofReal y)).Same (ofReal (w := w) dy) ∧ L (g, h) = dx + dy

/-- `jvpX` and `jvpY` are correct JVP rules for the binary `f` at every pair of finite arrays. -/
def IsJvp₂Of {sx : Shape m} {sy : Shape m'} {t : Shape n} {tx : Shape k} {ty : Shape k'}
    (jvpX : Tensor (.float wx) (Index sx) → Tensor (.float wx) (Index sx) →
      Tensor (.float wy) (Index sy) → Tensor d₁ (Index tx))
    (jvpY : Tensor (.float wy) (Index sy) → Tensor (.float wx) (Index sx) →
      Tensor (.float wy) (Index sy) → Tensor d₂ (Index ty))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t)) : Prop :=
  IsJvp₂OfOn jvpX jvpY f Set.univ

/-- `jvp` is a correct JVP rule for the binary `f` at every pair of finite arrays in `dom`,
for a rule that takes the tangents of both arguments together,
as the rules stored directly in `ad.primitive_jvps` do:
near `(x, y)`, `f` computes a real function `F`, which is differentiable at `(x, y)`,
and `jvp g h x y` is the derivative of `F` at `(x, y)` applied to the finite tangents `(g, h)`.
The tangents come first, then the primals. -/
def IsJointJvp₂OfOn {sx : Shape m} {sy : Shape m'} {t : Shape n} {t' : Shape k}
    (jvp : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) →
      Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor d (Index t'))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t))
    (dom : Set ((Index sx → ℝ) × (Index sy → ℝ))) : Prop :=
  ∀ x y, (x, y) ∈ dom →
    ∃ (F : (Index sx → ℝ) × (Index sy → ℝ) → Index t → ℝ)
      (L : (Index sx → ℝ) × (Index sy → ℝ) →L[ℝ] Index t → ℝ),
    (∀ᶠ p in nhds (x, y), f (ofReal p.1) (ofReal p.2) = ofReal (F p)) ∧ HasFDerivAt F L (x, y) ∧
    ∀ g h, (jvp (ofReal g) (ofReal h) (ofReal x) (ofReal y)).Same (ofReal (w := w) (L (g, h)))

/-- `jvp` is a correct JVP rule for the binary `f`, taking both tangents together,
at every pair of finite arrays. -/
def IsJointJvp₂Of {sx : Shape m} {sy : Shape m'} {t : Shape n} {t' : Shape k}
    (jvp : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) →
      Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor d (Index t'))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t)) : Prop :=
  IsJointJvp₂OfOn jvp f Set.univ

/-! ## JVP rules of ternary primitives -/

/-- `jvpA`, `jvpB` and `jvpC` are correct JVP rules for the ternary `f`
at every triple of finite arrays in `dom`, as `IsJvp₂OfOn` for three arguments:
near `(a, b, c)`, `f` computes a real function `F`, which is differentiable at `(a, b, c)`,
the rules return finite arrays of the output shape,
and their sum is the derivative of `F` at `(a, b, c)` applied to the finite tangents.
Each rule takes the tangent of its argument, then the three primals, as in `ad.defjvp`. -/
def IsJvp₃OfOn {ra rb rc r ra' rb' rc' : ℕ} {sa : Shape ra} {sb : Shape rb} {sc : Shape rc}
    {t : Shape r} {ta : Shape ra'} {tb : Shape rb'} {tc : Shape rc'} {wa wb wc : FloatWidth}
    (jvpA : Tensor (.float wa) (Index sa) → Tensor (.float wa) (Index sa) →
      Tensor (.float wb) (Index sb) → Tensor (.float wc) (Index sc) → Tensor d₁ (Index ta))
    (jvpB : Tensor (.float wb) (Index sb) → Tensor (.float wa) (Index sa) →
      Tensor (.float wb) (Index sb) → Tensor (.float wc) (Index sc) → Tensor d₂ (Index tb))
    (jvpC : Tensor (.float wc) (Index sc) → Tensor (.float wa) (Index sa) →
      Tensor (.float wb) (Index sb) → Tensor (.float wc) (Index sc) → Tensor d₃ (Index tc))
    (f : Tensor (.float wa) (Index sa) → Tensor (.float wb) (Index sb) →
      Tensor (.float wc) (Index sc) → Tensor (.float w) (Index t))
    (dom : Set ((Index sa → ℝ) × (Index sb → ℝ) × (Index sc → ℝ))) : Prop :=
  ∀ a b c, (a, b, c) ∈ dom →
    ∃ (F : (Index sa → ℝ) × (Index sb → ℝ) × (Index sc → ℝ) → Index t → ℝ)
      (L : (Index sa → ℝ) × (Index sb → ℝ) × (Index sc → ℝ) →L[ℝ] Index t → ℝ),
    (∀ᶠ p in nhds (a, b, c), f (ofReal p.1) (ofReal p.2.1) (ofReal p.2.2) = ofReal (F p)) ∧
    HasFDerivAt F L (a, b, c) ∧
    ∀ ga gb gc, ∃ da db dc : Index t → ℝ,
      (jvpA (ofReal ga) (ofReal a) (ofReal b) (ofReal c)).Same (ofReal (w := w) da) ∧
      (jvpB (ofReal gb) (ofReal a) (ofReal b) (ofReal c)).Same (ofReal (w := w) db) ∧
      (jvpC (ofReal gc) (ofReal a) (ofReal b) (ofReal c)).Same (ofReal (w := w) dc) ∧
      L (ga, gb, gc) = da + db + dc

/-- `jvpA`, `jvpB` and `jvpC` are correct JVP rules for the ternary `f`
at every triple of finite arrays. -/
def IsJvp₃Of {ra rb rc r ra' rb' rc' : ℕ} {sa : Shape ra} {sb : Shape rb} {sc : Shape rc}
    {t : Shape r} {ta : Shape ra'} {tb : Shape rb'} {tc : Shape rc'} {wa wb wc : FloatWidth}
    (jvpA : Tensor (.float wa) (Index sa) → Tensor (.float wa) (Index sa) →
      Tensor (.float wb) (Index sb) → Tensor (.float wc) (Index sc) → Tensor d₁ (Index ta))
    (jvpB : Tensor (.float wb) (Index sb) → Tensor (.float wa) (Index sa) →
      Tensor (.float wb) (Index sb) → Tensor (.float wc) (Index sc) → Tensor d₂ (Index tb))
    (jvpC : Tensor (.float wc) (Index sc) → Tensor (.float wa) (Index sa) →
      Tensor (.float wb) (Index sb) → Tensor (.float wc) (Index sc) → Tensor d₃ (Index tc))
    (f : Tensor (.float wa) (Index sa) → Tensor (.float wb) (Index sb) →
      Tensor (.float wc) (Index sc) → Tensor (.float w) (Index t)) : Prop :=
  IsJvp₃OfOn jvpA jvpB jvpC f Set.univ

/-! ## Zero JVP rules -/

/-- The zero tangent of `ad.defjvp_zero` is correct for the binary `f` at every pair of finite arrays
in `dom`: near `(x, y)`, `f` is constant.
The output of `f` may have any dtype. For a boolean output, whose tangent has dtype `float0`,
local constancy is what the zero tangent claims: it makes the chain rule through `select_n` correct.
`jax/_src/interpreters/ad.py:1041`:
```
def zero_jvp(primitive, primals, tangents, **params):
  r = primitive.bind(*primals, **params)
  return r, p2tz(r)
``` -/
def IsZeroJvp₂OfOn {sx : Shape m} {sy : Shape m'} {t : Shape n}
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor d (Index t))
    (dom : Set ((Index sx → ℝ) × (Index sy → ℝ))) : Prop :=
  ∀ x y, (x, y) ∈ dom →
    ∀ᶠ p in nhds (x, y), f (ofReal p.1) (ofReal p.2) = f (ofReal x) (ofReal y)

/-- The zero tangent is correct for the binary `f` at every pair of finite arrays. -/
def IsZeroJvp₂Of {sx : Shape m} {sy : Shape m'} {t : Shape n}
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor d (Index t)) : Prop :=
  IsZeroJvp₂OfOn f Set.univ

/-! ## Transpose rules

The names follow JAX, which calls these rules transposes.
The correctness notions below state that a rule is the adjoint of the linear `f`
with respect to the standard inner product `∑ i, u i * x i` of finite arrays,
the precise term for maps between Euclidean spaces of any shapes.
With this inner product, the adjoint is the transpose of the matrix of `f`
and a cotangent has the shape of the output of `f`.
-/

/-- `tr` is a correct transpose rule for `f`, that is, the adjoint of `f`:
on finite arrays, `f` returns a finite array, `tr` returns a finite array of the shape of the input,
and pairing a cotangent `ct` with `f x` equals pairing `tr ct` with `x`.
A cotangent has the dtype of the tangents of the output of `f`,
and `tr` must return the dtype of the tangents of its input.

The cotangent comes first, as in the rules passed to `ad.deflinear2`. -/
def IsTransposeOf {s : Shape m} {s' : Shape m'} {t : Shape n}
    (tr : Tensor (.float w') (Index t) → Tensor d (Index s'))
    (f : Tensor (.float w) (Index s) → Tensor (.float w') (Index t)) : Prop :=
  ∀ (ct : Index t → ℝ) (x : Index s → ℝ), ∃ (y : Index t → ℝ) (u : Index s → ℝ),
    f (ofReal x) = ofReal y ∧ (tr (ofReal ct)).Same (ofReal (w := w) u) ∧
    ∑ j, ct j * y j = ∑ i, u i * x i

/-- `trX` and `trY` are correct transpose rules for the binary `f`, which is linear in both arguments
together: on finite arrays, `f` returns a finite array, `trX` and `trY` return finite arrays
of the shapes of the inputs, and pairing a cotangent `ct` with `f x y`
equals pairing `trX ct` with `x` plus pairing `trY ct` with `y`.
The cotangent comes first, as in the rules of `ad.primitive_transposes`. -/
def IsTranspose₂Of {sx : Shape m} {sy : Shape m'} {sx' : Shape k} {sy' : Shape k'} {t : Shape n}
    (trX : Tensor (.float w) (Index t) → Tensor d₁ (Index sx'))
    (trY : Tensor (.float w) (Index t) → Tensor d₂ (Index sy'))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t)) :
    Prop :=
  ∀ (ct : Index t → ℝ) (x : Index sx → ℝ) (y : Index sy → ℝ),
    ∃ (z : Index t → ℝ) (u : Index sx → ℝ) (v : Index sy → ℝ),
    f (ofReal x) (ofReal y) = ofReal z ∧ (trX (ofReal ct)).Same (ofReal (w := wx) u) ∧
    (trY (ofReal ct)).Same (ofReal (w := wy) v) ∧ ∑ j, ct j * z j = ∑ i, u i * x i + ∑ i, v i * y i

/-- `tr` is a correct transpose rule for the binary `f`, which is linear in both arguments together,
whichever of its arguments are linear inputs.

A rule of `ad.primitive_transposes` takes the cotangent and then the arguments of the primitive.
An argument that is a linear input is an `ad.UndefinedPrimal`, here `none`; any other argument is its value.
The rule returns one entry per argument, a cotangent or `None`, and `backward_pass` ignores
the entries of the arguments that are not linear inputs (`write_cotangent`, `jax/_src/interpreters/ad.py:359`).
The outer `none` is a raised exception.

The rule is correct if there are correct transpose rules `trX` and `trY` of `f` such that,
whenever some argument is a linear input, the rule does not raise
and returns `trX ct` and `trY ct` for the arguments that are linear inputs. -/
def IsPartialTranspose₂Of {sx : Shape m} {sy : Shape m'} {sx' : Shape k} {sy' : Shape k'}
    {t : Shape n}
    (tr : Tensor (.float w) (Index t) → Option (Tensor (.float wx) (Index sx)) →
      Option (Tensor (.float wy) (Index sy)) →
      Option (Option (Tensor d₁ (Index sx')) × Option (Tensor d₂ (Index sy'))))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t)) :
    Prop :=
  ∃ (trX : Tensor (.float w) (Index t) → Tensor d₁ (Index sx'))
    (trY : Tensor (.float w) (Index t) → Tensor d₂ (Index sy')),
    IsTranspose₂Of trX trY f ∧
    ∀ (ct : Index t → ℝ) x y, x = none ∨ y = none →
      ∃ u v, tr (ofReal ct) x y = some (u, v) ∧
        (x = none → u = some (trX (ofReal ct))) ∧ (y = none → v = some (trY (ofReal ct)))

/-! ## Forwarding rules -/

/-- `fwd` is a correct forwarding rule for the unary `f`:
if `fwd` forwards the operand to the result, the operand is the result of `f`.

A rule of `pe.forwarding_rules`, `jax/_src/interpreters/partial_eval.py:1648`,
looks at the parameters and input shapes of an equation, not at the input values.
For each output it returns the index of an input that replaces it, or `None`.
For a unary primitive, `true` is `[0]` and `false` is `[None]`. -/
def IsFwdOf {s : Shape m} {t : Shape n} (fwd : Bool) (f : Tensor d (Index s) → Tensor d' (Index t)) :
    Prop :=
  fwd → ∀ x : Tensor d (Index s), x.Same (f x)

/-! ## Arrays of any shape -/

/-- An array of any rank, shape and dtype, as Python code holds it.
A rule that branches on the shapes of its arguments is a function of such arrays
that returns an `Option`: `none` is a raised exception. -/
structure Arr where
  {rank : ℕ}
  shape : Shape rank
  {dtype : DType}
  val : Tensor dtype (Index shape)

/-- `np.shape(a)` -/
def Arr.shapeList (a : Arr) : List ℕ := List.ofFn a.shape

/-- `a` and `b` are the same array, of the same dtype. -/
def Arr.Same (a b : Arr) : Prop := Tensor.Same a.val b.val

/-- Entry `i` of `a` along `axis`: the array `a[:, …, i, …, :]`, which has one axis less.
`none` if `a` has no axis `axis` or the axis has no entry `i`. -/
def Arr.unbatch (a : Arr) (axis i : ℕ) : Option Arr :=
  match a with
  | @Arr.mk 0 _ _ _ => none
  | @Arr.mk (n + 1) shape _ val =>
    if h : axis < n + 1 then
      if hi : i < shape ⟨axis, h⟩ then
        some ⟨Fin.removeNth ⟨axis, h⟩ shape,
          fun j => val (Fin.insertNth (α := fun d => Fin (shape d)) ⟨axis, h⟩ ⟨i, hi⟩ j)⟩
      else none
    else none

/-! ## Batching rules -/

/-- An argument or the result of a batching rule: an array and its batch axis,
which is `none` if the array is not batched. -/
structure Batched where
  val : Arr
  bdim : Option ℕ

/-- The array for entry `i` of the batch: `x` itself if it is not batched. -/
def Batched.get (x : Batched) (i : ℕ) : Option Arr :=
  match x.bdim with
  | none => some x.val
  | some d => x.val.unbatch d i

/-- The batch axis of `x`, if it has one, is an axis of size `size`. -/
def Batched.HasSize (x : Batched) (size : ℕ) : Prop :=
  ∀ d ∈ x.bdim, ∃ h : d < x.val.rank, x.val.shape ⟨d, h⟩ = size

/-- `rule` is a correct batching rule for the unary `f`, as `IsBatchOf₃` for one argument:
for every nonempty batch on whose entries `f` does not raise,
the rule does not raise, and entry `i` of its result is `f` of entry `i` of the argument. -/
def IsBatchOf₁ (rule : Batched → Option Batched) (f : Arr → Option Arr) : Prop :=
  ∀ (size : ℕ) (a : Batched), 0 < size → a.HasSize size → a.bdim.isSome →
    (∀ i < size, ∃ ai, a.get i = some ai ∧ (f ai).isSome) →
    ∃ y, rule a = some y ∧ ∀ i < size, ∃ ai r yi, a.get i = some ai ∧ f ai = some r ∧
      y.get i = some yi ∧ yi.Same r

/-- `rule` is a correct batching rule for the ternary `f`.

`vmap` maps `f` over a batch of `size` entries.
A rule of `batching.primitive_batchers` takes the arguments with their batch axes
and returns the result with its batch axis.
It is called if some argument is batched.

The rule is correct if, for every nonempty batch on whose entries `f` does not raise,
the rule does not raise, and entry `i` of its result is `f` of the entries `i` of the arguments. -/
def IsBatchOf₃ (rule : Batched → Batched → Batched → Option Batched)
    (f : Arr → Arr → Arr → Option Arr) : Prop :=
  ∀ (size : ℕ) (a b c : Batched), 0 < size → a.HasSize size → b.HasSize size → c.HasSize size →
    (a.bdim.isSome ∨ b.bdim.isSome ∨ c.bdim.isSome) →
    (∀ i < size, ∃ ai bi ci, a.get i = some ai ∧ b.get i = some bi ∧ c.get i = some ci ∧
      (f ai bi ci).isSome) →
    ∃ y, rule a b c = some y ∧ ∀ i < size, ∃ ai bi ci r yi, a.get i = some ai ∧
      b.get i = some bi ∧ c.get i = some ci ∧ f ai bi ci = some r ∧ y.get i = some yi ∧ yi.Same r

/-! ## Lemmas -/

theorem ofReal_injective {ι : Type} : Function.Injective (ofReal (w := w) (ι := ι)) :=
  fun _ _ h => funext fun i => Scalar.fin.inj (congrFun h i)

theorem Tensor.Same.of_eq {s : Shape n} {a b : Tensor d (Index s)} (h : a = b) : a.Same b :=
  ⟨rfl, rfl, fun _ => rfl, fun i j hij => by rw [h, Index.ext_val (j := j) hij]; rfl⟩

/-- Arrays of equal dtype and rank are the same if their shapes and entries agree. -/
theorem Tensor.same_of_shape_eq {s t : Shape n} {a : Tensor d (Index s)} {b : Tensor d (Index t)}
    (hs : ∀ e, s e = t e) (hv : ∀ i j, (∀ e, (i e : ℕ) = j e) → a i = b j) : a.Same b :=
  ⟨rfl, rfl, hs, hv⟩

/-- Entries of the same arrays of the same dtype agree. -/
theorem Tensor.Same.apply_eq {s : Shape m} {t : Shape n} {a : Tensor d (Index s)}
    {b : Tensor d (Index t)} (h : a.Same b) :
    ∃ hmn : m = n, (∀ e, s e = t (Fin.cast hmn e)) ∧
      ∀ i j, (∀ e, (i e : ℕ) = j (Fin.cast hmn e)) → a i = b j := by
  obtain ⟨_, hmn, hs, hv⟩ := h
  exact ⟨hmn, hs, hv⟩

/-- An array is the same as at most one array of a given dtype and shape. -/
theorem Tensor.Same.unique {s : Shape m} {t : Shape n} {a : Tensor d (Index s)}
    {b c : Tensor d' (Index t)} (hb : a.Same b) (hc : a.Same c) : b = c := by
  obtain ⟨rfl, rfl, hs, hv⟩ := hb
  obtain ⟨_, _, _, hv'⟩ := hc
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

/-- Entry `i` of an array along its axis `p`. -/
theorem Arr.unbatch_mk (shape : Shape (n + 1)) (val : Tensor d (Index shape)) (p : Fin (n + 1))
    {i : ℕ} (hi : i < shape p) :
    Arr.unbatch ⟨shape, val⟩ p i = some ⟨Fin.removeNth p shape,
      fun j => val (Fin.insertNth (α := fun d => Fin (shape d)) p ⟨i, hi⟩ j)⟩ := by
  simp [Arr.unbatch, hi, p.isLt]

/-- A linear primitive is its own JVP rule.
`ad.deflinear2` registers `linear_jvp`, `jax/_src/interpreters/ad.py:942`,
which applies the primitive to the tangent: `primitive.bind(*tangents, **params)`. -/
theorem isJvpOf_of_linear {s : Shape m} {t : Shape n}
    {f : Tensor (.float w) (Index s) → Tensor (.float w') (Index t)}
    (F : (Index s → ℝ) →ₗ[ℝ] Index t → ℝ) (hF : ∀ x, f (ofReal x) = ofReal (F x)) :
    IsJvpOf (fun g _ => f g) f := by
  intro x _
  let L : (Index s → ℝ) →L[ℝ] Index t → ℝ := LinearMap.toContinuousLinearMap F
  exact ⟨F, L, .of_forall hF, L.hasFDerivAt, fun g => .of_eq (hF g)⟩

/-- A JVP rule is correct if it is correct entry by entry:
near `x`, every entry of `f` computes a real function that is differentiable at `x`,
and the entry of `jvp` is its derivative applied to the tangent. -/
theorem isJvpOfOn_of_apply {s : Shape m} {t : Shape n}
    {jvp : Tensor (.float w) (Index s) → Tensor (.float w) (Index s) → Tensor (.float w') (Index t)}
    {f : Tensor (.float w) (Index s) → Tensor (.float w') (Index t)} {dom : Set (Index s → ℝ)}
    (h : ∀ x ∈ dom, ∀ g j, ∃ (Φ : (Index s → ℝ) → ℝ) (Φ' : (Index s → ℝ) →L[ℝ] ℝ),
      (∀ᶠ y in nhds x, f (ofReal y) j = .fin (Φ y)) ∧ HasFDerivAt Φ Φ' x ∧
        jvp (ofReal g) (ofReal x) j = .fin (Φ' g)) :
    IsJvpOfOn jvp f dom := by
  intro x hx
  choose Φ Φ' hev hd _ using fun j => h x hx 0 j
  refine ⟨fun y j => Φ j y, ContinuousLinearMap.pi Φ', ?_, hasFDerivAt_pi.2 hd,
    fun g => .of_eq (funext fun j => ?_)⟩
  · filter_upwards [Filter.eventually_all.2 hev] with y hy
    exact funext hy
  · obtain ⟨Ψ, Ψ', hevΨ, hdΨ, ht⟩ := h x hx g j
    have hΨ : Ψ =ᶠ[nhds x] Φ j := by
      filter_upwards [hev j, hevΨ] with y h1 h2
      exact Scalar.fin.inj (h2.symm.trans h1)
    rw [ht, hdΨ.unique ((hd j).congr_of_eventuallyEq hΨ)]
    rfl

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
