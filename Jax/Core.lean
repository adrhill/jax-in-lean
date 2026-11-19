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

/-! ## Symbolic zeros

JAX passes a tangent or cotangent that is known to be zero as a symbolic zero, `ad_util.Zero`,
which carries the aval of the array that it stands for and no entries.
`jax/_src/ad_util.py:76`:
```
class Zero:
  __slots__ = ['aval']
  def __init__(self, aval: core.AbstractValue):
    self.aval = aval
```
-/

/-- The abstract value of an array, `core.ShapedArray`: its shape and dtype. -/
structure Aval where
  shape : List ℕ
  dtype : DType
  deriving DecidableEq

/-- The aval of an array of dtype `d` and shape `s`. -/
def Aval.of (d : DType) (s : Shape n) : Aval := ⟨List.ofFn s, d⟩

/-- A tangent or cotangent as JAX passes it to and from a rule:
a symbolic zero of an aval, or an array of dtype `d` with index type `ι`.
The aval of a symbolic zero is data, as in JAX: the rule that creates it chooses it. -/
inductive Tan (d : DType) (ι : Type) where
  | zero (aval : Aval)
  | arr (x : Tensor d ι)

/-- `instantiate_zeros`, `jax/_src/interpreters/ad.py:1048`, for a floating-point tangent:
a symbolic zero becomes the array of zeros `zeros_like_aval(aval)`.
The zeros have the index type of the tangent, the shape of the aval wherever JAX instantiates. -/
def Tan.inst {w : FloatWidth} {ι : Type} : Tan (.float w) ι → Tensor (.float w) ι
  | .zero _ => 0
  | .arr x => x

/-- The tangent `a` is the floating-point array `b`:
an array that is the same as `b`, or a symbolic zero with the aval of `b` where `b` is zero. -/
def Tan.Same {s : Shape m} {t : Shape n} (a : Tan d (Index s)) (b : Tensor (.float w) (Index t)) : Prop :=
  match a with
  | .zero aval => aval = .of (.float w) t ∧ b = 0
  | .arr x => x.Same b

/-- The tangent of a floating-point argument of shape `s` in a pattern of symbolic zeros:
the symbolic zero of its aval if `z`, and the finite array `g` otherwise.
It stands for the real tangent `if z then 0 else g`. -/
def Tan.ofReal (z : Bool) {s : Shape n} (g : Index s → ℝ) : Tan (.float w) (Index s) :=
  if z then .zero (.of (.float w) s) else .arr (Jax.ofReal g)

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
and each rule, applied to a finite tangent of its argument,
returns the derivative of `F` at `(x, y)` applied to that tangent and a zero tangent of the other argument.
Both rules must return arrays of the shape and dtype of the output of `f`, the dtype of its tangents.

`ad.defjvp` takes one rule per argument. `standard_jvp` evaluates the rule of each argument
whose tangent is not a symbolic zero, and adds the results with `add_tangents`,
which returns the other argument for a symbolic zero, starting from the symbolic zero of the output.
`jax/_src/interpreters/ad.py:978`:
```
def standard_jvp(jvprules, primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  tangents_out = [rule(t, *primals, **params) for rule, t in zip(jvprules, tangents)
                  if rule is not None and type(t) is not Zero]
  return val_out, functools.reduce(add_tangents, tangents_out, p2tz(val_out))
```
So the result is correct for every pattern of symbolic zero tangents if and only if each rule is:
if both rules are evaluated, the derivative applied to `(g, h)` is the sum of their results by linearity.
`standardJvp₂_correct` states this for rules that return arrays of the same type. -/
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
    (∀ g, (jvpX (ofReal g) (ofReal x) (ofReal y)).Same (ofReal (w := w) (L (g, 0)))) ∧
    ∀ h, (jvpY (ofReal h) (ofReal x) (ofReal y)).Same (ofReal (w := w) (L (0, h)))

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
and for every pattern of symbolic zero tangents, `jvp g h x y` is the derivative of `F` at `(x, y)`
applied to the finite tangents `(g, h)`, with zeros at the symbolic zeros.
A symbolic zero that the rule returns must have the aval of the output of `f`.
The tangents come first, then the primals.

`JVPTrace.process_primitive`, `jax/_src/interpreters/ad.py:551`, does not call the rule
if every tangent is a symbolic zero:
```
    if (all(type(t) is Zero for t in tangents_in) and
        not primitive.ref_allocating and
        not any(isinstance(typeof(x), AbstractRef) for x in primals_in)):
      avals = tuple(core.typeof(x) for x in primals_in)
      return primitive.bind_with_trace(self.parent_trace, primals_in, avals, params)
``` -/
def IsJointJvp₂OfOn {sx : Shape m} {sy : Shape m'} {t : Shape n} {t' : Shape k}
    (jvp : Tan (.float wx) (Index sx) → Tan (.float wy) (Index sy) →
      Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tan d (Index t'))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t))
    (dom : Set ((Index sx → ℝ) × (Index sy → ℝ))) : Prop :=
  ∀ x y, (x, y) ∈ dom →
    ∃ (F : (Index sx → ℝ) × (Index sy → ℝ) → Index t → ℝ)
      (L : (Index sx → ℝ) × (Index sy → ℝ) →L[ℝ] Index t → ℝ),
    (∀ᶠ p in nhds (x, y), f (ofReal p.1) (ofReal p.2) = ofReal (F p)) ∧ HasFDerivAt F L (x, y) ∧
    ∀ (zg zh : Bool) g h, zg = false ∨ zh = false →
      (jvp (.ofReal zg g) (.ofReal zh h) (ofReal x) (ofReal y)).Same
        (ofReal (w := w) (L (if zg then 0 else g, if zh then 0 else h)))

/-- `jvp` is a correct JVP rule for the binary `f`, taking both tangents together,
at every pair of finite arrays. -/
def IsJointJvp₂Of {sx : Shape m} {sy : Shape m'} {t : Shape n} {t' : Shape k}
    (jvp : Tan (.float wx) (Index sx) → Tan (.float wy) (Index sy) →
      Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tan d (Index t'))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t)) : Prop :=
  IsJointJvp₂OfOn jvp f Set.univ

/-- `jvp` is a correct JVP rule for the binary `f` with two results at every pair of finite arrays in `dom`,
for a rule that takes the tangents of both arguments together, as `IsJointJvp₂OfOn` for each result.
The rule returns the tangents of both results, and `none` if it raises:
for every pattern of symbolic zero tangents that `JVPTrace.process_primitive` passes to it,
the rule does not raise, and each tangent it returns is the derivative of that result. -/
def IsJointJvp₂₂OfOn {sx : Shape m} {sy : Shape m'} {t : Shape n} {t' : Shape n'} {u : Shape k}
    {u' : Shape k'}
    (jvp : Tan (.float wx) (Index sx) → Tan (.float wy) (Index sy) →
      Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) →
      Option (Tan d₁ (Index u) × Tan d₂ (Index u')))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) →
      Tensor (.float wx) (Index t) × Tensor (.float wy) (Index t'))
    (dom : Set ((Index sx → ℝ) × (Index sy → ℝ))) : Prop :=
  ∀ x y, (x, y) ∈ dom →
    ∃ (F : (Index sx → ℝ) × (Index sy → ℝ) → Index t → ℝ)
      (F' : (Index sx → ℝ) × (Index sy → ℝ) → Index t' → ℝ)
      (L : (Index sx → ℝ) × (Index sy → ℝ) →L[ℝ] Index t → ℝ)
      (L' : (Index sx → ℝ) × (Index sy → ℝ) →L[ℝ] Index t' → ℝ),
    (∀ᶠ p in nhds (x, y), f (ofReal p.1) (ofReal p.2) = (ofReal (F p), ofReal (F' p))) ∧
    HasFDerivAt F L (x, y) ∧ HasFDerivAt F' L' (x, y) ∧
    ∀ (zg zh : Bool) g h, zg = false ∨ zh = false →
      ∃ a b, jvp (.ofReal zg g) (.ofReal zh h) (ofReal x) (ofReal y) = some (a, b) ∧
        a.Same (ofReal (w := wx) (L (if zg then 0 else g, if zh then 0 else h))) ∧
        b.Same (ofReal (w := wy) (L' (if zg then 0 else g, if zh then 0 else h)))

/-- `jvp` is a correct JVP rule for the binary `f` with two results at every pair of finite arrays. -/
def IsJointJvp₂₂Of {sx : Shape m} {sy : Shape m'} {t : Shape n} {t' : Shape n'} {u : Shape k}
    {u' : Shape k'}
    (jvp : Tan (.float wx) (Index sx) → Tan (.float wy) (Index sy) →
      Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) →
      Option (Tan d₁ (Index u) × Tan d₂ (Index u')))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) →
      Tensor (.float wx) (Index t) × Tensor (.float wy) (Index t')) : Prop :=
  IsJointJvp₂₂OfOn jvp f Set.univ

/-! ## JVP rules of ternary primitives -/

/-- `jvpA`, `jvpB` and `jvpC` are correct JVP rules for the ternary `f`
at every triple of finite arrays in `dom`, as `IsJvp₂OfOn` for three arguments:
near `(a, b, c)`, `f` computes a real function `F`, which is differentiable at `(a, b, c)`,
and each rule, applied to a finite tangent of its argument,
returns the derivative of `F` at `(a, b, c)` applied to that tangent and zero tangents of the others.
`standard_jvp` adds the results of the rules whose tangents are not symbolic zeros,
so the sum is correct for every pattern of symbolic zeros if and only if each rule is.
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
    (∀ ga, (jvpA (ofReal ga) (ofReal a) (ofReal b) (ofReal c)).Same (ofReal (w := w) (L (ga, 0, 0)))) ∧
    (∀ gb, (jvpB (ofReal gb) (ofReal a) (ofReal b) (ofReal c)).Same (ofReal (w := w) (L (0, gb, 0)))) ∧
    ∀ gc, (jvpC (ofReal gc) (ofReal a) (ofReal b) (ofReal c)).Same (ofReal (w := w) (L (0, 0, gc)))

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

/-! ## JVP rules on symbolic zeros

The rules of `ad.defjvp` and `ad.deflinear2` take arrays.
These are the functions of `jax/_src/interpreters/ad.py` that pass them tangents
that may be symbolic zeros, as `JVPTrace.process_primitive` does for a jaxpr.
-/

/-- `standard_jvp` for a unary primitive `f`, `jax/_src/interpreters/ad.py:978`:
the rule is evaluated for an array tangent,
and a symbolic zero tangent gives the symbolic zero of the output of `f`, `p2tz(val_out)`. -/
def standardJvp₁ {s : Shape m} {t : Shape n} {t' : Shape n'}
    (jvp : Tensor (.float w) (Index s) → Tensor (.float w) (Index s) → Tensor (.float w') (Index t'))
    (_f : Tensor (.float w) (Index s) → Tensor (.float w') (Index t))
    (g : Tan (.float w) (Index s)) (x : Tensor (.float w) (Index s)) : Tan (.float w') (Index t') :=
  match g with
  | .zero _ => .zero (.of (.float w') t)
  | .arr g => .arr (jvp g x)

/-- `standard_jvp` for a binary primitive `f` whose two rules return arrays of the same type,
`jax/_src/interpreters/ad.py:978`:
the rule of each argument with an array tangent is evaluated,
`add_tangents` adds the results with `add_jaxvals`, an elementwise `add`,
and the sum of no results is the symbolic zero of the output of `f`, `p2tz(val_out)`.
`jax/_src/interpreters/ad.py:996`:
```
def add_tangents(x, y):
  if type(x) is Zero:
    return y
  elif type(y) is Zero:
    return x
  else:
    return add_jaxvals(x, y)
``` -/
def standardJvp₂ {sx : Shape m} {sy : Shape m'} {t : Shape n} {t' : Shape n'}
    (jvpX : Tensor (.float wx) (Index sx) → Tensor (.float wx) (Index sx) →
      Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t'))
    (jvpY : Tensor (.float wy) (Index sy) → Tensor (.float wx) (Index sx) →
      Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t'))
    (_f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t))
    (g : Tan (.float wx) (Index sx)) (h : Tan (.float wy) (Index sy))
    (x : Tensor (.float wx) (Index sx)) (y : Tensor (.float wy) (Index sy)) : Tan (.float w) (Index t') :=
  match g, h with
  | .zero _, .zero _ => .zero (.of (.float w) t)
  | .arr g, .zero _ => .arr (jvpX g x y)
  | .zero _, .arr h => .arr (jvpY h x y)
  | .arr g, .arr h => .arr (jvpX g x y + jvpY h x y)

/-- `linear_jvp` of `ad.deflinear2` for a binary primitive `f`, which applies `f` to the tangents
after `instantiate_zeros`, `jax/_src/interpreters/ad.py:942`:
```
def linear_jvp(primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  if all(type(tangent) is Zero for tangent in tangents):
    if primitive.multiple_results:
      return val_out, map(p2tz, val_out)
    return val_out, p2tz(val_out)
  else:
    tangents = map(instantiate_zeros, tangents)
    return val_out, primitive.bind(*tangents, **params)
``` -/
def linearJvp₂ {sx : Shape m} {sy : Shape m'} {t : Shape n}
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t))
    (g : Tan (.float wx) (Index sx)) (h : Tan (.float wy) (Index sy)) : Tan (.float w) (Index t) :=
  match g, h with
  | .zero _, .zero _ => .zero (.of (.float w) t)
  | g, h => .arr (f g.inst h.inst)

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

/-- `tr` is a correct transpose rule for `f` that takes and returns symbolic zeros:
for an array cotangent, it is a correct transpose rule as in `IsTransposeOf`,
and for the symbolic zero cotangent of the output of `f`, it returns a zero cotangent of the input,
an array of zeros or a symbolic zero of the aval of the input.

`backward_pass` passes a symbolic zero to a rule of `ad.primitive_transposes`
if no cotangent reaches the output of an equation, `ValAccum.freeze`, `jax/_src/interpreters/ad.py:452`,
and `ct_check` raises if a cotangent does not have the shape of the input it is for.
Rules registered with `ad.deflinear2` or `ad.fancy_transposes` never see a symbolic zero:
`linear_transpose2`, `jax/_src/interpreters/ad.py:964`, returns symbolic zeros itself,
and their rules are stated with `IsTransposeOf`. -/
def IsTanTransposeOf {s : Shape m} {s' : Shape m'} {t : Shape n}
    (tr : Tan (.float w') (Index t) → Tan d (Index s'))
    (f : Tensor (.float w) (Index s) → Tensor (.float w') (Index t)) : Prop :=
  (tr (.zero (.of (.float w') t))).Same (0 : Tensor (.float w) (Index s)) ∧
  ∀ (ct : Index t → ℝ) (x : Index s → ℝ), ∃ (y : Index t → ℝ) (u : Index s → ℝ),
    f (ofReal x) = ofReal y ∧ (tr (.arr (ofReal ct))).Same (ofReal (w := w) u) ∧
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

The cotangent and the cotangents that the rule returns may be symbolic zeros, as in `IsTanTransposeOf`.

The rule is correct if there are correct transpose rules `trX` and `trY` of `f` such that,
whenever some argument is a linear input, the rule does not raise,
returns `trX ct` and `trY ct` for the arguments that are linear inputs if `ct` is an array,
and zero cotangents of their avals if `ct` is the symbolic zero of the output of `f`. -/
def IsPartialTranspose₂Of {sx : Shape m} {sy : Shape m'} {sx' : Shape k} {sy' : Shape k'}
    {t : Shape n}
    (tr : Tan (.float w) (Index t) → Option (Tensor (.float wx) (Index sx)) →
      Option (Tensor (.float wy) (Index sy)) →
      Option (Option (Tan d₁ (Index sx')) × Option (Tan d₂ (Index sy'))))
    (f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t)) :
    Prop :=
  ∃ (trX : Tensor (.float w) (Index t) → Tensor d₁ (Index sx'))
    (trY : Tensor (.float w) (Index t) → Tensor d₂ (Index sy')),
    IsTranspose₂Of trX trY f ∧
    ∀ x y, x = none ∨ y = none →
      (∀ ct : Index t → ℝ, ∃ u v, tr (.arr (ofReal ct)) x y = some (u, v) ∧
        (x = none → u = some (.arr (trX (ofReal ct)))) ∧ (y = none → v = some (.arr (trY (ofReal ct))))) ∧
      ∃ u v, tr (.zero (.of (.float w) t)) x y = some (u, v) ∧
        (x = none → ∃ a, u = some a ∧ a.Same (0 : Tensor (.float wx) (Index sx))) ∧
        (y = none → ∃ b, v = some b ∧ b.Same (0 : Tensor (.float wy) (Index sy)))

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

/-- The sum of two arrays of the same type that are finite arrays is their sum. -/
theorem Tensor.Same.add {s : Shape m} {t : Shape n} {a b : Tensor (.float w) (Index s)}
    {u v : Index t → ℝ} (ha : a.Same (ofReal (w := w) u)) (hb : b.Same (ofReal (w := w) v)) :
    (a + b).Same (ofReal (w := w) (u + v)) := by
  obtain ⟨hmn, hs, hva⟩ := ha.apply_eq
  obtain ⟨_, -, hvb⟩ := hb.apply_eq
  refine ⟨rfl, hmn, hs, fun i j hij => ?_⟩
  show a i + b i = _
  rw [hva i j hij, hvb i j hij]
  rfl

@[simp] theorem Tan.same_arr {s : Shape m} {t : Shape n} (x : Tensor d (Index s))
    (b : Tensor (.float w) (Index t)) : (Tan.arr x).Same b ↔ x.Same b := Iff.rfl

@[simp] theorem Tan.ofReal_true {s : Shape n} (g : Index s → ℝ) :
    Tan.ofReal (w := w) true g = .zero (.of (.float w) s) := rfl

@[simp] theorem Tan.ofReal_false {s : Shape n} (g : Index s → ℝ) :
    Tan.ofReal (w := w) false g = .arr (Jax.ofReal g) := rfl

/-- The symbolic zero of the output of `f` stands for the derivative applied to zero tangents. -/
theorem Tan.same_zero_of_eq_zero {s : Shape m} {t : Shape n} {x : Index t → ℝ} (hx : x = 0) :
    (Tan.zero (d := d) (ι := Index s) (.of (.float w) t)).Same (Jax.ofReal (w := w) x) :=
  ⟨rfl, by subst hx; rfl⟩

/-- `standard_jvp` of correct rules that return arrays of the same type is correct
for every pattern of symbolic zero tangents. -/
theorem IsJvp₂OfOn.standardJvp₂ {sx : Shape m} {sy : Shape m'} {t : Shape n} {t' : Shape n'}
    {jvpX : Tensor (.float wx) (Index sx) → Tensor (.float wx) (Index sx) →
      Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t')}
    {jvpY : Tensor (.float wy) (Index sy) → Tensor (.float wx) (Index sx) →
      Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t')}
    {f : Tensor (.float wx) (Index sx) → Tensor (.float wy) (Index sy) → Tensor (.float w) (Index t)}
    {dom : Set ((Index sx → ℝ) × (Index sy → ℝ))} (h : IsJvp₂OfOn jvpX jvpY f dom) :
    IsJointJvp₂OfOn (standardJvp₂ jvpX jvpY f) f dom := by
  intro x y hxy
  obtain ⟨F, L, hF, hL, hX, hY⟩ := h x y hxy
  refine ⟨F, L, hF, hL, fun zg zh g k _ => ?_⟩
  cases zg <;> cases zh
  · have hsum : L (g, k) = L (g, 0) + L (0, k) := by rw [← map_add]; simp
    simpa [Jax.standardJvp₂, hsum] using (hX g).add (hY k)
  · simpa [Jax.standardJvp₂] using hX g
  · simpa [Jax.standardJvp₂] using hY k
  · exact Tan.same_zero_of_eq_zero (by simp only [↓reduceIte, Prod.mk_zero_zero, map_zero])

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
