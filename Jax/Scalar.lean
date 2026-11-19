module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Basic.Real.Sign
public import Jax.DType

/-!
# Scalars

The entries of an array: a real number, an infinity or NaN.

Finite values are exact reals: there is no rounding and no overflow.
Operations on the special values follow IEEE 754, as [StableHLO](https://openxla.org/stablehlo/spec) prescribes.
Zero is unsigned and behaves like `+0`, so `1 / 0 = +∞`.

A scalar has the width `w` of its float dtype. The width does not change any value,
and the operations below are the same for every width.
It makes arrays of different widths different types,
so that Lean rejects an argument of one width where a primitive expects another.
-/

@[expose] public section

namespace Jax

/-- An array entry of a float dtype of width `w`: a real number, `+∞`, `-∞` or NaN. -/
inductive Scalar (w : FloatWidth) where
  | fin (x : ℝ)
  | posInf
  | negInf
  | nan

namespace Scalar

variable {w : FloatWidth}

noncomputable section

instance : Zero (Scalar w) := ⟨fin 0⟩
instance : One (Scalar w) := ⟨fin 1⟩

/-- The infinity with the sign of `x`, and NaN for `x = 0`. -/
def infWithSign (x : ℝ) : Scalar w := if 0 < x then posInf else if x < 0 then negInf else nan

/-- Addition. `∞ + -∞` is NaN. -/
protected def add : Scalar w → Scalar w → Scalar w
  | fin x, fin y => fin (x + y)
  | fin _, posInf | posInf, fin _ | posInf, posInf => posInf
  | fin _, negInf | negInf, fin _ | negInf, negInf => negInf
  | _, _ => nan

/-- Negation. -/
protected def neg : Scalar w → Scalar w
  | fin x => fin (-x)
  | posInf => negInf
  | negInf => posInf
  | nan => nan

/-- Multiplication. `0 * ±∞` is NaN. -/
protected def mul : Scalar w → Scalar w → Scalar w
  | fin x, fin y => fin (x * y)
  | fin x, posInf | posInf, fin x => infWithSign x
  | fin x, negInf | negInf, fin x => infWithSign (-x)
  | posInf, posInf | negInf, negInf => posInf
  | posInf, negInf | negInf, posInf => negInf
  | _, _ => nan

/-- Division. `x / 0` is an infinity with the sign of `x`, and `0 / 0` and `±∞ / ±∞` are NaN. -/
protected def div : Scalar w → Scalar w → Scalar w
  | fin x, fin y => if y = 0 then infWithSign x else fin (x / y)
  | fin _, posInf | fin _, negInf => fin 0
  | posInf, fin y => if 0 ≤ y then posInf else negInf
  | negInf, fin y => if 0 ≤ y then negInf else posInf
  | _, _ => nan

instance : Add (Scalar w) := ⟨Scalar.add⟩
instance : Neg (Scalar w) := ⟨Scalar.neg⟩
instance : Mul (Scalar w) := ⟨Scalar.mul⟩
instance : Div (Scalar w) := ⟨Scalar.div⟩

/-- Sine. NaN at the infinities. -/
protected def sin : Scalar w → Scalar w
  | fin x => fin (Real.sin x)
  | _ => nan

/-- Cosine. NaN at the infinities. -/
protected def cos : Scalar w → Scalar w
  | fin x => fin (Real.cos x)
  | _ => nan

/-- Natural logarithm. `log 0 = -∞`, and NaN at negative numbers. -/
protected def log : Scalar w → Scalar w
  | fin x => if 0 < x then fin (Real.log x) else if x = 0 then negInf else nan
  | posInf => posInf
  | _ => nan

/-- Exponential. `exp(-∞) = 0` and `exp(+∞) = +∞`. -/
protected def exp : Scalar w → Scalar w
  | fin x => fin (Real.exp x)
  | posInf => posInf
  | negInf => fin 0
  | nan => nan

/-- Square root. NaN at negative numbers and at `-∞`, and `sqrt(+∞) = +∞`. -/
protected def sqrt : Scalar w → Scalar w
  | fin x => if 0 ≤ x then fin (Real.sqrt x) else nan
  | posInf => posInf
  | _ => nan

open scoped Classical in
/-- Power, as `pow` of IEEE 754.
`x ^ 0 = 1` and `1 ^ y = 1` for every `x` and `y`, even NaN.
For finite `x` and `y`, `x ^ y` is the real power if `x > 0` or `y` is an integer,
`0 ^ y` is `0` for `y > 0` and `+∞` for `y < 0`,
and `x ^ y` is NaN for `x < 0` and a non-integer `y`.
`(-∞) ^ y` is `-∞` for a positive odd integer `y`, and `(-1) ^ ±∞ = 1`. -/
protected def pow (x y : Scalar w) : Scalar w :=
  if y = 0 ∨ x = 1 then 1 else
  match x, y with
  | fin a, fin b =>
    if a = 0 then (if 0 < b then 0 else posInf)
    else if 0 < a ∨ ∃ k : ℤ, b = k then fin (a ^ b) else nan
  | fin a, posInf => if |a| < 1 then 0 else if a = -1 then 1 else posInf
  | fin a, negInf => if |a| < 1 then posInf else if a = -1 then 1 else 0
  | posInf, fin b => if b < 0 then 0 else posInf
  | negInf, fin b => if b < 0 then 0 else if ∃ k : ℤ, b = 2 * k + 1 then negInf else posInf
  | posInf, posInf | negInf, posInf => posInf
  | posInf, negInf | negInf, negInf => 0
  | _, _ => nan

/-- Maximum. NaN if either argument is NaN. -/
protected def max : Scalar w → Scalar w → Scalar w
  | fin x, fin y => fin (x ⊔ y)
  | nan, _ | _, nan => nan
  | posInf, _ | _, posInf => posInf
  | negInf, y => y
  | x, negInf => x

/-- Minimum. NaN if either argument is NaN. -/
protected def min : Scalar w → Scalar w → Scalar w
  | fin x, fin y => fin (x ⊓ y)
  | nan, _ | _, nan => nan
  | negInf, _ | _, negInf => negInf
  | posInf, y => y
  | x, posInf => x

/-- Absolute value. `|±∞| = +∞`. -/
protected def abs : Scalar w → Scalar w
  | fin x => fin |x|
  | nan => nan
  | _ => posInf

/-- Sign: `-1`, `0` or `1`, and NaN at NaN. -/
protected def sign : Scalar w → Scalar w
  | fin x => fin (Real.sign x)
  | posInf => fin 1
  | negInf => fin (-1)
  | nan => nan

/-- Floor, the largest integer that is at most `x`. The infinities are unchanged. -/
protected def floor : Scalar w → Scalar w
  | fin x => fin ⌊x⌋
  | x => x

/-- The real number `q` rounded toward zero. -/
def truncReal (q : ℝ) : ℝ := if 0 ≤ q then ⌊q⌋ else ⌈q⌉

/-- Remainder, as `fmod` of C: `x - d * y`, where `d` is `x / y` rounded toward zero.
The result has the sign of `x`.
`x rem 0` and `±∞ rem y` are NaN, and `x rem ±∞ = x` for a finite `x`. -/
protected def rem : Scalar w → Scalar w → Scalar w
  | fin x, fin y => if y = 0 then nan else fin (x - truncReal (x / y) * y)
  | fin x, posInf | fin x, negInf => fin x
  | _, _ => nan

/-- The comparison `x > y`. False if either argument is NaN. -/
protected def gt : Scalar w → Scalar w → Prop
  | fin x, fin y => x > y
  | posInf, fin _ | posInf, negInf | fin _, negInf => True
  | _, _ => False

/-- The comparison `x == y`. False if either argument is NaN. -/
protected def eq (x y : Scalar w) : Prop := x = y ∧ x ≠ nan

/-- The comparison `x != y`. True if either argument is NaN. -/
protected def ne (x y : Scalar w) : Prop := ¬ x.eq y

/-- The comparison `x < y`. False if either argument is NaN. -/
protected def lt (x y : Scalar w) : Prop := y.gt x

/-- The comparison `x >= y`. False if either argument is NaN. -/
protected def ge (x y : Scalar w) : Prop := x.gt y ∨ x.eq y

/-- The comparison `x <= y`. False if either argument is NaN. -/
protected def le (x y : Scalar w) : Prop := y.gt x ∨ x.eq y

/-- The same value at the width `w'`. Values are exact, so a conversion between widths changes no value. -/
def cast {w' : FloatWidth} : Scalar w → Scalar w'
  | fin x => fin x
  | posInf => posInf
  | negInf => negInf
  | nan => nan

/-! ## Finite values -/

@[simp] theorem cast_fin {w' : FloatWidth} (x : ℝ) : (fin x : Scalar w).cast = (fin x : Scalar w') := rfl
@[simp] theorem fin_add_fin (x y : ℝ) : (fin x : Scalar w) + fin y = fin (x + y) := rfl
@[simp] theorem fin_mul_fin (x y : ℝ) : (fin x : Scalar w) * fin y = fin (x * y) := rfl
@[simp] theorem neg_fin (x : ℝ) : -(fin x : Scalar w) = fin (-x) := rfl
@[simp] theorem sin_fin (x : ℝ) : (fin x : Scalar w).sin = fin (Real.sin x) := rfl
@[simp] theorem cos_fin (x : ℝ) : (fin x : Scalar w).cos = fin (Real.cos x) := rfl
@[simp] theorem max_fin (x y : ℝ) : (fin x : Scalar w).max (fin y) = fin (x ⊔ y) := rfl
@[simp] theorem min_fin (x y : ℝ) : (fin x : Scalar w).min (fin y) = fin (x ⊓ y) := rfl
@[simp] theorem gt_fin (x y : ℝ) : (fin x : Scalar w).gt (fin y) ↔ x > y := Iff.rfl
@[simp] theorem eq_fin (x y : ℝ) : (fin x : Scalar w).eq (fin y) ↔ x = y := by simp [Scalar.eq]
@[simp] theorem exp_fin (x : ℝ) : (fin x : Scalar w).exp = fin (Real.exp x) := rfl
@[simp] theorem abs_fin (x : ℝ) : (fin x : Scalar w).abs = fin |x| := rfl
@[simp] theorem sign_fin (x : ℝ) : (fin x : Scalar w).sign = fin (Real.sign x) := rfl
@[simp] theorem floor_fin (x : ℝ) : (fin x : Scalar w).floor = fin ⌊x⌋ := rfl
@[simp] theorem ne_fin (x y : ℝ) : (fin x : Scalar w).ne (fin y) ↔ x ≠ y := by simp [Scalar.ne]
@[simp] theorem lt_fin (x y : ℝ) : (fin x : Scalar w).lt (fin y) ↔ x < y := Iff.rfl
@[simp] theorem ge_fin (x y : ℝ) : (fin x : Scalar w).ge (fin y) ↔ x ≥ y := by
  simp [Scalar.ge, le_iff_lt_or_eq, eq_comm]
@[simp] theorem le_fin (x y : ℝ) : (fin x : Scalar w).le (fin y) ↔ x ≤ y := by
  simp [Scalar.le, le_iff_lt_or_eq]
theorem zero_def : (0 : Scalar w) = fin 0 := rfl
theorem one_def : (1 : Scalar w) = fin 1 := rfl

theorem fin_div_fin (x : ℝ) {y : ℝ} (hy : y ≠ 0) : (fin x : Scalar w) / fin y = fin (x / y) := by
  show (if y = 0 then infWithSign x else fin (x / y)) = _
  simp [hy]

theorem rem_fin (x : ℝ) {y : ℝ} (hy : y ≠ 0) :
    (fin x : Scalar w).rem (fin y) = fin (x - truncReal (x / y) * y) := by
  simp [Scalar.rem, hy]

theorem sqrt_fin {x : ℝ} (hx : 0 ≤ x) : (fin x : Scalar w).sqrt = fin (Real.sqrt x) := by
  simp [Scalar.sqrt, hx]

theorem sqrt_fin_of_neg {x : ℝ} (hx : x < 0) : (fin x : Scalar w).sqrt = nan := by
  simp [Scalar.sqrt, hx.not_ge]

theorem log_fin {x : ℝ} (hx : 0 < x) : (fin x : Scalar w).log = fin (Real.log x) := by
  show (if 0 < x then fin (Real.log x) else _) = _
  simp [hx]

/-- The power of finite values is the real power if `x > 0`,
or if `y` is an integer and not both `x = 0` and `y < 0`. -/
theorem pow_fin {x y : ℝ} (h : 0 < x ∨ (∃ k : ℤ, y = k) ∧ (x ≠ 0 ∨ 0 ≤ y)) :
    (fin x : Scalar w).pow (fin y) = fin (x ^ y) := by
  unfold Scalar.pow
  by_cases hy : y = 0
  · subst hy
    simp [zero_def, one_def, Real.rpow_zero]
  by_cases hx1 : x = 1
  · subst hx1
    simp [one_def, Real.one_rpow]
  have hc : ¬ ((fin y : Scalar w) = 0 ∨ (fin x : Scalar w) = 1) := by simp [zero_def, one_def, hy, hx1]
  simp only [hc, ↓reduceIte]
  by_cases hx : x = 0
  · subst hx
    have : 0 < y := by
      rcases h with h | ⟨_, h | h⟩
      · exact absurd h (lt_irrefl 0)
      · exact absurd rfl h
      · exact lt_of_le_of_ne h (Ne.symm hy)
    simp [this, Real.zero_rpow hy, zero_def]
  · simp [hx, h.imp_right (·.1)]

/-- Addition is associative and commutative because finite values do not round. -/
instance : AddCommMonoid (Scalar w) where
  add_assoc := by
    rintro (a | _ | _ | _) (b | _ | _ | _) (c | _ | _ | _) <;>
      first | rfl | exact congrArg fin (add_assoc a b c)
  zero_add := by
    rintro (a | _ | _ | _) <;> first | rfl | exact congrArg fin (zero_add a)
  add_zero := by
    rintro (a | _ | _ | _) <;> first | rfl | exact congrArg fin (add_zero a)
  add_comm := by
    rintro (a | _ | _ | _) (b | _ | _ | _) <;> first | rfl | exact congrArg fin (add_comm a b)
  nsmul := nsmulRec

/-- Maximum is commutative, also with NaN and the infinities. -/
instance : Std.Commutative (Scalar.max (w := w)) where
  comm := by
    rintro (a | _ | _ | _) (b | _ | _ | _) <;> first | rfl | exact congrArg fin (sup_comm a b)

/-- Maximum is associative, also with NaN and the infinities. -/
instance : Std.Associative (Scalar.max (w := w)) where
  assoc := by
    rintro (a | _ | _ | _) (b | _ | _ | _) (c | _ | _ | _) <;>
      first | rfl | exact congrArg fin (sup_assoc a b c)

/-- The sum of finite values is finite. -/
theorem fin_sum {ι : Type} (S : Finset ι) (f : ι → ℝ) :
    (fin (∑ i ∈ S, f i) : Scalar w) = ∑ i ∈ S, fin (f i) :=
  map_sum (⟨⟨fin, rfl⟩, fun _ _ => rfl⟩ : ℝ →+ Scalar w) f S

end

end Scalar

namespace DType

/-- The entries of an array of dtype `d`. -/
abbrev Elem : DType → Type
  | bool => Bool
  | int _ => ℤ
  | float w => Scalar w
  | float0 => Unit

end DType

end Jax
