module

public import Jax.Lax.Primal

/-!
# Comparisons

Differentiation rules of the comparisons `eq`, `ne`, `ge`, `gt`, `le` and `lt` of `jax/_src/lax/lax.py`.
They share one file because they share their rule and their proofs.

Each comparison has the zero tangent of `ad.defjvp_zero`, `jax/_src/lax/lax.py:5716` to `5738`.
It is correct where no two compared entries are equal: there the comparison is locally constant.
At a tie no rule is correct, because the comparison jumps.
This is a convention of JAX and not a bug, as for `max` at a tie.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {sx sy : Shape n}

/-- The pairs of finite arrays in which no two compared entries are equal. -/
def noTies (h : Broadcastable sx sy) : Set ((Index sx → ℝ) × (Index sy → ℝ)) :=
  {p | ∀ j : Index (broadcastShape sx sy),
    p.1 (j.unbroadcast (broadcastsTo_left sx sy)) ≠ p.2 (j.unbroadcast h.right)}

/-! ## Proofs -/

/-- A comparison of finite values that depends only on their order is locally constant
where the compared entries differ. -/
private theorem isZeroJvp₂OfOn_binop (h : Broadcastable sx sy) (op : Scalar → Scalar → Bool)
    (hop : ∀ a b a' b' : ℝ, (a < b ↔ a' < b') → (b < a ↔ b' < a') →
      op (.fin a) (.fin b) = op (.fin a') (.fin b')) :
    IsZeroJvp₂OfOn (binop op h) (noTies h) := by
  intro x y hxy
  let ux : Index (broadcastShape sx sy) → Index sx :=
    fun j => j.unbroadcast (broadcastsTo_left sx sy)
  let uy : Index (broadcastShape sx sy) → Index sy := fun j => j.unbroadcast h.right
  have hj : ∀ j, ∀ᶠ p in nhds (x, y),
      binop op h (ofReal p.1) (ofReal p.2) j = binop op h (ofReal x) (ofReal y) j := by
    intro j
    rcases lt_or_gt_of_ne (hxy j) with hlt | hlt
    · have hopen : IsOpen {p : (Index sx → ℝ) × (Index sy → ℝ) | p.1 (ux j) < p.2 (uy j)} :=
        isOpen_lt (by fun_prop) (by fun_prop)
      filter_upwards [hopen.mem_nhds hlt] with p hp
      exact hop _ _ _ _ (iff_of_true hp hlt) (iff_of_false hp.not_gt hlt.not_gt)
    · have hopen : IsOpen {p : (Index sx → ℝ) × (Index sy → ℝ) | p.2 (uy j) < p.1 (ux j)} :=
        isOpen_lt (by fun_prop) (by fun_prop)
      filter_upwards [hopen.mem_nhds hlt] with p hp
      exact hop _ _ _ _ (iff_of_false hp.not_gt hlt.not_gt) (iff_of_true hp hlt)
  filter_upwards [Filter.eventually_all.2 hj] with p hp
  exact funext hp

/-- A comparison that jumps at a tie has no correct zero tangent there. -/
private theorem not_isZeroJvp₂Of_binop (op : Scalar → Scalar → Bool)
    (hjump : ∃ᶠ t in nhds (0 : ℝ), op (.fin t) (.fin 0) ≠ op (.fin 0) (.fin 0)) :
    ¬ IsZeroJvp₂Of (binop op (.refl ![1])) := by
  intro hz
  let path : ℝ → (Index ![1] → ℝ) × (Index ![1] → ℝ) := fun t => (fun _ => t, 0)
  have hpath : Filter.Tendsto path (nhds 0) (nhds (0, 0)) := by
    have hc : Continuous path := by fun_prop
    exact hc.tendsto 0
  let j₀ : Index (broadcastShape ![1] ![1]) := fun _ => ⟨0, by simp [broadcastShape]⟩
  exact hjump ((hpath.eventually (hz 0 0 (Set.mem_univ _))).mono fun t ht hne =>
    hne (congrFun ht j₀))

private theorem frequently_pos {P : ℝ → Prop} (h : ∀ t, 0 < t → P t) :
    ∃ᶠ t in nhds (0 : ℝ), P t :=
  (eventually_nhdsWithin_of_forall (s := Set.Ioi 0) h).frequently.filter_mono nhdsWithin_le_nhds

private theorem frequently_neg {P : ℝ → Prop} (h : ∀ t, t < 0 → P t) :
    ∃ᶠ t in nhds (0 : ℝ), P t :=
  (eventually_nhdsWithin_of_forall (s := Set.Iio 0) h).frequently.filter_mono nhdsWithin_le_nhds

private theorem eq_iff_not_lt {a b : ℝ} : a = b ↔ ¬ a < b ∧ ¬ b < a :=
  ⟨fun h => by simp [h], fun ⟨h1, h2⟩ => le_antisymm (not_lt.1 h2) (not_lt.1 h1)⟩

/-! ## Correctness -/

/-- The zero tangent of `eq` is correct where no two compared entries are equal. -/
theorem eqJvp_correct (h : Broadcastable sx sy) : IsZeroJvp₂OfOn (eq h) (noTies h) :=
  isZeroJvp₂OfOn_binop h _ fun a b a' b' h1 h2 => by
    simp only [Scalar.eq_fin, decide_eq_decide]
    rw [eq_iff_not_lt, eq_iff_not_lt, h1, h2]

/-- The zero tangent of `ne` is correct where no two compared entries are equal. -/
theorem neJvp_correct (h : Broadcastable sx sy) : IsZeroJvp₂OfOn (ne h) (noTies h) :=
  isZeroJvp₂OfOn_binop h _ fun a b a' b' h1 h2 => by
    simp only [Scalar.ne_fin, decide_eq_decide]
    rw [ne_eq, ne_eq, eq_iff_not_lt, eq_iff_not_lt, h1, h2]

/-- The zero tangent of `ge` is correct where no two compared entries are equal. -/
theorem geJvp_correct (h : Broadcastable sx sy) : IsZeroJvp₂OfOn (ge h) (noTies h) :=
  isZeroJvp₂OfOn_binop h _ fun a b a' b' h1 _ => by
    simp only [Scalar.ge_fin, ge_iff_le, ← not_lt, h1]

/-- The zero tangent of `gt` is correct where no two compared entries are equal. -/
theorem gtJvp_correct (h : Broadcastable sx sy) : IsZeroJvp₂OfOn (gt h) (noTies h) :=
  isZeroJvp₂OfOn_binop h _ fun a b a' b' _ h2 => by
    simp only [Scalar.gt_fin, gt_iff_lt, h2]

/-- The zero tangent of `le` is correct where no two compared entries are equal. -/
theorem leJvp_correct (h : Broadcastable sx sy) : IsZeroJvp₂OfOn (le h) (noTies h) :=
  isZeroJvp₂OfOn_binop h _ fun a b a' b' _ h2 => by
    simp only [Scalar.le_fin, ← not_lt, h2]

/-- The zero tangent of `lt` is correct where no two compared entries are equal. -/
theorem ltJvp_correct (h : Broadcastable sx sy) : IsZeroJvp₂OfOn (lt h) (noTies h) :=
  isZeroJvp₂OfOn_binop h _ fun a b a' b' h1 _ => by
    simp only [Scalar.lt_fin, h1]

/-! ## Refutations at a tie -/

/-- The zero tangent of `eq` is wrong at the tie `x = y = 0` of shape `(1,)`. -/
theorem eqJvp_not_correct : ¬ IsZeroJvp₂Of (eq (.refl ![1])) :=
  not_isZeroJvp₂Of_binop _ (frequently_pos fun t ht => by simp [ht.ne'])

/-- The zero tangent of `ne` is wrong at the tie `x = y = 0` of shape `(1,)`. -/
theorem neJvp_not_correct : ¬ IsZeroJvp₂Of (ne (.refl ![1])) :=
  not_isZeroJvp₂Of_binop _ (frequently_pos fun t ht => by simp [ht.ne'])

/-- The zero tangent of `ge` is wrong at the tie `x = y = 0` of shape `(1,)`. -/
theorem geJvp_not_correct : ¬ IsZeroJvp₂Of (ge (.refl ![1])) :=
  not_isZeroJvp₂Of_binop _ (frequently_neg fun t ht => by simp [ht])

/-- The zero tangent of `gt` is wrong at the tie `x = y = 0` of shape `(1,)`. -/
theorem gtJvp_not_correct : ¬ IsZeroJvp₂Of (gt (.refl ![1])) :=
  not_isZeroJvp₂Of_binop _ (frequently_pos fun t ht => by simp [ht])

/-- The zero tangent of `le` is wrong at the tie `x = y = 0` of shape `(1,)`. -/
theorem leJvp_not_correct : ¬ IsZeroJvp₂Of (le (.refl ![1])) :=
  not_isZeroJvp₂Of_binop _ (frequently_pos fun t ht => by simp [ht])

/-- The zero tangent of `lt` is wrong at the tie `x = y = 0` of shape `(1,)`. -/
theorem ltJvp_not_correct : ¬ IsZeroJvp₂Of (lt (.refl ![1])) :=
  not_isZeroJvp₂Of_binop _ (frequently_neg fun t ht => by simp [ht])

end Jax.Lax
