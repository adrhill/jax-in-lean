module

public import Mathlib.Data.Finset.Sort
public import Mathlib.Data.Fintype.Pi

/-!
# Shapes and indices

A shape of rank `n` gives the size of each of its `n` axes.
The JAX shape `(2, 3)` is `![2, 3] : Shape 2`.
-/

@[expose] public section

namespace Jax

/-- An array shape of rank `n`. -/
abbrev Shape (n : ℕ) := Fin n → ℕ

/-- A position in an array of shape `s`: one index per axis. -/
abbrev Index {n : ℕ} (s : Shape n) := (d : Fin n) → Fin (s d)

variable {m n : ℕ}

/-- The index that keeps axis `e d` of `i` as its axis `d`. -/
def Index.proj {s : Shape n} (e : Fin m → Fin n) (i : Index s) : Index (s ∘ e) := fun d => i (e d)

/-! ## Broadcasting -/

/-- An array of shape `s` can be broadcast to shape `t`: every axis has size 1 or the size in `t`. -/
def BroadcastsTo (s t : Shape n) : Prop := ∀ d, s d = 1 ∨ s d = t d

/-- The entry of an array of shape `s` that is read at position `j` once it is broadcast to shape `t`.
[StableHLO `broadcast_in_dim`](https://openxla.org/stablehlo/spec#broadcast_in_dim):
`operand_index[d] = 0` if `dim(operand, d) = 1`, and `result_index[d]` otherwise. -/
def Index.unbroadcast {s t : Shape n} (h : BroadcastsTo s t) (j : Index t) : Index s :=
  fun d => if h1 : s d = 1 then ⟨0, by omega⟩ else ⟨j d, by have := h d; have := (j d).isLt; omega⟩

/-- Two shapes can be broadcast against each other: on every axis the sizes are equal or one is 1. -/
def Broadcastable (s t : Shape n) : Prop := ∀ d, s d = t d ∨ s d = 1 ∨ t d = 1

/-- The shape of the broadcast of arrays of shapes `s` and `t`.
`jax/_src/lax/lax.py:121`: `# if all dims are equal (or 1), the result is the non-1 size` -/
def broadcastShape (s t : Shape n) : Shape n := fun d => if s d = 1 then t d else s d

/-- The axes that remain when `axes` are removed, in increasing order.
This is `np.delete(np.arange(n), axes)`. -/
def keptAxes (axes : Finset (Fin n)) : Fin axesᶜ.card → Fin n := axesᶜ.orderEmbOfFin rfl

/-! ## Lemmas -/

theorem BroadcastsTo.refl (s : Shape n) : BroadcastsTo s s := fun _ => Or.inr rfl

theorem Broadcastable.refl (s : Shape n) : Broadcastable s s := fun _ => Or.inl rfl

theorem Broadcastable.symm {s t : Shape n} (h : Broadcastable s t) : Broadcastable t s :=
  fun d => by have := h d; omega

theorem broadcastsTo_left (s t : Shape n) : BroadcastsTo s (broadcastShape s t) :=
  fun d => by unfold broadcastShape; split <;> simp_all

theorem Broadcastable.right {s t : Shape n} (h : Broadcastable s t) :
    BroadcastsTo t (broadcastShape s t) :=
  fun d => by unfold broadcastShape; have := h d; split <;> omega

theorem broadcastable_left (s t : Shape n) : Broadcastable s (broadcastShape s t) :=
  fun d => by unfold broadcastShape; split <;> simp_all

theorem Index.ext_val {s : Shape n} {i j : Index s} (h : ∀ d, (i d : ℕ) = j d) : i = j :=
  funext fun d => Fin.ext (h d)

theorem Index.unbroadcast_val {s t : Shape n} (h : BroadcastsTo s t) (j : Index t) (d : Fin n) :
    (j.unbroadcast h d : ℕ) = if s d = 1 then 0 else (j d : ℕ) := by
  unfold Index.unbroadcast
  split <;> rfl

theorem Index.unbroadcast_eq_of_val {s t : Shape n} (h : BroadcastsTo s t) (j : Index t)
    (i : Index s) (hij : ∀ d, (j d : ℕ) = i d) : j.unbroadcast h = i := by
  refine Index.ext_val fun d => ?_
  rw [Index.unbroadcast_val]
  have := (i d).isLt
  split <;> have := hij d <;> omega

theorem Index.unbroadcast_refl {s : Shape n} (j : Index s) :
    j.unbroadcast (BroadcastsTo.refl s) = j :=
  Index.unbroadcast_eq_of_val _ j j fun _ => rfl

theorem Index.unbroadcast_congr {s t t' : Shape n} (h : BroadcastsTo s t) (h' : BroadcastsTo s t')
    (j : Index t) (j' : Index t') (hjj : ∀ d, (j d : ℕ) = j' d) :
    j.unbroadcast h = j'.unbroadcast h' := by
  refine Index.ext_val fun d => ?_
  rw [Index.unbroadcast_val, Index.unbroadcast_val, hjj d]

end Jax
