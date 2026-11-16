module

public import Jax.Lax.Primal

/-!
# `select_n`

Differentiation rules of the `select_n` primitive of `jax/_src/lax/lax.py`,
for a boolean `which` and two cases.

For a fixed `which`, `select_n` is linear in the cases, and both rules are correct everywhere.
`which` is boolean and has no tangent. Where it is computed from the input,
the chain rule through `which` holds where the computation of `which` is locally constant,
which the zero tangents of the comparisons claim (`Jax/Lax/Compare.lean`).
-/

@[expose] public section

namespace Jax.Lax

variable {n n' k : ℕ} {sw : Shape k} {s0 : Shape n} {s1 : Shape n'}

/-- The JVP rule of `select_n`, which selects the tangents of the cases.
The tangents come first, then the cases. `which` has no tangent.
Symbolic zero tangents are not modelled: `_select_jvp` replaces them by zeros.
`jax/_src/lax/lax.py:8972`:
```
def _select_jvp(primals, tangents):
  which, *case_primals = primals
  case_tangents = tangents[1:]
  out = select_n(which, *case_primals)
  if all(type(t) is ad_util.Zero for t in case_tangents):
    out_dot = ad_util.Zero(case_tangents[0].aval)
  else:
    z = _zeros(next(t for t in case_tangents if type(t) is not ad_util.Zero))
    case_tangents = [z if type(t) is ad_util.Zero else t for t in case_tangents]
    out_dot = select_n(which, *case_tangents)
  return out, out_dot
``` -/
def selectNJvp (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0) (which : Index sw → Bool)
    (caseDot0 : Tensor (Index s0)) (caseDot1 : Tensor (Index s1)) (_case0 : Tensor (Index s0))
    (_case1 : Tensor (Index s1)) : Tensor (Index s0) :=
  selectN hc hw which caseDot0 caseDot1

/-- The transpose rule of `select_n` for its first case: the cotangent where `which` is false.
`select(p, a, b)` is `select_n(p, b, a)`, and `zeros = full_like(ct, 0)` has the shape of `ct`.
`jax/_src/lax/lax.py:8906`:
```
def _select_transpose_rule(ct, which, *cases):
  assert not ad.is_undefined_primal(which)
  if type(ct) is ad_util.Zero:
    ...
  else:
    zeros = full_like(ct, 0)
    if dtypes.dtype(which) == np.dtype(np.bool_):
      ct0 = select(which, zeros, ct) if ad.is_undefined_primal(cases[0]) else None
      ct1 = select(which, ct, zeros) if ad.is_undefined_primal(cases[1]) else None
      return (None, ct0, ct1)
``` -/
def selectNTranspose0 (hw : WhichShape sw s0) (which : Index sw → Bool) (ct : Tensor (Index s0)) :
    Tensor (Index s0) :=
  selectN rfl hw which ct 0

/-- The transpose rule of `select_n` for its second case: the cotangent where `which` is true.
`jax/_src/lax/lax.py:8917`: `ct1 = select(which, ct, zeros) if ad.is_undefined_primal(cases[1]) else None` -/
def selectNTranspose1 (hw : WhichShape sw s0) (which : Index sw → Bool) (ct : Tensor (Index s0)) :
    Tensor (Index s0) :=
  selectN rfl hw which 0 ct

/-! ## Proofs -/

/-- `select_n` on finite arrays, a linear map of the pair of cases. -/
private noncomputable def selectNReal (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0)
    (which : Index sw → Bool) : (Index s0 → ℝ) × (Index s1 → ℝ) →L[ℝ] Index s0 → ℝ :=
  ContinuousLinearMap.pi fun j => if which (Index.ofWhich hw j) then
    ContinuousLinearMap.proj (Index.castShape hc j) ∘L ContinuousLinearMap.snd ℝ _ _
  else ContinuousLinearMap.proj j ∘L ContinuousLinearMap.fst ℝ _ _

private theorem selectN_ofReal (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0)
    (which : Index sw → Bool) (x : Index s0 → ℝ) (y : Index s1 → ℝ) :
    selectN hc hw which (ofReal x) (ofReal y) = ofReal (selectNReal hc hw which (x, y)) := by
  funext j
  simp only [selectN, selectNReal, ofReal, ContinuousLinearMap.pi_apply]
  split_ifs <;> rfl

/-! ## Correctness -/

/-- The JVP rule of `select_n` computes the derivative of `select_n` for a fixed `which`. -/
theorem selectNJvp_correct (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0)
    (which : Index sw → Bool) :
    IsJointJvp₂Of (selectNJvp hc hw which) (selectN hc hw which) := by
  intro x y _
  let L := selectNReal hc hw which
  exact ⟨L, L, .of_forall fun p => selectN_ofReal hc hw which p.1 p.2, L.hasFDerivAt,
    fun g h => .of_eq (selectN_ofReal hc hw which g h)⟩

/-- The transpose rules of `select_n` transpose `select_n` for a fixed `which`. -/
theorem selectNTranspose_correct (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0)
    (which : Index sw → Bool) :
    IsTranspose₂Of (selectNTranspose0 hw which) (selectNTranspose1 hw which)
      (selectN hc hw which) := by
  intro ct x y
  let w : Index s0 → Bool := fun j => which (Index.ofWhich hw j)
  obtain rfl := rank_eq_of_ofFn_eq hc
  have hc' : ∀ d, s1 d = s0 d := fun d => size_eq_of_ofFn_eq hc d
  let e : Index s0 ≃ Index s1 := Equiv.piCongrRight fun d => finCongr (hc' d).symm
  have he : ∀ j, e j = Index.castShape hc j := fun _ => rfl
  refine ⟨selectNReal hc hw which (x, y), fun j => if w j then 0 else ct j,
    fun i => if w (e.symm i) then ct (e.symm i) else 0, selectN_ofReal hc hw which x y,
    .of_eq (funext fun j => ?_), ?_, ?_⟩
  · simp only [selectNTranspose0, selectN, ofReal, w]
    split_ifs <;> rfl
  · refine ⟨rfl, fun d => (hc' d).symm, fun j i hij => ?_⟩
    have hi : i = e j := Index.ext_val fun d => (hij d).symm
    subst hi
    simp only [selectNTranspose1, selectN, ofReal, Equiv.symm_apply_apply, w]
    split_ifs <;> rfl
  · rw [← Fintype.sum_equiv e (fun j => (if w (e.symm (e j)) then ct (e.symm (e j)) else 0) *
      y (e j)) _ fun _ => rfl, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [Equiv.symm_apply_apply]
    simp only [selectNReal, ContinuousLinearMap.pi_apply, w, he]
    split_ifs <;> simp

end Jax.Lax
