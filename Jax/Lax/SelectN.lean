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

variable {n n' k : ℕ} {sw : Shape k} {s0 : Shape n} {s1 : Shape n'} {w : FloatWidth}

/-- The JVP rule of `select_n`, which selects the tangents of the cases.
The tangents come first, then the cases. `which` has no tangent.
If one tangent is a symbolic zero, it is replaced by zeros of the shape of the other,
which the shape check of `select_n` makes the shape of both cases.
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
def selectNJvp (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0) (which : Tensor .bool (Index sw))
    (caseDot0 : Tan (.float w) (Index s0)) (caseDot1 : Tan (.float w) (Index s1)) (_case0 : Tensor (.float w) (Index s0))
    (_case1 : Tensor (.float w) (Index s1)) : Tan (.float w) (Index s0) :=
  match caseDot0, caseDot1 with
  | .zero aval, .zero _ => .zero aval
  | .zero _, .arr t1 => .arr (selectN hc hw which 0 t1)
  | .arr t0, .zero _ => .arr (selectN hc hw which t0 0)
  | .arr t0, .arr t1 => .arr (selectN hc hw which t0 t1)

/-- The cotangent of the first case: the cotangent where `which` is false.
`select(p, a, b)` is `select_n(p, b, a)`, and `zeros = full_like(ct, 0)` has the shape of `ct`.
`jax/_src/lax/lax.py:8916`: `ct0 = select(which, zeros, ct) if ad.is_undefined_primal(cases[0]) else None` -/
def selectNTranspose0 (hw : WhichShape sw s0) (which : Tensor .bool (Index sw)) (ct : Tensor (.float w) (Index s0)) :
    Tensor (.float w) (Index s0) :=
  selectN rfl hw which ct 0

/-- The cotangent of the second case: the cotangent where `which` is true.
`jax/_src/lax/lax.py:8917`: `ct1 = select(which, ct, zeros) if ad.is_undefined_primal(cases[1]) else None` -/
def selectNTranspose1 (hw : WhichShape sw s0) (which : Tensor .bool (Index sw)) (ct : Tensor (.float w) (Index s0)) :
    Tensor (.float w) (Index s0) :=
  selectN rfl hw which 0 ct

/-- The transpose rule of `select_n`, for a boolean `which` and two cases.
A case that is a linear input is `none`, as in `IsPartialTranspose₂Of`, and so is its cotangent in the result
if it is not one. The cotangent `None` of `which`, which is never a linear input, is dropped.
For a symbolic zero cotangent, the cotangent of a case is the symbolic zero of the aval of the case.
`jax/_src/lax/lax.py:8906`:
```
def _select_transpose_rule(ct, which, *cases):
  assert not ad.is_undefined_primal(which)
  if type(ct) is ad_util.Zero:
    return [None] + [ad_util.Zero(c.aval) if ad.is_undefined_primal(c) else None
                     for c in cases]
  elif len(cases) == 1:
    return (None, ct if ad.is_undefined_primal(cases[0]) else None)
  else:
    zeros = full_like(ct, 0)
    if dtypes.dtype(which) == np.dtype(np.bool_):
      ct0 = select(which, zeros, ct) if ad.is_undefined_primal(cases[0]) else None
      ct1 = select(which, ct, zeros) if ad.is_undefined_primal(cases[1]) else None
      return (None, ct0, ct1)
    ...
``` -/
def selectNTranspose (hw : WhichShape sw s0) (which : Tensor .bool (Index sw)) (ct : Tan (.float w) (Index s0))
    (case0 : Option (Tensor (.float w) (Index s0))) (case1 : Option (Tensor (.float w) (Index s1))) :
    Option (Option (Tan (.float w) (Index s0)) × Option (Tan (.float w) (Index s0))) :=
  match ct with
  | .zero _ => some (if case0.isNone then some (.zero (.of (.float w) s0)) else none,
      if case1.isNone then some (.zero (.of (.float w) s1)) else none)
  | .arr ct => some (if case0.isNone then some (.arr (selectNTranspose0 hw which ct)) else none,
      if case1.isNone then some (.arr (selectNTranspose1 hw which ct)) else none)

/-! ## Proofs -/

/-- `select_n` on finite arrays, a linear map of the pair of cases. -/
private noncomputable def selectNReal (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0)
    (which : Tensor .bool (Index sw)) : (Index s0 → ℝ) × (Index s1 → ℝ) →L[ℝ] Index s0 → ℝ :=
  ContinuousLinearMap.pi fun j => if which (Index.ofWhich hw j) then
    ContinuousLinearMap.proj (Index.castShape hc j) ∘L ContinuousLinearMap.snd ℝ _ _
  else ContinuousLinearMap.proj j ∘L ContinuousLinearMap.fst ℝ _ _

private theorem selectN_ofReal (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0)
    (which : Tensor .bool (Index sw)) (x : Index s0 → ℝ) (y : Index s1 → ℝ) :
    selectN (d := .float w) hc hw which (ofReal x) (ofReal y) = ofReal (selectNReal hc hw which (x, y)) := by
  funext j
  simp only [selectN, selectNReal, ofReal, ContinuousLinearMap.pi_apply]
  split_ifs <;> rfl

/-! ## Correctness -/

/-- The JVP rule of `select_n` computes the derivative of `select_n` for a fixed `which`. -/
theorem selectNJvp_correct (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0)
    (which : Tensor .bool (Index sw)) :
    IsJointJvp₂Of (selectNJvp hc hw which) (selectN (d := .float w) hc hw which) := by
  intro x y _
  let L := selectNReal hc hw which
  refine ⟨L, L, .of_forall fun p => selectN_ofReal hc hw which p.1 p.2, L.hasFDerivAt,
    fun zg zh g k hz => ?_⟩
  cases zg <;> cases zh
  · exact .of_eq (selectN_ofReal hc hw which g k)
  · exact .of_eq (selectN_ofReal hc hw which g 0)
  · exact .of_eq (selectN_ofReal hc hw which 0 k)
  · simp at hz

/-- The cotangents of the two cases transpose `select_n` for a fixed `which`. -/
theorem selectN_isTranspose₂Of (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0)
    (which : Tensor .bool (Index sw)) :
    IsTranspose₂Of (selectNTranspose0 hw which) (selectNTranspose1 hw which)
      (selectN (d := .float w) hc hw which) := by
  intro ct x y
  let wh : Index s0 → Bool := fun j => which (Index.ofWhich hw j)
  obtain rfl := rank_eq_of_ofFn_eq hc
  have hc' : ∀ d, s1 d = s0 d := fun d => size_eq_of_ofFn_eq hc d
  let e : Index s0 ≃ Index s1 := Equiv.piCongrRight fun d => finCongr (hc' d).symm
  have he : ∀ j, e j = Index.castShape hc j := fun _ => rfl
  refine ⟨selectNReal hc hw which (x, y), fun j => if wh j then 0 else ct j,
    fun i => if wh (e.symm i) then ct (e.symm i) else 0, selectN_ofReal hc hw which x y,
    .of_eq (funext fun j => ?_), ?_, ?_⟩
  · simp only [selectNTranspose0, selectN, ofReal, wh]
    split_ifs <;> rfl
  · refine ⟨rfl, rfl, fun d => (hc' d).symm, fun j i hij => ?_⟩
    have hi : i = e j := Index.ext_val fun d => (hij d).symm
    subst hi
    simp only [selectNTranspose1, selectN, ofReal, Equiv.symm_apply_apply, wh]
    split_ifs <;> rfl
  · rw [← Fintype.sum_equiv e (fun j => (if wh (e.symm (e j)) then ct (e.symm (e j)) else 0) *
      y (e j)) _ fun _ => rfl, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [Equiv.symm_apply_apply]
    simp only [selectNReal, ContinuousLinearMap.pi_apply, wh, he]
    split_ifs <;> simp

/-- The transpose rule of `select_n` transposes `select_n` for a fixed `which`,
whichever cases are linear inputs, also for a symbolic zero cotangent. -/
theorem selectNTranspose_correct (hc : List.ofFn s1 = List.ofFn s0) (hw : WhichShape sw s0)
    (which : Tensor .bool (Index sw)) :
    IsPartialTranspose₂Of (selectNTranspose hw which) (selectN (d := .float w) hc hw which) := by
  refine ⟨_, _, selectN_isTranspose₂Of hc hw which, fun x y _ => ⟨fun ct => ?_, ?_⟩⟩
  · exact ⟨_, _, rfl, fun hx => by simp [hx], fun hy => by simp [hy]⟩
  · exact ⟨_, _, rfl, fun hx => by simp [hx, Tan.Same], fun hy => by simp [hy, Tan.Same]⟩

end Jax.Lax
