import Jax
import Lean

/-!
# Audit

Run by `scripts/audit.sh` as `lake env lean Audit.lean`, from the root of the repository.
It fails if

* a declaration of the model depends on anything but the three standard axioms of Lean,
* a theorem of the registry below does not exist, or is proven but registered as refuted, or the reverse,
* a theorem about a rule, or a refutation, in `Jax/Lax/` or `Jax/Jaxpr/` is not in the registry,
* a definition of a rule file occurs in no statement of the registry, so that nothing is said about it,
  unless it is listed as unsettled or as a helper of proofs,
* a refutation of the registry names a bug record that does not exist,
  or a bug record names a refutation that the registry does not list for it.

This file is not a module, so that it sees the proofs of the model and not only its statements.
-/

open Lean Elab Command

namespace Jax.Audit

/-- Whether a theorem proves a property or refutes one, and what a refutation means for JAX. -/
inductive Status where
  /-- The theorem proves the property. -/
  | proven
  /-- The theorem refutes the property, and `bugs/<id>.json` shows the refutation on JAX. -/
  | bug (id : String)
  /-- The theorem refutes the property, and this is not a bug in JAX, for the given reason. -/
  | notBug (reason : String)

/-- One settled property: a theorem and its status. -/
structure Entry where
  thm : Name
  status : Status

/-- The function is not differentiable at the counterexample, so no rule can be correct there. -/
def notDifferentiable : Status := .notBug "the function is not differentiable there"

/-- At the counterexample the function returns NaN or an infinity for real arguments,
so it is not a function of real arrays and no rule can be correct for it. -/
def notReal : Status := .notBug "the function does not map real arrays to real arrays there"

/-- The counterexample breaks the promise of `GatherScatterMode.PROMISE_IN_BOUNDS`. -/
def brokenPromise : Status := .notBug "the indices break the promise of PROMISE_IN_BOUNDS"

/-- Every settled property of `Jax/Lax/` and `Jax/Jaxpr/`, in the order of the files. -/
def registry : List Entry := [
  -- `Jax/Lax/BroadcastInDim.lean`
  ⟨``Jax.Lax.broadcastInDimJvp_correct, .proven⟩,
  ⟨``Jax.Lax.broadcastInDimTranspose_correct, .proven⟩,
  ⟨``Jax.Lax.broadcastInDimTranspose_not_correct, .bug "broadcast-in-dim-transpose"⟩,
  ⟨``Jax.Lax.broadcastInDimFwd_correct, .proven⟩,
  ⟨``Jax.Lax.broadcastInDimFwd_not_correct, .bug "broadcast-in-dim-forwarding"⟩,
  -- `Jax/Lax/Clamp.lean`
  ⟨``Jax.Lax.clampJvpScalar_not_defined, .bug "clamp-jvp-scalar-bounds"⟩,
  ⟨``Jax.Lax.clampJvp_correct, .proven⟩,
  ⟨``Jax.Lax.clampJvp_not_correct, .bug "clamp-jvp-swapped-bounds"⟩,
  ⟨``Jax.Lax.clampBatch_not_correct, .bug "clamp-batch-scalar-operand"⟩,
  -- `Jax/Lax/Compare.lean`
  ⟨``Jax.Lax.eqJvp_correct, .proven⟩,
  ⟨``Jax.Lax.neJvp_correct, .proven⟩,
  ⟨``Jax.Lax.geJvp_correct, .proven⟩,
  ⟨``Jax.Lax.gtJvp_correct, .proven⟩,
  ⟨``Jax.Lax.leJvp_correct, .proven⟩,
  ⟨``Jax.Lax.ltJvp_correct, .proven⟩,
  ⟨``Jax.Lax.eqJvp_not_correct, notDifferentiable⟩,
  ⟨``Jax.Lax.neJvp_not_correct, notDifferentiable⟩,
  ⟨``Jax.Lax.geJvp_not_correct, notDifferentiable⟩,
  ⟨``Jax.Lax.gtJvp_not_correct, notDifferentiable⟩,
  ⟨``Jax.Lax.leJvp_not_correct, notDifferentiable⟩,
  ⟨``Jax.Lax.ltJvp_not_correct, notDifferentiable⟩,
  -- `Jax/Lax/ConvertElementType.lean`
  ⟨``Jax.Lax.convertFloatJvp_correct, .proven⟩,
  ⟨``Jax.Lax.convertFloatTranspose_correct, .proven⟩,
  -- `Jax/Lax/Cos.lean`
  ⟨``Jax.Lax.cosJvp_correct, .proven⟩,
  -- `Jax/Lax/Cumsum.lean`
  ⟨``Jax.Lax.cumsumJvp_correct, .proven⟩,
  ⟨``Jax.Lax.cumsumTranspose_correct, .proven⟩,
  ⟨``Jax.Lax.cumsumBatch_correct, .proven⟩,
  -- `Jax/Lax/Div.lean`
  ⟨``Jax.Lax.divJvp_correct, .proven⟩,
  ⟨``Jax.Lax.divTranspose_correct, .proven⟩,
  ⟨``Jax.Lax.div_zero_not_isTransposeOf, notReal⟩,
  -- `Jax/Lax/Exp.lean`
  ⟨``Jax.Lax.expJvp_correct, .proven⟩,
  -- `Jax/Lax/Gather.lean`
  ⟨``Jax.Lax.gatherJvp_correct, .proven⟩,
  ⟨``Jax.Lax.gather_not_isJvpOf, notReal⟩,
  ⟨``Jax.Lax.gatherTranspose_not_defined, .bug "gather-transpose-empty-indices"⟩,
  ⟨``Jax.Lax.gatherTranspose_correct, .proven⟩,
  ⟨``Jax.Lax.gatherTranspose_promiseInBounds_not_correct, brokenPromise⟩,
  -- `Jax/Lax/IntegerPow.lean`
  ⟨``Jax.Lax.integerPowJvp_correct, .proven⟩,
  -- `Jax/Lax/Log.lean`
  ⟨``Jax.Lax.logJvp_correct, .proven⟩,
  -- `Jax/Lax/Max.lean`
  ⟨``Jax.Lax.maxJvp_correct, .proven⟩,
  ⟨``Jax.Lax.maxJvp_not_correct, notDifferentiable⟩,
  -- `Jax/Lax/Mul.lean`
  ⟨``Jax.Lax.mulJvp_correct, .proven⟩,
  ⟨``Jax.Lax.mulTransposeLeft_correct, .proven⟩,
  ⟨``Jax.Lax.mulTransposeRight_correct, .proven⟩,
  -- `Jax/Lax/Pad.lean`
  ⟨``Jax.Lax.padJvp_correct, .proven⟩,
  ⟨``Jax.Lax.padTranspose_correct, .proven⟩,
  ⟨``Jax.Lax.padTranspose_not_correct, .bug "pad-transpose-padding-value"⟩,
  -- `Jax/Lax/Polynomial.lean`
  ⟨``Jax.Lax.polynomialJvp_correct, .proven⟩,
  ⟨``Jax.Lax.polynomialJvp_not_correct, .bug "polynomial-jvp-broadcast"⟩,
  ⟨``Jax.Lax.polynomial_unroll_not_independent, .bug "polynomial-unroll-infinity"⟩,
  -- `Jax/Lax/Pow.lean`
  ⟨``Jax.Lax.powJvp_correct, .proven⟩,
  ⟨``Jax.Lax.powJvpLeft_correct, .proven⟩,
  ⟨``Jax.Lax.powIntJvpLeft_correct, .proven⟩,
  ⟨``Jax.Lax.powJvpLeft_not_correct, .bug "pow-jvp-zero-exponent"⟩,
  -- `Jax/Lax/Reduce.lean`
  ⟨``Jax.Lax.reduceJvp_correct, .proven⟩,
  ⟨``Jax.Lax.reduceJvp_not_correct, .bug "reduce-jvp-noncommutative"⟩,
  ⟨``Jax.Lax.reduceJvp_correct_needs_comm, .bug "reduce-jvp-noncommutative"⟩,
  -- `Jax/Lax/ReduceMax.lean`
  ⟨``Jax.Lax.reduceMaxJvp_correct, .proven⟩,
  -- `Jax/Lax/ReduceSum.lean`
  ⟨``Jax.Lax.reduceSumJvp_correct, .proven⟩,
  ⟨``Jax.Lax.reduceSumTranspose_correct, .proven⟩,
  -- `Jax/Lax/Rem.lean`
  ⟨``Jax.Lax.remJvp_correct, .proven⟩,
  ⟨``Jax.Lax.remJvp_not_correct, notDifferentiable⟩,
  -- `Jax/Lax/ScatterAdd.lean`
  ⟨``Jax.Lax.scatterAddJvp_correct, .proven⟩,
  ⟨``Jax.Lax.scatterAdd_isTranspose₂Of, .proven⟩,
  ⟨``Jax.Lax.scatterAddTranspose_correct, .proven⟩,
  ⟨``Jax.Lax.scatterAddTranspose_not_correct, .bug "scatter-add-transpose-empty-operand"⟩,
  ⟨``Jax.Lax.scatterAddTranspose_promiseInBounds_not_correct, brokenPromise⟩,
  -- `Jax/Lax/ScatterSub.lean`
  ⟨``Jax.Lax.scatterSubJvp_correct, .proven⟩,
  ⟨``Jax.Lax.scatterSubTranspose_correct, .proven⟩,
  ⟨``Jax.Lax.scatterSubTranspose_not_correct, .bug "scatter-add-transpose-empty-operand"⟩,
  -- `Jax/Lax/SelectN.lean`
  ⟨``Jax.Lax.selectNJvp_correct, .proven⟩,
  ⟨``Jax.Lax.selectNTranspose_correct, .proven⟩,
  -- `Jax/Lax/Sin.lean`
  ⟨``Jax.Lax.sinJvp_correct, .proven⟩,
  -- `Jax/Lax/Slice.lean`
  ⟨``Jax.Lax.sliceJvp_correct, .proven⟩,
  ⟨``Jax.Lax.sliceTranspose_correct, .proven⟩,
  -- `Jax/Lax/Sqrt.lean`
  ⟨``Jax.Lax.sqrtJvp_correct, .proven⟩,
  ⟨``Jax.Lax.sqrt_not_isJvpOfOn, notDifferentiable⟩,
  -- `Jax/Jaxpr/MaxInitial.lean`
  ⟨``Jax.Jaxpr.maxInitial_hasZeroJvp, .proven⟩,
  ⟨``Jax.Jaxpr.maxInitialJvp_not_correct, .bug "reduce-max-jvp-empty-axis"⟩,
  ⟨``Jax.Jaxpr.maxInitialJvp_not_correct_vector, .bug "reduce-max-jvp-empty-axis"⟩,
  -- `Jax/Jaxpr/SelectNan.lean`
  ⟨``Jax.Jaxpr.selectNan_eq_one, .proven⟩,
  ⟨``Jax.Jaxpr.selectNanVjp_nan, .proven⟩,
  ⟨``Jax.Jaxpr.selectNanVjp_finite, .proven⟩,
  ⟨``Jax.Jaxpr.mul_nan_not_isTransposeOf, notReal⟩,
  -- `Jax/Jaxpr/SqrtRelu.lean`
  ⟨``Jax.Jaxpr.sqrtRelu_hasZeroJvp, .proven⟩,
  ⟨``Jax.Jaxpr.sqrtReluJvp_not_correct,
    .notBug "no rule of `sqrt` is correct at 0, where the jaxpr evaluates it (jax-ml/jax#6636)"⟩,
  ⟨``Jax.Jaxpr.sqrtReluVjp_of_neg, .proven⟩
]

/-- Definitions of rule files about which no registered statement says anything yet, and what is missing.
The audit accepts them and counts them. An entry that a registered statement mentions is an error. -/
def unsettled : List (Name × String) := [
  (``Jax.Lax.clampJvpMinScalar, "`clampJvpScalar_not_defined` is about its shape check; no theorem about its values"),
  (``Jax.Lax.clampJvpMaxScalar, "`clampJvpScalar_not_defined` is about its shape check; no theorem about its values")
]

/-- Definitions of rule files that only proofs use, and that are not private because another file needs them.
An entry that a registered statement mentions is an error. -/
def proofHelpers : List Name := [
  ``Jax.Lax.IsMonoidOp.toCommMonoid,
  ``Jax.Lax.scatterAddReal,
  ``Jax.Lax.scatterSubReal
]

/-! ## Checks -/

/-- The correctness notions of `Jax/Core.lean`. A theorem that states one of them is about a rule. -/
def notions : List Name := [
  ``IsJvpOfOn, ``IsJvpOf, ``IsJvp₂OfOn, ``IsJvp₂Of, ``IsJointJvp₂OfOn, ``IsJointJvp₂Of,
  ``IsJvp₃OfOn, ``IsJvp₃Of, ``IsZeroJvp₂OfOn, ``IsZeroJvp₂Of, ``IsTransposeOf, ``IsTranspose₂Of,
  ``IsPartialTranspose₂Of, ``IsFwdOf, ``IsBatchOf₁, ``IsBatchOf₃]

/-- The module that declares `n`, if it is a module of the model. -/
def modelModule? (env : Environment) (n : Name) : Option Name := do
  let idx ← env.getModuleIdxFor? n
  let m ← env.header.moduleNames[idx.toNat]?
  if m == `Jax || (`Jax).isPrefixOf m then some m else none

/-- Whether `m` is a file of rules: `Jax/Lax/<Name>.lean` or `Jax/Jaxpr/<Name>.lean`,
but not the primal semantics, which have no statements of their own. -/
def isRuleModule (m : Name) : Bool :=
  ((`Jax.Lax).isPrefixOf m || (`Jax.Jaxpr).isPrefixOf m) &&
    m != `Jax.Lax.Primal && m != `Jax.Lax.Dynamic

/-- A declaration that a person wrote and that other files can use. -/
def isWritten (env : Environment) (n : Name) : Bool :=
  !n.isInternalDetail && !isPrivateName n && !Meta.isMatcherCore env n && !isAuxRecursor env n &&
    !env.isProjectionFn n && !Meta.isInstanceCore env n

/-- The conclusion of a statement, below its hypotheses. -/
def conclusion (type : Expr) : Expr := type.getForallBody

/-- Whether a statement is a refutation: its conclusion is a negation. -/
def isRefutation (type : Expr) : Bool := (conclusion type).isAppOf ``Not

/-- Whether a statement is about a rule: its conclusion is a correctness notion. -/
def isAboutRule (type : Expr) : Bool :=
  match (conclusion type).getAppFn.constName? with
  | some c => notions.contains c
  | none => false

/-- Every constant on which a declaration of the model depends, through statements and proofs. -/
partial def dependencies (env : Environment) (roots : Array Name) : NameSet := Id.run do
  let mut seen : NameSet := {}
  let mut todo := roots
  while h : 0 < todo.size do
    let n := todo[todo.size - 1]
    todo := todo.pop
    if seen.contains n then continue
    seen := seen.insert n
    let some ci := env.find? n | continue
    todo := todo ++ ci.type.getUsedConstants
    if let some v := ci.value? (allowOpaque := true) then todo := todo ++ v.getUsedConstants
    match ci with
    | .inductInfo i => todo := todo ++ i.ctors.toArray
    | _ => pure ()
  return seen

/-- The definitions of the model that the statements `roots` talk about:
those that occur in a statement, and those that these definitions are made of. -/
partial def mentioned (env : Environment) (roots : Array Name) : NameSet := Id.run do
  let mut seen : NameSet := {}
  let mut todo : Array Name := #[]
  for r in roots do
    if let some ci := env.find? r then todo := todo ++ ci.type.getUsedConstants
  while h : 0 < todo.size do
    let n := todo[todo.size - 1]
    todo := todo.pop
    if seen.contains n || (modelModule? env n).isNone then continue
    seen := seen.insert n
    let some ci := env.find? n | continue
    if ci matches .thmInfo _ then continue
    todo := todo ++ ci.type.getUsedConstants
    if let some v := ci.value? then todo := todo ++ v.getUsedConstants
  return seen

/-- The names of the bug records and the refutations that they give, from `bugs/*.json`. -/
def bugRecords : IO (List (String × Option String)) := do
  let mut out := []
  for f in (← System.FilePath.readDir "bugs") do
    if f.path.extension != some "json" then continue
    let json ← IO.ofExcept (Json.parse (← IO.FS.readFile f.path))
    let thm := (json.getObjValAs? String "lean_refutation").toOption
    out := (f.path.fileStem.getD "", thm) :: out
  return out

elab "#audit" : command => do
  let env ← getEnv
  let mut errors : Array MessageData := #[]
  let model := env.constants.toList.filter fun (n, _) => (modelModule? env n).isSome

  -- Axioms.
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let deps := dependencies env (model.map (·.1)).toArray
  let extra := deps.toList.filter fun n =>
    (env.find? n matches some (.axiomInfo _)) && !allowed.contains n
  unless extra.isEmpty do
    errors := errors.push m!"the model depends on {extra} besides the standard axioms"
    for (n, ci) in model do
      if ci matches .thmInfo _ then
        let own := (← liftCoreM (collectAxioms n)).filter (!allowed.contains ·)
        unless own.isEmpty do errors := errors.push m!"  {n} depends on {own}"

  -- The registry against the theorems.
  let mut registered : NameSet := {}
  for e in registry do
    if registered.contains e.thm then errors := errors.push m!"{e.thm} is registered twice"
    registered := registered.insert e.thm
    match env.find? e.thm with
    | some (.thmInfo ci) =>
      match e.status, isRefutation ci.type with
      | .proven, true => errors := errors.push m!"{e.thm} is registered as proven but is a negation"
      | .bug _, false | .notBug _, false =>
        errors := errors.push m!"{e.thm} is registered as refuted but is not a negation"
      | _, _ => pure ()
    | _ => errors := errors.push m!"{e.thm} is registered but is not a theorem"
  for (n, ci) in model do
    let some m := modelModule? env n | continue
    if isRuleModule m && isWritten env n && !registered.contains n then
      if let .thmInfo t := ci then
        if isRefutation t.type then
          errors := errors.push m!"the refutation {n} is not in the registry"
        else if isAboutRule t.type then
          errors := errors.push m!"the theorem {n} about a rule is not in the registry"

  -- Every definition of a rule file occurs in a registered statement.
  let covered := mentioned env (registry.map (·.thm)).toArray
  let excused := unsettled.map (·.1) ++ proofHelpers
  for (n, ci) in model do
    let some m := modelModule? env n | continue
    if isRuleModule m && isWritten env n && (ci matches .defnInfo _) && !covered.contains n
        && !excused.contains n then
      errors := errors.push m!"no registered statement mentions the definition {n}"
  for n in excused do
    if covered.contains n then
      errors := errors.push m!"{n} is listed as unsettled or as a proof helper, \
        but a registered statement mentions it"
    unless (env.find? n matches some (.defnInfo _)) && ((modelModule? env n).map isRuleModule == some true) do
      errors := errors.push m!"{n} is listed as unsettled or as a proof helper, \
        but is not a definition of a rule file"

  -- The registry against the bug records.
  let records ← bugRecords
  for e in registry do
    if let .bug id := e.status then
      unless records.any (·.1 == id) do
        errors := errors.push m!"{e.thm} names the bug record {id}, which does not exist"
  for (id, thm) in records do
    let some thm := thm | continue
    let listed := registry.any fun e =>
      e.thm.toString == thm && (match e.status with | .bug id' => id' == id | _ => false)
    unless listed do
      errors := errors.push
        m!"bugs/{id}.json names the refutation {thm}, which the registry does not list for it"

  unless errors.isEmpty do
    throwError "audit failed:\n{MessageData.joinSep (errors.toList.map fun e => m!"- {e}") "\n"}"
  let count (p : Status → Bool) := (registry.filter (p ·.status)).length
  logInfo m!"audit: {model.length} declarations depend only on the standard axioms; \
    {count (· matches .proven)} properties proven, {count (· matches .bug _)} refuted as bugs, \
    {count (· matches .notBug _)} refuted without a bug, {unsettled.length} definitions unsettled"

end Jax.Audit

#audit
