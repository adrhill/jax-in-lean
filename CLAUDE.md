# jax-in-lean

A Lean 4 model of JAX with proofs about it, used to find bugs in JAX.
Sister repository of `../asdex`.

## Structure

```
Jax.lean, Jax/      # Model of JAX and proofs about it
├── Scalar.lean     # Array entries: reals, infinities and NaN
├── DType.lean      # Dtypes: bool, int and float widths, float0, and tangent dtypes
├── Shape.lean      # Shape, Index, broadcasting
├── Core.lean       # Tensor, IsJvpOf, IsJvp₂Of, IsTransposeOf and variants, Arr, IsBatchOf₁, IsBatchOf₃
├── Lax/Primal.lean # Primal semantics of every primitive, and helpers that rules call
├── Lax/Dynamic.lean # The primitives on arrays of any shape and dtype: `none` if a check fails
├── Lax/<Name>.lean # One file per lax primitive: differentiation rules and proofs
│                   # (the six comparisons share Lax/Compare.lean)
└── Jaxpr/<Name>.lean # Jaxprs of several primitives, e.g. from upstream issues
Audit.lean          # Axiom check, and the registry of every settled property
bugs/<name>.json    # One record per JAX bug, with its reproducer
bugs/CLAUDE.md      # How to write a bug record
scripts/audit.sh    # Forbidden-token scan, build, Audit.lean, bug reproducers, shapes.py
scripts/bugs.py     # Checks the bug records and runs their reproducers
scripts/shapes.py   # Shape- and dtype-level differential test of the model against JAX
gen/                # Generated Lean files of scripts/shapes.py (gitignored)
pyproject.toml      # Pins the modelled JAX version
```

## Development

All tools come from the Nix flake. Outside a `direnv` shell, prefix commands with `nix develop -c`.

```bash
lake build           # build and check all proofs
scripts/audit.sh     # required before every commit
uv sync              # install the pinned JAX into .venv
uv run python ...    # run a reproducer against real JAX
prek install         # once: run ruff and ty from `prek.toml` on every commit
```

The pinned JAX source is readable at `.venv/lib/python3.15/site-packages/jax/`.

## Modelling rules

- **Exact scalars.** An entry of a float array of width `w` is a `Scalar w`: a real number, `+∞`, `-∞` or NaN.
  Special values follow IEEE 754. Finite values are exact: no rounding, no overflow, no signed zero.
  An integer entry is an exact `ℤ`. Unsigned integers and `jax_enable_x64` are not modelled.
- **Shapes.** A shape `(m, n)` is `![m, n] : Shape 2`, and an array of dtype `d` and shape `s` is a `Tensor d (Index s)`.
  Shape checks that JAX does at trace time are proof arguments of the primitive.
- **Dtypes.** A dtype (`Jax/DType.lean`) is `bool`, `int` or `float` of a width, or `float0`.
  Arguments that a primitive requires to have the same dtype share the dtype in its type,
  and dtypes that it does not accept are not representable.
  A constant in a rule has the dtype that JAX gives it, e.g. a Python float is `float32`.
  An equality of dtypes that a rule needs is a proof argument of the rule, as for shapes.
  The entries of a float array, `Scalar w`, carry the width, so Lean rejects an argument of another width.
  Weak types are dropped: they never change a dtype or whether a `lax` primitive raises.
- **Checks of rules.** A shape or dtype check inside a rule that does not follow from the checks of the primitive
  is a proof argument of the rule: the rule raises where it is false.
  A theorem `foo_not_defined` shows that it does not follow.
- **Linear inputs.** A transpose rule that handles several linear inputs takes every argument as an `Option`,
  where `none` is an `ad.UndefinedPrimal`, and returns `none` if it raises (`IsPartialTranspose₂Of`).
- **Branching on shapes.** A rule that branches on the shapes of its arguments, such as a batching rule,
  is a function of `Arr`, an array of any shape and dtype, and returns `none` if it raises (`IsBatchOf₁`, `IsBatchOf₃`).
  It calls the primitives of `Lax/Dynamic.lean`, which wrap those of `Lax/Primal.lean`.
- **Booleans.** A boolean array of shape `s` is a `Tensor .bool (Index s)`, whose tangent dtype is `float0`.
- **Indices.** An array of indices of shape `s`, as `gather` and `scatter` take it, is a `Tensor (.int wi) (Index s)`,
  whose tangent dtype is `float0`.
  `scripts/shapes.py` also compares the positions that `gather` and `scatter_add` read and write with JAX on concrete indices.
- **Computed shapes.** A rule has the output shape and dtype that its own code computes, even if it only equals the expected one propositionally.
  Correctness notions compare arrays with `Tensor.Same`, so a rule that returns the wrong shape or dtype is refutable.
- **Jaxpr level.** Model `lax` primitives and jaxprs, not Python or `jax.numpy`.
- **Mirror the source.** Differentiation rules (`ad.defjvp`, `ad.deflinear`, transpose rules)
  are transcribed from the JAX source as written, including argument order, not from the textbook.
  If JAX's rule is wrong, the model must be wrong in the same way.
- **Cite the source.** Every mirrored definition cites `jax/_src/<file>.py:<line>` of the pinned version
  and quotes the rule if it is short.
- **Shapes and dtypes are tested.** `scripts/shapes.py` traces every modelled function and rule in JAX
  and checks with `decide` that the shapes and dtypes in the model's types agree,
  and that JAX rejects an input exactly when a proof argument is false.
  It checks with `#check` and `#check_failure` that JAX rejects arguments of different dtypes exactly when Lean does.
  It also decides in Lean that every JVP rule of the model returns a tangent of the primal result,
  and every transpose rule a cotangent of its input, with their shapes and tangent dtypes;
  a violation is a bug, listed in `DOMAIN_BUGS` once it has a record.
  A new primitive or rule gets cases there. Model definitions must reduce by `decide` on concrete shapes.
- **Primal semantics** are not defined in Python but by XLA.
  Take them from the `lax` docstring and the [StableHLO spec](https://github.com/openxla/stablehlo/blob/main/docs/spec.md).
- **Dropped parameters** that have no meaning over the reals (e.g. `accuracy`) are omitted and stay visible in the quoted source.

## Proof rules

- No `sorry`, `admit`, `native_decide`, `axiom`, `unsafe` or `implemented_by`.
  Only the standard axioms `propext`, `Classical.choice` and `Quot.sound` may appear.
  `Audit.lean` collects the axioms of every declaration of the model and fails on any other.
- Every theorem about a rule and every refutation in `Jax/Lax/` and `Jax/Jaxpr/` has an entry
  in the registry of `Audit.lean`: proven, refuted with the bug record that shows it on JAX,
  or refuted with the reason why it is not a bug.
  Every definition of a rule file occurs in a registered statement, or is listed there as unsettled.
- The first theorem for a primitive `foo` is `fooJvp_correct`, stated with `IsJvpOf` or `IsJvp₂Of`,
  or with `IsJvpOfOn` or `IsJvp₂OfOn` and the largest domain on which it holds.
- Never weaken a statement to make it provable.
  If a property is false, prove its negation with a concrete counterexample and keep both the statement and the refutation.
- Bugs are found in Lean: a refuted statement about the model, or a domain check of `scripts/shapes.py`.
  Never by testing JAX against itself.
- A refutation is a JAX bug only with a Python reproducer that shows it on the pinned JAX.
  Every bug gets a record `bugs/<name>.json` that follows `scripts/bug.schema.json`.
  It reads as a self-contained issue on the JAX repository, as `bugs/CLAUDE.md` describes.
  If the model and real JAX disagree, the model is wrong: fix the model.
- Theorem statements and model definitions are the trusted part and are reviewed by the user.
  Keep them short and readable, and point out new or changed ones when reporting work.

## Conventions

- Lean files use the module system: `module`, `public import`, `@[expose] public section`.
- Names follow Mathlib: `camelCase` definitions, `snake_case` theorems, primitive names as in `lax`.
- Semantic line breaks in prose (one sentence or clause per line).
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for all commit messages.
