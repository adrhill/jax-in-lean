# jax-in-lean

A Lean 4 model of JAX with proofs about it, used to find bugs in JAX.
Sister repository of `../asdex`.

## Structure

```
Jax.lean, Jax/      # Model of JAX and proofs about it
├── Core.lean       # Tensor, IsJvpOf
└── Lax/<Name>.lean # One file per lax primitive: primal, JVP rule, proofs
scripts/audit.sh    # Forbidden-token scan + build
pyproject.toml      # Pins the modelled JAX version
```

## Development

All tools come from the Nix flake. Outside a `direnv` shell, prefix commands with `nix develop -c`.

```bash
lake build           # build and check all proofs
scripts/audit.sh     # required before every commit
uv sync              # install the pinned JAX into .venv
uv run python ...    # run a reproducer against real JAX
```

The pinned JAX source is readable at `.venv/lib/python3.15/site-packages/jax/`.

## Modelling rules

- **Reals only.** Arrays are `Tensor ι := ι → ℝ` over a finite index type `ι`.
  A shape `(m, n)` is `Fin m × Fin n`. No floats, NaN, infinities or integer dtypes for now.
- **Jaxpr level.** Model `lax` primitives and jaxprs, not Python or `jax.numpy`.
- **Mirror the source.** Differentiation rules (`ad.defjvp`, `ad.deflinear`, transpose rules)
  are transcribed from the JAX source as written, including argument order, not from the textbook.
  If JAX's rule is wrong, the model must be wrong in the same way.
- **Cite the source.** Every mirrored definition cites `jax/_src/<file>.py:<line>` of the pinned version
  and quotes the rule if it is short.
- **Primal semantics** are not defined in Python but by XLA.
  Take them from the `lax` docstring and the [StableHLO spec](https://github.com/openxla/stablehlo/blob/main/docs/spec.md).
- **Dropped parameters** that have no meaning over the reals (e.g. `accuracy`) are omitted and stay visible in the quoted source.

## Proof rules

- No `sorry`, `admit`, `native_decide`, `axiom`, `unsafe` or `implemented_by`.
  Only the standard axioms `propext`, `Classical.choice` and `Quot.sound` may appear.
- The first theorem for a primitive `foo` is `fooJvp_correct : IsJvpOf fooJvp foo`.
- Never weaken a statement to make it provable.
  If a property is false, prove its negation with a concrete counterexample and keep both the statement and the refutation.
- A refutation is a JAX bug only with a Python reproducer that shows it on the pinned JAX.
  If the model and real JAX disagree, the model is wrong: fix the model.
- Theorem statements and model definitions are the trusted part and are reviewed by the user.
  Keep them short and readable, and point out new or changed ones when reporting work.

## Conventions

- Lean files use the module system: `module`, `public import`, `@[expose] public section`.
- Names follow Mathlib: `camelCase` definitions, `snake_case` theorems, primitive names as in `lax`.
- Semantic line breaks in prose (one sentence or clause per line).
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for all commit messages.
