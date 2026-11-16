# jax-in-lean

A model of [JAX](https://github.com/jax-ml/jax) in [Lean 4](https://lean-lang.org), with machine-checked proofs about it.
Sister repository of [asdex](https://github.com/adrhill/asdex).

The goal is to find bugs in JAX by proving what it promises.
The approach follows [julia-in-lean](https://github.com/MilesCranmer/julia-in-lean):
mirror the source in Lean, state properties, and settle each one as proven or refuted.

## Layout

| Path | Contents |
|---|---|
| `Jax/Scalar.lean`, `Jax/Shape.lean`, `Jax/Core.lean` | Array entries, shapes, and what it means for a rule to be correct. |
| `Jax/Lax/Primal.lean` | What each `lax` primitive computes. |
| `Jax/Lax/<Name>.lean` | The differentiation rules of one primitive, with proofs. |
| `bugs/` | The JAX bugs found so far, one JSON record each, with a reproducer. |
| `scripts/audit.sh` | Rejects `sorry`, new axioms and other escape hatches, builds, and runs the reproducers. |
| `pyproject.toml` | Pins the JAX version being modelled. |

## Scope

- Arrays hold exact real numbers, infinities and NaN. Rounding, overflow and signed zeros are out of scope for now.
- Shapes, axes and broadcasting are modelled, so a rule that returns the wrong shape can be refuted.
- The model lives at the level of jaxprs and `lax` primitives, not Python or `jax.numpy`.
- The first property proven for every primitive is that its JVP rule computes its derivative.

## What a proof means

A theorem here is about the Lean model, which is written by hand and can be wrong.
What has to be trusted is the model, the theorem statements and the Lean kernel, not the proofs.
A property refuted in Lean counts as a JAX bug only once a Python script reproduces it on the pinned JAX.

## Setup

Requires [Nix](https://nixos.org) with flakes.

```bash
nix develop           # or `direnv allow`
uv sync               # installs the pinned JAX into .venv
lake exe cache get    # downloads prebuilt Mathlib
lake build
scripts/audit.sh
```

Lean itself is installed by `elan` on first use, in the version given by `lean-toolchain`.

## Related work

- [julia-in-lean](https://github.com/MilesCranmer/julia-in-lean) does the same for Julia,
  and this repository follows its approach.
- [SciLean](https://github.com/lecopivo/SciLean) is a Lean 4 library for scientific computing
  that implements automatic differentiation as code transformations backed by formalized mathematics.
- [Automatic Certified Differentiation in Lean](https://arxiv.org/abs/2610.36396) (Shahin)
  generates derivatives whose correctness the Lean kernel checks.
- [You Only Linearize Once](https://arxiv.org/abs/2204.10923)
  ([Radul, Paszke, Johnson and Maclaurin, POPL 2023](https://popl23.sigplan.org/details/POPL-2023-popl-research-papers/43/You-Only-Linearize-Once-Tangents-Transpose-to-Gradients))
  describes how JAX gets reverse mode from forward mode by unzipping and transposing.
  This is why JAX's differentiation rules are JVP and transpose rules, which are what this repository models.
