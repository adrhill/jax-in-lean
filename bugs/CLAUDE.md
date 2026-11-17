# Bug reports

Each `<name>.json` is one bug in JAX, following `../scripts/bug.schema.json`.
A record must work as an issue on the JAX GitHub repository, posted without any editing:
someone who has never heard of this repository
should understand the bug, run the reproducer, and see where to fix it.

## From record to issue

| Issue                 | Record                                                   |
|-----------------------|----------------------------------------------------------|
| Title                 | `title`                                                  |
| Description           | `summary`                                                |
| Reproducer and output | `reproducer`, with its `# desired:` and `# actual:` lines |
| Cause                 | `cause`, with `source` as a link                         |
| Suggested fix         | `fix`, if not `null`                                     |
| JAX version           | `jax_version`                                            |

`lean` and `lean_refutation` appear only on our dashboard, never in the issue.
`lean` is the only field that may mention Lean, the model, theorems or files of this repository.
`scripts/bugs.py` rejects such references in the other fields.
To point to another bug, list it in `related` and describe what it does in the text.
Never write its file name.

## Fields

- **`title`**: one line that states the wrong behaviour and when it happens,
  e.g. "`vmap` of `clamp` raises if only `max` is batched and `min` and the operand are scalars".
  Say "silently" if the result is wrong without an error.
- **`summary`**: what goes wrong, from the user's side, in a few sentences.
  1. What the primitive promises, from its docstring or the StableHLO spec, if it is not obvious.
  2. The input that triggers the bug, and what JAX does instead:
     a wrong result, a wrong shape, an exception, or a crash.
  3. What still works (the primal, forward mode, nearby inputs),
     and whether `jnp` code reaches the bug.
  Name no internal function here: that belongs in `cause`.
- **`cause`**: the faulty code in JAX or XLA and why it is wrong.
  Name the function, quote the few lines that matter in a fenced block if that is clearer than words,
  and cite other places as `jax/_src/<file>.py:<line>` of the pinned version.
  If the bug is in XLA, start with "In XLA." and say what JAX emits.
- **`fix`**: the smallest change that fixes it, in one or two sentences, or `null` if none is clear.
  If the fix belongs to XLA, start with "In XLA." and give a workaround in JAX if there is one.
- **`lean`**: what the model proves about the rule, and which theorem refutes it.
  If it is not modelled, say why in one sentence.

## Reproducer

- A plain script that runs on its own with `import jax`, `jax.numpy as jnp` and `from jax import lax`.
- The smallest input that shows the bug, with small hand-picked numbers so the right answer is checkable by hand.
- First show what works (`# output:`), then what fails (`# desired:` and `# actual:`),
  and then any variants worth knowing, such as the `jnp` function that reaches the bug.
- One comment line above each step says what it shows, in plain words.
- Catch an expected exception and print `"<Type>: <message>"`, so the script finishes and the message is visible.
- `scripts/bugs.py` runs it and checks that it prints exactly the `actual` lines.

## Style

- Concise and plain. Short sentences, active voice, no hedging and no filler.
  Every sentence should tell a JAX maintainer something they need.
- Semicolons and em-dashes are allowed, but should be rare.
  A sentence usually reads better split in two, or with its clauses joined by a comma, a colon or a conjunction.
- Use the terms of JAX: primitive, jaxpr, JVP rule, transpose rule, batching rule, primal, tangent, cotangent.
  Do not use our terms (`IsJvpOf`, proof argument, refutation, `Arr`).
- Write code, shapes and names in backticks: `lax.clamp`, `(3, 2)`, `broadcast_dimensions`.
- Semantic line breaks: one sentence or clause per array entry.
  An empty entry starts a new paragraph.
- Read the record once as a stranger would, and cut anything they would skip.

`broadcast-in-dim-forwarding.json` and `clamp-batch-scalar-operand.json` are good examples.
