# Bug reports

Each `<name>.json` is one bug in JAX, following `../scripts/bug.schema.json`.
A record must work as an issue on the JAX GitHub repository, posted without any editing:
someone who has never heard of this repository
should understand the bug, run the reproducer, and see where to fix it.

## From record to issue

The sections of the issue, in this order:

| Issue             | Record                                                    |
|-------------------|-----------------------------------------------------------|
| Title             | `title`                                                   |
| Description       | `description`                                             |
| Expected behavior | `expected`                                                |
| Reproducer        | `reproducer`, with its `# desired:` and `# actual:` lines |
| Cause             | `cause`, with `source` as a link                          |
| Suggested fix     | `fix`, if not `null`                                      |
| JAX version       | `jax_version`                                             |

The keys of a record follow the order of the schema, which is that of the issue,
with `lean` and `related` before `jax_version`. `scripts/bugs.py` checks this.

`lean`, `lean_refutation`, `related` and the triage fields `priority`, `duplicate_search` and `maybe_intended`
appear only on our dashboard, never in the issue.
`lean` is the only field that may mention Lean, the model, theorems or files of this repository.
`scripts/bugs.py` rejects such references in the other fields.
To point to another bug, list it in `related` and describe what it does in the text.
Never write its file name.

## Fields

- **`title`**: one line that states the wrong behaviour and when it happens,
  e.g. "`vmap` of `clamp` raises if only `max` is batched and `min` and the operand are scalars".
  Say "silently" if the result is wrong without an error.
- **`description`**: what goes wrong, from the user's side, in two to four sentences.
  1. The input that triggers the bug, and what JAX does:
     a wrong result, a wrong shape, an exception, or a crash.
  2. What still works (the primal, forward mode, nearby inputs),
     and whether `jnp` code reaches the bug.
  State the symptom, not the rule it breaks: that belongs in `expected`.
  Name no internal function here: that belongs in `cause`.
- **`expected`**: what JAX should do instead, and on what authority,
  in one to three sentences:
  the docstring, the StableHLO spec, the mathematics,
  or another path of JAX that already gets it right (the primal, forward mode, `jnp`, NumPy).
  If more than one answer is defensible, name them and say what any fix must ensure.
  The `# desired:` lines of the reproducer are instances of it.
  It comes before the reproducer, so it names concrete inputs instead of the reproducer's variables.
- **`cause`**: the faulty code in JAX or XLA and why it is wrong.
  Name the function, quote the few lines that matter in a fenced block if that is clearer than words,
  and cite other places as `jax/_src/<file>.py:<line>` of the pinned version.
  If the bug is in XLA, start with "In XLA." and say what JAX emits.
- **`fix`**: the smallest change that fixes it, in one or two sentences, or `null` if none is clear.
  If the fix belongs to XLA, start with "In XLA." and give a workaround in JAX if there is one.
- **`lean`**: what the model proves about the rule, and which theorem refutes it.
  If it is not modelled, say why in one sentence.

## Triage

Three fields say whether and how urgently a bug is worth an issue. All three are judgments that the user reviews.

- **`priority`**: `P0` to `P3`, on the scale of the schema. Choose the lower priority if in doubt.
  It depends on how ordinary code reaches the bug, so say in `description` whether `jnp` does.
- **`duplicate_search`**: search the issues and pull requests of jax-ml/jax for the bug before adding a record,
  and those of openxla/xla and openxla/stablehlo if the cause is in XLA.
  Record every query, also those that found nothing, and what was found about the same code.
  An issue for the same bug goes into `upstream_issue`.
  Repeat the search before filing and update the date.
- **`maybe_intended`**: why a maintainer could call the behaviour intended, or `null`.
  Fill it in whenever `expected` rests on our reading and not on a written promise:
  a docstring that can be read two ways, an input that only `lax` accepts, a value that no document fixes.
  A doubt that concerns several records or the model is a question in `../QUESTIONS.md` instead.

Before filing, also run the reproducers on the latest release, `uv run python scripts/bugs.py --jax latest`.
If a bug no longer shows there, find the change that fixed it, and set `status`, `fixed_in` and `upstream_issue`.

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
- Before adding a record, read the full report concatenated in the order of the issue:
  title, description, expected behavior, reproducer, cause, fix and JAX version, then `lean`.
  Read it as a stranger would, and cut anything they would skip.
  Check that each section follows from the ones before it and does not repeat them.

`broadcast-in-dim-forwarding.json` and `clamp-batch-scalar-operand.json` are good examples.
