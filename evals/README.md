# PM admin-config evals

A fixed set of 48 product-manager questions about Django admin configuration,
with checkable expected answers, used to measure whether CodeAtlas answers this
class of question well enough to point PMs at.

Target service: **frodo** (`riderappcontent`), whose Django admin is at
`/riderappcontent/`. Gold answers are pinned to frodo commit `0bb7d9f32`.

---

## 1. Why this exists

PMs ask operational questions — "how do I reduce the geofence radius for
certain clients?" These are *procedures*, not explanations. A correct answer
names an admin page, a field, the scope it applies to, and what happens after
you save. A wrong answer sends someone to change the wrong row in production.

Three decisions this set is meant to unblock:

1. **Can PMs be pointed at CodeAtlas for admin-config questions at all?**
2. **Does the audience-cache shortcut degrade procedural answers?** Commit
   `bc8eed1` builds a product answer by rewriting a cached *dev* answer. A dev
   answer about fee logic may never mention the admin page, so the product
   answer cannot either. Testable, and the runner has a dedicated arm for it.
3. **Does per-repo retrieval tuning pay?** `synonyms`, `keyword_boosts` and
   `pre_search_instruction` in `app/retrieval/config_schema.py` are tunable
   today with no way to tell whether a change helped.

## 2. Where the questions came from

Two independent sources, deliberately:

- **Admin surface** — the live staging admin was walked and its schema read off
  unbound `/add/` forms. Key names came from the two key-value config
  changelists (`RMS.config`, 194 rows; `rider_app.configparams`, 17). No config
  *values* and no record data were read.
- **Code truth** — the frodo repo at `0bb7d9f32`, read separately, with
  `file:line` citations recorded on each item as `gold_citations`.

**These two sources disagree far more often than expected**, and the
disagreement turned out to be the most valuable thing in the set. Eleven items
are tagged `kind: admin_vs_code`: the admin shows an editable control and the
code ignores it.

> ### Warning, from experience
> Two items in the first draft had **wrong gold** because an admin-only
> inference was promoted without a code check. A hidden input behind a JS
> upload widget looked like a missing field; a `DateField` rendering as
> `type=text` looked like unvalidated free text. Both would have *penalised
> correct answers*.
>
> **Never write gold from the admin HTML alone.** A wrong gold is worse than
> no item, because it trains you to fix things that are not broken.

## 3. How an item is built

```yaml
- id: s3-cache-ignores-is-active
  scope_class: S3
  kind: admin_vs_code          # this item is an admin/code disagreement
  band: B3                     # phrasing difficulty, see below
  priority: critical
  question: >-
    Two rows exist for the same city and fee, one active and one
    inactive. Which amount do riders actually get charged?
  expect:                      # each key = ONE binary assertion, judged
    answer: whichever row was saved most recently, active or not
    must_mention:
      - the cache key does not include the active flag
    must_not: answer "the active one" from the database constraint
  must_not_contain: [".py", "line "]        # regex, no judge needed
  evidence_paths: ["**/RMS/**"]             # retrieval recall, from the trace
  gold_citations: [common/redis_keys.py:91] # where the truth was verified
```

Every key under `expect` is graded as a **separate binary assertion**, not as
one holistic score. That is what makes a failure actionable.

### Two axes

**`scope_class`** — the shape of the configuration:

| | |
|---|---|
| `K1` | key–value config row (the dominant pattern: 211 keys across two tables) |
| `S3` | composite override table, scoped by city / client / category |
| `S3b` | referral surge config — the one model with `help_text` |
| `S2` | per-entity field |
| `C` | content (banners, carousel, notices) |
| `L` | lookup table |

**`band`** — how far the question's wording is from the code's:

| | | tests |
|---|---|---|
| `B1` | uses the admin's own label | baseline — should be ~100% |
| `B2` | business synonym | `synonyms`, `keyword_boosts` |
| `B3` | a symptom, no config vocabulary | `pre_search_instruction` |
| `B4` | wrong-but-plausible vocabulary | disambiguation, clarification |

**Read B1 first.** If B1 is below ~0.9 the class is broken, not the
vocabulary, and every other row is uninterpretable.

## 4. Running it

```bash
pip install -r evals/requirements.txt
```

```bash
export CODEATLAS_EVAL_PASSWORD='...'
python evals/run_eval.py --base-url http://localhost:8000 \
    --workspace frodo --username you --reps 3 --arms fresh,audience
```

Useful flags: `--only <substring>` to run one item, `--priority critical` for
the five that matter most, `--branch <id>` to pin an indexed branch.
`CODEATLAS_EVAL_COOKIE` works instead of a password if you already have a
session. Credentials are read from the environment and never from argv.

Output lands in `evals/runs/<timestamp>/`:

| file | |
|---|---|
| `results.jsonl` | one row per item × rep × arm, with the graded deterministic checks |
| `judge_tasks.jsonl` | payloads for the facet-correctness step |
| `meta.json` | workspace, branch, reps, arms, source commit |

### What the runner grades itself

- **Retrieval recall** — did `agent_trace` ever open a file matching
  `evidence_paths`? Deterministic, no judge, nearly free.
- **Audience compliance** — does a product-team answer leak `.py` filenames,
  line numbers or class names? Pure regex. Asserts that the guardrail in
  `app/llm/client.py::_clean_product_answer` actually holds.

### What it does not

**Facet correctness needs a judge**, so the runner writes `judge_tasks.jsonl`
instead of calling an LLM itself. That keeps the runner dependency-light and
the judge swappable. Each task carries the `expect` block and this instruction:
grade each key separately, *and do not reward hedging* — an item whose `expect`
states a definite answer fails if the response only expresses uncertainty.

Validate any judge against ~20 hand-labelled items before trusting it.

## 5. Caching — read this before believing a number

CodeAtlas has three layers that will short-circuit an eval
(`app/ask_service.py:139`): a per-session cache, a per-repo cache, and the
product-team audience-evidence reuse path. All live **in process**
(`app/conversations.py`), so:

> **For a true cold run, restart the CodeAtlas server first.** The runner
> prints this reminder but cannot enforce it.

The runner turns this into a feature via `--arms`:

- **`fresh`** — asks as `product_team` directly.
- **`audience`** — asks as `dev_team` *first*, then `product_team`. That
  populates the dev cache and forces the product answer through
  `answer_from_cached_audience_evidence`. It is the precise way to exercise
  the `bc8eed1` path rather than hoping to hit it.

Comparing the two arms answers decision #2 in section 1.

**Compare arms paired by `item_id`, never in aggregate.** Paired comparison
counts only the items that flip between arms, which collapses variance — 9
items flipping with 8 going one way is a real signal at this sample size,
where the unpaired maths would call it noise.

## 6. Reading the scorecard

The runner prints retrieval and audience per class, per kind and per band, plus
an arm comparison and a list of items unstable across reps. Facet columns stay
blank until judging runs.

| symptom | what to fix |
|---|---|
| B1 below ~0.9 | the class is broken. Stop and fix this first |
| B3 low, B1 high | vocabulary gap → `synonyms`, `pre_search_instruction` |
| retrieval high, facets low | found the page, missed the scoping → prompt or context size |
| `admin_vs_code` low | answering from the admin surface without checking call sites. **The headline number** |
| abstention high *but* `s3-units` and `s3-client-axis-exists` failing | over-refusal, not calibration. A regression, not a win |
| `kv-new-suffix` passes but `s2-geo-newer-field-does-not-win` fails | a convention was generalised instead of checked |
| `kv-aliased-constant-indirection` fails | retrieval does not follow indirection |

### Abstention is not "say I don't know"

Ten items expect the answer *"this cannot be determined here."* Every one of
them **also requires the constructive half** — naming where the answer actually
lives, or who to ask. A bare refusal fails.

Without that rule the metric rewards unhelpful non-answers and you optimise
straight into a system that refuses everything. Two items exist purely as
counterweights: `s3-units` (the unit *is* knowable from code — hedging fails)
and `s3-client-axis-exists` (a client axis *does* exist here — refusing fails).

## 7. The items worth knowing about

Five are `priority: critical`, all `admin_vs_code`:

- **`kv-dead-disabled-keys`** — twelve `*_disbaled_city_ids` keys have zero
  Python references; they survive only in a test fixture. A PM can see them,
  edit them, and nothing happens. Finding them at all needs typo-tolerant
  retrieval (all twelve misspell "disabled"); answering needs the judgement
  that a live-looking admin row is dead.
- **`s3-deactivate-does-not-disable`** — unticking "is active" on a city's fee
  row does not stop the charge: the fallback default returns sentinel id `-1`,
  which satisfies the caller's gate.
- **`s3-cache-ignores-is-active`** — the Redis key omits `is_active` and
  `post_save` writes unconditionally, so saving the *inactive* row poisons the
  cache for 24h. The DB has `unique_together` including `is_active`, which
  makes "the active one wins" sound defensible and wrong.
- **`s2-geo-both-pairs-read`** — `CityReferralConfig` has two coordinate pairs,
  both read, nothing keeping them in sync. The numeric pair decides which
  centres a rider sees; the text pair computes the distance and is what the app
  displays. Edit one and the UI looks right while the matching is wrong.
- **`s2-geo-newer-field-does-not-win`** — paired with `kv-new-suffix`. There,
  the newer `_new` key supersedes its twin cleanly. Here, the newer float field
  *lost* — a half-finished 2022 migration only one query ever used. Passing
  both means the convention was checked, not generalised.

Over-transfer of a learned rule is the dominant failure mode this set is built
to catch. `s3-client-axis-exists` and `kv-cross-table-collision` are the other
two traps of that shape.

## 8. Maintaining it

- **Pin the revision.** `meta.source_commit` records the frodo commit the gold
  was verified against. Gold is only valid for a commit; CodeAtlas must be
  indexing a matching revision.
- **Prune non-discriminating items.** After a run, anything that passes 3/3 in
  every arm carries no information and costs full price forever. Expect to keep
  roughly three quarters of what you write.
- **Watch the cells, not the total.** A cell needs 10–15 items to mean
  anything. Currently healthy: `admin_vs_code` 11, `S3` 13, `K1` 11,
  abstention 10, `C` 10. Thin, read pooled: `S3b` 6, `S2` 6, `L` 2.
- **Items beat repeats.** Item-level variance dominates run-level flakiness.
  Keep 3 reps; never trade items for more reps.
- **Spot-check cited gold.** It was produced by an agent reading source. The
  `file:line` citations make verification cheap — start with the five critical
  items.

## 9. Known gaps

- The `B1` phrasings are educated guesses at PM vocabulary. Replace them with
  real questions once question logging lands — CodeAtlas does not currently
  persist question text anywhere durable.
- Roughly 30 of frodo's 49 admin models are unexamined, mostly
  `access_management`, `rider_training` and `bike_taxi`.
- `referralsurgeconfig` is effectively dead code — zero live call sites, and
  the one live reader hard-codes `None`. Its items are kept because the admin
  exposes it and a complete answer says so.
- Only frodo is covered. Nothing here is specific to frodo except the items
  themselves; the method transfers to any indexed Django repo.
