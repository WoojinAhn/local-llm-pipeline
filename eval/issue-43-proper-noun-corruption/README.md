# Proper-noun corruption evaluation (#43)

Measures how well each pipeline configuration preserves Korean proper nouns.

The 3-stage pipeline (KO→EN → English analysis → EN→KO) was observed producing
confident, well-formatted lists of Korean historical figures who do not exist. This
harness exists so that finding is reproducible and so #44 (Korea-domain routing) can be
regression-tested rather than assumed to work.

## Layout

| Path | Purpose |
|---|---|
| `cases.jsonl` | Queries with hand-curated canonical entity lists. Includes control (non-Korean) and holdout cases. |
| `run.py` | Runs a configuration, capturing **every stage** and full environment metadata. |
| `score.py` | Canonical recall (automatic) + entity validity (annotation-driven). |
| `isolate.py` | Three controlled experiments separating translation / reasoner / back-translation. |
| `annotations.jsonl` | Human verdicts on entities the models produced. |
| `outputs/` | Raw run records. |

## The two metrics are not interchangeable

**Canonical recall** — of the entities we expected, how many appeared. Fully automatic.
It **understates every configuration**, because model answers are summaries while the
canonical lists are exhaustive. Use it for comparison between configurations, never as
an accuracy rate.

**Entity validity** — of the names the extractor **detected**, how many are real. This is
a **detected-candidate lower bound, not TP/FP/FN**: a fabrication the extractor misses
never enters the denominator, so these counts cannot be converted into a fabrication
rate. It is **not** automatic and must not be made automatic. An entity outside the canonical list is
not evidence of fabrication — it is frequently a real figure the list omits. In the first
run, `한명회`, `유길준`, `김윤식`, `고니시` and others fell outside the canonical lists and
are all real people. Only `annotations.jsonl` decides validity; unannotated entities are
reported as `unreviewed`.

Conflating these two produced a real error during this investigation: a 4x recall gap was
described as "4x less corruption", which the data does not support.

## Known limitations

- The person-name extractor is regex over Korean text and is noisy in both directions.
  Korean surname syllables are common in ordinary vocabulary, so a first version matched
  large numbers of non-names. The current version still surfaces non-entities — this is
  why extraction feeds annotation rather than a verdict.
- The extractor was fixed at 3 syllables in two places. Recall was fixed first (it now
  matches canonical entities directly); the outside-candidate pattern kept the same blind
  spot until later, so 2-syllable fabrications were invisible on the FP side. Now 2-4
  syllables, which raises noise — acceptable, because candidates go to a human.
- Provenance is split: `generation_sha256_16` (run.py + cases.jsonl) and
  `scoring_sha256_16` (score.py). They were bundled at first, which meant a scorer-only
  edit invalidated `--resume` for raw generations that were still perfectly valid.
- Canonical lists are hand-curated and incomplete by construction.
- n=6 Korea-domain cases (plus 2 holdout). Enough to establish that the effect is
  systematic, not enough to rank configurations precisely.
- Holdout cases (`"holdout": true`) are excluded by default and must **not** be used when
  tuning the `KOREA:yes/no` router in #44, or they stop being holdout.

## Running

```bash
cd eval/issue-43-proper-noun-corruption
python run.py three-stage-gpt-oss                             # search off, candidate profile
python run.py three-stage-qwen36 --profile production         # production token limits
python run.py single-exaone --search parity --profile production    # settings only
python run.py three-stage-gpt-oss --resume                    # continue an interrupted run
python score.py                                               # default tag
python score.py --tag search-parity-production

# entities needing a verdict
python score.py --candidates three-stage-gpt-oss >> annotations.jsonl
```

Generation is greedy (mlx-lm's default `temp=0`), so runs are reproducible without a
seed. `run.py` records library versions, platform, and per-stage timing, token counts
and peak memory.

For provenance it records `repo_commit` **plus** `repo_dirty`, `dirty_paths`,
`harness_sha256_16` and `prompts_sha256_16`. `repo_commit` alone is misleading: a run
made while the harness was uncommitted records the *previous* HEAD, which does not
identify the code that produced it. Check `repo_dirty` before trusting `repo_commit`.

Long runs save after every case (atomic replace) and support `--resume`, so an
interrupted multi-hour run is not lost. `--resume` refuses to append when
`config_detail`, `search`, `profile`, `harness_sha256_16` or `prompts_sha256_16` differ
from the existing file, so results from different code never mix into one record.

## Isolating the damage stage — still open

Observational runs establish the defect is **reasoner-independent**: two different
reasoners scored identically while the wrapper-free path scored roughly 4x higher (v1
figures — see the comparability caveats below). They do **not** establish which step
loses the entities. "Reasoner-independent" and "not the
reasoner" are different claims — any reasoner handed romanized Korean names may
confabulate.

`isolate.py` separates the three candidates:

```bash
python isolate.py a   # KO -> EN -> KO, names only
python isolate.py b   # correct English -> KO only
python isolate.py c   # fixed English Q, both reasoners
```

Not yet run.

Experiment A is a **necessary, not sufficient** check on the translator: names surviving
a newline-separated list does not clear translation, because names in running prose carry
different context and segmentation.

## Two axes — neither one alone is "production"

**`--search off`** (default) disables web search; the `SEARCH:` judgment is recorded but
not acted on. Reproducible, and corruption is attributable to the pipeline rather than to
whatever the web returned.
**`--search parity`** mirrors production's *search behaviour only*: search runs when the
judge says yes, Korean snippets are translated to English, and
`build_search_context_prompt()` wraps the reasoner input.

**`--profile production`** — translate 2000 / reason 4000, the limits `mlx-pipeline.py`
uses today (`_stream_qwen`, `_stream_reasoner`).
**`--profile candidate`** (default) — translate 4000 / reason 8000, headroom for Qwen3.6,
which does not close `</think>` within 4000 on analytical prompts (#40).

`--search parity --profile production` matches production's **settings**. It does not
make a run "the shipped pipeline" — two gaps remain, and metadata records all three
facts separately:

| Field | Meaning |
|---|---|
| `settings_equivalent_to_production` | search + token limits match |
| `config_matches_shipped` | the models are Qwen3-14B + gpt-oss-120b in 3 stages |
| `reproduces_conversation_state` | always `false` — see below |

Production accumulates `_reasoner_history` and reuses `prompt_cache` across turns. This
harness runs every case cold and independent, so no configuration here reproduces a
multi-turn session. `single-exaone --search parity --profile production` is a
candidate at production settings — not production.

Output files are named `run-search-<search>-<profile>.json`; compare only within one tag.

An earlier revision named these `control`/`parity` and claimed parity mirrored
production. It did not — the generation profile diverged at 4000/8000 while production
used 2000/4000.

## Baseline results — v1 and v2 are NOT directly comparable

Do not put these in one table and read a trend off it. Three things changed.

**Denominator.** `ko-03`'s canonical list went from 7 entries to 6: `元均` was a
hanja duplicate of `원균` (a mistake), and `와키자카` is a Japanese commander, out of
scope for a Korean-proper-noun eval. `황진` was added. Total 43 → 42.

**Reasoner budget.** v1 used `max_tokens=1600`, which truncated Qwen3.6 mid-reasoning.
v2 defaults to the `candidate` profile at 8000. This changed latency by ~2.8x, so v1
latencies measure truncation, not throughput. Note that neither matches production's
4000 — use `--profile production` for that.

**Recall method.** v1 derived recall from the general extractor, whose plain-text
pattern is fixed at 3 syllables. Two-syllable names (`이익`, `원균`, `권율`, `일연`,
`고종`) were counted only when a model happened to bold them, so v1 recall is
formatting-dependent and understated. v2 matches canonical entities directly.

### v1 (2026-08-06, final answers only, 1600-token reasoner, extractor-derived recall)

| Configuration | Recall /43 | Avg latency |
|---|---|---|
| `single-exaone` | 16 | 60.5 s |
| `three-stage-gpt-oss` | 4 | 42.5 s |
| `three-stage-qwen36` | 4 | 25.6 s |

### v2 (this harness, `search-off-candidate`, direct canonical matching)

| Configuration | Recall /42 | Avg latency |
|---|---|---|
| `three-stage-qwen36` | not yet run | — |
| `three-stage-gpt-oss` | not yet run | — |
| `single-exaone` | not yet run | — |

An earlier v2 attempt is kept at `outputs/three-stage-qwen36/v2pre-superseded.json`
(recall 10/42, 70.9 s avg). **Superseded, do not cite.** It truncated the reasoner
output at 6000 characters before back-translation — production does not — predates the
control/parity split, and its `repo_commit` points at the pre-harness HEAD.

The v1 records are kept at `outputs/*/v1-final-only.json`. They lack per-stage
intermediates and cannot support stage-level analysis.

`outputs/english-reasoner-ab.json` holds a separate English analytical A/B across
gpt-oss-120b, Qwen3.6-35B-A3B and EXAONE-4.0-32B, used to decide whether EXAONE could
serve as a general reasoner (it could not: 110 s/task vs 27 s).

## Setup note

`mlx-community/EXAONE-4.0-32B-4bit` does not load unmodified. Its `config.json` sets
`sliding_window_pattern: "LLLG"`; `mlx-lm`'s `exaone4.py` indexes that as a string while
`transformers` 5.9.0 applies integer modulo to it and crashes. Patch the cached config to
keep the string and pin `layer_types` explicitly so transformers skips its computed
branch:

```python
c["sliding_window_pattern"] = "LLLG"
c["layer_types"] = ["sliding_attention" if (i + 1) % 4 else "full_attention"
                    for i in range(c["num_hidden_layers"])]
```
