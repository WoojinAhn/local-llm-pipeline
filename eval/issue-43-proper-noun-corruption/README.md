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
| `control_questions.jsonl` | Hand-written English questions with the same intent and no canonical-name leakage. |
| `control_clean_english.py` | Runs the clean-English reasoner + back-translation control at production limits. |
| `analyze_transitions.py` | Compares per-person reasoner (R) to final translation (T) transitions without model calls. |
| `canonical_aliases.py` | Canonical alias data and the boundary-aware matcher, shared by the control harness and the analyzer. |
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
python control_clean_english.py --dry-run                     # 24-call matrix, no model load
python control_clean_english.py                               # clean-English control
python control_clean_english.py --resume                      # resume an interrupted current-schema run
python score.py --tag control-clean-english-production
python analyze_transitions.py                                 # read-only R -> T analysis

# entities needing a verdict
python score.py --candidates three-stage-gpt-oss >> annotations.jsonl
```

Generation is greedy (mlx-lm's default `temp=0`), so runs are reproducible without a
seed. `run.py` records library versions, platform, and per-stage timing, token counts
and peak memory.

For provenance it records `repo_commit` plus the fields below. `repo_commit` alone is
misleading: a run made while the harness was uncommitted records the *previous* HEAD,
which does not identify the code that produced it.

| Field | Meaning |
|---|---|
| `source_dirty`, `source_dirty_paths` | Uncommitted changes **excluding** `eval/*/outputs/**`. This is the field that bears on reproducibility — check it before trusting `repo_commit`. |
| `repo_dirty`, `dirty_paths` | Whole-tree state, original meaning, kept so records written before the split stay readable. |
| `generation_sha256_16` | Hash of `run.py` + `cases.jsonl` — what produced the raw results. |
| `scoring_sha256_16` | Hash of `score.py`. Deliberately separate: a scorer edit must not invalidate valid raw generations. |

A sequential sweep writes results between configurations, so every run after the first
saw the tree dirtied by its own earlier output and recorded `repo_dirty=true` with no
source change. `source_dirty` excludes expected generated artifacts; anything else — a
touched `run.py`, `cases.jsonl`, `prompts.py`, pipeline source — still counts.

Records written before this split (`outputs/*/run-search-off-candidate.json` from
2026-08-07) carry only `repo_dirty`. Two of the three read `true` for exactly this
reason; their `dirty_paths` show only sibling `outputs/**` files. They are not
retroactively rewritten.

Long runs save after every case (atomic replace) and support `--resume`, so an
interrupted multi-hour run is not lost. `--resume` refuses to append when
`config_detail`, `search`, `profile`, `generation_sha256_16` or `prompts_sha256_16` differ
from the existing file, so results from different code never mix into one record.

The committed clean-control outputs are complete legacy schema-1 records. At generation
time the alias table lived inside `control_clean_english.py`, so the recorded script hash
covered it. They must not be resumed with the schema-2 harness or backfilled with the
current alias-file hash. Their raw stage records also retain a legacy `stopped_at_cap`
field derived from post-generation retokenization; it is not a recorded stop reason and
must not be interpreted as one. Current runs write `retokenized_near_limit` instead.

## Isolating the damage stage

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

Experiments A-C have not yet been run. The separate clean-English control described
below has run; it holds the reasoner prompt intent constant without handing the model
any canonical names.

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

### v3 (`search-off-production`) — shipped token limits

Same harness and case set as v2, with `--profile production` (translate 2000 / reason
4000, the limits `mlx-pipeline.py` uses). Denominator 42 canonical entities over 6
Korea-domain cases; both configurations completed all 9 non-holdout cases with
`source_dirty=false`, generation hash `f0fad4af3d93f000`. The figures below use the
suffix-aware scorer at scoring hash `677a4d573089930e`.

| Configuration | Recall | Miss rate | Avg latency (Korea) |
|---|---|---|---|
| `three-stage-gpt-oss` | 16/42 (38.1%) | 26/42 (61.9%) | 79.5 s |
| `three-stage-qwen36` | 12/42 (28.6%) | 30/42 (71.4%) | 68.1 s |

These same raw artifacts previously scored 15/42 and 9/42. An A/B using the scorer
from `7aeb6d9` and the current scorer on the unchanged JSON produced 15→16 for gpt-oss
and 9→12 for Qwen3.6. The difference comes from `def8942`, which admits attached Korean
particles and titles in canonical matching; it is not a generation improvement.

### Candidate vs production, same configurations

| Configuration | Profile | Recall | Avg latency (Korea) |
|---|---|---|---|
| `three-stage-gpt-oss` | candidate (4000/8000) | 17/42 (40.5%) | 114.2 s |
| `three-stage-gpt-oss` | **production (2000/4000)** | **16/42 (38.1%)** | **79.5 s** |
| `three-stage-qwen36` | candidate (4000/8000) | 12/42 (28.6%) | 76.0 s |
| `three-stage-qwen36` | **production (2000/4000)** | **12/42 (28.6%)** | **68.1 s** |

The shipped limits cost gpt-oss one canonical entity and save 35 s per Korea query.
Qwen3.6 is unchanged at 12/42 — the extra budget in the candidate profile did not buy it
any additional Korean proper nouns.

**These runs are `search=off`, so they are neither settings-equivalent nor
conversation-state-equivalent to production.** Only the generation profile matches; the
metadata records `settings_equivalent_to_production=false` for both, because settings
equivalence additionally requires `--search parity`. And
`reproduces_conversation_state` is `false` here as everywhere in this harness: production
accumulates `_reasoner_history` and reuses `prompt_cache` across turns, while every case
here runs cold. Read this as a token-budget comparison holding the reasoner axis fixed,
nothing more.

### Clean-English control (`control-clean-english-production`)

The observational arm hands each reasoner the actual KO→EN output. The control replaces
only that input with a hand-written English question of the same intent, checked to
contain none of the 42 canonical people, then runs reason + EN→KO at the production
limits. Six cases × two reasoners × two calls produced exactly 24 calls. Both output
files record commit `7788e5d`, `source_dirty=false`, and generation hash
`f0fad4af3d93f000`.

The current scorer gives the following full as-run totals. Its `INCOMPLETE RUN` banner
for the control files is expected: the 24-call budget covers the six Korea cases but not
the three `ctl-*` cases. Those omitted cases contain no canonical Korean people, so the
Korea-case denominator remains 42; `analyze_transitions.py` also reconciles these totals
directly with `score.py`.

| Reasoner | Actual KO→EN | Clean English | Net |
|---|---:|---:|---:|
| gpt-oss | 16/42 (38.1%) | 19/42 (45.2%) | +3 |
| Qwen3.6 | 12/42 (28.6%) | 15/42 (35.7%) | +3 |

Canonical names are absent from both question arms by design, so Q is a leakage guard,
not a causal transition axis. The useful transition is R→T: whether the English
reasoner output names a canonical person, and whether that person survives in the final
Korean text. R uses the `alias-boundary` matcher: Hangul-boundary canonical names plus
`CONVENTIONAL_ALIASES` matched as whole names. Alias parts may be joined directly or
separated by in-name whitespace and at most one hyphen, so `Park Jiwon`, `Park Ji-won`
and gpt-oss's U+202F and U+2011 forms all match. A word-hyphen extension on either side
is rejected, so neither `Yi Ik-hwan` nor `Hwan-Yi Ik` counts as `Yi Ik`. T uses
`score.find_canonical` unchanged.

| Reasoner | Arm | n | R | T | P(T\|R) | T rescue / loss |
|---|---|---:|---:|---:|---:|---:|
| gpt-oss | actual | 42 | 19 | 16 | 84.21% | — |
| gpt-oss | clean control | 42 | 23 | 19 | 82.61% | +7 / −4 |
| Qwen3.6 | actual | 42 | 19 | 12 | 63.16% | — |
| Qwen3.6 | clean control | 42 | 19 | 15 | 78.95% | +6 / −3 |

This primary table keeps failures because they are part of the shipped-limit behavior.
For a symmetric closed-pair sensitivity, the analyzer drops a case from both arms when
either reason output never closes: Qwen3.6 `ko-03` fails in the actual arm and `ko-04`
fails in the control. Over the remaining 28 people, Qwen3.6 moves from R=14, T=10
(71.43%) to R=15, T=12 (80.00%), with T rescue/loss +4/−2. gpt-oss has no unclosed
reason block, so its sensitivity figures equal the primary table.

The clean question raises gpt-oss reason-stage coverage from 19→23 but leaves Qwen3.6
unchanged at 19; it changes which Qwen3.6 names appear, with R rescue/loss +5/−5. Final
recall rises by three for both reasoners, so degraded KO→EN questions contribute to the
loss, but the effect is not monotonic. Loss also remains after R. For clean-control
gpt-oss, decoded-text retokenization is at or above the nominal 2000-token EN→KO limit
on 5/6 cases, while the remaining case ends mid-sentence at 1997. The artifacts do not
record actual generation token counts or stop reasons, so exact limit incidence cannot be
established. The clean control therefore narrows the cross-reasoner P(T|R) gap, but the
small sample, asymmetric model failures, and possible translation truncation do not
support claiming that the gap disappears or that KO→EN is the only cause.

### Reason blocks at or above the shipped 4000-token limit

Evidence taken from the raw `stages.reason` records, not from the `unterminated_think`
flag — see the caveat below.

| Configuration | Reason stage at or above the 4000-token limit (retokenized) | Reasoning block left unclosed |
|---|---|---|
| `three-stage-gpt-oss` | 1/9 (`ctl-03`) | 0/9 — `<\|channel\|>final<\|message\|>` present in all 9 |
| `three-stage-qwen36` | 4/9 (`ko-03`, `ko-04`, `ko-06`, `ctl-03`) | **1/9 (`ko-03`)** — no `</think>` |

Token counts are `len(tokenizer.encode(raw))` computed after generation, not the
generation counter. A value at the limit indicates possible budget exhaustion but does
not prove it; re-encoding decoded text need not reproduce the original count, and the
harness records no stop reason.

On `ko-03` (임진왜란) Qwen3.6's reason block reaches 4000 tokens when its decoded text
is re-encoded, carries no closure marker, and ends mid-word. What reached
back-translation was therefore raw reasoning presented as analysis, and the case scored
1/6. Budget exhaustion is the natural reading, but it remains an inference.

**Caveat — `unterminated_think` reports 0/9 for both and is unreliable for Qwen3.6.**
`has_unterminated_think()` requires a literal `<think>` in the completion, but Qwen3.6's
chat template opens the block in the *prompt*, so the completion contains only the
closing tag. The flag can therefore never fire for this model. The numbers above come
from checking `</think>` presence and token counts directly. Fixing the detector is out
of scope for this commit.

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
