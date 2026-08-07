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

**Entity validity** — of the names a model actually produced, how many are real. This is
**not** automatic and must not be made automatic. An entity outside the canonical list is
not evidence of fabrication — it is frequently a real figure the list omits. In the first
run, `한명회`, `유길준`, `김윤식`, `고니시` and others fell outside the canonical lists and
are all real people. Only `annotations.jsonl` decides validity; unannotated entities are
reported as `unreviewed`.

Conflating these two produced a real error during this investigation: a 4x recall gap was
described as "4x less corruption", which the data does not support.

## Known limitations

- The person-name extractor is regex over Korean text and is noisy in both directions.
  Korean surname syllables are common in ordinary vocabulary, so a first version matched
  large numbers of non-names. The current version is tighter but still surfaces
  non-entities — this is why extraction feeds annotation rather than a verdict.
- Canonical lists are hand-curated and incomplete by construction.
- n=6 Korea-domain cases (plus 2 holdout). Enough to establish that the effect is
  systematic, not enough to rank configurations precisely.
- Holdout cases (`"holdout": true`) are excluded by default and must **not** be used when
  tuning the `KOREA:yes/no` router in #44, or they stop being holdout.

## Running

```bash
python eval/issue-43-proper-noun-corruption/run.py three-stage-gpt-oss
python eval/issue-43-proper-noun-corruption/run.py three-stage-qwen36
python eval/issue-43-proper-noun-corruption/run.py single-exaone
python eval/issue-43-proper-noun-corruption/score.py

# entities needing a verdict
python eval/issue-43-proper-noun-corruption/score.py --candidates three-stage-gpt-oss \
  >> eval/issue-43-proper-noun-corruption/annotations.jsonl
```

Generation is greedy (mlx-lm's default `temp=0`), so runs are reproducible without a
seed. `run.py` records library versions, repo commit, platform, and per-stage timing,
token counts and peak memory.

## Isolating the damage stage — still open

Observational runs establish the defect is **reasoner-independent**: two different
reasoners score identically (4/43) while the wrapper-free path scores 16/43. They do
**not** establish which step loses the entities. "Reasoner-independent" and "not the
reasoner" are different claims — any reasoner handed romanized Korean names may
confabulate.

`isolate.py` separates the three candidates:

```bash
python eval/issue-43-proper-noun-corruption/isolate.py a   # KO -> EN -> KO, names only
python eval/issue-43-proper-noun-corruption/isolate.py b   # correct English -> KO only
python eval/issue-43-proper-noun-corruption/isolate.py c   # fixed English Q, both reasoners
```

Not yet run.

## Baseline results (2026-08-06, v1 harness — final answers only)

Canonical recall over 6 Korea-domain cases, 43 canonical entities:

| Configuration | Recall | Avg latency |
|---|---|---|
| `single-exaone` | 16/43 | 60.5 s |
| `three-stage-gpt-oss` | 4/43 | 42.5 s |
| `three-stage-qwen36` | 4/43 | 25.6 s |

`outputs/*/v1-final-only.json` are those original records. They lack per-stage
intermediates — `run.py` in this directory supersedes the script that produced them and
captures stages. Re-run before relying on stage-level analysis.

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
