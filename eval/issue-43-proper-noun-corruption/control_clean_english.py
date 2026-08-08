"""Clean-English control for #43: does the reasoner underperform because KO->EN
degraded the question, or because it does not produce the entities at all?

The observational runs hold nothing fixed — the reasoner receives whatever the
translator produced. This experiment replaces only that input with a hand-written
English question of the same intent, keeps everything else at the production profile,
and re-runs the reasoner and the back-translation.

    actual arm (existing artifact, no new calls) : KO -> [KO2EN] -> reason -> EN2KO
    control arm (this script, 24 calls)          :        clean EN -> reason -> EN2KO

Comparing R (entity produced in English) and T (entity survives to Korean) across the
two arms separates a degraded question from a reasoner that never had the entity.

Call budget is exactly 24 and is asserted before anything loads:
  6 cases x 2 reasoners = 12 reason calls, then 12 EN2KO calls on the same translator.
The KO2EN stage is never re-called; its text is copied from the production artifacts
purely for the record.

Design constraints enforced in code, not by convention:
  - clean questions carry no canonical person name, in Hangul, in any listed conventional
    alias, or in Revised Romanization across spacing and hyphenation (verified at load;
    see check_no_leakage). Romanizations that are not RR-derived — McCune-Reischauer
    "Sin Suk-chu" for 신숙주, say — are covered only by CONVENTIONAL_ALIASES, which is
    hand-maintained; check_alias_coverage proves it is non-empty per person, not complete.
  - both reasoners receive a byte-identical system+user prompt per case, recorded as
    prompt_sha256_16
  - production profile only (reason 4000 / translate 2000)
  - `retokenized_near_limit` on each stage records whether len(tokenizer.encode(raw))
    reached that limit. It is an indicator, not a stop reason: the harness never
    captures why generation ended, and re-encoding decoded text need not reproduce the
    original count. An earlier name for this field asserted a stop reason and read
    False on a reason block that had in fact exhausted its budget without closing.
  - every call is stateless: each request is a fresh two-message exchange (system +
    user) with no accumulated history and no prompt cache reuse. Models are loaded once
    per phase for speed — statelessness comes from the messages, not from reloading.

isolate.py's experiment C is deliberately not reused. It fixes a single hand-written
question unrelated to the case set, scores nothing, and hardcodes an 8000-token budget,
so it cannot produce a per-case R/T transition at the production profile.

Scoring, verified against score.score_config rather than assumed:

    python score.py --tag control-clean-english-production

score.py's CLI compares against the full non-holdout case set (9), so it prints an
INCOMPLETE banner listing ctl-01..ctl-03 — this experiment covers only the 6 Korea
cases, because the 24-call budget has no room for the control-domain three and they
carry no canonical people anyway. The banner is about case coverage only: the recall
denominator stays 42 either way, since score_config skips non-Korea cases when
counting. Passing just the 6 Korea cases to score_config reports complete.

Usage:
    python control_clean_english.py --dry-run     # matrix + budget, loads no model
    python control_clean_english.py               # runs the 24 calls
    python control_clean_english.py --resume      # continue an interrupted run
"""
import hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run as R  # reuse call/save/env_metadata/profiles; importing loads no model
from canonical_aliases import (CONVENTIONAL_ALIASES, contains_alias,
                               flat_letters as _flat, romanize, romanize_spaced)

EXPERIMENT = "control-clean-english"
PROFILE = "production"
REASONERS = ("three-stage-gpt-oss", "three-stage-qwen36")
SOURCE_TAG = "run-search-off-production.json"
EXPECTED_CALLS = 24

# 1 = alias table inlined in this script, so script_sha256_16 covered it.
# 2 = alias table lives in canonical_aliases.py and is hashed separately.
# 3 = the source artifact's content hash is carried, not just its filename. Schema 2
#     pinned source_tag only, so regenerating the actual arm and resuming appended
#     results built from a different KO->EN input to results built from the old one,
#     and the mixed file still passed check_provenance downstream (it records one sha).
# An older record cannot be upgraded: the hash that would prove which inputs it used was
# never written. Backfilling today's value would assert something unverified, so the
# record is refused with an explanation instead.
CONFIG_DETAIL_SCHEMA = 3

def load_cases():
    cases = [json.loads(l) for l in open(f"{HERE}/cases.jsonl") if l.strip()]
    return [c for c in cases if c["domain"] == "korea" and not c.get("holdout")]


def load_questions():
    return {r["id"]: r["question"]
            for r in (json.loads(l) for l in open(f"{HERE}/control_questions.jsonl") if l.strip())}


def check_alias_coverage(cases):
    """Every canonical person must have at least one conventional alias listed."""
    return [(c["id"], e) for c in cases for e in c["canonical_people"]
            if not CONVENTIONAL_ALIASES.get(e)]


def check_no_leakage(cases, questions):
    """A clean question must not hand the reasoner an answer.

    Checked against the whole canonical set, not just the case's own people: a name
    leaked into a neighbouring question would still contaminate that case's arm.
    """
    problems = []
    all_people = {e for c in cases for e in c["canonical_people"]}
    for c in cases:
        q = questions[c["id"]]
        if any("가" <= ch <= "힣" for ch in q):
            problems.append((c["id"], "contains Hangul"))
        for e in sorted(all_people):
            if e in q:
                problems.append((c["id"], f"canonical name {e}"))
            for alias in CONVENTIONAL_ALIASES.get(e, []):
                if contains_alias(q, alias):
                    problems.append((c["id"], f"conventional alias {e} -> {alias!r}"))
            rom = romanize(e)
            # Spaced so syllables match across spacing and hyphenation; the length floor
            # stays on the fused form, keeping very short names (이익 -> "iik") out.
            if len(rom) >= 5 and contains_alias(q, romanize_spaced(e)):
                problems.append((c["id"], f"mechanical romanization {e} ({rom})"))
    return problems


def sha16(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def file_sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def source_refs():
    refs = {}
    for name in REASONERS:
        p = f"{HERE}/outputs/{name}/{SOURCE_TAG}"
        refs[name] = dict(path=os.path.relpath(p, HERE), sha256_16=file_sha16(p))
    return refs


_SOURCE_CACHE = {}


def actual_ko2en(name, case_id):
    """The KO2EN text the observational arm actually used. Copied, never re-generated.

    Cached: this is read once per case per reasoner and the source files are large
    enough that re-parsing them twelve times is pure waste.
    """
    if name not in _SOURCE_CACHE:
        d = json.load(open(f"{HERE}/outputs/{name}/{SOURCE_TAG}"))
        _SOURCE_CACHE[name] = {x["id"]: x["stages"]["ko2en"]["text"] for x in d["results"]}
    return _SOURCE_CACHE[name][case_id]


def plan(cases, questions):
    """The full call matrix. Built before any model loads so the budget is checkable."""
    rows = []
    for name in REASONERS:
        for c in cases:
            q = questions[c["id"]]
            rows.append(dict(id=c["id"], reasoner=name,
                             prompt_sha256_16=sha16(R.prompts.REASONER_SYSTEM + "\x00" + q),
                             question=q))
    return rows


# One file per reasoner, in the layout score.py already reads
# (outputs/<config>/run-<tag>.json), so the corrected scorer needs no changes:
#     python score.py --tag control-clean-english-production
SCORE_TAG = f"{EXPERIMENT}-{PROFILE}"


# Overridable so the smoke test can exercise the real write path without touching
# the committed outputs tree.
OUT_ROOT = f"{HERE}/outputs"


def dest_path(reasoner):
    short = reasoner.replace("three-stage-", "")
    return f"{OUT_ROOT}/control-clean-{short}/run-{SCORE_TAG}.json"


def build_config_detail(reasoner):
    """Provenance that must match for a resume to be allowed.

    run.RESUME_KEYS covers search/profile/generation/prompts hashes but knows nothing
    about this experiment's own inputs, so the script, alias data, question set and
    source artifact are carried here — check_resumable compares config_detail as a whole.

    source_tag alone is a filename and says nothing about content: regenerating the
    actual arm leaves it unchanged, so the source hash is what actually gates a resume.
    """
    return dict(experiment=EXPERIMENT, reasoner=reasoner, profile=PROFILE,
                config_detail_schema=CONFIG_DETAIL_SCHEMA,
                source_tag=SOURCE_TAG,
                source_sha256_16=file_sha16(f"{HERE}/outputs/{reasoner}/{SOURCE_TAG}"),
                script_sha256_16=file_sha16(f"{HERE}/control_clean_english.py"),
                aliases_sha256_16=file_sha16(f"{HERE}/canonical_aliases.py"),
                questions_sha256_16=file_sha16(f"{HERE}/control_questions.jsonl"))


def init_payloads(env, resume):
    """Build or resume the per-reasoner payloads.

    Refuses to touch an existing output unless --resume is given: without this an
    accidental re-run silently discards the previous results and spends another 24
    calls.
    """
    payloads = {}
    for name in REASONERS:
        d = dest_path(name)
        cd = build_config_detail(name)
        if os.path.exists(d):
            if resume:
                prev = json.load(open(d)).get("config_detail", {})
                if prev and prev.get("config_detail_schema") != CONFIG_DETAIL_SCHEMA:
                    raise SystemExit(
                        f"{os.path.relpath(d, HERE)} was written under config_detail "
                        f"schema {prev.get('config_detail_schema', 1)}; this script writes "
                        f"schema {CONFIG_DETAIL_SCHEMA}.\n"
                        "    Schema 1 inlined the alias table in the script; schema 2 "
                        "hashes canonical_aliases.py separately; schema 3\n"
                        "    additionally pins the source artifact's content, not just "
                        "its filename. The missing hashes cannot be\n"
                        "    re-derived, so resuming would silently mix two different "
                        "alias tables or two different KO->EN sources.\n"
                        "    If the run is already complete no resume is needed; "
                        "otherwise start a fresh run.")
            if not resume:
                raise SystemExit(
                    f"refusing to overwrite {os.path.relpath(d, HERE)}\n"
                    "    Pass --resume to continue it, or delete the file to start over.\n"
                    "    Re-running without --resume would discard results and spend "
                    "another 24 model calls.")
            existing = json.load(open(d))
            R.check_resumable(existing, env, cd)
            payloads[name] = existing
            continue
        payloads[name] = dict(
            config=f"control-clean-{name.replace('three-stage-', '')}",
            experiment=EXPERIMENT, config_detail=cd, env=env, results=[])
        os.makedirs(os.path.dirname(d), exist_ok=True)
    return payloads


def main():
    dry = "--dry-run" in sys.argv
    resume = "--resume" in sys.argv
    cases, questions = load_cases(), load_questions()

    missing_alias = check_alias_coverage(cases)
    if missing_alias:
        raise SystemExit(
            "CONVENTIONAL_ALIASES is incomplete, so the leakage check cannot be trusted:\n  " +
            "\n  ".join(f"{i}: {e}" for i, e in missing_alias))

    want, have = {c["id"] for c in cases}, set(questions)
    if want != have:
        raise SystemExit(
            "control_questions.jsonl does not match the case set:\n"
            f"    missing questions : {sorted(want - have)}\n"
            f"    unexpected ids    : {sorted(have - want)}")

    problems = check_no_leakage(cases, questions)
    if problems:
        raise SystemExit("clean questions leak canonical answers:\n  " +
                         "\n  ".join(f"{i}: {w}" for i, w in problems))

    rows = plan(cases, questions)
    n_reason, n_en2ko = len(rows), len(rows)
    total = n_reason + n_en2ko
    if total != EXPECTED_CALLS:
        raise SystemExit(f"call budget is {total}, expected {EXPECTED_CALLS}")

    R.GEN.update(R.PROFILES[PROFILE])
    env = R.env_metadata("off", PROFILE, EXPERIMENT)
    env.update(experiment=EXPERIMENT, arm="control-clean-english",
               questions_sha256_16=file_sha16(f"{HERE}/control_questions.jsonl"),
               planned_calls=dict(reason=n_reason, en2ko=n_en2ko, total=total),
               source_artifacts=source_refs())

    if dry:
        print(f"=== {EXPERIMENT} · dry run · NO MODEL LOADED ===")
        print(f"profile={PROFILE}  reason_max={R.GEN['reason_max_tokens']}  "
              f"translate_max={R.GEN['translate_max_tokens']}")
        print(f"leakage check: PASS ({len(cases)} questions)")
        print(f"budget: reason {n_reason} + en2ko {n_en2ko} = {total} (limit {EXPECTED_CALLS})")
        print("dest (one per reasoner, score.py layout):")
        for _n in REASONERS: print(f"  {os.path.relpath(dest_path(_n), HERE)}")
        print(f"score with: python score.py --tag {SCORE_TAG}")
        print(f"\n{'case':7s} {'reasoner':22s} {'prompt_sha':12s} question")
        for r in rows:
            print(f"{r['id']:7s} {r['reasoner']:22s} {r['prompt_sha256_16']:12s} "
                  f"{r['question'][:58]}...")
        ids = sorted({r["id"] for r in rows})
        for cid in ids:
            shas = {r["prompt_sha256_16"] for r in rows if r["id"] == cid}
            assert len(shas) == 1, (cid, shas)
        print("\nbyte-identical prompt across both reasoners: PASS (one sha per case)")
        print("source artifacts (KO2EN reused, never re-called):")
        for k, v in env["source_artifacts"].items():
            print(f"  {k:22s} {v['path']}  sha={v['sha256_16']}")
        return

    payloads = init_payloads(env, resume)
    done = {(r["id"], name) for name, p in payloads.items() for r in p["results"]}

    # Phase 1 — reason, grouped by model so each reasoner loads once.
    for name in REASONERS:
        todo = [r for r in rows if r["reasoner"] == name and (r["id"], name) not in done]
        if not todo:
            continue
        cfg = R.CONFIGS[name]
        m, t = R.load(cfg["reasoner"])
        for r in todo:
            s = R.call(m, t, R.prompts.REASONER_SYSTEM, r["question"],
                       R.GEN["reason_max_tokens"], thinking=cfg.get("reasoner_thinking"))
            s["retokenized_near_limit"] = s["tokens"] >= R.GEN["reason_max_tokens"]
            payloads[name]["results"].append(dict(
                id=r["id"], reasoner=name, model_id=cfg["reasoner"],
                prompt_sha256_16=r["prompt_sha256_16"], clean_question=r["question"],
                actual_ko2en=actual_ko2en(name, r["id"]),
                stages=dict(reason=s)))
            R.save(dest_path(name), payloads[name])
            print(f"  reason {r['id']} {name}: {s['secs']}s "
                  f"{'[NEAR-LIMIT]' if s['retokenized_near_limit'] else ''}"
                  f"{' [UNTERMINATED]' if s['unterminated_think'] else ''}", flush=True)
        del m

    # Phase 2 — one translator load for all back-translations.
    todo = [(name, r) for name, p in payloads.items() for r in p["results"]
            if "en2ko" not in r["stages"]]
    if todo:
        m, t = R.load(R.TRANSLATOR)
        for name, r in todo:
            s = R.call(m, t, R.prompts.TRANSLATE_EN_TO_KO, r["stages"]["reason"]["text"],
                       R.GEN["translate_max_tokens"], thinking=False)
            s["retokenized_near_limit"] = s["tokens"] >= R.GEN["translate_max_tokens"]
            r["stages"]["en2ko"] = s
            r["final"] = s["text"]
            r["total_secs"] = round(r["stages"]["reason"]["secs"] + s["secs"], 2)
            R.save(dest_path(name), payloads[name])
            print(f"  en2ko  {r['id']} {name}: {s['secs']}s "
                  f"{'[NEAR-LIMIT]' if s['retokenized_near_limit'] else ''}", flush=True)
    for name in REASONERS:
        d = dest_path(name)
        print(f"wrote {os.path.relpath(d, HERE) if d.startswith(HERE) else d}")


if __name__ == "__main__":
    main()
