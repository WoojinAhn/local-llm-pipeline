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
  - clean questions carry no canonical person name, in Hangul or romanized (verified at
    load; see check_no_leakage)
  - both reasoners receive a byte-identical system+user prompt per case, recorded as
    prompt_sha256_16
  - production profile only (reason 4000 / translate 2000)
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

EXPERIMENT = "control-clean-english"
PROFILE = "production"
REASONERS = ("three-stage-gpt-oss", "three-stage-qwen36")
SOURCE_TAG = "run-search-off-production.json"
EXPECTED_CALLS = 24

# Conventional romanizations for all 42 canonical people. Mechanical Revised
# Romanization is not enough on its own: it yields "I Sunsin" and "Gim Busik" while the
# forms that would actually leak an answer are "Yi Sun-sin" and "Kim Bu-sik". Both checks
# run; this table is the binding one.
CONVENTIONAL_ALIASES = {
    "유형원": ["Yu Hyeong-won", "Yu Hyongwon", "Ryu Hyongwon"],
    "이익": ["Yi Ik", "Lee Ik", "I Ik"],
    "박지원": ["Park Ji-won", "Pak Chiwon", "Bak Jiwon"],
    "박제가": ["Park Je-ga", "Pak Chega", "Bak Jega"],
    "홍대용": ["Hong Dae-yong", "Hong Taeyong"],
    "정약용": ["Jeong Yak-yong", "Chong Yagyong", "Jung Yakyong", "Dasan"],
    "유득공": ["Yu Deuk-gong", "Yu Tukkong"],
    "이덕무": ["Yi Deok-mu", "Lee Deokmu", "Yi Tongmu"],
    "김정희": ["Kim Jeong-hui", "Kim Jung-hee", "Kim Chonghui", "Chusa"],
    "세종": ["Sejong", "King Sejong", "Sejong the Great"],
    "정인지": ["Jeong In-ji", "Chong Inji"],
    "신숙주": ["Shin Suk-ju", "Sin Sukchu", "Shin Sukju"],
    "성삼문": ["Seong Sam-mun", "Song Sammun", "Sung Sammun"],
    "최항": ["Choe Hang", "Choi Hang"],
    "박팽년": ["Park Paeng-nyeon", "Pak Paengnyon"],
    "최만리": ["Choe Man-ri", "Choi Man-ri", "Choi Mal-li", "Choe Malli"],
    "정창손": ["Jeong Chang-son", "Chong Changson"],
    "이순신": ["Yi Sun-sin", "Yi Sun-shin", "Lee Sun-shin", "Yi Sunshin"],
    "원균": ["Won Gyun", "Weon Gyun", "Won Kyun"],
    "권율": ["Gwon Yul", "Kwon Yul", "Kwon Yool"],
    "이억기": ["Yi Eok-gi", "Lee Eok-gi", "Yi Okki"],
    "정운": ["Jeong Un", "Chong Un", "Jung Woon"],
    "황진": ["Hwang Jin", "Hwang Chin"],
    "김종직": ["Kim Jong-jik", "Kim Chongjik"],
    "조광조": ["Jo Gwang-jo", "Cho Kwang-jo", "Jo Gwangjo"],
    "김일손": ["Kim Il-son", "Kim Ilson"],
    "유자광": ["Yu Ja-gwang", "Yu Chagwang", "Ryu Jagwang"],
    "남곤": ["Nam Gon", "Nam Kon"],
    "심정": ["Sim Jeong", "Shim Jung", "Sim Chong"],
    "연산군": ["Yeonsangun", "Yonsangun", "Yeonsan-gun", "Prince Yeonsan"],
    "중종": ["Jungjong", "Chungjong", "King Jungjong"],
    "김부식": ["Kim Bu-sik", "Kim Busik", "Kim Pusik"],
    "일연": ["Ilyeon", "Iryeon", "Il-yeon"],
    "인종": ["Injong", "King Injong"],
    "각훈": ["Gakhun", "Kakhun", "Gak-hun"],
    "김옥균": ["Kim Ok-gyun", "Kim Okkyun"],
    "박영효": ["Park Yeong-hyo", "Pak Yonghyo", "Park Young-hyo"],
    "홍영식": ["Hong Yeong-sik", "Hong Yongsik"],
    "서광범": ["Seo Gwang-beom", "So Kwangbom"],
    "서재필": ["Seo Jae-pil", "So Chaepil", "Philip Jaisohn"],
    "민영익": ["Min Yeong-ik", "Min Yongik"],
    "고종": ["Gojong", "Kojong", "King Gojong", "Emperor Gojong"],
}

# Mechanical Revised Romanization, kept as a second net under the table above.
_I = ['g','kk','n','d','tt','r','m','b','pp','s','ss','','j','jj','ch','k','t','p','h']
_V = ['a','ae','ya','yae','eo','e','yeo','ye','o','wa','wae','oe','yo','u','wo','we','wi','yu','eu','ui','i']
_F = ['','k','k','k','n','n','n','t','l','l','l','l','l','l','l','l','m','p','p','t','t','ng','t','t','k','t','p','t']


def romanize(s):
    out = []
    for ch in s:
        c = ord(ch) - 0xAC00
        out.append(_I[c // 588] + _V[(c % 588) // 28] + _F[c % 28] if 0 <= c < 11172 else ch)
    return "".join(out)


def load_cases():
    cases = [json.loads(l) for l in open(f"{HERE}/cases.jsonl") if l.strip()]
    return [c for c in cases if c["domain"] == "korea" and not c.get("holdout")]


def load_questions():
    return {r["id"]: r["question"]
            for r in (json.loads(l) for l in open(f"{HERE}/control_questions.jsonl") if l.strip())}


def _flat(s):
    return re.sub(r"[^a-z]", "", s.lower())


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
        flat = _flat(q)
        if any("가" <= ch <= "힣" for ch in q):
            problems.append((c["id"], "contains Hangul"))
        for e in sorted(all_people):
            if e in q:
                problems.append((c["id"], f"canonical name {e}"))
            for alias in CONVENTIONAL_ALIASES.get(e, []):
                if _flat(alias) and _flat(alias) in flat:
                    problems.append((c["id"], f"conventional alias {e} -> {alias!r}"))
            rom = romanize(e)
            if len(rom) >= 5 and rom.lower() in flat:
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
    about this experiment's own inputs, so the script and the question set are carried
    here — check_resumable compares config_detail as a whole.
    """
    return dict(experiment=EXPERIMENT, reasoner=reasoner, profile=PROFILE,
                source_tag=SOURCE_TAG,
                script_sha256_16=file_sha16(f"{HERE}/control_clean_english.py"),
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
            s["stopped_at_cap"] = s["tokens"] >= R.GEN["reason_max_tokens"]
            payloads[name]["results"].append(dict(
                id=r["id"], reasoner=name, model_id=cfg["reasoner"],
                prompt_sha256_16=r["prompt_sha256_16"], clean_question=r["question"],
                actual_ko2en=actual_ko2en(name, r["id"]),
                stages=dict(reason=s)))
            R.save(dest_path(name), payloads[name])
            print(f"  reason {r['id']} {name}: {s['secs']}s "
                  f"{'[CAP]' if s['stopped_at_cap'] else ''}"
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
            s["stopped_at_cap"] = s["tokens"] >= R.GEN["translate_max_tokens"]
            r["stages"]["en2ko"] = s
            r["final"] = s["text"]
            r["total_secs"] = round(r["stages"]["reason"]["secs"] + s["secs"], 2)
            R.save(dest_path(name), payloads[name])
            print(f"  en2ko  {r['id']} {name}: {s['secs']}s "
                  f"{'[CAP]' if s['stopped_at_cap'] else ''}", flush=True)
    for name in REASONERS:
        d = dest_path(name)
        print(f"wrote {os.path.relpath(d, HERE) if d.startswith(HERE) else d}")


if __name__ == "__main__":
    main()
