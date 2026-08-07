"""Score #43 runs.

Two metrics are computed separately and must not be conflated:

  1. Canonical recall  — of the entities we expected, how many appeared.
                         Fully automatic. Understates every config, because answers
                         are summaries while canonical lists are exhaustive.

  2. Entity validity   — of the entities the extractor *detected*, how many are real.
                         A DETECTED-CANDIDATE LOWER BOUND, not TP/FP/FN — an
                         undetected fabrication never enters the denominator, so this
                         cannot be converted into a fabrication rate. NOT automatic. An entity outside the canonical list is
                         *not* evidence of fabrication: it may be a real figure the
                         list omits. Validity comes from annotations.jsonl, which a
                         human fills in. Unannotated entities are reported as
                         `unreviewed`, never silently counted as fabricated.

Usage:
    python score.py                              # default tag search-off-candidate
    python score.py --tag search-parity-production
    python score.py --candidates <cfg>           # emit entities needing a verdict

Only compare configs sharing one tag. A mismatched harness or prompts hash across
the scored set is reported as a warning.
"""
import json, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

SURNAMES = ("김이박최정강조윤장임한오서신권황안송류유전홍고문양손배백허남심노하"
            "곽성차주우구석선설마길연방위표명기반왕금옥육인맹제모탁국여진어편봉대걸")
# Inflectional endings — a Korean given name never ends in these.
ENDINGS = set("을를이가은는의에서로와과도만고며여다한할함들적성화기인된대라며야죠음임든지네요")
STOPWORDS = {
    "조선", "고려", "신라", "백제", "한국", "중국", "일본", "청나라", "명나라", "성리학", "유교",
    "양반", "왕권", "정치", "제도", "문화", "사회", "경제", "학문", "사상", "개혁", "전쟁", "정변",
    "문자", "한글", "한문", "한자", "백성", "신하", "국가", "권력", "기반", "배경", "주요", "인물",
    "전개", "과정", "차이", "편찬", "기록", "설화", "전설", "신분", "유학", "서양", "무역", "상업",
    "농업", "토지", "군사", "국방", "외교", "재정", "세금", "신분제", "훈구파", "사림파", "북학파",
    "개화파", "실학", "정음", "이두", "표기", "연구", "학파", "계열", "중인", "상민", "천민",
    "방법론", "정통성", "연대기", "왕조사", "편찬자", "표기법", "문화권", "한국어", "국학의",
}
BOLD = re.compile(r"\*\*([가-힣]{2,4})\*\*")
GLOSS = re.compile(r"([가-힣]{2,4})\s*\(\s*[一-鿿]{2,4}\s*\)")
# 2-4 syllables. An earlier version was fixed at 3, so 2-syllable fabrications
# (the 이익/원균 shape) were invisible on the FP side exactly as they had been on
# the recall side. Widening raises noise, which is fine: candidates go to a human.
PLAIN = re.compile(rf"(?<![가-힣])([{SURNAMES}][가-힣]{{1,3}})(?![가-힣])")
WORK = re.compile(r"[《<〈『]([가-힣A-Za-z0-9 ]{2,20})[》>〉』]")
# Korean attaches particles and honorific titles straight onto a name with no space
# (성삼문과, 김종직의, 세종대왕). A bare non-Hangul boundary rejects all of those, which
# undercounted every configuration. Closed lists only, and a non-Hangul boundary is still
# required after the suffix — so 이익 does not match inside 이익환, and the one-syllable
# corruptions this eval exists to catch stay excluded.
PARTICLES = ("이라고", "으로", "께서", "에게", "한테", "보다", "처럼", "같이", "부터", "까지",
             "라고", "에서", "은", "는", "이", "가", "을", "를", "의", "에", "와", "과",
             "도", "만", "로", "라", "야", "여", "및", "등")
TITLES = ("대왕", "장군", "선생", "황제", "임금", "왕", "공")


def _alt(words):
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


# A title and a particle stack: 세종대왕의. Both optional, and a non-Hangul boundary is
# still required after them.
NAME_SUFFIX = rf"(?:{_alt(TITLES)})?(?:{_alt(PARTICLES)})?"


def plausible_name(tok):
    return (len(tok) >= 2 and tok not in STOPWORDS
            and tok[-1] not in ENDINGS and tok[0] in SURNAMES)


def find_canonical(text, canon):
    """Recall: match each expected entity directly, with Hangul boundaries.

    Must NOT go through extract_people(). That extractor's PLAIN pattern is
    fixed at 3 syllables, so plain-text 2-syllable names (이익, 원균, 권율,
    일연, 고종) were only ever counted when the model happened to bold them.
    That made recall depend on formatting and silently understated every
    configuration.

    Boundary lookarounds also do the discrimination that matters here: 이익
    matches 이익, but not 이익환 — which is one of the fabrications this eval
    exists to catch.

    The right boundary additionally admits an attached particle or honorific
    title (see NAME_SUFFIX). Requiring a bare non-Hangul boundary rejected
    성삼문과, 김종직의 and 세종대왕, and it did so unevenly across configurations,
    so the gap between them was overstated as well as the absolute numbers.
    """
    return {e for e in canon
            if re.search(rf"(?<![가-힣]){re.escape(e)}{NAME_SUFFIX}(?![가-힣])", text)}


def extract_people(text):
    """Candidate extractor for the *outside-canonical* set only.

    Deliberately imprecise and formatting-dependent — its output feeds human
    annotation, never a recall figure. See module docstring.
    """
    cands = set(BOLD.findall(text)) | set(GLOSS.findall(text)) | set(PLAIN.findall(text))
    return {c for c in cands if plausible_name(c)}


def _dirty_label(env):
    """Never relabel a legacy whole-tree value as source-scoped.

    Records written before the source/whole-tree split carry only `repo_dirty`, which
    counts generated outputs/** as dirt. Printing that as `source_dirty` would restate
    the exact false positive the split removed.
    """
    if "source_dirty" in env:
        return f"source_dirty={env['source_dirty']}"
    return f"repo_dirty(legacy, counts outputs/)={env.get('repo_dirty')}"


def load_annotations():
    path = f"{HERE}/annotations.jsonl"
    ann = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                ann[(r["case_id"], r["entity"])] = r["verdict"]
    return ann


def score_config(name, cases, ann, tag="search-off-candidate"):
    path = f"{HERE}/outputs/{name}/run-{tag}.json"
    run = json.load(open(path))
    by_id = {c["id"]: c for c in cases}
    expected = {c["id"] for c in cases}
    present = {r["id"] for r in run["results"]}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    tp = fn = 0
    validity = Counter()
    per_case = []
    for r in run["results"]:
        c = by_id[r["id"]]
        if c["domain"] != "korea":
            continue
        canon = set(c["canonical_people"])
        hit = find_canonical(r["final"], canon)
        tp += len(hit); fn += len(canon - hit)
        outside = sorted(extract_people(r["final"]) - canon)
        verdicts = Counter(ann.get((r["id"], e), "unreviewed") for e in outside)
        validity.update(verdicts)
        validity["valid"] += len(hit)  # canonical hits are valid by construction
        per_case.append(dict(id=r["id"], recall=f"{len(hit)}/{len(canon)}",
                             hit=sorted(hit), outside=outside,
                             secs=r["total_secs"], verdicts=dict(verdicts)))
    return dict(config=name, recall_tp=tp, recall_fn=fn,
                recall=round(tp / (tp + fn), 3) if tp + fn else None,
                validity=dict(validity), per_case=per_case,
                missing_cases=missing, unexpected_cases=extra,
                complete=not missing and not extra)


def main():
    cases = [json.loads(l) for l in open(f"{HERE}/cases.jsonl") if l.strip()]
    # run.py excludes holdout unless --holdout is passed; the completeness check has to
    # use the same case set or every default run reports as incomplete.
    if "--holdout" not in sys.argv:
        cases = [c for c in cases if not c.get("holdout")]
    ann = load_annotations()

    if "--candidates" in sys.argv:
        name = sys.argv[sys.argv.index("--candidates") + 1]
        tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "search-off-candidate"
        s = score_config(name, cases, ann, tag)
        for pc in s["per_case"]:
            for e in pc["outside"]:
                if (pc["id"], e) not in ann:
                    print(json.dumps(dict(case_id=pc["id"], entity=e, verdict="",
                                          note=""), ensure_ascii=False))
        return

    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "search-off-candidate"
    configs = [d for d in sorted(os.listdir(f"{HERE}/outputs"))
               if os.path.isdir(f"{HERE}/outputs/{d}")
               and os.path.exists(f"{HERE}/outputs/{d}/run-{tag}.json")]
    if not configs:
        print(f"no run-{tag}.json found under outputs/ — run run.py first")
        return
    seen = set()
    for name in configs:
        s = score_config(name, cases, ann, tag)
        env = json.load(open(f"{HERE}/outputs/{name}/run-{tag}.json"))["env"]
        seen.add((env.get("generation_sha256_16") or env.get("harness_sha256_16"),
                  env.get("prompts_sha256_16")))
        print(f"\n===== {s['config']} "
              f"[search={env.get('search')} profile={env.get('profile')}"
              f"{' PRODUCTION-EQUIVALENT' if env.get('production_equivalent') else ''}] "
              f"gen={env.get('generation_sha256_16') or env.get('harness_sha256_16')} "
              f"{_dirty_label(env)} =====")
        if not s["complete"]:
            print("  *** INCOMPLETE RUN — SCORES BELOW ARE NOT COMPARABLE ***")
            if s["missing_cases"]:
                print(f"      missing cases: {s['missing_cases']}")
            if s["unexpected_cases"]:
                print(f"      unexpected cases: {s['unexpected_cases']}")
        print(f"  canonical recall : {s['recall_tp']}/{s['recall_tp'] + s['recall_fn']}"
              f"  ({s['recall']})")
        print(f"  entity validity  : {s['validity']}   [lower bound — see note]")
        for pc in s["per_case"]:
            print(f"   {pc['id']} {pc['recall']:>5}  {pc['secs']:>6}s  hit={pc['hit']}")
            if pc["outside"]:
                print(f"        outside-canonical: {pc['outside']}")
    if len(seen) > 1:
        print("\nWARNING: configs above were produced by different harness/prompt "
              f"revisions {sorted(seen)} — not directly comparable.")
    print("\nNote: entities outside the canonical list are NOT fabrications by default.")
    print("Run --candidates <config> and annotate them in annotations.jsonl.")
    print("Validity counts are a DETECTED-CANDIDATE LOWER BOUND, not TP/FP/FN: the")
    print("extractor only surfaces names it matches, so undetected fabrications are")
    print("absent from the denominator. Never report these as a fabrication rate.")


if __name__ == "__main__":
    main()
