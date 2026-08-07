"""Score #43 runs.

Two metrics are computed separately and must not be conflated:

  1. Canonical recall  — of the entities we expected, how many appeared.
                         Fully automatic. Understates every config, because answers
                         are summaries while canonical lists are exhaustive.

  2. Entity validity   — of the entities the model actually produced, how many are
                         real. NOT automatic. An entity outside the canonical list is
                         *not* evidence of fabrication: it may be a real figure the
                         list omits. Validity comes from annotations.jsonl, which a
                         human fills in. Unannotated entities are reported as
                         `unreviewed`, never silently counted as fabricated.

Usage:
    python score.py                      # score all configs present
    python score.py --candidates <cfg>   # emit unannotated entities for review
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
PLAIN = re.compile(rf"(?<![가-힣])([{SURNAMES}][가-힣]{{2}})(?![가-힣])")
WORK = re.compile(r"[《<〈『]([가-힣A-Za-z0-9 ]{2,20})[》>〉』]")


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
    """
    return {e for e in canon
            if re.search(rf"(?<![가-힣]){re.escape(e)}(?![가-힣])", text)}


def extract_people(text):
    """Candidate extractor for the *outside-canonical* set only.

    Deliberately imprecise and formatting-dependent — its output feeds human
    annotation, never a recall figure. See module docstring.
    """
    cands = set(BOLD.findall(text)) | set(GLOSS.findall(text)) | set(PLAIN.findall(text))
    return {c for c in cands if plausible_name(c)}


def load_annotations():
    path = f"{HERE}/annotations.jsonl"
    ann = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                ann[(r["case_id"], r["entity"])] = r["verdict"]
    return ann


def score_config(name, cases, ann, mode="control"):
    path = f"{HERE}/outputs/{name}/run-{mode}.json"
    run = json.load(open(path))
    by_id = {c["id"]: c for c in cases}
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
                validity=dict(validity), per_case=per_case)


def main():
    cases = [json.loads(l) for l in open(f"{HERE}/cases.jsonl") if l.strip()]
    ann = load_annotations()

    if "--candidates" in sys.argv:
        name = sys.argv[sys.argv.index("--candidates") + 1]
        mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "control"
        s = score_config(name, cases, ann, mode)
        for pc in s["per_case"]:
            for e in pc["outside"]:
                if (pc["id"], e) not in ann:
                    print(json.dumps(dict(case_id=pc["id"], entity=e, verdict="",
                                          note=""), ensure_ascii=False))
        return

    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "control"
    configs = [d for d in sorted(os.listdir(f"{HERE}/outputs"))
               if os.path.isdir(f"{HERE}/outputs/{d}")
               and os.path.exists(f"{HERE}/outputs/{d}/run-{mode}.json")]
    if not configs:
        print(f"no run-{mode}.json found under outputs/ — run run.py first")
        return
    for name in configs:
        s = score_config(name, cases, ann, mode)
        env = json.load(open(f"{HERE}/outputs/{name}/run-{mode}.json"))["env"]
        print(f"\n===== {s['config']} [{env['mode']}] "
              f"harness={env.get('harness_sha256_16')} "
              f"dirty={env.get('repo_dirty')} =====")
        print(f"  canonical recall : {s['recall_tp']}/{s['recall_tp'] + s['recall_fn']}"
              f"  ({s['recall']})")
        print(f"  entity validity  : {s['validity']}")
        for pc in s["per_case"]:
            print(f"   {pc['id']} {pc['recall']:>5}  {pc['secs']:>6}s  hit={pc['hit']}")
            if pc["outside"]:
                print(f"        outside-canonical: {pc['outside']}")
    print("\nNote: entities outside the canonical list are NOT fabrications by default.")
    print("Run --candidates <config> and annotate them in annotations.jsonl.")


if __name__ == "__main__":
    main()
