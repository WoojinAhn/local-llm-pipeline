"""Per-person R -> T transition across the actual and clean-control arms (#43).

Reads two existing artifact sets and calls no model. Nothing here generates text; if
this script ever loads a model, that is a bug and test_transitions.py fails.

    actual arm  : outputs/three-stage-*/run-search-off-production.json
    control arm : outputs/control-clean-*/run-control-clean-english-production.json

For each reasoner x case x canonical person:

    R  the reasoner's English output names the person — canonical Hangul with Hangul
       boundaries, or any CONVENTIONAL_ALIASES form matched as a whole word. Between the
       parts of an alias only in-name whitespace and at most one hyphen are allowed, so
       gpt-oss's U+202F narrow no-break space and U+2011 non-breaking hyphen both pass
       while "Yi. Ik" and a Markdown cell break like "King** | Jungjong" do not. A
       trailing or leading word-hyphen extension is rejected, so "Yi Ik-sun" is not 이익.
    T  score.find_canonical on the final Korean text — the committed scorer, unchanged,
       so totals here reconcile with `score.py --tag ...`.

Q is deliberately not a transition axis. Both arms are constant zero: the control by
construction (the leakage guard rejects any clean question containing an answer) and the
actual arm because its KO->EN question text never named these people either — that stage
output is what is checked, not the original Korean query. It is verified
as a guard and reported as such, never as a stage that lost anything.

Two aggregates are reported, and they must not be mixed:

  PRIMARY (as-run)      every case, n=42 per reasoner, nothing dropped. This is the
                        headline: it counts truncation failures as the outcomes they are.

  SENSITIVITY           drops a case from BOTH arms when EITHER arm's reason block never
  (closed pairs)        closed. The union rule is the point: dropping only the arm that
                        failed would be outcome-dependent exclusion and would flatter
                        whichever arm was spared. An earlier version of this script keyed
                        exclusion on the control arm alone and did exactly that.

A cell with R=0 and T=1 is reported as `unmatched` rather than impossible. Back-
translation can introduce a correct Korean name the reasoner's English never spelled out
— from context, or from a Hangul gloss the alias matcher did not catch — so such a cell
is an anomaly worth inspecting, not a contradiction.

    python analyze_transitions.py
"""
import hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import score
from canonical_aliases import CONVENTIONAL_ALIASES, names_person

# Data-only imports on purpose. control_clean_english pulls in run.py and therefore MLX,
# which this analysis never needs; canonical_aliases and score are pure Python, so the
# analyzer runs on an interpreter with no MLX installed. test_transitions proves it in a
# subprocess with the mlx packages blocked.

MATCHER = ("alias-boundary (Hangul-boundary canonical + CONVENTIONAL_ALIASES matched as "
           "whole words; in-name whitespace and at most one hyphen may separate alias "
           "parts; word-hyphen extensions on either side are rejected)")
ACTUAL_TAG = "search-off-production"
CONTROL_TAG = "control-clean-english-production"
ARMS = {
    "gpt-oss": ("three-stage-gpt-oss", "control-clean-gpt-oss"),
    "qwen36": ("three-stage-qwen36", "control-clean-qwen36"),
}
# Mirrors run.py's THINK_SPLIT exactly. Duplicated rather than imported because run.py
# imports MLX; test_transitions pins the two patterns to the same source string. The
# Harmony marker must be the full "<|channel|>final<|message|>": the bare prefix also
# appears in a truncated header, so matching it alone would score an unfinished block
# as closed.
CLOSURE = re.compile(r"</think>|<\|channel\|>final<\|message\|>|assistantfinal")


def load_arm(config, tag):
    d = json.load(open(f"{HERE}/outputs/{config}/run-{tag}.json"))
    return {r["id"]: r for r in d["results"]}, d["env"]


def non_holdout_cases():
    cases = [json.loads(l) for l in open(f"{HERE}/cases.jsonl") if l.strip()]
    return [c for c in cases if not c.get("holdout")]


def korea_cases():
    return [c for c in non_holdout_cases() if c["domain"] == "korea"]


def unterminated(stage):
    """No closing marker means the model never left its reasoning block."""
    return not CLOSURE.search(stage["raw"])


def near_limit(stage, limit):
    """Retokenized length reached the limit — an indicator, not a stop reason.

    `stage["tokens"]` is len(tokenizer.encode(raw)) recorded after the fact, not the
    generation counter, so a value at the limit is evidence the budget was probably
    exhausted rather than proof of it. Re-encoding decoded text need not reproduce the
    original token count.
    """
    return stage["tokens"] >= limit


def check_provenance(reasoner, act_cfg, aenv, cenv):
    """Refuse to compare artifacts that were not produced by the same harness.

    A silent mismatch here would compare a control against an actual arm generated by
    different code, prompts or limits, and every rescue/loss number would be noise.
    """
    problems = []
    for key in ("generation_sha256_16", "prompts_sha256_16"):
        if aenv.get(key) != cenv.get(key):
            problems.append(f"{key}: actual={aenv.get(key)} control={cenv.get(key)}")
    ref = (cenv.get("source_artifacts") or {}).get(act_cfg)
    if not ref:
        problems.append(f"control env has no source_artifacts entry for {act_cfg}")
    else:
        path = f"{HERE}/outputs/{act_cfg}/run-{ACTUAL_TAG}.json"
        actual_sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        if ref.get("sha256_16") != actual_sha:
            problems.append(
                f"control was generated against a different actual artifact: "
                f"recorded={ref.get('sha256_16')} on-disk={actual_sha}")
    if problems:
        raise SystemExit(f"provenance mismatch for {reasoner}:\n  " + "\n  ".join(problems))


def collect():
    rows, meta = [], {}
    for reasoner, (act_cfg, ctl_cfg) in ARMS.items():
        A, aenv = load_arm(act_cfg, ACTUAL_TAG)
        C, cenv = load_arm(ctl_cfg, CONTROL_TAG)
        check_provenance(reasoner, act_cfg, aenv, cenv)
        # Each arm is measured against its own recorded limits. Applying one arm's
        # profile to the other would silently mislabel truncation.
        a_r_lim = aenv["generation"]["reason_max_tokens"]
        a_e_lim = aenv["generation"]["translate_max_tokens"]
        c_r_lim = cenv["generation"]["reason_max_tokens"]
        c_e_lim = cenv["generation"]["translate_max_tokens"]
        for c in korea_cases():
            a, b = A[c["id"]], C[c["id"]]
            a_un = unterminated(a["stages"]["reason"])
            c_un = unterminated(b["stages"]["reason"])
            meta[(reasoner, c["id"])] = dict(
                actual_unterminated=a_un, control_unterminated=c_un,
                closed_pair=not (a_un or c_un),
                a_r_near=near_limit(a["stages"]["reason"], a_r_lim),
                a_e_near=near_limit(a["stages"]["en2ko"], a_e_lim),
                c_r_near=near_limit(b["stages"]["reason"], c_r_lim),
                c_e_near=near_limit(b["stages"]["en2ko"], c_e_lim),
                limits=dict(actual=dict(reason=a_r_lim, en2ko=a_e_lim),
                            control=dict(reason=c_r_lim, en2ko=c_e_lim)))
            for p in c["canonical_people"]:
                rows.append(dict(
                    reasoner=reasoner, case=c["id"], person=p,
                    closed_pair=not (a_un or c_un),
                    aQ=names_person(a["stages"]["ko2en"]["text"], p),
                    cQ=names_person(b["clean_question"], p),
                    aR=names_person(a["stages"]["reason"]["text"], p),
                    aT=bool(score.find_canonical(a["final"], [p])),
                    cR=names_person(b["stages"]["reason"]["text"], p),
                    cT=bool(score.find_canonical(b["final"], [p]))))
    return rows, meta


def aggregate(rows, reasoner, closed_only):
    sub = [r for r in rows if r["reasoner"] == reasoner
           and (r["closed_pair"] or not closed_only)]
    out = dict(n=len(sub))
    for arm in ("a", "c"):
        R = sum(r[f"{arm}R"] for r in sub)
        T = sum(r[f"{arm}T"] for r in sub)
        RT = sum(r[f"{arm}R"] and r[f"{arm}T"] for r in sub)
        out[arm] = dict(R=R, T=T, RT=RT, unmatched=sum((not r[f"{arm}R"]) and r[f"{arm}T"]
                                                       for r in sub),
                        p=(RT / R * 100) if R else None)
    return out, sub


def rescue_loss(sub, axis):
    gain = [r for r in sub if not r[f"a{axis}"] and r[f"c{axis}"]]
    lose = [r for r in sub if r[f"a{axis}"] and not r[f"c{axis}"]]
    return gain, lose


def show_layer(title, rows, closed_only):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    for rn in ARMS:
        agg, sub = aggregate(rows, rn, closed_only)
        print(f"\n  {rn}  n={agg['n']}")
        for arm, label in (("a", "actual "), ("c", "control")):
            d = agg[arm]
            p = f"{d['RT']}/{d['R']} = {d['p']:.2f}%" if d["p"] is not None else "n/a"
            print(f"    {label}  R={d['R']:2d}  T={d['T']:2d}  P(T|R)={p:18s} unmatched={d['unmatched']}")
        for axis in ("R", "T"):
            gain, lose = rescue_loss(sub, axis)
            print(f"    {axis}: rescue +{len(gain)}  loss -{len(lose)}")
            if gain:
                print(f"        +{[(g['case'], g['person']) for g in gain]}")
            if lose:
                print(f"        -{[(l['case'], l['person']) for l in lose]}")


def main():
    rows, meta = collect()
    print("=" * 78)
    print("R -> T transition, actual vs clean-English control (#43)")
    print("=" * 78)
    print(f"matcher : {MATCHER}")
    print("T       : score.find_canonical (committed scorer, unmodified)")

    print("\n[0] Q guard — not a transition axis")
    aq, cq = sum(r["aQ"] for r in rows), sum(r["cQ"] for r in rows)
    print(f"  control Q = {cq}/{len(rows)} (leakage guard)")
    print(f"  actual  Q = {aq}/{len(rows)} (KO->EN question text never named these people)")
    if aq or cq:
        raise SystemExit("Q is non-zero; the leakage guard or the case set changed")
    print("  both constant zero -> Q carries no signal and is not aggregated")

    print("\n[1] truncation census (detected from raw, not hardcoded)")
    any_fail = False
    for (rn, cid), m in sorted(meta.items()):
        if m["actual_unterminated"] or m["control_unterminated"]:
            any_fail = True
            who = []
            if m["actual_unterminated"]:
                who.append("actual")
            if m["control_unterminated"]:
                who.append("control")
            n = sum(1 for r in rows if r["reasoner"] == rn and r["case"] == cid)
            print(f"  {rn} {cid}: reason never closed on {'+'.join(who)} "
                  f"-> {n} people, kept in PRIMARY, dropped from SENSITIVITY")
    if not any_fail:
        print("  none")

    show_layer("[2] PRIMARY (as-run) — nothing excluded", rows, closed_only=False)
    show_layer("[3] SENSITIVITY (closed pairs) — union exclusion, either arm's failure "
               "drops the case from both", rows, closed_only=True)

    print("\n[4] per-case   (x = dropped from SENSITIVITY only)")
    print("    NEAR = retokenized length reached that arm's own limit; an indicator of")
    print("    budget exhaustion, not a recorded stop reason.")
    print(f"  {'reasoner':9s} {'case':7s} {'n':>2s} | {'aR':>3s} {'aT':>3s} {'rNEA':>4s} "
          f"{'eNEA':>4s} | {'cR':>3s} {'cT':>3s} {'rNEA':>4s} {'eNEA':>4s} | dT")
    for rn in ARMS:
        for c in korea_cases():
            sub = [r for r in rows if r["reasoner"] == rn and r["case"] == c["id"]]
            m = meta[(rn, c["id"])]
            aT = sum(r["aT"] for r in sub); cT = sum(r["cT"] for r in sub)
            print(f"  {rn:9s} {c['id']:7s} {len(sub):>2d} | "
                  f"{sum(r['aR'] for r in sub):>3d} {aT:>3d} "
                  f"{'NEAR' if m['a_r_near'] else '':>4s} {'NEAR' if m['a_e_near'] else '':>4s} | "
                  f"{sum(r['cR'] for r in sub):>3d} {cT:>3d} "
                  f"{'NEAR' if m['c_r_near'] else '':>4s} {'NEAR' if m['c_e_near'] else '':>4s} | "
                  f"{cT - aT:+d}{'' if m['closed_pair'] else ' x'}")

    print("\n[5] scorer totals over all 42 — unfiltered, reconciles with score.py")
    cases = non_holdout_cases()
    for rn, (act_cfg, ctl_cfg) in ARMS.items():
        for label, cfg, tag in (("actual", act_cfg, ACTUAL_TAG),
                                ("control", ctl_cfg, CONTROL_TAG)):
            s = score.score_config(cfg, cases, {}, tag)
            print(f"  {rn:8s} {label:8s} {s['recall_tp']}/"
                  f"{s['recall_tp'] + s['recall_fn']}  complete={s['complete']}")


if __name__ == "__main__":
    main()
