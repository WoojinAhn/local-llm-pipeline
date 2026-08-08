"""Regression tests for analyze_transitions.py and the alias it depends on.

Pins the six aggregates the analysis is quoted on, the Q guard, the detected exclusion,
and — most importantly — that the analyzer never touches a model.

    python eval/issue-43-proper-noun-corruption/test_transitions.py
"""
import json, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_transitions as A
import canonical_aliases as CA

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("박영효 carries the observed 'Park Yung-hyo' form")
def _():
    assert "Park Yung-hyo" in CA.CONVENTIONAL_ALIASES["박영효"]
    assert A.names_person("reform officials including Park Yung-hyo met", "박영효")


@check("연산군 carries the observed 'King Yeonsan' form")
def _():
    assert "King Yeonsan" in CA.CONVENTIONAL_ALIASES["연산군"]
    assert A.names_person("purges ordered under King Yeonsan", "연산군")


@check("alias matching does not cross word boundaries (short-alias negatives)")
def _():
    # "I Ik" is a listed 이익 alias; a letters-only substring matcher matched it inside
    # unrelated words, so the matcher is word-bounded.
    assert "I Ik" in CA.CONVENTIONAL_ALIASES["이익"]
    for text in ("Silhak thinkers", "the Yi Ikjae family", "Iikonomou wrote",
                 "Yi Ikhwan is fabricated"):
        assert not A.names_person(text, "이익"), text
    for text in ("Yi Ik argued", "by Yi Ik.", "Yi Ik's essays"):
        assert A.names_person(text, "이익"), text


@check("closure detection matches run.py's splitter source, without importing it")
def _():
    run_src = open(f"{A.HERE}/run.py").read()
    m = re.search(r"THINK_SPLIT = re\.compile\(r\"(.+?)\"\)", run_src)
    assert m, "could not locate THINK_SPLIT in run.py"
    assert A.CLOSURE.pattern == m.group(1), (A.CLOSURE.pattern, m.group(1))
    # the bare Harmony prefix must not count as closed
    assert A.unterminated({"raw": "reasoning <|channel|>final"})
    assert not A.unterminated({"raw": "x <|channel|>final<|message|> answer"})
    assert not A.unterminated({"raw": "think </think> answer"})
    assert not A.unterminated({"raw": "analysis assistantfinal answer"})


@check("a reasoner that never closes at all is not read as truncated")
def _():
    # Qwen3.6 opens <think> from the chat template, so the opened-block precondition
    # run.py uses is absent from every completion — including the genuinely truncated
    # ones, which must stay flagged. The separating signal is the arm's own behaviour.
    closing = [{"stages": {"reason": {"raw": "a </think> b"}}},
               {"stages": {"reason": {"raw": "mid-sentence and cut"}}}]
    never = [{"stages": {"reason": {"raw": "plain answer"}}},
             {"stages": {"reason": {"raw": "another plain answer"}}}]
    assert A.emits_closure(closing) and not A.emits_closure(never)
    cut = {"raw": "mid-sentence and cut"}
    assert A.unterminated(cut, A.emits_closure(closing)), "truncation must stay flagged"
    assert not A.unterminated(cut, A.emits_closure(never)), \
        "a non-reasoning model's every case would be dropped from SENSITIVITY"


@check("the two real truncated Qwen3.6 stages have no opener, so run.py's test cannot be reused")
def _():
    rows, meta = A.collect()
    for cfg, tag, cid in (("three-stage-qwen36", A.ACTUAL_TAG, "ko-03"),
                          ("control-clean-qwen36", A.CONTROL_TAG, "ko-04")):
        raw = A.load_arm(cfg, tag)[0][cid]["stages"]["reason"]["raw"]
        assert "<think>" not in raw and "<|channel|>analysis" not in raw, cid
        assert not A.CLOSURE.search(raw), cid
    # Mirroring run.py's precondition literally would unflag both and erase SENSITIVITY.
    assert meta[("qwen36", "ko-03")]["actual_unterminated"] is True
    assert meta[("qwen36", "ko-04")]["control_unterminated"] is True


@check("near_limit is retokenized-length semantics, not a recorded stop reason")
def _():
    assert A.near_limit({"tokens": 4000}, 4000)
    assert A.near_limit({"tokens": 4001}, 4000)
    assert not A.near_limit({"tokens": 3999}, 4000)
    assert "near_limit" in A.__dict__ and not hasattr(A, "capped")
    doc = A.near_limit.__doc__ or ""
    assert "not the generation counter" in doc or "generation counter" in doc


@check("provenance mismatch is fail-loud")
def _():
    aenv = {"generation_sha256_16": "a" * 16, "prompts_sha256_16": "b" * 16}
    good_sha = __import__("hashlib").sha256(
        open(f"{A.HERE}/outputs/three-stage-gpt-oss/run-{A.ACTUAL_TAG}.json", "rb")
        .read()).hexdigest()[:16]
    ok = dict(aenv, source_artifacts={"three-stage-gpt-oss": {"sha256_16": good_sha}})
    A.check_provenance("gpt-oss", "three-stage-gpt-oss", aenv, ok)  # must not raise
    for broken, needle in (
            (dict(ok, generation_sha256_16="z" * 16), "generation_sha256_16"),
            (dict(ok, prompts_sha256_16="z" * 16), "prompts_sha256_16"),
            (dict(aenv, source_artifacts={}), "no source_artifacts"),
            (dict(aenv, source_artifacts={"three-stage-gpt-oss": {"sha256_16": "0" * 16}}),
             "different actual artifact")):
        try:
            A.check_provenance("gpt-oss", "three-stage-gpt-oss", aenv, broken)
        except SystemExit as e:
            assert needle in str(e), (needle, str(e))
            continue
        raise AssertionError(f"mismatch not caught: {needle}")


@check("alias matching is insensitive to the U+2011 hyphen gpt-oss emits")
def _():
    assert A.names_person("Admiral Yi Sun‑sin led the fleet", "이순신")
    assert A.names_person("Admiral Yi Sun-sin led the fleet", "이순신")


@check("Q is zero on both arms and is not a transition axis")
def _():
    rows, _ = A.collect()
    assert sum(r["aQ"] for r in rows) == 0
    assert sum(r["cQ"] for r in rows) == 0


@check("truncation is detected from the artifact, not hardcoded")
def _():
    _, meta = A.collect()
    assert meta[("qwen36", "ko-03")]["actual_unterminated"] is True
    assert meta[("qwen36", "ko-03")]["control_unterminated"] is False
    assert meta[("qwen36", "ko-04")]["control_unterminated"] is True
    assert meta[("qwen36", "ko-04")]["actual_unterminated"] is False


@check("sensitivity exclusion is the UNION of both arms, never one arm alone")
def _():
    _, meta = A.collect()
    dropped = {k for k, v in meta.items() if not v["closed_pair"]}
    # ko-03 fails on actual, ko-04 on control: keying on either arm alone would be
    # outcome-dependent and would flatter whichever arm was spared.
    assert dropped == {("qwen36", "ko-03"), ("qwen36", "ko-04")}, dropped
    for k, v in meta.items():
        assert v["closed_pair"] == (not (v["actual_unterminated"]
                                         or v["control_unterminated"])), k


@check("PRIMARY (as-run, nothing excluded) matches the verified figures")
def _():
    rows, _ = A.collect()
    want = {  # reasoner: (n, aR, aT, aP, cR, cT, cP)
        "gpt-oss": (42, 19, 16, 84.21, 23, 19, 82.61),
        "qwen36": (42, 19, 12, 63.16, 19, 15, 78.95),
    }
    for rn, (n, aR, aT, aP, cR, cT, cP) in want.items():
        agg, _ = A.aggregate(rows, rn, closed_only=False)
        assert agg["n"] == n, (rn, agg["n"])
        assert (agg["a"]["R"], agg["a"]["T"]) == (aR, aT), (rn, agg["a"])
        assert (agg["c"]["R"], agg["c"]["T"]) == (cR, cT), (rn, agg["c"])
        assert round(agg["a"]["p"], 2) == aP, (rn, agg["a"]["p"])
        assert round(agg["c"]["p"], 2) == cP, (rn, agg["c"]["p"])


@check("SENSITIVITY (closed pairs) matches the verified figures")
def _():
    rows, _ = A.collect()
    want = {
        "gpt-oss": (42, 19, 16, 84.21, 23, 19, 82.61),   # no failures, unchanged
        "qwen36": (28, 14, 10, 71.43, 15, 12, 80.00),
    }
    for rn, (n, aR, aT, aP, cR, cT, cP) in want.items():
        agg, _ = A.aggregate(rows, rn, closed_only=True)
        assert agg["n"] == n, (rn, agg["n"])
        assert (agg["a"]["R"], agg["a"]["T"]) == (aR, aT), (rn, agg["a"])
        assert (agg["c"]["R"], agg["c"]["T"]) == (cR, cT), (rn, agg["c"])
        assert round(agg["a"]["p"], 2) == aP, (rn, agg["a"]["p"])
        assert round(agg["c"]["p"], 2) == cP, (rn, agg["c"]["p"])


@check("rescue/loss is reported separately per layer")
def _():
    rows, _ = A.collect()
    prim = {rn: {ax: tuple(len(x) for x in A.rescue_loss(A.aggregate(rows, rn, False)[1], ax))
                 for ax in ("R", "T")} for rn in A.ARMS}
    sens = {rn: {ax: tuple(len(x) for x in A.rescue_loss(A.aggregate(rows, rn, True)[1], ax))
                 for ax in ("R", "T")} for rn in A.ARMS}
    assert prim["qwen36"]["R"] == (5, 5) and prim["qwen36"]["T"] == (6, 3), prim["qwen36"]
    assert sens["qwen36"]["R"] == (4, 3) and sens["qwen36"]["T"] == (4, 2), sens["qwen36"]
    # gpt-oss has no truncation, so the two layers must agree exactly
    assert prim["gpt-oss"] == sens["gpt-oss"], (prim["gpt-oss"], sens["gpt-oss"])


@check("no unmatched R=0 / T=1 cell remains on either arm or layer")
def _():
    rows, _ = A.collect()
    for closed_only in (False, True):
        for rn in A.ARMS:
            agg, _ = A.aggregate(rows, rn, closed_only)
            assert agg["a"]["unmatched"] == 0, (rn, closed_only, "actual")
            assert agg["c"]["unmatched"] == 0, (rn, closed_only, "control")


@check("scorer totals reconcile: 16/42, 12/42, 19/42, 15/42")
def _():
    import score
    cases = A.non_holdout_cases()
    want = {("gpt-oss", "actual"): 16, ("gpt-oss", "control"): 19,
            ("qwen36", "actual"): 12, ("qwen36", "control"): 15}
    for rn, (act, ctl) in A.ARMS.items():
        for label, cfg, tag in (("actual", act, A.ACTUAL_TAG),
                                ("control", ctl, A.CONTROL_TAG)):
            s = score.score_config(cfg, cases, {}, tag)
            assert s["recall_tp"] == want[(rn, label)], (rn, label, s["recall_tp"])
            assert s["recall_tp"] + s["recall_fn"] == 42, s


@check("analyzer imports nothing model-related, transitively")
def _():
    seen, stack = set(), ["analyze_transitions"]
    banned = {"mlx", "mlx_lm", "mlx_vlm", "run", "control_clean_english", "isolate"}
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = f"{A.HERE}/{mod}.py"
        if not os.path.exists(path):
            continue
        for m in re.finditer(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)",
                             open(path).read(), re.M):
            name = m.group(1).split(".")[0]
            assert name not in banned, f"{mod}.py imports {name}"
            stack.append(name)
    assert "canonical_aliases" in seen and "score" in seen, seen


@check("analyzer runs in a subprocess with MLX blocked")
def _():
    blocker = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name.split('.')[0] in ('mlx', 'mlx_lm', 'mlx_vlm'):\n"
        "            raise ImportError('MLX blocked for this test: ' + name)\n"
        "        return None\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        return self.find_module(name, path)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.argv = ['analyze_transitions.py']\n"
        f"exec(open({A.HERE + '/analyze_transitions.py'!r}).read(), "
        "{'__name__': '__main__', '__file__': " f"{A.HERE + '/analyze_transitions.py'!r}" "})\n")
    r = subprocess.run([sys.executable, "-c", blocker], capture_output=True, text=True,
                       cwd=A.HERE)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "PRIMARY" in r.stdout and "SENSITIVITY" in r.stdout
    assert "16/42" in r.stdout and "12/42" in r.stdout


@check("alias separators: only in-name whitespace and a single hyphen")
def _():
    # positives that must keep working — both observed in the real artifacts
    assert A.names_person("scholar-official Kim\u202fBu\u2011sik compiled it", "김부식")
    assert A.names_person("Admiral Yi Sun\u2011sin", "이순신")
    assert A.names_person("Park Jiwon wrote", "박지원")          # joined form
    # negatives: separators that are not part of a name
    for text, person in (("Toegye Yi. Ik was", "이익"),
                         ("Yi -- Ik", "이익"),
                         ("Nam, Gon", "남곤"),
                         ("| **King** | Jungjong |", "중종"),
                         ("Yi\n\nIk", "이익"),
                         ("Sim (Jeong)", "심정")):
        assert not CA.contains_alias(text, _first_alias(person)), (text, person)


def _first_alias(person):
    """The multi-word alias whose separator is under test."""
    for a in CA.CONVENTIONAL_ALIASES[person]:
        if len(a.split()) > 1:
            return a
    raise AssertionError(person)


@check("a name preceded by word+hyphen is not sliced out of a longer token")
def _():
    # The left guard was written as hyphen-then-word, which never fires: for
    # "Hwan-Yi Ik" the two characters before the match are "n-", not "-Y".
    for text, person in (("Hwan-Yi Ik", "이익"),
                         ("Choe-Hwang Jin", "황진"),
                         ("Jo-Nam Gon", "남곤")):
        assert not A.names_person(text, person), (text, person)
    # a hyphen that is not preceded by a word character is not an extension
    for text, person in (("-Yi Ik", "이익"), ("(Yi Ik)", "이익"), ("Yi Ik", "이익")):
        assert A.names_person(text, person), (text, person)


@check("fabricated suffix extensions are rejected, possessives are kept")
def _():
    # 이익환 is the one-syllable fabrication shape score.py exists to catch; the hyphenated
    # romanization must not be read as 이익.
    for text, person in (("Yi Ik-hwan was fabricated", "이익"),
                         ("Yi Ikhwan is fabricated", "이익"),
                         ("Hwang Jin-i the poet", "황진"),
                         ("Nam Gon-hui", "남곤"),
                         ("Sim Jeong-ho", "심정")):
        assert not A.names_person(text, person), (text, person)
    for text, person in (("Yi Ik's essays", "이익"),
                         ("Yi Ik\u2019s essays", "이익"),
                         ("by Yi Ik.", "이익"),
                         ("Hwang Jin fought", "황진")):
        assert A.names_person(text, person), (text, person)


@check("per-arm limits: control limits are never reused for the actual arm")
def _():
    """Non-tautological: the two arms are given deliberately different limits.

    Both real arms ran the production profile, so asserting 4000/2000 on each cannot
    distinguish reading each arm's own env from reusing one arm's for both.
    """
    real = A.load_arm

    def doctored(config, tag):
        rows, env = real(config, tag)
        env = json.loads(json.dumps(env))
        if tag == A.CONTROL_TAG:                       # control gets distinct limits
            env["generation"]["reason_max_tokens"] = 8000
            env["generation"]["translate_max_tokens"] = 4000
        return rows, env

    A.load_arm = doctored
    try:
        _, meta = A.collect()
    finally:
        A.load_arm = real
    for (rn, cid), m in meta.items():
        assert m["limits"]["actual"] == {"reason": 4000, "en2ko": 2000}, (rn, cid, m["limits"])
        assert m["limits"]["control"] == {"reason": 8000, "en2ko": 4000}, (rn, cid, m["limits"])
    # The flags must follow each arm's own limit. qwen36 ko-04 reason is 3998, which is
    # below 4000 and below 8000, so it cannot tell the two apart — a boundary value is
    # required. gpt-oss control ko-03 is reason=4000 / en2ko=2000: True under the
    # production limits, False under the injected control limits. If c_r_near were
    # computed from the actual arm's 4000/2000 it would read True here.
    assert meta[("qwen36", "ko-04")]["c_r_near"] is False
    assert meta[("gpt-oss", "ko-03")]["c_r_near"] is False, "reason flag used the wrong limit"
    assert meta[("gpt-oss", "ko-03")]["c_e_near"] is False, "en2ko flag used the wrong limit"
    # the same case under the real production limits must be True on both, so the
    # assertions above are discriminating rather than vacuously false
    real_rows, real_meta = A.collect()
    assert real_meta[("gpt-oss", "ko-03")]["c_r_near"] is True
    assert real_meta[("gpt-oss", "ko-03")]["c_e_near"] is True

    # The actual arm needs its own boundary cases, one per stage. An earlier version
    # compared gpt-oss actual ko-03 reason, which is 3207 and therefore False under both
    # 4000 and 8000 — it could not catch the mirror regression where a_*_near is computed
    # from the control arm's limits. These two straddle the boundary:
    #   qwen36 actual ko-06 reason = 4001  -> True at 4000, False at 8000
    #   gpt-oss actual ko-03 en2ko = 2000  -> True at 2000, False at 4000
    assert meta[("qwen36", "ko-06")]["a_r_near"] is True, "a_r_near used the control limit"
    assert meta[("gpt-oss", "ko-03")]["a_e_near"] is True, "a_e_near used the control limit"
    assert real_meta[("qwen36", "ko-06")]["a_r_near"] is True
    assert real_meta[("gpt-oss", "ko-03")]["a_e_near"] is True


@check("no field or label claims a recorded stop reason")
def _():
    # This file is excluded on purpose: it holds these strings as the denylist itself,
    # which is not a claim about the data. The implementation files are what must be clean.
    banned = ("stopped_at_cap", "[CAP]", "stop_reason")
    for name in ("analyze_transitions.py", "control_clean_english.py",
                 "canonical_aliases.py"):
        src = open(f"{A.HERE}/{name}").read()
        for token in banned:
            assert token not in src, f"{name} still contains {token!r}"


def main():
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
