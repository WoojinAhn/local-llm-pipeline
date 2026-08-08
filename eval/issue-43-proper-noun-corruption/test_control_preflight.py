"""Preflight tests for control_clean_english.py — no real model is loaded.

Guards the properties that make the control valid: an exact call budget, no leaked
answers, one identical prompt per case across reasoners, and a resume path that accepts
a matching payload while refusing a changed one or an accidental overwrite.

The resume and smoke checks drive the real functions (init_payloads, main) rather than
re-implementing their logic, with R.load/R.call replaced by counting fakes.

    python eval/issue-43-proper-noun-corruption/test_control_preflight.py
"""
import contextlib, io, json, os, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import control_clean_english as C
import run as R

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def fresh_env():
    R.GEN.update(R.PROFILES[C.PROFILE])
    env = R.env_metadata("off", C.PROFILE, C.EXPERIMENT)
    env.update(experiment=C.EXPERIMENT, arm="control-clean-english",
               questions_sha256_16=C.file_sha16(f"{C.HERE}/control_questions.jsonl"))
    return env


class FakeModels:
    """Stands in for R.load/R.call and counts every call the run makes."""

    def __init__(self):
        self.calls = []

    def install(self, tmp_root):
        self._saved = (R.load, R.call, C.OUT_ROOT)
        R.load = lambda repo: (f"model:{repo}", f"tok:{repo}")
        R.call = self._call
        C.OUT_ROOT = tmp_root

    def restore(self):
        R.load, R.call, C.OUT_ROOT = self._saved

    def _call(self, model, tok, system, user, max_tokens, thinking=None):
        self.calls.append(dict(model=model, max_tokens=max_tokens))
        return dict(raw="스텁 출력 세종대왕", text="스텁 출력 세종대왕", secs=0.01,
                    tokens=10, unterminated_think=False, peak_gb=0.0)


@check("call budget is exactly 24 (12 reason + 12 en2ko)")
def _():
    rows = C.plan(C.load_cases(), C.load_questions())
    assert len(rows) == 12, len(rows)
    assert len(rows) * 2 == C.EXPECTED_CALLS


@check("6 Korea cases, 2 reasoners, no duplicates")
def _():
    rows = C.plan(C.load_cases(), C.load_questions())
    assert len({r["id"] for r in rows}) == 6
    assert len({r["reasoner"] for r in rows}) == 2
    assert len({(r["id"], r["reasoner"]) for r in rows}) == 12


@check("prompt is byte-identical across reasoners, distinct across cases")
def _():
    rows = C.plan(C.load_cases(), C.load_questions())
    by_case = {}
    for r in rows:
        by_case.setdefault(r["id"], set()).add(r["prompt_sha256_16"])
    assert all(len(v) == 1 for v in by_case.values()), by_case
    assert len({next(iter(v)) for v in by_case.values()}) == 6


@check("all 42 canonical people have a conventional alias listed")
def _():
    missing = C.check_alias_coverage(C.load_cases())
    assert missing == [], missing
    assert sum(len(v) for v in C.CONVENTIONAL_ALIASES.values()) >= 42


@check("clean questions leak no canonical person name")
def _():
    assert C.check_no_leakage(C.load_cases(), C.load_questions()) == []


@check("leakage check fires on a planted Hangul name")
def _():
    q = dict(C.load_questions())
    q["ko-01"] = "Explain the work of 정약용 in late Joseon."
    problems = C.check_no_leakage(C.load_cases(), q)
    assert any("Hangul" in w or "정약용" in w for _, w in problems), problems


@check("leakage check fires on a planted conventional alias (Yi Sun-sin)")
def _():
    q = dict(C.load_questions())
    q["ko-03"] = "Explain the naval battles led by Yi Sun-sin during the Imjin War."
    problems = C.check_no_leakage(C.load_cases(), q)
    assert any("Yi Sun-sin" in w for _, w in problems), problems


@check("leakage check fires on Kim Bu-sik, Park Ji-won and Choi Man-ri")
def _():
    for cid, planted, needle in (("ko-05", "Kim Bu-sik compiled it.", "Kim Bu-sik"),
                                 ("ko-01", "Park Ji-won wrote widely.", "Park Ji-won"),
                                 ("ko-02", "Choi Man-ri objected.", "Choi Man-ri")):
        q = dict(C.load_questions())
        q[cid] = planted
        problems = C.check_no_leakage(C.load_cases(), q)
        assert any(needle in w for _, w in problems), (cid, needle, problems)


@check("mechanical romanization alone would have missed these aliases")
def _():
    # Why the table exists. RR yields "gimbusik"/"bakjiwon"/"choemanri" while the forms
    # that would actually leak are "Kim Bu-sik"/"Park Ji-won"/"Choi Man-ri".
    # (이순신 is not a valid example: RR "isunsin" happens to sit inside "yisunsin".)
    for ko, alias in (("김부식", "Kim Bu-sik"),
                      ("박지원", "Park Ji-won"),
                      ("최만리", "Choi Man-ri")):
        assert C.romanize(ko).lower() not in C._flat(alias), (ko, alias)
        assert any(C._flat(a) == C._flat(alias) for a in C.CONVENTIONAL_ALIASES[ko]), ko


@check("production profile only: reason 4000 / translate 2000")
def _():
    p = R.PROFILES[C.PROFILE]
    assert p["reason_max_tokens"] == 4000 and p["translate_max_tokens"] == 2000, p


@check("source artifacts exist and KO2EN is reusable without a call")
def _():
    assert set(C.source_refs()) == set(C.REASONERS)
    for name in C.REASONERS:
        for c in C.load_cases():
            assert C.actual_ko2en(name, c["id"]).strip()


@check("config_detail carries script and questions hashes")
def _():
    cd = C.build_config_detail(C.REASONERS[0])
    assert cd["script_sha256_16"] and cd["questions_sha256_16"]
    assert cd["questions_sha256_16"] == C.file_sha16(f"{C.HERE}/control_questions.jsonl")


@check("config_detail pins the source artifact's content, not just its filename")
def _():
    for name in C.REASONERS:
        cd = C.build_config_detail(name)
        assert cd["source_sha256_16"] == C.file_sha16(
            f"{C.HERE}/outputs/{name}/{C.SOURCE_TAG}"), name
    # Distinct arms read distinct sources; one shared hash would defeat the check.
    assert len({C.build_config_detail(n)["source_sha256_16"] for n in C.REASONERS}) == \
        len(C.REASONERS)


@check("init_payloads: a regenerated source artifact is refused on --resume")
def _():
    env = fresh_env()
    with tempfile.TemporaryDirectory() as tmp:
        saved, C.OUT_ROOT = C.OUT_ROOT, tmp
        try:
            p = C.init_payloads(env, resume=False)
            for name in C.REASONERS:
                R.save(C.dest_path(name), p[name])
            # Same source_tag, different content — exactly what regenerating the actual
            # arm produces. Schema 2 pinned only the filename and let this through.
            real_sha = C.file_sha16
            C.file_sha16 = lambda p: ("cafebabecafebabe" if p.endswith(C.SOURCE_TAG)
                                      else real_sha(p))
            try:
                C.init_payloads(env, resume=True)
            except SystemExit as e:
                assert "config_detail" in str(e), e
                return
            finally:
                C.file_sha16 = real_sha
            raise AssertionError("resume accepted a different source artifact")
        finally:
            C.OUT_ROOT = saved


@check("init_payloads: fresh run creates empty per-reasoner payloads")
def _():
    env = fresh_env()
    with tempfile.TemporaryDirectory() as tmp:
        saved, C.OUT_ROOT = C.OUT_ROOT, tmp
        try:
            p = C.init_payloads(env, resume=False)
            assert set(p) == set(C.REASONERS)
            assert all(x["results"] == [] for x in p.values())
        finally:
            C.OUT_ROOT = saved


@check("init_payloads: existing output without --resume is refused")
def _():
    env = fresh_env()
    with tempfile.TemporaryDirectory() as tmp:
        saved, C.OUT_ROOT = C.OUT_ROOT, tmp
        try:
            p = C.init_payloads(env, resume=False)
            for name in C.REASONERS:
                R.save(C.dest_path(name), p[name])
            try:
                C.init_payloads(env, resume=False)
            except SystemExit as e:
                assert "refusing to overwrite" in str(e), e
                return
            raise AssertionError("existing output was silently overwritten")
        finally:
            C.OUT_ROOT = saved


@check("init_payloads: matching payload resumes")
def _():
    env = fresh_env()
    with tempfile.TemporaryDirectory() as tmp:
        saved, C.OUT_ROOT = C.OUT_ROOT, tmp
        try:
            p = C.init_payloads(env, resume=False)
            p[C.REASONERS[0]]["results"].append({"id": "ko-01"})
            for name in C.REASONERS:
                R.save(C.dest_path(name), p[name])
            again = C.init_payloads(env, resume=True)
            assert len(again[C.REASONERS[0]]["results"]) == 1
        finally:
            C.OUT_ROOT = saved


@check("init_payloads: changed profile is refused on resume")
def _():
    env = fresh_env()
    with tempfile.TemporaryDirectory() as tmp:
        saved, C.OUT_ROOT = C.OUT_ROOT, tmp
        try:
            p = C.init_payloads(env, resume=False)
            for name in C.REASONERS:
                p[name]["env"] = dict(env, profile="candidate")
                R.save(C.dest_path(name), p[name])
            try:
                C.init_payloads(env, resume=True)
            except SystemExit as e:
                assert "profile" in str(e), e
                return
            raise AssertionError("changed profile was accepted")
        finally:
            C.OUT_ROOT = saved


@check("init_payloads: changed question set is refused on resume")
def _():
    env = fresh_env()
    with tempfile.TemporaryDirectory() as tmp:
        saved, C.OUT_ROOT = C.OUT_ROOT, tmp
        try:
            p = C.init_payloads(env, resume=False)
            for name in C.REASONERS:
                cd = dict(p[name]["config_detail"], questions_sha256_16="deadbeefdeadbeef")
                p[name]["config_detail"] = cd
                R.save(C.dest_path(name), p[name])
            try:
                C.init_payloads(env, resume=True)
            except SystemExit as e:
                assert "config_detail" in str(e), e
                return
            raise AssertionError("changed question set was accepted")
        finally:
            C.OUT_ROOT = saved


@check("smoke: non-dry run makes exactly 24 calls and writes 6 rows per reasoner")
def _():
    fake = FakeModels()
    tmp = tempfile.mkdtemp()
    argv = sys.argv[:]
    try:
        fake.install(tmp)
        sys.argv = ["control_clean_english.py"]
        with contextlib.redirect_stdout(io.StringIO()):
            C.main()
        assert len(fake.calls) == 24, len(fake.calls)
        reason = [c for c in fake.calls if c["max_tokens"] == 4000]
        en2ko = [c for c in fake.calls if c["max_tokens"] == 2000]
        assert len(reason) == 12 and len(en2ko) == 12, (len(reason), len(en2ko))
        for name in C.REASONERS:
            d = json.load(open(C.dest_path(name)))
            assert len(d["results"]) == 6, (name, len(d["results"]))
            assert {r["id"] for r in d["results"]} == {f"ko-0{i}" for i in range(1, 7)}
            for r in d["results"]:
                assert r["final"] and "reason" in r["stages"] and "en2ko" in r["stages"]
                assert r["actual_ko2en"].strip()
        # resume must add no calls
        before = len(fake.calls)
        sys.argv = ["control_clean_english.py", "--resume"]
        with contextlib.redirect_stdout(io.StringIO()):
            C.main()
        assert len(fake.calls) == before, f"resume made {len(fake.calls) - before} extra calls"
    finally:
        sys.argv = argv
        fake.restore()
        shutil.rmtree(tmp, ignore_errors=True)


@check("smoke: output is shaped for score.py (config dir, id, final)")
def _():
    fake = FakeModels()
    tmp = tempfile.mkdtemp()
    argv = sys.argv[:]
    try:
        fake.install(tmp)
        sys.argv = ["control_clean_english.py"]
        with contextlib.redirect_stdout(io.StringIO()):
            C.main()
        sys.path.insert(0, C.HERE)
        import score
        cases = [c for c in (json.loads(l) for l in open(f"{C.HERE}/cases.jsonl") if l.strip())
                 if not c.get("holdout")]
        saved_here, score.HERE = score.HERE, tmp.rstrip("/").replace("/outputs", "")
        try:
            for name in C.REASONERS:
                d = json.load(open(C.dest_path(name)))
                # the fields score_config depends on
                for r in d["results"]:
                    assert isinstance(r["id"], str) and isinstance(r["final"], str)
                assert d["config"].startswith("control-clean-")
                assert d["env"]["profile"] == "production"
            # the stub text contains 세종대왕 -> the corrected matcher must see 세종
            hit = score.find_canonical("스텁 출력 세종대왕", ["세종"])
            assert hit == {"세종"}, hit
        finally:
            score.HERE = saved_here
    finally:
        sys.argv = argv
        fake.restore()
        shutil.rmtree(tmp, ignore_errors=True)


def _run_main_expecting_exit(needle, **patches):
    """Drive the real main() with a patched module attribute and expect a refusal."""
    fake = FakeModels()
    tmp = tempfile.mkdtemp()
    argv, saved = sys.argv[:], {k: getattr(C, k) for k in patches}
    try:
        fake.install(tmp)
        for k, v in patches.items():
            setattr(C, k, v)
        sys.argv = ["control_clean_english.py"]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                C.main()
        except SystemExit as e:
            assert needle in str(e), f"expected {needle!r} in {e}"
            assert fake.calls == [], f"guard fired after {len(fake.calls)} calls"
            return
        raise AssertionError(f"main() did not refuse; expected {needle!r}")
    finally:
        sys.argv = argv
        for k, v in saved.items():
            setattr(C, k, v)
        fake.restore()
        shutil.rmtree(tmp, ignore_errors=True)


@check("main refuses before loading when an alias is missing")
def _():
    thinned = {k: v for k, v in C.CONVENTIONAL_ALIASES.items() if k != "이순신"}
    _run_main_expecting_exit("CONVENTIONAL_ALIASES is incomplete",
                             CONVENTIONAL_ALIASES=thinned)


@check("main refuses before loading when a question is missing")
def _():
    q = C.load_questions()
    q.pop("ko-03")
    _run_main_expecting_exit("missing questions", load_questions=lambda: q)


@check("main refuses before loading on an extra question id")
def _():
    q = C.load_questions()
    q["ko-99"] = "An extra question that matches no case."
    _run_main_expecting_exit("unexpected ids", load_questions=lambda: q)


@check("smoke: score.score_config scores both configs as complete over 6 Korea rows")
def _():
    fake = FakeModels()
    tmp = tempfile.mkdtemp()
    argv = sys.argv[:]
    try:
        fake.install(f"{tmp}/outputs")
        sys.argv = ["control_clean_english.py"]
        with contextlib.redirect_stdout(io.StringIO()):
            C.main()
        sys.path.insert(0, C.HERE)
        import score
        korea = C.load_cases()
        saved_here, score.HERE = score.HERE, tmp
        try:
            for name in C.REASONERS:
                cfg = f"control-clean-{name.replace('three-stage-', '')}"
                s = score.score_config(cfg, korea, {}, C.SCORE_TAG)
                assert s["complete"] is True, (cfg, s["missing_cases"], s["unexpected_cases"])
                assert len(s["per_case"]) == 6, (cfg, len(s["per_case"]))
                assert s["recall_tp"] + s["recall_fn"] == 42, s
                # the stub answer contains 세종대왕; 세종 is canonical only in ko-02
                assert s["recall_tp"] == 1, s["recall_tp"]
                assert [p["id"] for p in s["per_case"] if p["hit"]] == ["ko-02"], s["per_case"]
        finally:
            score.HERE = saved_here
    finally:
        sys.argv = argv
        fake.restore()
        shutil.rmtree(tmp, ignore_errors=True)


@check("smoke: scoring the full non-holdout set reports exactly the 3 control ids missing")
def _():
    fake = FakeModels()
    tmp = tempfile.mkdtemp()
    argv = sys.argv[:]
    try:
        fake.install(f"{tmp}/outputs")
        sys.argv = ["control_clean_english.py"]
        with contextlib.redirect_stdout(io.StringIO()):
            C.main()
        sys.path.insert(0, C.HERE)
        import score
        all_cases = [c for c in (json.loads(l) for l in
                                 open(f"{C.HERE}/cases.jsonl") if l.strip())
                     if not c.get("holdout")]
        saved_here, score.HERE = score.HERE, tmp
        try:
            cfg = f"control-clean-{C.REASONERS[0].replace('three-stage-', '')}"
            s = score.score_config(cfg, all_cases, {}, C.SCORE_TAG)
            assert s["complete"] is False
            assert s["missing_cases"] == ["ctl-01", "ctl-02", "ctl-03"], s["missing_cases"]
            assert s["unexpected_cases"] == []
            # the banner is about coverage only; the recall denominator is untouched
            assert s["recall_tp"] + s["recall_fn"] == 42, s
        finally:
            score.HERE = saved_here
    finally:
        sys.argv = argv
        fake.restore()
        shutil.rmtree(tmp, ignore_errors=True)


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
