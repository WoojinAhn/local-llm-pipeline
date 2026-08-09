"""Run pipeline configurations over the #43 case set, capturing every stage.

Unlike the throwaway harness this replaces, per-stage intermediates are recorded, so
the corruption can be localized to KO->EN translation, the reasoner, or EN->KO
back-translation rather than only observed end-to-end.

Two independent axes. Neither name claims full production parity — see below.

--search  off      (default) web search disabled. The `SEARCH:` judgment is recorded but
                   not acted on. Deterministic; corruption is attributable to the
                   pipeline rather than to whatever the web returned that day.
          parity   the *search behaviour* mirrors `mlx-pipeline.py:pipeline()`: Brave+
                   Tavily run when the judge says yes, Korean snippets are translated to
                   English, and `build_search_context_prompt()` wraps the reasoner input.
                   Varies by run, network and API keys.

--profile production   translate 2000 / reason 4000 — the limits `mlx-pipeline.py`
                       actually uses today.
          candidate    translate 4000 / reason 8000 — headroom for Qwen3.6, which
                       exhausts 4000 mid-reasoning on analytical prompts (#40).

`--search parity` matches production's search *behaviour only*. It is NOT full
production parity: the generation profile is a separate axis, and `--profile candidate`
deliberately diverges. Only `--search parity --profile production` reproduces
`mlx-pipeline.py` as shipped.

Both settings pass the full reasoner output to back-translation, matching production. An
earlier version truncated it at 6000 characters, which production does not do.

Usage:
    python run.py three-stage-gpt-oss
    python run.py three-stage-qwen36 --profile candidate
    python run.py single-exaone --search parity --profile production --resume
"""
import hashlib, importlib.metadata, json, os, platform, re, subprocess, sys, time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import mlx.core as mx
from mlx_lm import load, generate
import prompts

TRANSLATOR = "mlx-community/Qwen3-14B-4bit"

CONFIGS = {
    "three-stage-gpt-oss": dict(kind="3stage", reasoner="mlx-community/gpt-oss-120b-4bit"),
    "three-stage-qwen36": dict(kind="3stage", reasoner="mlx-community/Qwen3.6-35B-A3B-4bit",
                               reasoner_thinking=True),
    "single-exaone": dict(kind="single", model="mlx-community/EXAONE-4.0-32B-4bit"),
}

PROFILES = {
    # Mirrors mlx-pipeline.py: _stream_qwen(max_tokens=2000), _stream_reasoner(max_tokens=4000)
    "production": dict(translate_max_tokens=2000, reason_max_tokens=4000,
                       single_max_tokens=4000),
    # Headroom for Qwen3.6, which does not close </think> within 4000 on analytical prompts
    "candidate": dict(translate_max_tokens=4000, reason_max_tokens=8000,
                      single_max_tokens=8000),
}
GEN = {}  # bound in main() from --profile

THINK_SPLIT = re.compile(r"</think>|<\|channel\|>final<\|message\|>|assistantfinal")


def strip_think(text):
    return THINK_SPLIT.split(text)[-1]


def has_unterminated_think(text):
    """Qwen3.6 can exhaust the budget mid-reasoning; the block then never closes."""
    return ("<think>" in text or "<|channel|>analysis" in text) and not THINK_SPLIT.search(text)


# Provenance is split so a scorer-only edit does not invalidate raw generations.
# Bundling them meant touching score.py made every prior run un-resumable.
GENERATION_FILES = ("run.py", "cases.jsonl")
SCORING_FILES = ("score.py",)

# Paths a run is expected to write. A sequential sweep dirties the tree with its own
# earlier results, which made every config after the first record repo_dirty=true even
# though no source had changed. Only these are excluded; anything else still counts.
GENERATED_RE = re.compile(r"^eval/[^/]+/outputs/")


def _hash(files):
    h = hashlib.sha256()
    for f in sorted(files):
        p = f"{HERE}/{f}"
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def env_metadata(search, profile, name):
    # Distribution metadata, not the module attribute: `mlx` exposes no top-level
    # `__version__` (it lives under `mlx.core`), so an attribute lookup recorded the core
    # as null in every artifact ever written. These three decide the numerics, so a missing
    # one is a refusal rather than a null — an unattributable record is worse than no record.
    def ver(dist):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError as e:
            raise RuntimeError(
                f"cannot record a version for {dist!r}, which the harness depends on. "
                f"Runs are only comparable within a fixed stack; refusing to write an "
                f"artifact that cannot be attributed to one."
            ) from e

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", ROOT, *args], text=True).strip()
        except Exception:
            return None

    # NOT via git(): that helper strips the whole output, eating the leading status
    # column of a modified-tracked line (" M path"), which shifted every such path by
    # one character. Untracked lines ("?? path") have no leading space, so the defect
    # only ever showed on the source changes that matter most.
    def git_raw(*args):
        try:
            return subprocess.check_output(["git", "-C", ROOT, *args], text=True)
        except Exception:
            return ""

    # The single-model path sends no system prompt at all (`call(m, t, None, ...)`
    # below), so the locale-dependent reasoner prompt cannot reach its output.
    # Recording a hash there would assert a prompt that was never in play, and gate
    # its --resume on a value that means nothing for it. An unrecognised name is the
    # clean-English control passing its experiment name; that arm does use the
    # prompt, so anything not known to be single-model records.
    uses_reasoner_prompt = CONFIGS.get(name, {}).get("kind") != "single"

    dirty = git_raw("status", "--porcelain")
    # Porcelain v1: 2 status columns + a space, then the path. Renames read "old -> new";
    # take the destination, which is the path that actually differs.
    all_paths = [l[3:].split(" -> ")[-1].strip('"')
                 for l in dirty.splitlines() if len(l) > 3]
    source_paths = [p for p in all_paths if not GENERATED_RE.match(p)]
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        search=search,
        profile=profile,
        settings_equivalent_to_production=(search == "parity" and profile == "production"),
        # Settings equivalence is NOT pipeline equivalence. Two gaps remain even at
        # --search parity --profile production:
        #   config — mlx-pipeline.py ships Qwen3-14B + gpt-oss-120b in 3 stages. Any
        #            other config here is a candidate, not production, whatever the flags.
        #   state  — production accumulates _reasoner_history and reuses prompt_cache
        #            across turns. This harness runs each case cold and independent.
        config_matches_shipped=(name == "three-stage-gpt-oss"),
        reproduces_conversation_state=False,
        repo_commit=git("rev-parse", "HEAD"),
        # `source_dirty` is the one that bears on reproducibility. `repo_dirty` keeps its
        # original whole-tree meaning so records written before this split stay readable.
        source_dirty=bool(source_paths),
        source_dirty_paths=source_paths,
        repo_dirty=bool(all_paths),
        dirty_paths=all_paths,
        generation_sha256_16=_hash(GENERATION_FILES),
        scoring_sha256_16=_hash(SCORING_FILES),
        prompts_sha256_16=hashlib.sha256(
            open(f"{ROOT}/prompts.py", "rb").read()).hexdigest()[:16],
        # The file hash above cannot attribute the reasoner prompt any more: #67 made
        # reasoner_system() resolve the user's locale at call time, so two machines
        # running identical source send different system prompts. Record the resolved
        # value's hash, and the locale that produced it, or the record is unattributable.
        # Null for configs that send no system prompt — see uses_reasoner_prompt above.
        reasoner_system_sha256_16=(
            hashlib.sha256(prompts.reasoner_system().encode()).hexdigest()[:16]
            if uses_reasoner_prompt else None),
        reasoner_system_location=(
            prompts.current_location_context() if uses_reasoner_prompt else None),
        python=sys.version.split()[0],
        platform=platform.platform(), machine=platform.machine(),
        mlx=ver("mlx"), mlx_lm=ver("mlx-lm"), transformers=ver("transformers"),
        generation=dict(sampler="greedy (temp=0 default)", profile=profile, **GEN),
    )


def call(model, tok, system, user, max_tokens, thinking=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    kw = {} if thinking is None else {"enable_thinking": thinking}
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
    mx.reset_peak_memory()
    t0 = time.time()
    raw = generate(model, tok, p, max_tokens=max_tokens, verbose=False)
    return dict(raw=raw, text=strip_think(raw).strip(), secs=round(time.time() - t0, 2),
                tokens=len(tok.encode(raw)),
                unterminated_think=has_unterminated_think(raw),
                peak_gb=round(mx.get_peak_memory() / 1e9, 2))


def do_search(tm, tt, ko_query, en_query):
    """Mirror the search half of mlx-pipeline.py:pipeline()."""
    from web_search import search_both, format_search_context
    ko_results, en_results = search_both(ko_query, en_query)
    translated = None
    if ko_results:
        snippets = "\n".join(f"{i}. [{r['title']}] {r['snippet']}"
                             for i, r in enumerate(ko_results, 1))
        s = call(tm, tt, prompts.TRANSLATE_KO_TO_EN, snippets,
                 GEN["translate_max_tokens"], thinking=False)
        translated = re.sub(r"SEARCH:\s*(yes|no)", "", s["text"]).strip()
        ko_results = [{"title": "Korean sources (translated)", "url": "",
                       "snippet": translated}]
    return format_search_context(ko_results, en_results), translated, \
        len(ko_results) + len(en_results)


def run_case_3stage(c, tm, tt, rm, rt, think, search):
    t0 = time.time()
    s1 = call(tm, tt, prompts.TRANSLATE_KO_TO_EN, c["query"], GEN["translate_max_tokens"],
              thinking=False)
    en_q = re.sub(r"SEARCH:\s*(yes|no)", "", s1["text"]).strip()
    search_flag = bool(re.search(r"SEARCH:\s*yes", s1["text"], re.I))

    searched = None
    reason_input = en_q
    if search == "parity" and search_flag:
        ctx, translated, hits = do_search(tm, tt, c["query"], en_q)
        searched = dict(hits=hits, translated_ko_snippets=translated, context=ctx)
        reason_input = prompts.build_search_context_prompt(ctx, en_q)

    s2 = call(rm, rt, prompts.reasoner_system(), reason_input, GEN["reason_max_tokens"],
              thinking=think)
    # Production passes the full analysis; do not truncate.
    s3 = call(tm, tt, prompts.TRANSLATE_EN_TO_KO, s2["text"],
              GEN["translate_max_tokens"], thinking=False)
    return dict(id=c["id"], query=c["query"], total_secs=round(time.time() - t0, 2),
                search_judged=search_flag, search_performed=searched is not None,
                search=searched, stages=dict(ko2en=s1, reason=s2, en2ko=s3),
                final=s3["text"])


def save(dest, payload):
    """Atomic write — a long run must not be lost or half-written on interrupt."""
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, dest)


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# Scoring provenance is deliberately excluded — it does not affect raw generations.
RESUME_KEYS = ("search", "profile", "generation_sha256_16", "prompts_sha256_16",
               "reasoner_system_sha256_16")


def check_resumable(payload, env, cfg):
    """Refuse to append results produced by different code or settings.

    Without this, --resume silently mixes runs from different harness revisions or
    generation profiles into one file, and the resulting numbers are unattributable.
    """
    prev = payload.get("env", {})
    diffs = [(k, prev.get(k), env[k]) for k in RESUME_KEYS if prev.get(k) != env[k]]
    if payload.get("config_detail") != cfg:
        diffs.append(("config_detail", payload.get("config_detail"), cfg))
    if diffs:
        lines = "\n".join(f"    {k}: existing={o!r} current={n!r}" for k, o, n in diffs)
        raise SystemExit(
            "refusing to --resume: the existing run was produced by different "
            f"code or settings.\n{lines}\n"
            "    Start a fresh run, or delete the existing file to overwrite.")


def main():
    name = sys.argv[1]
    search = arg("--search", "off")
    profile = arg("--profile", "candidate")
    assert search in ("off", "parity"), search
    assert profile in PROFILES, profile
    GEN.update(PROFILES[profile])

    resume = "--resume" in sys.argv
    cfg = CONFIGS[name]

    cases = [json.loads(l) for l in open(f"{HERE}/cases.jsonl") if l.strip()]
    if "--holdout" not in sys.argv:
        cases = [c for c in cases if not c.get("holdout")]

    env = env_metadata(search, profile, name)
    dest = f"{HERE}/outputs/{name}/run-search-{search}-{profile}.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    payload = dict(config=name, config_detail=cfg, env=env, results=[])
    if resume and os.path.exists(dest):
        existing = json.load(open(dest))
        check_resumable(existing, env, cfg)
        payload = existing
        done = {r["id"] for r in payload["results"]}
        cases = [c for c in cases if c["id"] not in done]
        print(f"resuming — {len(done)} cases already recorded", flush=True)

    print(f"=== {name} [search={search} profile={profile}] ({len(cases)} to run) ===",
          flush=True)
    if not cases:
        return

    if cfg["kind"] == "3stage":
        tm, tt = load(TRANSLATOR)
        rm, rt = load(cfg["reasoner"])
        think = cfg.get("reasoner_thinking")
        for c in cases:
            row = run_case_3stage(c, tm, tt, rm, rt, think, search)
            payload["results"].append(row)
            save(dest, payload)  # per-case, not at the end
            flag = "  [UNTERMINATED THINK]" if row["stages"]["reason"]["unterminated_think"] else ""
            print(f"  {c['id']}: {row['total_secs']}s{flag}", flush=True)
    else:
        m, t = load(cfg["model"])
        for c in cases:
            s = call(m, t, None, c["query"], GEN["single_max_tokens"])
            payload["results"].append(dict(id=c["id"], query=c["query"], total_secs=s["secs"],
                                           search_judged=None, search_performed=False,
                                           search=None, stages=dict(direct=s), final=s["text"]))
            save(dest, payload)
            print(f"  {c['id']}: {s['secs']}s", flush=True)

    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
