"""Run pipeline configurations over the #43 case set, capturing every stage.

Unlike the throwaway harness this replaces, per-stage intermediates are recorded, so
the corruption can be localized to KO->EN translation, the reasoner, or EN->KO
back-translation rather than only observed end-to-end.

Two modes, because they answer different questions and must not be mixed:

  control  (default) — web search disabled. The `SEARCH:` judgment is recorded but not
                       acted on. Deterministic and reproducible; corruption is
                       attributable to the pipeline rather than to whatever the web
                       returned that day. This is NOT what production does.

  parity             — mirrors `mlx-pipeline.py:pipeline()`: when the judge says yes,
                       Brave+Tavily run, Korean snippets are translated to English, and
                       `build_search_context_prompt()` wraps the reasoner input. Results
                       vary run to run and depend on network and API keys.

Both modes now pass the full reasoner output to back-translation, matching production.
An earlier version truncated it at 6000 characters, which production does not do.

Usage:
    python run.py three-stage-gpt-oss
    python run.py three-stage-qwen36 --mode parity
    python run.py single-exaone --resume
"""
import hashlib, json, os, platform, re, subprocess, sys, time
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

GEN = dict(translate_max_tokens=4000, reason_max_tokens=8000, single_max_tokens=8000)

THINK_SPLIT = re.compile(r"</think>|<\|channel\|>final<\|message\|>|assistantfinal")


def strip_think(text):
    return THINK_SPLIT.split(text)[-1]


def has_unterminated_think(text):
    """Qwen3.6 can exhaust the budget mid-reasoning; the block then never closes."""
    return ("<think>" in text or "<|channel|>analysis" in text) and not THINK_SPLIT.search(text)


def harness_hash():
    """Identify the harness itself.

    `repo_commit` alone is not enough: a run made while the harness was uncommitted
    records the *previous* HEAD, which does not identify the code that produced it.
    """
    h = hashlib.sha256()
    for f in sorted(("run.py", "score.py", "isolate.py", "cases.jsonl")):
        p = f"{HERE}/{f}"
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def env_metadata(mode):
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", ROOT, *args], text=True).strip()
        except Exception:
            return None

    dirty = git("status", "--porcelain")
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        repo_commit=git("rev-parse", "HEAD"),
        repo_dirty=bool(dirty),
        dirty_paths=[l[3:] for l in dirty.splitlines()] if dirty else [],
        harness_sha256_16=harness_hash(),
        prompts_sha256_16=hashlib.sha256(
            open(f"{ROOT}/prompts.py", "rb").read()).hexdigest()[:16],
        python=sys.version.split()[0],
        platform=platform.platform(), machine=platform.machine(),
        mlx=ver("mlx"), mlx_lm=ver("mlx_lm"), transformers=ver("transformers"),
        generation=dict(sampler="greedy (temp=0 default)", **GEN),
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


def run_case_3stage(c, tm, tt, rm, rt, think, mode):
    t0 = time.time()
    s1 = call(tm, tt, prompts.TRANSLATE_KO_TO_EN, c["query"], GEN["translate_max_tokens"],
              thinking=False)
    en_q = re.sub(r"SEARCH:\s*(yes|no)", "", s1["text"]).strip()
    search_flag = bool(re.search(r"SEARCH:\s*yes", s1["text"], re.I))

    search = None
    reason_input = en_q
    if mode == "parity" and search_flag:
        ctx, translated, hits = do_search(tm, tt, c["query"], en_q)
        search = dict(hits=hits, translated_ko_snippets=translated, context=ctx)
        reason_input = prompts.build_search_context_prompt(ctx, en_q)

    s2 = call(rm, rt, prompts.REASONER_SYSTEM, reason_input, GEN["reason_max_tokens"],
              thinking=think)
    # Production passes the full analysis; do not truncate.
    s3 = call(tm, tt, prompts.TRANSLATE_EN_TO_KO, s2["text"],
              GEN["translate_max_tokens"], thinking=False)
    return dict(id=c["id"], query=c["query"], total_secs=round(time.time() - t0, 2),
                search_judged=search_flag, search_performed=search is not None,
                search=search, stages=dict(ko2en=s1, reason=s2, en2ko=s3),
                final=s3["text"])


def save(dest, payload):
    """Atomic write — a long run must not be lost or half-written on interrupt."""
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, dest)


def main():
    name = sys.argv[1]
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "control"
    assert mode in ("control", "parity"), mode
    resume = "--resume" in sys.argv
    cfg = CONFIGS[name]

    cases = [json.loads(l) for l in open(f"{HERE}/cases.jsonl") if l.strip()]
    if "--holdout" not in sys.argv:
        cases = [c for c in cases if not c.get("holdout")]

    dest = f"{HERE}/outputs/{name}/run-{mode}.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    payload = dict(config=name, config_detail=cfg, env=env_metadata(mode), results=[])
    if resume and os.path.exists(dest):
        payload = json.load(open(dest))
        done = {r["id"] for r in payload["results"]}
        cases = [c for c in cases if c["id"] not in done]
        print(f"resuming — {len(done)} cases already recorded", flush=True)

    print(f"=== {name} [{mode}] ({len(cases)} to run) ===", flush=True)
    if not cases:
        return

    if cfg["kind"] == "3stage":
        tm, tt = load(TRANSLATOR)
        rm, rt = load(cfg["reasoner"])
        think = cfg.get("reasoner_thinking")
        for c in cases:
            row = run_case_3stage(c, tm, tt, rm, rt, think, mode)
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
