"""Run pipeline configurations over the #43 case set, capturing every stage.

Unlike the throwaway harness this replaces, per-stage intermediates are recorded, so
the corruption can be localized to KO->EN translation, the reasoner, or EN->KO
back-translation rather than only observed end-to-end.

Usage:
    python eval/issue-43-proper-noun-corruption/run.py three-stage-gpt-oss
    python eval/issue-43-proper-noun-corruption/run.py single-exaone
"""
import json, os, platform, re, subprocess, sys, time
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

# Generation is greedy so runs are reproducible without a seed. mlx-lm samples
# argmax when temp is 0, which is generate()'s default.
GEN = dict(translate_max_tokens=2000, reason_max_tokens=8000, single_max_tokens=8000)

THINK_SPLIT = re.compile(r"</think>|<\|channel\|>final<\|message\|>|assistantfinal")


def strip_think(text):
    return THINK_SPLIT.split(text)[-1]


def has_unterminated_think(text):
    """Qwen3.6 can exhaust the budget mid-reasoning; the block then never closes."""
    return ("<think>" in text or "<|channel|>analysis" in text) and not THINK_SPLIT.search(text)


def env_metadata():
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = None
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        repo_commit=commit,
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
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


def run_3stage(cfg, cases):
    tm, tt = load(TRANSLATOR)
    rm, rt = load(cfg["reasoner"])
    think = cfg.get("reasoner_thinking")
    rows = []
    for c in cases:
        t0 = time.time()
        s1 = call(tm, tt, prompts.TRANSLATE_KO_TO_EN, c["query"], GEN["translate_max_tokens"],
                  thinking=False)
        en_q = re.sub(r"SEARCH:\s*(yes|no)", "", s1["text"]).strip()
        search_flag = bool(re.search(r"SEARCH:\s*yes", s1["text"], re.I))
        s2 = call(rm, rt, prompts.REASONER_SYSTEM, en_q, GEN["reason_max_tokens"], thinking=think)
        s3 = call(tm, tt, prompts.TRANSLATE_EN_TO_KO, s2["text"][:6000],
                  GEN["translate_max_tokens"], thinking=False)
        rows.append(dict(id=c["id"], query=c["query"], total_secs=round(time.time() - t0, 2),
                         search_judged=search_flag,
                         stages=dict(ko2en=s1, reason=s2, en2ko=s3),
                         final=s3["text"]))
        print(f"  {c['id']}: {rows[-1]['total_secs']}s"
              f"{'  [UNTERMINATED THINK]' if s2['unterminated_think'] else ''}", flush=True)
    return rows


def run_single(cfg, cases):
    m, t = load(cfg["model"])
    rows = []
    for c in cases:
        s = call(m, t, None, c["query"], GEN["single_max_tokens"])
        rows.append(dict(id=c["id"], query=c["query"], total_secs=s["secs"],
                         search_judged=None, stages=dict(direct=s), final=s["text"]))
        print(f"  {c['id']}: {s['secs']}s", flush=True)
    return rows


def main():
    name = sys.argv[1]
    cfg = CONFIGS[name]
    cases = [json.loads(l) for l in open(f"{HERE}/cases.jsonl") if l.strip()]
    if "--holdout" not in sys.argv:
        cases = [c for c in cases if not c.get("holdout")]
    print(f"=== {name} ({len(cases)} cases) ===", flush=True)
    rows = run_3stage(cfg, cases) if cfg["kind"] == "3stage" else run_single(cfg, cases)
    out = dict(config=name, config_detail=cfg, env=env_metadata(), results=rows)
    dest = f"{HERE}/outputs/{name}/run.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
