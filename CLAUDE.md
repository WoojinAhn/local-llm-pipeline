# CLAUDE.md

## Overview

Local LLM pipelines that run on 128GB unified memory (Apple Silicon) without model swapping. Two pipelines:

1. **mlx-pipeline** (3-stage): Qwen3-14B (translation) + GPT-OSS 120B (analysis). Produces Korean analysis with no Hanja contamination.
2. **multimodal** (single model): Gemma 4 31B. Text+image, Korean-native (no translation wrapper).

See `README.md` for product-level details, model specs, and benchmarks. This file is the agent operating manual.

## Commands

Always use the project venv interpreter (`.venv/bin/python` or activate first). Models load on first run.

```bash
# Setup (once)
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 3-stage pipeline (Qwen translate -> GPT-OSS analyze -> Qwen translate)
python3 mlx-pipeline.py                 # interactive
python3 mlx-pipeline.py "질문"           # one-shot
#   flags: --reasoner-only (GPT-OSS English analysis only)
#          --qwen-only      (Qwen Korean chat only)
#          --translate-only (translation only, no analysis)
#   interactive: /search <q>, /nosearch <q>, /reset, /help, quit|exit

# Multimodal pipeline (Gemma 4, text+image)
python3 multimodal.py                            # interactive
python3 multimodal.py "Describe this" -i pic.jpg # one-shot with image
python3 multimodal.py --text-only "..."          # no image
#   flags: --no-search, --max-tokens N
#   interactive: /image <path>, /clear, /search (toggle), /reset, /quit

# Legacy LM Studio pipeline (requires LM Studio running on :1234)
python3 llm-pipeline.py
```

The pipelines themselves have no test suite — verify changes by running the relevant one and inspecting output (Korean cleanliness, search injection, channel filtering). The eval harness under `eval/proper-noun-preservation/` does have tests. They load no model, but `test_control_preflight.py` imports `run.py` and therefore needs `mlx` — use the venv interpreter, not the system `python3`:

```bash
.venv/bin/python eval/proper-noun-preservation/test_score.py             # scorer
.venv/bin/python eval/proper-noun-preservation/test_control_preflight.py # control harness
.venv/bin/python eval/proper-noun-preservation/test_transitions.py       # alias matcher + analyzer
```

## Setup / Environment

- Dependencies: `mlx-lm`, `mlx-vlm`, `rich` (`pip install -r requirements.txt`).
- Secrets in `.env` (loaded by `env_loader.load_env()`, gitignored): `BRAVE_API_KEY`, `TAVILY_API_KEY` (search — optional, skipped gracefully if unset), `HF_TOKEN` (avoids HuggingFace rate limits / gated-model access).
- Model path resolution: LM Studio cache (`~/.lmstudio/models/`) → HuggingFace cache (`~/.cache/huggingface/hub/`) → HuggingFace ID fallback.
- Offline run: set `HF_HUB_OFFLINE=1` once models are cached.

## Architecture

### mlx-pipeline.py (primary, mlx-lm direct inference)

- GPT-OSS 120B 4-bit (~65GB): English analysis/reasoning only. Qwen3-14B 4-bit (~7.7GB): bidirectional KO↔EN translation. Both loaded together (~73GB; ~55GB free on 128GB).
- Flow: KO→EN translate → English analysis → EN→KO translate.
- Search (optional): Qwen judges need via `SEARCH:yes/no` during translation. Brave (Korean) + Tavily (English) run in parallel. **Korean results are translated to English by Qwen before injection** — the reasoner ignores Korean context otherwise.
- Conversation context: mlx-lm `prompt_cache` accumulates reasoner history (KV-cache reuse). `/reset` clears it.
- Output filtering: GPT-OSS harmony format hides the `analysis` channel, shows only `final` (`filter_thinking_harmony()`).
- Rendering: Rich (colored step labels, Markdown final output, Rule separators).

### multimodal.py (Gemma 4, mlx-vlm direct inference)

- Gemma 4 31B 4-bit (~17GB), Korean-native, no translation wrapper.
- Search (optional): Gemma judges need + rewrites query (date resolution, keyword optimization) in one inference. Brave + Tavily in parallel.
- Current date auto-injected into the system prompt.
- Thinking filter: strips Gemma 4 `<|channel>thought` blocks.

### Other

- FLUX.2 image gen (`flux-2-swift-mlx`): built via `setup-flux.sh` (Xcode source build; prebuilt binary has a missing-metallib issue). Models cached in `~/Library/Caches/models/`. FLUX.2-dev is HF-gated (needs token + access approval).
- llm-pipeline.py (legacy): DeepSeek R1 70B + Qwen 3 32B 2-stage via LM Studio OpenAI-compatible API (requires model swapping). Python stdlib only.

## Key Files

- `mlx-pipeline.py` — 3-stage pipeline (mlx-lm).
- `multimodal.py` — Gemma 4 multimodal pipeline (mlx-vlm).
- `prompts.py` — shared prompts (date injection, search judge/query gen, citation enforcement, thinking filters). Edit here to change system prompts for both pipelines.
- `web_search.py` — `brave_search()`, `tavily_search()`, `search_both()`, `format_search_context()`.
- `env_loader.py` — loads `.env`.
- `setup-flux.sh` — FLUX.2 CLI build script.

## Conventions

- README.md keeps Korean and English sections equivalent — editing one side requires the same edit on the other. Keep the `[English](#english) | [한국어](#한국어)` anchors.
- System prompts live in `prompts.py` constants/functions (shared by both pipelines).
- File issue before non-trivial code changes; commit format `[#issue] type: description`.

## Gotchas

- **Ollama M5 Max Metal crash** (ollama#14432): only with `brew install ollama` (source build). Use `brew install --cask ollama` (prebuilt) — works.
- **GPT-OSS is not Korean-native** → the 3-stage translation wrapper is required, not optional.
- **Reasoner ignores Korean search results** → always translate Korean results to English before injecting.
- **Reasoner speculates without search** on factual queries → web search integration is the fix.
- **Qwen Hanja contamination**: Qwen3-14B 4-bit clean (0/10 tested); larger Qwen mix Hanja — Qwen3.5-27B 4-bit mixed Chinese idioms, Qwen3.6-27B substitutes raw Hanja mid-word (rejected as Gemma replacement, #34). Re-test any Qwen swap. This rules Qwen3.6 out of Korean-facing slots only. #34's other blocker — mlx-lm being unable to load `qwen3_5_text` — is gone: 0.31.3 ships `qwen3_5`/`qwen3_5_moe` and loads them, so Qwen3.6-35B-A3B is live as an English-only reasoner (#40).
- **Translation round-trip fabricates Korean proper nouns**: KO→EN→analysis→EN→KO invents plausible historical figures (신숙주→신석주, 원균→원경, plus wholly invented names with dates). Reasoner-independent in direction but not in degree, and a clean-English control recovers part of it — measured in #43, harness under `eval/proper-noun-preservation/`. Do not treat Korea-domain answers from the 3-stage path as reliable.
- **GPT-OSS long-context memory pressure**: ~65GB weights + KV cache can strain 128GB on long sessions.
- **mlx-lm has `gemma4_text` (text only)**; Gemma 4 multimodal still needs mlx-vlm.
- **mlx-vlm is pinned far behind**: 0.5.0 carries 59 architecture packages, main/0.6.10 carries 168. `deepseek_v4`, `kimi_k3`, `minimax_m3`, `solar_open` exist only upstream, so "MLX cannot load X" is worth re-checking against the current release before believing it (#42).
- **EXAONE-4.0-32B will not load unpatched**: its `sliding_window_pattern` is `"LLLG"`, which mlx-lm indexes as a string while transformers 5.9.0 applies integer modulo to it. Patch the cached config to keep the string and pin `layer_types` explicitly — procedure in the #43 harness README.
- **Stage `tokens` in eval artifacts is `len(tokenizer.encode(raw))` recorded after generation**, not a generation counter and not a stop reason. A value at the limit indicates probable budget exhaustion; re-encoding decoded text need not reproduce the original count (one record retokenizes to 4001 against a 4000 limit).
- **LM Studio CLI (`lms`)** not on PATH: `/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms`.

## Backlog (tracked as issues)

- `requirements.txt` pins nothing and does not declare `mlx`/`transformers` at all — #63. A rebuilt venv can silently land on the mlx 0.32.0 stack that #42 rolled back, and the harness's committed baseline is only comparable within a fixed stack. Complementary to #62: #62 makes the stack observable after the fact, #63 makes it reproducible in advance.
- Korea-domain routing around the translation wrapper — #44, blocked on #40 for memory headroom. The recall table in #44's body is stale (it predates the scorer fixes) and overstates the gap ~4x; corrected in a comment there — at the same profile it is EXAONE 25/42 vs 3-stage+gpt-oss 17/42, not 16/43 vs 4/43. Its "reasoner-independent" claim is also wrong; degree differs (gpt-oss 17 vs Qwen3.6 12).
- Reasoner swap to Qwen3.6-35B-A3B — #40. Memory 65.9GB → 19.7GB; speed roughly neutral once thinking tokens are counted, and it *lowers* Korean recall, so it is not a free win. The spike is measured and closed out in a comment; the issue stays open as the tracker for the swap itself, which #44's memory budget depends on.
- Parked: mlx-vlm 0.5.0 → 0.6.10 — #42. The upgrade drags the shared mlx core (0.31.2 → 0.32.0) and so puts the 3-stage pipeline in the blast radius, but it unlocks nothing we currently want: `exaone4` upstream is a text-only shell (#44's EXAONE path already runs through patched mlx-lm), and `deepseek_v4`'s only consumer #41 is itself parked. Re-entry conditions in the issue.
- Parked: speculative decoding for the GPT-OSS reasoner (gpt-oss-20b draft) — #35, and KV cache quantization (`kv_bits`) — #39. Both act on the *default* reasoner path, so both stay valid as long as GPT-OSS is the default; they are superseded only if #40 promotes Qwen3.6-35B-A3B to default rather than flag-gated. They are not redundant with #40 — #39 targets KV growth, not weights, and gpt-oss vs Qwen3.6 is a wall-clock tie (27.20s vs 27.76s over 5 prompts in `english-reasoner-ab.json`).
- Hy-MT2 (Tencent, WMT lineage; supersedes the earlier Hunyuan-MT 7B note) as a translation-stage candidate. Not a drop-in: the current Qwen3-14B slot also does the `SEARCH:yes/no` judgment, which a dedicated MT model cannot.
- Result-to-file save option; streaming translation output; Textual TUI — #8.
- Parked: DeepSeek V4 Flash at 2.4-bit — #41. Fits 92.8GB but needs the mlx-vlm path and the quant is unvalidated. Also gated on #42, which is itself parked.
- Closed as infeasible: GLM-5.2 (#38) — 395GB at mxfp4, no quantization path fits 128GB.
