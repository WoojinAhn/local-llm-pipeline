#!/usr/bin/env python3
"""
Triple-stage MLX pipeline: translate → analyze → translate.

Qwen3-14B (translator) and DeepSeek R1 70B (analyst) are loaded
simultaneously — no model swap needed. DeepSeek maintains conversation
context across turns via mlx-lm prompt cache.

Usage:
  python3 mlx-pipeline.py "한국어 질문"
  python3 mlx-pipeline.py                    # interactive mode
  python3 mlx-pipeline.py --deepseek-only    # DeepSeek analysis (English in/out)
  python3 mlx-pipeline.py --qwen-only        # Qwen conversation (Korean)
  python3 mlx-pipeline.py --translate-only   # Translation only (no analysis)
"""

import os
import sys
import time

from env_loader import load_env
load_env()

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

from prompts import (
    DEEPSEEK_SYSTEM,
    TRANSLATE_KO_TO_EN,
    TRANSLATE_EN_TO_KO,
    build_search_context_prompt,
    filter_thinking_deepseek,
)

# --- Model paths (LM Studio cache first, HuggingFace fallback) ---
_LMSTUDIO = os.path.expanduser("~/.lmstudio/models")

_DEEPSEEK_LOCAL = os.path.join(_LMSTUDIO, "mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit")
DEEPSEEK_ID = _DEEPSEEK_LOCAL if os.path.isdir(_DEEPSEEK_LOCAL) else "mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit"

QWEN_ID = "mlx-community/Qwen3-14B-4bit"

# System prompts imported from prompts.py

# --- Models (loaded once at startup) ---
_deepseek_model = None
_deepseek_tokenizer = None
_deepseek_cache = None
_deepseek_history = []  # English conversation history for DeepSeek

_qwen_model = None
_qwen_tokenizer = None


def load_models():
    """Load both models into memory."""
    global _deepseek_model, _deepseek_tokenizer, _deepseek_cache
    global _qwen_model, _qwen_tokenizer

    print(f"  Loading DeepSeek R1 ({DEEPSEEK_ID})...", flush=True)
    start = time.time()
    _deepseek_model, _deepseek_tokenizer = load(DEEPSEEK_ID)
    _deepseek_cache = make_prompt_cache(_deepseek_model)
    print(f"  DeepSeek loaded in {time.time() - start:.1f}s", flush=True)

    print(f"  Loading Qwen3-14B ({QWEN_ID})...", flush=True)
    start = time.time()
    _qwen_model, _qwen_tokenizer = load(QWEN_ID)
    print(f"  Qwen loaded in {time.time() - start:.1f}s", flush=True)


def _stream_and_collect(model, tokenizer, prompt, max_tokens=2000,
                        stream=True, think_start=False, prompt_cache=None):
    """Stream generation, filter thinking blocks, return clean text."""
    raw_parts = []
    in_think = think_start
    for response in stream_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, prompt_cache=prompt_cache,
    ):
        raw_parts.append(response.text)
        if "<think>" in response.text:
            in_think = True
        if in_think:
            if "</think>" in response.text:
                in_think = False
            continue
        if stream:
            print(response.text, end="", flush=True)

    if stream:
        print(flush=True)

    return filter_thinking_deepseek("".join(raw_parts))


def translate(text, direction="ko2en", stream=False):
    """Translate text. For ko2en, returns (translation, needs_search) tuple."""
    system = TRANSLATE_KO_TO_EN if direction == "ko2en" else TRANSLATE_EN_TO_KO
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    prompt = _qwen_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=False,
    )
    raw = _stream_and_collect(
        _qwen_model, _qwen_tokenizer, prompt,
        max_tokens=2000, stream=stream,
    )

    if direction == "ko2en":
        needs_search = False
        translation = raw
        for line in raw.strip().split("\n"):
            stripped = line.strip()
            if stripped.startswith("SEARCH:"):
                needs_search = "yes" in stripped.lower()
                translation = raw[:raw.rfind(line)].strip()
                break
        return translation, needs_search

    return raw


def analyze(text, stream=True):
    """Analyze text using DeepSeek R1 with conversation context."""
    global _deepseek_history, _deepseek_cache

    # Build full conversation with history
    # First message includes system prompt merged into user message
    if not _deepseek_history:
        _deepseek_history.append({
            "role": "user",
            "content": f"{DEEPSEEK_SYSTEM}\n\n{text}",
        })
    else:
        _deepseek_history.append({"role": "user", "content": text})

    prompt = _deepseek_tokenizer.apply_chat_template(
        _deepseek_history, add_generation_prompt=True, tokenize=False,
    )

    result = _stream_and_collect(
        _deepseek_model, _deepseek_tokenizer, prompt,
        max_tokens=4000, stream=stream, think_start=True,
        prompt_cache=_deepseek_cache,
    )

    # Append assistant response to history
    _deepseek_history.append({"role": "assistant", "content": result})

    return result


def reset_context():
    """Reset DeepSeek conversation context."""
    global _deepseek_history, _deepseek_cache
    _deepseek_history = []
    if _deepseek_model is not None:
        _deepseek_cache = make_prompt_cache(_deepseek_model)
    print("  [Context reset]\n", flush=True)


def pipeline(query, force_search=None):
    """Triple-stage pipeline with optional web search.

    force_search: True=always, False=never, None=auto (Qwen judges)
    """
    # Stage 1: Translate + judge search
    print("  [1/4] Translating to English...", flush=True)
    english_query, needs_search = translate(query, direction="ko2en")
    print(f"  → {english_query}", flush=True)

    if force_search is not None:
        needs_search = force_search

    # Stage 2: Web search (if needed)
    search_context = ""
    if needs_search:
        print("  [2/4] Searching web...", flush=True)
        from web_search import search_both, format_search_context
        ko_results, en_results = search_both(query, english_query)
        hit_count = len(ko_results) + len(en_results)
        print(f"  → {hit_count} results found", flush=True)

        # Translate Korean snippets to English for DeepSeek
        if ko_results:
            print("  → Translating Korean results to English...", flush=True)
            ko_snippets = "\n".join(
                f"{i}. [{r['title']}] {r['snippet']}" for i, r in enumerate(ko_results, 1)
            )
            translated_snippets, _ = translate(ko_snippets, direction="ko2en")
            for i, r in enumerate(ko_results):
                r["snippet"] = ""  # clear original
            # Replace ko_results with single translated block
            ko_results = [{"title": "Korean sources (translated)", "url": "", "snippet": translated_snippets}]

        search_context = format_search_context(ko_results, en_results)
        print(flush=True)
    else:
        print("  [2/4] Search skipped\n", flush=True)

    # Stage 3: DeepSeek analysis
    print("  [3/4] DeepSeek R1 analyzing...", flush=True)
    if search_context:
        analysis_prompt = build_search_context_prompt(search_context, english_query)
    else:
        analysis_prompt = english_query
    english_analysis = analyze(analysis_prompt)

    # Stage 4: Translate to Korean
    print(f"\n  [4/4] Translating to Korean...", flush=True)
    korean_result = translate(english_analysis, direction="en2ko", stream=True)

    return f"""
{'='*60}
[English Analysis]
{'='*60}
{english_analysis}

{'='*60}
[Korean Translation]
{'='*60}
{korean_result}"""


def main():
    mode = "pipeline"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if "--deepseek-only" in flags:
        mode = "deepseek"
    elif "--qwen-only" in flags:
        mode = "qwen"
    elif "--translate-only" in flags:
        mode = "translate"

    query = " ".join(args) if args else None

    # Load models
    if mode in ("deepseek",):
        print("Loading DeepSeek R1...", flush=True)
        global _deepseek_model, _deepseek_tokenizer, _deepseek_cache
        _deepseek_model, _deepseek_tokenizer = load(DEEPSEEK_ID)
        _deepseek_cache = make_prompt_cache(_deepseek_model)
    elif mode in ("qwen", "translate"):
        print("Loading Qwen3-14B...", flush=True)
        global _qwen_model, _qwen_tokenizer
        _qwen_model, _qwen_tokenizer = load(QWEN_ID)
    else:
        load_models()

    if query is None:
        print(f"\nMLX Triple-Stage Pipeline (mode: {mode})")
        print("  /help 로 사용법 확인\n")

    while True:
        if query is None:
            try:
                user_input = input("질문> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n종료합니다.")
                break
            if not user_input or user_input in ("quit", "exit"):
                break
            if user_input == "/reset":
                reset_context()
                continue
            if user_input == "/help":
                print("""
  사용법:
    <질문>                한국어 질문 → 영어 분석 → 한국어 결과 (검색 자동 판별)
    /search <질문>        웹 검색 강제 실행
    /nosearch <질문>      웹 검색 건너뛰기
    /reset                대화 컨텍스트 초기화
    /help                 이 도움말 표시
    quit / exit           종료

  CLI 모드:
    --deepseek-only       DeepSeek 영어 분석만
    --qwen-only           Qwen 한국어 대화만
    --translate-only      번역만 (분석 없이)
""")
                continue
        else:
            user_input = query

        # Search override commands
        force_search = None
        if user_input.startswith("/search "):
            force_search = True
            user_input = user_input[8:]
        elif user_input.startswith("/nosearch "):
            force_search = False
            user_input = user_input[10:]

        if mode == "deepseek":
            analyze(user_input)
            print()
        elif mode == "qwen":
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Always respond in Korean using Hangul only."},
                {"role": "user", "content": user_input},
            ]
            prompt = _qwen_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            _stream_and_collect(_qwen_model, _qwen_tokenizer, prompt)
            print()
        elif mode == "translate":
            result, needs_search = translate(user_input, direction="ko2en")
            search_tag = "SEARCH:yes" if needs_search else "SEARCH:no"
            print(f"\n{result}\n{search_tag}\n")
        else:
            result = pipeline(user_input, force_search=force_search)
            print(result)
            print()

        if query is not None:
            break


if __name__ == "__main__":
    main()
