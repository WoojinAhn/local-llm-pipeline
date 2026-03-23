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
import re
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

# --- Model paths (LM Studio cache first, HuggingFace fallback) ---
_LMSTUDIO = os.path.expanduser("~/.lmstudio/models")

_DEEPSEEK_LOCAL = os.path.join(_LMSTUDIO, "mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit")
DEEPSEEK_ID = _DEEPSEEK_LOCAL if os.path.isdir(_DEEPSEEK_LOCAL) else "mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit"

QWEN_ID = "mlx-community/Qwen3-14B-4bit"

# --- System Prompts ---
DEEPSEEK_SYSTEM = """\
You are an expert analyst. Respond ONLY in English.
Provide thorough, structured analysis with clear reasoning.
Be concise but comprehensive. Use bullet points and sections when appropriate."""

TRANSLATE_KO_TO_EN = """\
You are a translator. Translate the following Korean text to natural English. \
Output ONLY the English translation, nothing else."""

TRANSLATE_EN_TO_KO = """\
You are a translator. Translate the following English text to natural Korean. \
Use pure Hangul only — never use Chinese characters (漢字) or Japanese characters. \
Proper nouns and technical terms may remain in English. \
Output ONLY the Korean translation, nothing else."""

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


def _filter_thinking(raw):
    """Remove <think>...</think> blocks from generated text."""
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
    raw = re.sub(r"^.*?</think>\s*", "", raw, flags=re.DOTALL)
    return raw.strip()


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

    return _filter_thinking("".join(raw_parts))


def translate(text, direction="ko2en", stream=False):
    """Translate text using Qwen3-14B (stateless, no cache)."""
    system = TRANSLATE_KO_TO_EN if direction == "ko2en" else TRANSLATE_EN_TO_KO
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    prompt = _qwen_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=False,
    )
    return _stream_and_collect(
        _qwen_model, _qwen_tokenizer, prompt,
        max_tokens=2000, stream=stream,
    )


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


def pipeline(query):
    """Triple-stage: Korean → English → Analysis → Korean."""
    # Stage 1: Translate Korean input to English
    print("  [1/3] Translating to English...", flush=True)
    english_query = translate(query, direction="ko2en")
    print(f"  → {english_query}\n", flush=True)

    # Stage 2: DeepSeek analysis in English (with conversation context)
    print("  [2/3] DeepSeek R1 analyzing...", flush=True)
    english_analysis = analyze(english_query)

    # Stage 3: Translate analysis to Korean
    print("\n  [3/3] Translating to Korean...", flush=True)
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
        print(f"\nMLX Triple-Stage Pipeline (mode: {mode}, quit: quit/exit, reset: /reset)\n")

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
        else:
            user_input = query

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
            result = translate(user_input, direction="ko2en")
            print(f"\n{result}\n")
        else:
            result = pipeline(user_input)
            print(result)
            print()

        if query is not None:
            break


if __name__ == "__main__":
    main()
