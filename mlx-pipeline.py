#!/usr/bin/env python3
"""
Triple-stage MLX pipeline: translate → analyze → translate.

Qwen3-14B (translator) and DeepSeek R1 70B (analyst) are loaded
simultaneously — no model swap needed.

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
_qwen_model = None
_qwen_tokenizer = None


def load_models():
    """Load both models into memory."""
    global _deepseek_model, _deepseek_tokenizer, _qwen_model, _qwen_tokenizer

    print(f"  Loading DeepSeek R1 ({DEEPSEEK_ID})...", flush=True)
    start = time.time()
    _deepseek_model, _deepseek_tokenizer = load(DEEPSEEK_ID)
    print(f"  DeepSeek loaded in {time.time() - start:.1f}s", flush=True)

    print(f"  Loading Qwen3-14B ({QWEN_ID})...", flush=True)
    start = time.time()
    _qwen_model, _qwen_tokenizer = load(QWEN_ID)
    print(f"  Qwen loaded in {time.time() - start:.1f}s", flush=True)


def _generate(model, tokenizer, messages, max_tokens=2000, stream=True,
              think_start=False, enable_thinking=None):
    """Generate a response, optionally streaming and filtering <think> blocks."""
    template_kwargs = {"add_generation_prompt": True, "tokenize": False}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    raw_parts = []
    in_think = think_start
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
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

    raw = "".join(raw_parts)
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
    raw = re.sub(r"^.*?</think>\s*", "", raw, flags=re.DOTALL)
    return raw.strip()


def translate(text, direction="ko2en", stream=False):
    """Translate text using Qwen3-14B."""
    if direction == "ko2en":
        system = TRANSLATE_KO_TO_EN
    else:
        system = TRANSLATE_EN_TO_KO

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    return _generate(_qwen_model, _qwen_tokenizer, messages,
                     max_tokens=2000, stream=stream, enable_thinking=False)


def analyze(text, stream=True):
    """Analyze text using DeepSeek R1 (English only)."""
    # DeepSeek: merge system into user message (chat template fix)
    messages = [
        {"role": "user", "content": f"{DEEPSEEK_SYSTEM}\n\n{text}"},
    ]
    return _generate(_deepseek_model, _deepseek_tokenizer, messages,
                     max_tokens=4000, stream=stream, think_start=True)


def pipeline(query):
    """Triple-stage: Korean → English → Analysis → Korean."""
    # Stage 1: Translate Korean input to English
    print("  [1/3] Translating to English...", flush=True)
    english_query = translate(query, direction="ko2en")
    print(f"  → {english_query}\n", flush=True)

    # Stage 2: DeepSeek analysis in English
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
        global _deepseek_model, _deepseek_tokenizer
        _deepseek_model, _deepseek_tokenizer = load(DEEPSEEK_ID)
    elif mode in ("qwen", "translate"):
        print("Loading Qwen3-14B...", flush=True)
        global _qwen_model, _qwen_tokenizer
        _qwen_model, _qwen_tokenizer = load(QWEN_ID)
    else:
        load_models()

    if query is None:
        print(f"\nMLX Triple-Stage Pipeline (mode: {mode}, quit: quit/exit)\n")

    while True:
        if query is None:
            try:
                user_input = input("질문> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n종료합니다.")
                break
            if not user_input or user_input in ("quit", "exit"):
                break
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
            _generate(_qwen_model, _qwen_tokenizer, messages, enable_thinking=False)
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
