#!/usr/bin/env python3
"""
Triple-stage MLX pipeline: translate → analyze → translate.

Qwen3-14B (translator) and GPT-OSS 120B (reasoner) are loaded
simultaneously — no model swap needed. The reasoner maintains
conversation context across turns via mlx-lm prompt cache.

Usage:
  python3 mlx-pipeline.py "한국어 질문"
  python3 mlx-pipeline.py                    # interactive mode
  python3 mlx-pipeline.py --reasoner-only    # GPT-OSS analysis (English in/out)
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
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from prompts import (
    REASONER_SYSTEM,
    TRANSLATE_KO_TO_EN,
    TRANSLATE_EN_TO_KO,
    build_search_context_prompt,
    filter_thinking_harmony,
)

console = Console()


def _stage(num, total, label):
    console.print(f"[bold cyan]\\[{num}/{total}][/] {label}")


def _info(msg):
    console.print(f"  [dim]{msg}[/]")

# --- Model paths (LM Studio cache first, HuggingFace fallback) ---
_LMSTUDIO = os.path.expanduser("~/.lmstudio/models")

_REASONER_LOCAL = os.path.join(_LMSTUDIO, "mlx-community/gpt-oss-120b-4bit")
REASONER_ID = _REASONER_LOCAL if os.path.isdir(_REASONER_LOCAL) else "mlx-community/gpt-oss-120b-4bit"

QWEN_ID = "mlx-community/Qwen3-14B-4bit"

HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"

# System prompts imported from prompts.py

# --- Models (loaded once at startup) ---
_reasoner_model = None
_reasoner_tokenizer = None
_reasoner_cache = None
_reasoner_history = []  # English conversation history for the reasoner

_qwen_model = None
_qwen_tokenizer = None


def load_models():
    """Load both models into memory."""
    global _reasoner_model, _reasoner_tokenizer, _reasoner_cache
    global _qwen_model, _qwen_tokenizer

    console.print(f"  [dim]Loading reasoner ({REASONER_ID})...[/]")
    start = time.time()
    _reasoner_model, _reasoner_tokenizer = load(REASONER_ID)
    _reasoner_cache = make_prompt_cache(_reasoner_model)
    console.print(f"  [green]Reasoner loaded[/] [dim]in {time.time() - start:.1f}s[/]")

    console.print(f"  [dim]Loading Qwen3-14B ({QWEN_ID})...[/]")
    start = time.time()
    _qwen_model, _qwen_tokenizer = load(QWEN_ID)
    console.print(f"  [green]Qwen loaded[/] [dim]in {time.time() - start:.1f}s[/]")


def _stream_qwen(model, tokenizer, prompt, max_tokens=2000, stream=True):
    """Stream Qwen generation (no thinking-block filtering)."""
    parts = []
    for response in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
    ):
        parts.append(response.text)
        if stream:
            print(response.text, end="", flush=True)
    if stream:
        print(flush=True)
    return "".join(parts).strip()


def _stream_reasoner(model, tokenizer, prompt, max_tokens=4000,
                     stream=True, prompt_cache=None):
    """Stream reasoner (harmony-format) generation.

    Suppresses the analysis channel; streams only the final channel to stdout.
    Returns the filtered final-channel text.
    """
    raw_parts = []
    in_final = False
    buffer = ""

    for response in stream_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, prompt_cache=prompt_cache,
    ):
        raw_parts.append(response.text)
        if in_final:
            if stream:
                print(response.text, end="", flush=True)
            continue
        buffer += response.text
        if HARMONY_FINAL_MARKER in buffer:
            in_final = True
            tail = buffer.split(HARMONY_FINAL_MARKER, 1)[1]
            buffer = ""
            if stream and tail:
                print(tail, end="", flush=True)

    if stream:
        print(flush=True)

    return filter_thinking_harmony("".join(raw_parts))


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
    raw = _stream_qwen(
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
    """Analyze text using the GPT-OSS reasoner with conversation context."""
    global _reasoner_history, _reasoner_cache

    if not _reasoner_history:
        _reasoner_history.append({"role": "system", "content": REASONER_SYSTEM})
    _reasoner_history.append({"role": "user", "content": text})

    prompt = _reasoner_tokenizer.apply_chat_template(
        _reasoner_history, add_generation_prompt=True, tokenize=False,
    )

    result = _stream_reasoner(
        _reasoner_model, _reasoner_tokenizer, prompt,
        max_tokens=4000, stream=stream,
        prompt_cache=_reasoner_cache,
    )

    _reasoner_history.append({"role": "assistant", "content": result})

    return result


def reset_context():
    """Reset reasoner conversation context."""
    global _reasoner_history, _reasoner_cache
    _reasoner_history = []
    if _reasoner_model is not None:
        _reasoner_cache = make_prompt_cache(_reasoner_model)
    console.print("  [yellow]\\[Context reset][/]\n")


def pipeline(query, force_search=None):
    """Triple-stage pipeline with optional web search.

    force_search: True=always, False=never, None=auto (Qwen judges)
    """
    # Stage 1: Translate + judge search
    _stage(1, 4, "Translating to English...")
    english_query, needs_search = translate(query, direction="ko2en")
    _info(f"→ {english_query}")

    if force_search is not None:
        needs_search = force_search

    # Stage 2: Web search (if needed)
    search_context = ""
    if needs_search:
        _stage(2, 4, "Searching web...")
        from web_search import search_both, format_search_context
        ko_results, en_results = search_both(query, english_query)
        hit_count = len(ko_results) + len(en_results)
        _info(f"→ {hit_count} results found")

        # Translate Korean snippets to English for the reasoner
        if ko_results:
            _info("→ Translating Korean results to English...")
            ko_snippets = "\n".join(
                f"{i}. [{r['title']}] {r['snippet']}" for i, r in enumerate(ko_results, 1)
            )
            translated_snippets, _ = translate(ko_snippets, direction="ko2en")
            for i, r in enumerate(ko_results):
                r["snippet"] = ""  # clear original
            # Replace ko_results with single translated block
            ko_results = [{"title": "Korean sources (translated)", "url": "", "snippet": translated_snippets}]

        search_context = format_search_context(ko_results, en_results)
        console.print()
    else:
        _stage(2, 4, "[dim]Search skipped[/]")
        console.print()

    # Stage 3: Reasoner — spinner during generation, markdown on finish
    _stage(3, 4, "GPT-OSS reasoning...")
    if search_context:
        analysis_prompt = build_search_context_prompt(search_context, english_query)
    else:
        analysis_prompt = english_query
    with console.status("[cyan]reasoning...[/]", spinner="dots"):
        english_analysis = analyze(analysis_prompt, stream=False)

    # Stage 4: Translate to Korean (also silent, rendered in the final block)
    _stage(4, 4, "Translating to Korean...")
    with console.status("[cyan]translating...[/]", spinner="dots"):
        korean_result = translate(english_analysis, direction="en2ko", stream=False)

    # Final rendered output — markdown for readability, no stream duplication
    console.print()
    console.print(Rule("English Analysis", style="blue"))
    console.print(Markdown(english_analysis))
    console.print(Rule("Korean Translation", style="blue"))
    console.print(Markdown(korean_result))
    return None


def main():
    mode = "pipeline"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if "--reasoner-only" in flags:
        mode = "reasoner"
    elif "--qwen-only" in flags:
        mode = "qwen"
    elif "--translate-only" in flags:
        mode = "translate"

    query = " ".join(args) if args else None

    # Load models
    if mode == "reasoner":
        console.print("[dim]Loading reasoner...[/]")
        global _reasoner_model, _reasoner_tokenizer, _reasoner_cache
        _reasoner_model, _reasoner_tokenizer = load(REASONER_ID)
        _reasoner_cache = make_prompt_cache(_reasoner_model)
    elif mode in ("qwen", "translate"):
        console.print("[dim]Loading Qwen3-14B...[/]")
        global _qwen_model, _qwen_tokenizer
        _qwen_model, _qwen_tokenizer = load(QWEN_ID)
    else:
        load_models()

    if query is None:
        console.print(f"\n[bold]MLX Triple-Stage Pipeline[/] [dim](mode: {mode})[/]")
        console.print("  [dim]/help 로 사용법 확인[/]\n")

    while True:
        if query is None:
            try:
                user_input = console.input("[bold green]질문>[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]종료합니다.[/]")
                break
            if not user_input or user_input in ("quit", "exit"):
                break
            if user_input == "/reset":
                reset_context()
                continue
            if user_input == "/help":
                console.print(Markdown("""
### 사용법

| 입력 | 동작 |
|------|------|
| `<질문>` | 한국어 질문 → 영어 분석 → 한국어 결과 (검색 자동 판별) |
| `/search <질문>` | 웹 검색 강제 실행 |
| `/nosearch <질문>` | 웹 검색 건너뛰기 |
| `/reset` | 대화 컨텍스트 초기화 |
| `/help` | 이 도움말 표시 |
| `quit` / `exit` | 종료 |

### CLI 모드

| 플래그 | 동작 |
|--------|------|
| `--reasoner-only` | GPT-OSS 영어 분석만 |
| `--qwen-only` | Qwen 한국어 대화만 |
| `--translate-only` | 번역만 (분석 없이) |
"""))
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

        if mode == "reasoner":
            # Streaming raw tokens to stdout — already readable, just separate turns
            analyze(user_input, stream=True)
            console.print()
            console.print(Rule(style="dim"))
        elif mode == "qwen":
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Always respond in Korean using Hangul only."},
                {"role": "user", "content": user_input},
            ]
            prompt = _qwen_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            _stream_qwen(_qwen_model, _qwen_tokenizer, prompt)
            console.print()
        elif mode == "translate":
            result, needs_search = translate(user_input, direction="ko2en")
            search_tag = "SEARCH:yes" if needs_search else "SEARCH:no"
            console.print(f"\n{result}\n[dim]{search_tag}[/]\n")
        else:
            pipeline(user_input, force_search=force_search)
            console.print()

        if query is not None:
            break


if __name__ == "__main__":
    main()
