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
import readline  # noqa: F401 - character-aware line editing; Darwin's tty erases by byte, splitting Hangul
import sys
import time
from dataclasses import dataclass

from env_loader import load_env
load_env()

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from prompts import (
    reasoner_system,
    TRANSLATE_KO_TO_EN,
    TRANSLATE_EN_TO_KO,
    build_search_context_prompt,
    location_judge_prompt,
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


def _drain(chunks, echo):
    """Run a chunk generator to completion, returning its value.

    `for` discards a generator's return value, so the StopIteration is caught by hand.
    """
    while True:
        try:
            chunk = next(chunks)
        except StopIteration as stop:
            return stop.value
        if echo:
            print(chunk, end="", flush=True)


def _qwen_chunks(model, tokenizer, prompt, max_tokens=2000):
    """Yield Qwen text as it is generated (no thinking-block filtering).

    Returns the joined text — the generation is a generator so that a caller wanting
    tokens as they appear and a caller wanting only the finished string share one path.
    """
    parts = []
    for response in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
    ):
        parts.append(response.text)
        yield response.text
    return "".join(parts).strip()


def _reasoner_chunks(model, tokenizer, prompt, max_tokens=4000, prompt_cache=None):
    """Yield reasoner (harmony-format) final-channel text as it is generated.

    The analysis channel is suppressed and never yielded. Returns the filtered text.
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
            yield response.text
            continue
        buffer += response.text
        if HARMONY_FINAL_MARKER in buffer:
            in_final = True
            tail = buffer.split(HARMONY_FINAL_MARKER, 1)[1]
            buffer = ""
            if tail:
                yield tail

    return filter_thinking_harmony("".join(raw_parts))


def _stream_qwen(model, tokenizer, prompt, max_tokens=2000, stream=True):
    """Generate with Qwen, optionally echoing to stdout. Returns the text."""
    text = _drain(_qwen_chunks(model, tokenizer, prompt, max_tokens), stream)
    if stream:
        print(flush=True)
    return text


def _stream_reasoner(model, tokenizer, prompt, max_tokens=4000,
                     stream=True, prompt_cache=None):
    """Generate with the reasoner, optionally echoing the final channel to stdout."""
    text = _drain(
        _reasoner_chunks(model, tokenizer, prompt, max_tokens, prompt_cache), stream)
    if stream:
        print(flush=True)
    return text


def _split_search_flag(raw):
    """Peel the SEARCH:yes/no line off a ko2en translation."""
    for line in raw.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("SEARCH:"):
            return raw[:raw.rfind(line)].strip(), "yes" in stripped.lower()
    return raw, False


def _translate_chunks(text, direction="ko2en"):
    """Yield translator text as it is generated.

    Returns (translation, needs_search) for ko2en, the translation otherwise.
    """
    system = TRANSLATE_KO_TO_EN if direction == "ko2en" else TRANSLATE_EN_TO_KO
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    prompt = _qwen_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=False,
    )
    raw = yield from _qwen_chunks(_qwen_model, _qwen_tokenizer, prompt, max_tokens=2000)

    if direction == "ko2en":
        return _split_search_flag(raw)
    return raw


def translate(text, direction="ko2en", stream=False):
    """Translate text. For ko2en, returns (translation, needs_search) tuple."""
    value = _drain(_translate_chunks(text, direction), stream)
    if stream:
        print(flush=True)
    return value


def judge_location(query):
    """Ask Qwen whether the query depends on the user's location.

    A separate call rather than another line in TRANSLATE_KO_TO_EN — folding it in
    there degraded both translation fidelity and the SEARCH judgment (#67).
    """
    messages = [{"role": "user", "content": location_judge_prompt(query)}]
    prompt = _qwen_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    raw = _stream_qwen(_qwen_model, _qwen_tokenizer, prompt, max_tokens=10, stream=False)
    return "local:yes" in raw.lower()


def _analyze_chunks(text):
    """Yield reasoner final-channel text as it is generated; return the analysis.

    Conversation history and the prompt cache are updated here, so any caller of this
    generator — streaming or not — keeps the same context bookkeeping.
    """
    global _reasoner_history, _reasoner_cache

    if not _reasoner_history:
        _reasoner_history.append({"role": "system", "content": reasoner_system()})
    _reasoner_history.append({"role": "user", "content": text})

    prompt = _reasoner_tokenizer.apply_chat_template(
        _reasoner_history, add_generation_prompt=True, tokenize=False,
    )

    result = yield from _reasoner_chunks(
        _reasoner_model, _reasoner_tokenizer, prompt,
        max_tokens=4000, prompt_cache=_reasoner_cache,
    )

    _reasoner_history.append({"role": "assistant", "content": result})

    return result


def analyze(text, stream=True):
    """Analyze text using the GPT-OSS reasoner with conversation context."""
    result = _drain(_analyze_chunks(text), stream)
    if stream:
        print(flush=True)
    return result


def reset_context():
    """Reset reasoner conversation context."""
    global _reasoner_history, _reasoner_cache
    _reasoner_history = []
    if _reasoner_model is not None:
        _reasoner_cache = make_prompt_cache(_reasoner_model)
    console.print("  [yellow]\\[Context reset][/]\n")


@dataclass
class PipelineResult:
    """Everything one pipeline run produced.

    Carries its own evidence: a saved or re-served answer should say what was asked in
    English, what was searched, and where the material came from — not just the prose.
    """
    query: str
    english_query: str
    english_analysis: str
    korean_result: str
    search_performed: bool
    search_queries: dict   # {"ko": ..., "en": ...} as issued, empty when no search ran
    sources: list          # [{"title", "url", "lang"}], in the order the engines returned
    timings: dict          # seconds per stage, plus "total"


def _tokens(stage, chunks):
    """Re-yield generated chunks as token events, passing the return value through."""
    while True:
        try:
            chunk = next(chunks)
        except StopIteration as stop:
            return stop.value
        yield ("token", (stage, chunk))


def pipeline_events(query, force_search=None):
    """Triple-stage pipeline with optional web search, as an event stream.

    Yields (kind, payload) pairs and writes nothing to the terminal:

        ("stage", (num, total, label))   a stage boundary
        ("info", text)                   an indented progress line
        ("blank", None)                  a blank line
        ("busy", label) / ("done", None) generation begins / ends
        ("token", (stage, text))         a chunk, as the model produces it
        ("result", PipelineResult)       the last event, always

    force_search: True=always, False=never, None=auto (Qwen judges)
    """
    started = time.time()
    timings = {}

    # Stage 1: Translate + judge search
    yield ("stage", (1, 4, "Translating to English..."))
    t0 = time.time()
    english_query, needs_search = yield from _tokens(
        "ko2en", _translate_chunks(query, direction="ko2en"))
    timings["ko2en"] = round(time.time() - t0, 2)
    yield ("info", f"→ {english_query}")

    if force_search is not None:
        needs_search = force_search

    # Stage 2: Web search (if needed)
    search_context = ""
    search_queries = {}
    sources = []
    if needs_search:
        yield ("stage", (2, 4, "Searching web..."))
        t0 = time.time()
        from web_search import search_both, format_search_context, localize_en_query
        en_search_query = localize_en_query(english_query, judge_location(query))
        search_queries = {"ko": query, "en": en_search_query}
        ko_results, en_results = search_both(query, en_search_query)
        hit_count = len(ko_results) + len(en_results)
        yield ("info", f"→ {hit_count} results found")

        # Captured before the Korean results are collapsed below: that step keeps the
        # snippet text and drops every Korean URL, so this is the last point where the
        # answer's Korean evidence can still be attributed.
        sources = [dict(title=r["title"], url=r["url"], lang="ko") for r in ko_results]
        sources += [dict(title=r["title"], url=r["url"], lang="en") for r in en_results]

        # Translate Korean snippets to English for the reasoner
        if ko_results:
            yield ("info", "→ Translating Korean results to English...")
            ko_snippets = "\n".join(
                f"{i}. [{r['title']}] {r['snippet']}" for i, r in enumerate(ko_results, 1)
            )
            translated_snippets, _ = yield from _tokens(
                "ko-snippets", _translate_chunks(ko_snippets, direction="ko2en"))
            for i, r in enumerate(ko_results):
                r["snippet"] = ""  # clear original
            # Replace ko_results with single translated block
            ko_results = [{"title": "Korean sources (translated)", "url": "", "snippet": translated_snippets}]

        search_context = format_search_context(ko_results, en_results)
        timings["search"] = round(time.time() - t0, 2)
        yield ("blank", None)
    else:
        yield ("stage", (2, 4, "[dim]Search skipped[/]"))
        yield ("blank", None)

    # Stage 3: Reasoner
    yield ("stage", (3, 4, "GPT-OSS reasoning..."))
    if search_context:
        analysis_prompt = build_search_context_prompt(search_context, english_query)
    else:
        analysis_prompt = english_query
    yield ("busy", "reasoning...")
    t0 = time.time()
    english_analysis = yield from _tokens("reason", _analyze_chunks(analysis_prompt))
    timings["reason"] = round(time.time() - t0, 2)
    yield ("done", None)

    # Stage 4: Translate to Korean
    yield ("stage", (4, 4, "Translating to Korean..."))
    yield ("busy", "translating...")
    t0 = time.time()
    korean_result = yield from _tokens(
        "en2ko", _translate_chunks(english_analysis, direction="en2ko"))
    timings["en2ko"] = round(time.time() - t0, 2)
    yield ("done", None)

    yield ("result", PipelineResult(
        query=query, english_query=english_query,
        english_analysis=english_analysis, korean_result=korean_result,
        search_performed=bool(needs_search), search_queries=search_queries,
        sources=sources, timings=dict(timings, total=round(time.time() - started, 2)),
    ))


def pipeline(query, force_search=None, on_event=None):
    """Run the triple-stage pipeline and return its PipelineResult.

    Silent unless `on_event` is given — pass ConsoleRenderer() for the terminal output.
    """
    result = None
    for kind, payload in pipeline_events(query, force_search=force_search):
        if kind == "result":
            result = payload
        elif on_event is not None:
            on_event(kind, payload)
    return result


class ConsoleRenderer:
    """Turns pipeline events into the terminal output.

    Token events are dropped: the final block re-renders through Markdown(), so
    echoing them here would print the answer twice (#64 owns that choice).
    """

    def __init__(self):
        self._status = None

    def __call__(self, kind, payload):
        if kind == "stage":
            _stage(*payload)
        elif kind == "info":
            _info(payload)
        elif kind == "blank":
            console.print()
        elif kind == "busy":
            self._status = console.status(f"[cyan]{payload}[/]", spinner="dots")
            self._status.start()
        elif kind == "done":
            self.close()

    def close(self):
        """Stop any live spinner. Idempotent — also the caller's cleanup on error."""
        if self._status is not None:
            self._status.stop()
            self._status = None


def render_result(result):
    """Markdown for readability, no stream duplication."""
    console.print()
    console.print(Rule("English Analysis", style="blue"))
    console.print(Markdown(result.english_analysis))
    console.print(Rule("Korean Translation", style="blue"))
    console.print(Markdown(result.korean_result))


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
            renderer = ConsoleRenderer()
            try:
                result = pipeline(user_input, force_search=force_search,
                                  on_event=renderer)
            finally:
                renderer.close()  # a spinner outlives an exception otherwise
            render_result(result)
            console.print()

        if query is not None:
            break


if __name__ == "__main__":
    main()
