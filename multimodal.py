#!/usr/bin/env python3
"""
Standalone multimodal script using mlx-vlm + Gemma 4.

Accepts text + image input, outputs analysis.
Optional web search (Brave + Tavily) with query rewriting.
Gemma 4 31B 4-bit (~17GB) runs comfortably on 128GB unified memory.

Usage:
  python3 multimodal.py "Describe this image" --image photo.jpg
  python3 multimodal.py "What's in this screenshot?" --image screenshot.png
  python3 multimodal.py                                # interactive mode
  python3 multimodal.py --text-only "Explain quantum computing"
  python3 multimodal.py --no-search "Analyze this argument"
"""

import argparse
import os
import sys
import time

import mlx.core as mx
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from prompts import (
    build_search_context_prompt,
    current_date_context,
    filter_thinking_gemma,
    gemma_system,
    parse_search_judge,
    search_judge_prompt,
)
from web_search import search_both, format_search_context

_LOCAL_MODEL = os.path.expanduser("~/.cache/huggingface/hub/mlx-community--gemma-4-31b-it-4bit")
MODEL_ID = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else "mlx-community/gemma-4-31b-it-4bit"


def load_model():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, processor = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


def judge_and_search(model, processor, config, query):
    """Judge search need and generate optimized queries in one call."""
    prompt = search_judge_prompt(query)
    formatted = apply_chat_template(processor, config, prompt, num_images=0)
    result = generate(model, processor, formatted, max_tokens=100, temperature=0.0, verbose=False)

    raw = filter_thinking_gemma(result.text)
    needs_search, ko_query, en_query = parse_search_judge(raw)

    if not needs_search:
        return ""

    # Fallback to original query if parsing failed
    ko_query = ko_query or query
    en_query = en_query or query

    print(f"  Searching (KO: {ko_query})")
    print(f"           (EN: {en_query})")
    t0 = time.time()
    ko_results, en_results = search_both(ko_query, en_query)
    elapsed = time.time() - t0
    total = len(ko_results) + len(en_results)
    print(f"  → {total} results ({elapsed:.1f}s)")

    return format_search_context(ko_results, en_results)


_conversation_history = []


def reset_context():
    """Reset conversation history."""
    global _conversation_history
    _conversation_history = []
    print("  [Context reset]\n")


def run_query(model, processor, config, prompt, image=None,
              max_tokens=2048, search_enabled=True):
    global _conversation_history

    images = [image] if image else None
    num_images = 1 if image else 0

    # Search step (text-only, no image queries)
    search_context = ""
    if search_enabled and not image:
        search_context = judge_and_search(model, processor, config, prompt)

    # Build user message with date + search context
    date_prefix = f"[{current_date_context()}]\n\n"
    if search_context:
        user_msg = date_prefix + build_search_context_prompt(search_context, prompt)
    else:
        user_msg = date_prefix + prompt

    # Build multi-turn conversation
    _conversation_history.append({"role": "user", "content": user_msg})

    formatted = apply_chat_template(
        processor, config, _conversation_history, num_images=num_images,
    )

    print("\n--- Response ---")
    t0 = time.time()
    raw_parts = []
    in_thinking = False

    for result in stream_generate(
        model, processor, formatted,
        image=images,
        max_tokens=max_tokens,
        temperature=0.7,
    ):
        text = result.text if hasattr(result, 'text') else str(result)
        raw_parts.append(text)

        # Suppress thinking blocks from streaming output
        if "<|channel>thought" in text:
            in_thinking = True
        if in_thinking:
            if "<channel|>" in text:
                in_thinking = False
            continue

        print(text, end="", flush=True)

    # Final cleanup via regex (handles token-boundary edge cases)
    full_text = filter_thinking_gemma("".join(raw_parts))
    elapsed = time.time() - t0
    token_count = len(raw_parts)
    tps = token_count / elapsed if elapsed > 0 else 0
    print(f"\n--- {token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s) ---\n")

    # Append assistant response to history
    _conversation_history.append({"role": "assistant", "content": full_text})


def interactive_mode(model, processor, config, search_enabled=True):
    search_status = "on" if search_enabled else "off"
    print(f"\nInteractive mode (search: {search_status}). Commands:")
    print("  /image <path>  — set image for next query")
    print("  /clear         — clear current image")
    print("  /search        — toggle web search on/off")
    print("  /reset         — reset conversation context")
    print("  /quit          — exit")
    print()

    current_image = None

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt == "/quit":
            break
        elif prompt.startswith("/image "):
            current_image = prompt[7:].strip()
            print(f"Image set: {current_image}")
            continue
        elif prompt == "/clear":
            current_image = None
            print("Image cleared.")
            continue
        elif prompt == "/search":
            search_enabled = not search_enabled
            print(f"Search: {'on' if search_enabled else 'off'}")
            continue
        elif prompt == "/reset":
            reset_context()
            continue

        run_query(model, processor, config, prompt,
                  image=current_image, search_enabled=search_enabled)


def main():
    parser = argparse.ArgumentParser(description="Multimodal inference with Gemma 4 + mlx-vlm")
    parser.add_argument("prompt", nargs="?", help="Text prompt")
    parser.add_argument("--image", "-i", help="Path or URL to image")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--text-only", action="store_true", help="Text-only mode (no image)")
    parser.add_argument("--no-search", action="store_true", help="Disable web search")
    args = parser.parse_args()

    model, processor = load_model()
    config = load_config(MODEL_ID)
    search_enabled = not args.no_search

    if args.prompt:
        run_query(model, processor, config, args.prompt,
                  image=None if args.text_only else args.image,
                  max_tokens=args.max_tokens, search_enabled=search_enabled)
    else:
        interactive_mode(model, processor, config, search_enabled=search_enabled)


if __name__ == "__main__":
    main()
