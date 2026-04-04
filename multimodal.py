#!/usr/bin/env python3
"""
Standalone multimodal script using mlx-vlm + Gemma 4.

Accepts text + image input, outputs analysis.
Optional web search (Brave + Tavily) for factual queries.
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

from web_search import search_both, format_search_context

_LOCAL_MODEL = os.path.expanduser("~/.cache/huggingface/hub/mlx-community--gemma-4-31b-it-4bit")
MODEL_ID = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else "mlx-community/gemma-4-31b-it-4bit"

SEARCH_JUDGE_PROMPT = """\
Does the following user query require up-to-date factual knowledge \
(recent events, current statistics, specific people/organizations, news, prices, dates) \
to answer accurately? Reply with ONLY "SEARCH:yes" or "SEARCH:no".

Query: {query}"""


def load_model():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, processor = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


def needs_search(model, processor, config, query):
    """Ask Gemma 4 whether this query needs web search."""
    prompt = SEARCH_JUDGE_PROMPT.format(query=query)
    formatted = apply_chat_template(processor, config, prompt, num_images=0)
    result = generate(model, processor, formatted, max_tokens=20, temperature=0.0, verbose=False)
    return "yes" in result.text.lower()


def do_search(query):
    """Run Brave (Korean) + Tavily (English) search in parallel."""
    print("  Searching...", end="", flush=True)
    t0 = time.time()
    ko_results, en_results = search_both(query, query)
    elapsed = time.time() - t0
    total = len(ko_results) + len(en_results)
    print(f" {total} results ({elapsed:.1f}s)")
    return format_search_context(ko_results, en_results)


def run_query(model, processor, config, prompt, image=None,
              max_tokens=2048, search_enabled=True):
    images = [image] if image else None
    num_images = 1 if image else 0

    # Search step (text-only, no image queries)
    search_context = ""
    if search_enabled and not image:
        if needs_search(model, processor, config, prompt):
            search_context = do_search(prompt)

    # Build final prompt with search context
    if search_context:
        full_prompt = (
            f"Use the following search results to answer accurately. "
            f"Cite sources when possible.\n\n"
            f"--- Search Results ---\n{search_context}\n"
            f"--- End Search Results ---\n\n{prompt}"
        )
    else:
        full_prompt = prompt

    formatted = apply_chat_template(
        processor, config, full_prompt, num_images=num_images,
    )

    print("\n--- Response ---")
    t0 = time.time()
    token_count = 0

    for result in stream_generate(
        model, processor, formatted,
        image=images,
        max_tokens=max_tokens,
        temperature=0.7,
    ):
        text = result.text if hasattr(result, 'text') else str(result)
        print(text, end="", flush=True)
        token_count += 1

    elapsed = time.time() - t0
    tps = token_count / elapsed if elapsed > 0 else 0
    print(f"\n--- {token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s) ---\n")


def interactive_mode(model, processor, config, search_enabled=True):
    search_status = "on" if search_enabled else "off"
    print(f"\nInteractive mode (search: {search_status}). Commands:")
    print("  /image <path>  — set image for next query")
    print("  /clear         — clear current image")
    print("  /search        — toggle web search on/off")
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
