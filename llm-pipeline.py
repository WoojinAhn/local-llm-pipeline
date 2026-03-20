#!/usr/bin/env python3
"""
로컬 LLM 듀얼 모델 파이프라인
- DeepSeek R1 70B: 영어로 분석/추론
- Qwen 3 32B: 한국어 번역 + 자연스러운 정리

사용법:
  python3 llm-pipeline.py "분석해줄 내용"
  python3 llm-pipeline.py                   # 대화형 모드
  python3 llm-pipeline.py --deepseek-only   # DeepSeek만 (영어 응답)
  python3 llm-pipeline.py --qwen-only       # Qwen만 (한국어 직접)
"""

import json
import sys
import urllib.request
import subprocess
import time

API_URL = "http://localhost:1234/v1/chat/completions"

DEEPSEEK_MODEL = "deepseek-r1-distill-llama-70b"
QWEN_MODEL = "qwen/qwen3-32b:2"

DEEPSEEK_SYSTEM = """You are an expert analyst. Respond ONLY in English.
Provide thorough, structured analysis with clear reasoning.
Be concise but comprehensive. Use bullet points and sections when appropriate."""

QWEN_SYSTEM = """You are a Korean translator and editor.
Translate the following English analysis into natural, fluent Korean (한국어).
Rules:
- Use pure Hangul only. Never use Chinese characters (漢字) or Japanese.
- Maintain the original structure (bullet points, sections).
- Use terms commonly used in Korean media and academia (e.g. "job displacement"→"일자리 대체", "universal basic income"→"기본소득", "self-service kiosk"→"무인 키오스크", "emotional intelligence"→"감성 지능").
- Avoid literal/word-for-word translation. Write as if originally authored in Korean.
- Do not add your own analysis — only translate and polish."""


def api_call(model: str, system: str, user: str, max_tokens: int = 2000) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[API 오류] {e}"


def get_loaded_model() -> str | None:
    try:
        result = subprocess.run(
            ["/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms", "ps"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout
        if DEEPSEEK_MODEL.split("/")[-1].split(":")[0] in output.lower():
            return "deepseek"
        if "qwen" in output.lower():
            return "qwen"
    except Exception:
        pass
    return None


def load_model(model_name: str) -> None:
    lms = "/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms"
    subprocess.run([lms, "unload", "--all"], capture_output=True, timeout=30)
    time.sleep(1)
    subprocess.run([lms, "load", model_name], capture_output=True, timeout=120)
    time.sleep(2)


def pipeline(query: str) -> str:
    # Step 1: DeepSeek R1 분석 (영어)
    current = get_loaded_model()
    if current != "deepseek":
        print("  [1/3] DeepSeek R1 로딩 중...", flush=True)
        load_model("deepseek-r1-distill-llama-70b")

    print("  [1/3] DeepSeek R1 분석 중... (영어)", flush=True)
    english_analysis = api_call(DEEPSEEK_MODEL, DEEPSEEK_SYSTEM, query)

    if english_analysis.startswith("[API 오류]"):
        return english_analysis

    # Step 2: Qwen 3 번역 (한국어)
    print("  [2/3] Qwen 3 로딩 중...", flush=True)
    load_model("qwen3-32b")

    print("  [3/3] Qwen 3 한국어 번역 중...", flush=True)
    korean_result = api_call(
        QWEN_MODEL,
        QWEN_SYSTEM,
        f"Translate this analysis to Korean:\n\n{english_analysis}",
    )

    return f"""{'='*60}
[DeepSeek R1 영어 원문]
{'='*60}
{english_analysis}

{'='*60}
[Qwen 3 한국어 번역]
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

    query = " ".join(args) if args else None

    if query is None:
        print("로컬 LLM 파이프라인 (종료: quit/exit)")
        print(f"모드: {mode}\n")

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
            current = get_loaded_model()
            if current != "deepseek":
                print("  DeepSeek R1 로딩 중...", flush=True)
                load_model("deepseek-r1-distill-llama-70b")
            print("  DeepSeek R1 분석 중...", flush=True)
            result = api_call(DEEPSEEK_MODEL, DEEPSEEK_SYSTEM, user_input)
        elif mode == "qwen":
            current = get_loaded_model()
            if current != "qwen":
                print("  Qwen 3 로딩 중...", flush=True)
                load_model("qwen3-32b")
            print("  Qwen 3 응답 중...", flush=True)
            result = api_call(
                QWEN_MODEL,
                "You are a helpful assistant. Always respond in Korean (한국어) using Hangul only.",
                user_input,
            )
        else:
            result = pipeline(user_input)

        print(f"\n{result}\n")

        if query is not None:
            break


if __name__ == "__main__":
    main()
