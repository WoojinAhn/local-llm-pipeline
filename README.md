# local-llm-pipeline

MacBook Pro M5 Max (128GB)에서 로컬 LLM 듀얼 모델 파이프라인.

DeepSeek R1 70B로 영어 분석 → Qwen 3 32B로 한국어 번역. 중국어/일본어 한자 혼입 없는 순수 한글 결과물을 얻기 위한 구조.

## 왜 듀얼 모델인가

| 모델 | 강점 | 약점 |
|------|------|------|
| DeepSeek R1 70B 8-bit | 추론/분석 최강 | 한국어 출력 시 한자 혼입 |
| Qwen 3 32B 4-bit | 한국어 자연스러움 | 분석 깊이 한계 |

각자의 강점만 조합: **DeepSeek이 영어로 깊이 분석 → Qwen이 한국어로 자연스럽게 번역**.

## 요구 사항

- macOS + Apple Silicon (M-series)
- [LM Studio](https://lmstudio.ai/) 설치 및 서버 실행 (`localhost:1234`)
- 모델 다운로드:
  - `lms get deepseek-r1-distill-llama-70b@8bit --mlx -y`
  - `lms get qwen3-32b --mlx -y`
- Python 3.10+

## 사용법

```bash
# 파이프라인 (DeepSeek 분석 → Qwen 번역)
python3 llm-pipeline.py "인공지능이 노동 시장에 미치는 영향을 분석해줘"

# 대화형 모드
python3 llm-pipeline.py

# DeepSeek만 (영어 응답)
python3 llm-pipeline.py --deepseek-only "Analyze the impact of AI on labor markets"

# Qwen만 (한국어 직접)
python3 llm-pipeline.py --qwen-only "오늘 할 일 정리해줘"
```

## 구조

```
User Input (한국어/영어)
    │
    ▼
[DeepSeek R1 70B] ── 영어로 분석 (~70GB 메모리)
    │
    ▼
[모델 스왑] ── ~10초
    │
    ▼
[Qwen 3 32B] ── 한국어 번역 (~18GB 메모리)
    │
    ▼
Output: 영어 원문 + 한국어 번역
```

## 제한 사항

- 모델 스왑에 ~10초 소요 (LM Studio는 동시에 하나의 LLM만 로딩)
- DeepSeek R1은 "thinking" 시간이 있어 복잡한 질문일수록 응답 지연
- Qwen 번역은 기능적 수준 (전문 번역가 수준은 아님)
- Ollama는 M5 Max Metal 크래시 이슈로 사용 불가 ([ollama#14432](https://github.com/ollama/ollama/issues/14432))

## License

MIT
