# local-llm-pipeline

[English](#english) | [한국어](#한국어)

---

## 한국어

MacBook Pro M5 Max (128GB)에서 로컬 LLM 듀얼 모델 파이프라인.

DeepSeek R1 70B로 영어 분석 → Qwen 3 32B로 한국어 번역. 중국어/일본어 한자 혼입 없는 순수 한글 결과물을 얻기 위한 구조.

### 장비 스펙

| 항목 | 사양 |
|------|------|
| 모델 | MacBook Pro (Mac17,6) |
| 칩 | Apple M5 Max |
| 메모리 | 128GB Unified Memory |
| 스토리지 | 4TB SSD |
| 메모리 대역폭 | 614 GB/s |
| OS | macOS 15 (Sequoia) |

### 왜 이 조합인가 — 의사결정 과정

**1. Ollama vs LM Studio**

처음에 Ollama를 설치했으나, M5 Max에서 Metal 백엔드 크래시 발생 ([ollama#14432](https://github.com/ollama/ollama/issues/14432)). 모든 모델이 `exit status 2`로 로딩 실패. LM Studio의 MLX 백엔드는 M5 Max에서 정상 동작하여 LM Studio로 전환.

**2. 모델 선택 — 분석 용도**

128GB 메모리에서 돌릴 수 있는 분석용 모델을 비교:

| 모델 | 양자화 | 메모리 | 분석력 | 한국어 | 선택 |
|------|--------|--------|--------|--------|------|
| Qwen 3 32B | 4-bit | ~18GB | A | A | 메모리 낭비 |
| Qwen 2.5 72B | 8-bit | ~75GB | A | A+ | 분석 깊이 부족 |
| DeepSeek R1 70B | 8-bit | ~70GB | **S** | B+ | **채택** |
| Mistral Large 123B | 4-bit | ~70GB | A+ | B- | 범용이라 R1보다 분석 약함 |
| Qwen 3 235B-A22B | 4-bit | ~130GB | S+ | A | 128GB에 안 들어감 |

DeepSeek R1 70B 8-bit를 채택한 이유: 같은 ~70GB 메모리로 추론 특화 모델(R1)이 범용 대형 모델(123B)보다 분석에서 우위. 8-bit 양자화로 품질 손실 최소화.

**3. 듀얼 모델 파이프라인 — 한국어 문제 해결**

DeepSeek R1에 한국어 출력을 시도했으나, 시스템 프롬프트를 아무리 강화해도 중국어 한자(`通常`, `季節`, `数值予報模型` 등)가 구조적으로 혼입됨. 이는 모델이 내부적으로 중국어로 "사고"하기 때문.

해결: DeepSeek은 영어로만 분석하고, Qwen 3가 한국어로 번역하는 파이프라인 구축. 결과물에서 한자 혼입 0% 달성.

**4. Qwen 3 32B — 왜 32B인가**

Qwen 공식 벤치마크에서 Qwen 3 32B ≈ Qwen 2.5 72B 성능 (학습 데이터 2배 + thinking 모드). 72B를 4-bit로 돌리는 것보다 32B를 그대로 쓰는 게 효율적. 번역 용도로는 충분.

### 요구 사항

- macOS + Apple Silicon (M-series)
- [LM Studio](https://lmstudio.ai/) 설치 및 서버 실행 (`localhost:1234`)
- 모델 다운로드:
  - `lms get deepseek-r1-distill-llama-70b@8bit --mlx -y`
  - `lms get qwen3-32b --mlx -y`
- Python 3.10+

### 사용법

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

### 구조

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

### 제한 사항

- 모델 스왑에 ~10초 소요 (LM Studio는 동시에 하나의 LLM만 로딩)
- DeepSeek R1은 "thinking" 시간이 있어 복잡한 질문일수록 응답 지연
- Qwen 번역은 기능적 수준 (전문 번역가 수준은 아님, 하지만 의미 전달 충분)
- Ollama는 M5 Max Metal 크래시 이슈로 사용 불가 ([ollama#14432](https://github.com/ollama/ollama/issues/14432))

---

## English

Dual local LLM pipeline on MacBook Pro M5 Max (128GB).

DeepSeek R1 70B analyzes in English → Qwen 3 32B translates to Korean. This architecture produces pure Hangul output without Chinese/Japanese character contamination.

### Hardware

| Spec | Value |
|------|-------|
| Machine | MacBook Pro (Mac17,6) |
| Chip | Apple M5 Max |
| Memory | 128GB Unified Memory |
| Storage | 4TB SSD |
| Memory Bandwidth | 614 GB/s |
| OS | macOS 15 (Sequoia) |

### Why Dual Models — Decision Log

**1. Ollama vs LM Studio**

Initially installed Ollama, but it crashes on M5 Max due to a Metal backend issue ([ollama#14432](https://github.com/ollama/ollama/issues/14432)). All models fail with `exit status 2`. LM Studio's MLX backend works correctly on M5 Max.

**2. Model Selection — Analysis Use Case**

Compared models that fit in 128GB for deep analysis:

| Model | Quant | Memory | Analysis | Korean | Decision |
|-------|-------|--------|----------|--------|----------|
| Qwen 3 32B | 4-bit | ~18GB | A | A | Underutilizes hardware |
| Qwen 2.5 72B | 8-bit | ~75GB | A | A+ | Insufficient analysis depth |
| DeepSeek R1 70B | 8-bit | ~70GB | **S** | B+ | **Selected** |
| Mistral Large 123B | 4-bit | ~70GB | A+ | B- | General-purpose, weaker than R1 for analysis |
| Qwen 3 235B-A22B | 4-bit | ~130GB | S+ | A | Doesn't fit in 128GB |

DeepSeek R1 70B 8-bit was chosen because a reasoning-specialized model (R1) outperforms a larger general-purpose model (123B) for analysis tasks at the same memory footprint. 8-bit quantization minimizes quality loss.

**3. Dual Pipeline — Solving the Korean Output Problem**

When DeepSeek R1 outputs Korean directly, Chinese characters (`通常`, `季節`, `数值予報模型`) leak into the response regardless of system prompt engineering. This is structural — the model "thinks" in Chinese internally.

Solution: DeepSeek analyzes in English only, Qwen 3 translates to Korean. Result: 0% Chinese character contamination.

**4. Why Qwen 3 32B (not 72B)**

Per Qwen's official benchmarks, Qwen 3 32B ≈ Qwen 2.5 72B in performance (2x training data + thinking mode). Running 32B is more efficient than running 72B at 4-bit for translation purposes.

### Requirements

- macOS + Apple Silicon (M-series)
- [LM Studio](https://lmstudio.ai/) with server running (`localhost:1234`)
- Models:
  - `lms get deepseek-r1-distill-llama-70b@8bit --mlx -y`
  - `lms get qwen3-32b --mlx -y`
- Python 3.10+

### Usage

```bash
# Full pipeline (DeepSeek analysis → Qwen translation)
python3 llm-pipeline.py "Analyze the impact of AI on labor markets"

# Interactive mode
python3 llm-pipeline.py

# DeepSeek only (English response)
python3 llm-pipeline.py --deepseek-only "question"

# Qwen only (Korean response)
python3 llm-pipeline.py --qwen-only "질문"
```

### Architecture

```
User Input (Korean/English)
    │
    ▼
[DeepSeek R1 70B] ── English analysis (~70GB memory)
    │
    ▼
[Model swap] ── ~10s
    │
    ▼
[Qwen 3 32B] ── Korean translation (~18GB memory)
    │
    ▼
Output: English original + Korean translation
```

### Limitations

- ~10s model swap overhead (LM Studio loads one LLM at a time)
- DeepSeek R1 has "thinking" latency — longer for complex queries
- Qwen translation is functional, not professional-grade (but sufficient for comprehension)
- Ollama unusable on M5 Max due to Metal crash ([ollama#14432](https://github.com/ollama/ollama/issues/14432))

## License

MIT
