# CLAUDE.md

## Overview

로컬 LLM 듀얼 모델 파이프라인. DeepSeek R1 70B (분석) + Qwen 3 32B (한국어 번역)을 조합하여 한자 혼입 없는 한국어 분석 결과물을 생성한다.

## Architecture

- **LM Studio** (`localhost:1234`): OpenAI 호환 API 서버, MLX 백엔드
- **DeepSeek R1 Distill Llama 70B 8-bit MLX**: 영어 분석/추론 전용. 한국어 직접 출력 시 중국어/일본어 한자가 혼입되는 구조적 문제가 있어 영어로만 사용.
- **Qwen 3 32B 4-bit MLX**: 한국어 번역 및 한국어 직접 대화용. 한자 혼입 없음.
- LM Studio는 동시에 하나의 LLM만 메모리에 로딩 가능. 모델 전환 시 unload → load 필요 (~10초).

## Key Files

- `llm-pipeline.py`: 메인 파이프라인 스크립트. 외부 의존성 없이 Python stdlib만 사용.

## README Convention

- README.md는 한국어/영문 양쪽 섹션을 동치로 유지해야 함.
- 한쪽을 수정하면 반드시 다른 쪽도 동일하게 갱신할 것.
- 상단에 `[English](#english) | [한국어](#한국어)` 앵커 링크 유지.

## Development Guidelines

- 외부 패키지 의존성 최소화 (pip install 없이 동작해야 함)
- LM Studio API는 OpenAI 호환 (`/v1/chat/completions`)
- 모델 식별자: `deepseek-r1-distill-llama-70b`, `qwen/qwen3-32b:2`
- 시스템 프롬프트 수정 시 `DEEPSEEK_SYSTEM`, `QWEN_SYSTEM` 상수 편집

## Known Issues

- Ollama는 M5 Max Metal 크래시로 사용 불가 (ollama#14432). LM Studio만 사용.
- DeepSeek R1의 thinking 과정에서 중국어가 나오는 건 정상 (출력에만 안 섞이면 됨).
- Qwen 번역 품질은 기능적 수준. 직역 투가 간혹 나타남 — QWEN_SYSTEM 프롬프트 개선으로 완화 가능.
- LM Studio CLI(`lms`)는 PATH에 자동 등록 안 됨. 전체 경로: `/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms`

## Improvement Ideas

- 모델 동시 로딩이 가능한 환경(vLLM, mlx-lm 직접 사용 등)으로 전환 시 스왑 시간 제거 가능
- 스트리밍 출력 지원
- Qwen 번역 프롬프트 튜닝 (용어집 확장, few-shot 예시 추가)
- 파이프라인 결과 파일 저장 옵션
