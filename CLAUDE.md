# CLAUDE.md

## Overview

로컬 LLM 삼단 파이프라인. Qwen3-14B (한영 번역) + DeepSeek R1 70B (영어 분석)을 동시 로딩하여 한자 혼입 없는 한국어 분석 결과물을 생성한다. 모델 스왑 없이 두 모델이 128GB 통합 메모리에 상시 적재.

## Architecture

### mlx-pipeline.py (Primary — mlx-lm 직접 추론)

- **DeepSeek R1 Distill Llama 70B 8-bit MLX** (~75GB): 영어 분석/추론 전용. S급 분석력.
- **Qwen3-14B 4-bit MLX** (~7.7GB): 한영 양방향 번역 전용. 한자 혼입 0% (10건 테스트 검증).
- 두 모델 동시 로딩 (~83GB). 128GB에서 여유 ~30GB.
- 삼단 파이프라인: 한→영 번역 → 영어 분석 → 영→한 번역.
- 선택적 웹 검색: Qwen3-14B가 번역 시 검색 필요 여부 판별 (SEARCH:yes/no). 검색 필요 시 Brave Search (한국어) + Tavily (영어) 병렬 실행. 한국어 검색 결과는 Qwen으로 영어 번역 후 DeepSeek 컨텍스트에 주입 (DeepSeek이 한국어 컨텍스트를 무시하는 이슈 우회).
- 대화 컨텍스트 유지: mlx-lm `prompt_cache`로 DeepSeek 대화 히스토리 누적 (KV 캐시 재사용, 이전 턴 재계산 없음). `/reset`으로 초기화.

### llm-pipeline.py (Legacy — LM Studio API)

- **LM Studio** (`localhost:1234`): OpenAI 호환 API 서버, MLX 백엔드
- DeepSeek R1 70B + Qwen 3 32B 이단 파이프라인 (모델 스왑 필요)
- LM Studio 앱 실행이 필요한 환경에서 사용

## Key Files

- `mlx-pipeline.py`: 삼단 파이프라인. mlx-lm 직접 추론. `pip install mlx-lm` 필요.
- `web_search.py`: 웹 검색 모듈. `brave_search()`, `tavily_search()`, `search_both()`, `format_search_context()` 제공.
- `llm-pipeline.py`: 이단 파이프라인 (레거시). LM Studio API. Python stdlib만 사용.

## README Convention

- README.md는 한국어/영문 양쪽 섹션을 동치로 유지해야 함.
- 한쪽을 수정하면 반드시 다른 쪽도 동일하게 갱신할 것.
- 상단에 `[English](#english) | [한국어](#한국어)` 앵커 링크 유지.

## Development Guidelines

- mlx-pipeline: `pip install mlx-lm` 필요 (venv 사용 권장)
- llm-pipeline: 외부 패키지 없이 Python stdlib만 사용
- 모델 경로: LM Studio 캐시(`~/.lmstudio/models/`) 우선, HuggingFace ID 폴백
- DeepSeek chat template: system role 대신 user message에 지시사항 병합 (mlx-lm에서 system role이 한국어 입력을 무시하는 이슈 우회)
- 시스템 프롬프트 수정: `DEEPSEEK_SYSTEM`, `TRANSLATE_KO_TO_EN`, `TRANSLATE_EN_TO_KO` 상수 편집
- 웹 검색 API 키: `BRAVE_API_KEY`, `TAVILY_API_KEY` 환경변수 (선택 — 미설정 시 검색 단계를 graceful skip)

## Known Issues

- Ollama M5 Max Metal 크래시 (ollama#14432): `brew install ollama`(소스 컴파일)에서만 발생. `brew install --cask ollama`(프리빌트 바이너리)로 설치하면 정상 동작.
- DeepSeek R1의 thinking 과정에서 중국어가 나오는 건 정상 (출력에서 `<think>` 블록을 필터링).
- DeepSeek R1 mlx-lm: chat template의 system role이 한국어 입력과 충돌 → user message에 병합하여 해결.
- Qwen 계열 한자 혼입 위험: Qwen3-14B 4-bit에서는 10건 테스트 결과 0건. 단, Qwen3.5-27B 4-bit에서는 중국어 성어 혼입 확인됨 — 모델 크기/양자화에 따라 달라질 수 있음.
- DeepSeek R1은 추론/분석 특화 모델. 사실 기반 지식 질의에서는 웹 검색 없이 가정법으로 답변하는 한계 있음 → 웹 검색 통합으로 해결.
- DeepSeek R1은 한국어 컨텍스트를 제대로 읽지 못함. 검색 결과가 한국어일 경우 반드시 영어로 번역 후 주입해야 함.
- LM Studio CLI(`lms`)는 PATH에 자동 등록 안 됨. 전체 경로: `/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms`

## Improvement Ideas

- Qwen 번역 프롬프트 튜닝 (용어집 확장, few-shot 예시 추가)
- 파이프라인 결과 파일 저장 옵션
- Hunyuan-MT 7B (WMT25 번역 대회 1위) MLX 변환 후 번역 모델 교체 검토
- 스트리밍 출력 개선 (번역 단계도 실시간 출력)
- 터미널 TUI 인터페이스 (Textual 기반) — #8
