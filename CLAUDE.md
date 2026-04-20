# CLAUDE.md

## Overview

로컬 LLM 파이프라인. 128GB 통합 메모리에서 모델 스왑 없이 동작하는 두 가지 파이프라인 제공:

1. **mlx-pipeline** (삼단): Qwen3-14B (번역) + GPT-OSS 120B (분석). 한자 혼입 없는 한국어 분석 결과물 생성.
2. **multimodal** (단일 모델): Gemma 4 31B. 텍스트+이미지 멀티모달 분석. 한국어 네이티브 지원으로 번역 불필요.

## Architecture

### mlx-pipeline.py (Primary — mlx-lm 직접 추론)

- **GPT-OSS 120B 4-bit MLX** (~65GB): 영어 분석/추론 전용. MoE (총 116.8B / active 5.1B), 128K 컨텍스트, MMLU-Pro 90.0.
- **Qwen3-14B 4-bit MLX** (~7.7GB): 한영 양방향 번역 전용. 한자 혼입 0% (10건 테스트 검증).
- 두 모델 동시 로딩 (~73GB). 128GB에서 여유 ~55GB.
- 삼단 파이프라인: 한→영 번역 → 영어 분석 → 영→한 번역.
- 선택적 웹 검색: Qwen3-14B가 번역 시 검색 필요 여부 판별 (SEARCH:yes/no). 검색 필요 시 Brave Search (한국어) + Tavily (영어) 병렬 실행. 한국어 검색 결과는 Qwen으로 영어 번역 후 분석 컨텍스트에 주입 (추론 모델이 한국어를 네이티브로 처리하지 않으므로 우회).
- 대화 컨텍스트 유지: mlx-lm `prompt_cache`로 추론 모델 대화 히스토리 누적 (KV 캐시 재사용, 이전 턴 재계산 없음). `/reset`으로 초기화.
- 출력 필터링: GPT-OSS harmony 포맷의 `analysis` 채널은 숨기고 `final` 채널만 표시.
- 출력 렌더링: Rich 라이브러리로 단계 표시(색상), 최종 분석/번역 결과(Markdown), 구분선(Rule) 렌더. `pipeline` 모드는 스피너+최종 Markdown, `--reasoner-only` 모드는 스트리밍+Rule 구분.

### multimodal.py (Gemma 4 — mlx-vlm 직접 추론)

- **Gemma 4 31B 4-bit MLX** (~17GB): 텍스트+이미지 멀티모달. 한국어 네이티브, 번역 파이프라인 불필요.
- 단일 모델 로딩 (~17GB). 128GB에서 여유 ~110GB.
- 선택적 웹 검색: Gemma 4가 검색 필요 여부 판별 + 검색 쿼리 리라이팅 (날짜 해석, 키워드 최적화) 한 번의 추론으로 처리. Brave (한국어) + Tavily (영어) 병렬 검색.
- 날짜 인식: 시스템 프롬프트에 현재 날짜/요일 자동 주입.
- thinking 필터링: Gemma 4 `<|channel>thought` 블록 자동 제거.

### FLUX.2 이미지 생성 (flux-2-swift-mlx)

- **FLUX.2 Klein 4B / Dev 32B**: 텍스트→이미지, 이미지→이미지 생성. MLX 네이티브 Swift 구현.
- `setup-flux.sh`로 Xcode 소스 빌드 필요 (프리빌트 바이너리는 metallib 누락 이슈).
- Gemma 4 (17GB) + FLUX.2-dev int4 (~32GB) = ~49GB. 128GB에서 동시 실행 가능.
- 모델 자동 다운로드: `~/Library/Caches/models/`에 캐시.
- FLUX.2-dev는 HuggingFace gated model — HF 토큰 + 모델 페이지에서 access 승인 필요.

### llm-pipeline.py (Legacy — LM Studio API)

- **LM Studio** (`localhost:1234`): OpenAI 호환 API 서버, MLX 백엔드
- DeepSeek R1 70B + Qwen 3 32B 이단 파이프라인 (모델 스왑 필요)
- LM Studio 앱 실행이 필요한 환경에서 사용

## Key Files

- `mlx-pipeline.py`: 삼단 파이프라인. mlx-lm 직접 추론. 의존성: `mlx-lm`, `rich` (`pip install -r requirements.txt`).
- `multimodal.py`: Gemma 4 멀티모달 파이프라인. mlx-vlm 직접 추론. `pip install mlx-vlm` 필요.
- `prompts.py`: 공유 프롬프트 모듈. 날짜 주입, 검색 판별/쿼리 생성, 인용 강제, thinking 필터 등.
- `web_search.py`: 웹 검색 모듈. `brave_search()`, `tavily_search()`, `search_both()`, `format_search_context()` 제공.
- `setup-flux.sh`: FLUX.2 CLI 빌드 스크립트. Metal Toolchain + Xcode 소스 빌드 + 바이너리 설치.
- `llm-pipeline.py`: 이단 파이프라인 (레거시). LM Studio API. Python stdlib만 사용.

## README Convention

- README.md는 한국어/영문 양쪽 섹션을 동치로 유지해야 함.
- 한쪽을 수정하면 반드시 다른 쪽도 동일하게 갱신할 것.
- 상단에 `[English](#english) | [한국어](#한국어)` 앵커 링크 유지.

## Development Guidelines

- mlx-pipeline: `pip install -r requirements.txt` (mlx-lm + rich, venv 사용 권장)
- multimodal: `pip install mlx-vlm` 필요 (venv 사용 권장). Gemma 4 모델은 HuggingFace에서 다운로드 (HF 토큰 권장).
- llm-pipeline: 외부 패키지 없이 Python stdlib만 사용
- 모델 경로: LM Studio 캐시(`~/.lmstudio/models/`) 우선, HuggingFace 캐시(`~/.cache/huggingface/hub/`), HuggingFace ID 폴백
- GPT-OSS는 harmony 포맷 — `<|channel|>analysis<|message|>…<|end|>` 에 CoT를, `<|channel|>final<|message|>…` 에 최종 답변을 출력. `filter_thinking_harmony()` 로 `final` 채널만 추출
- 시스템 프롬프트 수정: `prompts.py` 내 상수/함수 편집 (양쪽 파이프라인 공유)
- 웹 검색 API 키: `BRAVE_API_KEY`, `TAVILY_API_KEY` 환경변수 (선택 — 미설정 시 검색 단계를 graceful skip)

## Known Issues

- Ollama M5 Max Metal 크래시 (ollama#14432): `brew install ollama`(소스 컴파일)에서만 발생. `brew install --cask ollama`(프리빌트 바이너리)로 설치하면 정상 동작.
- GPT-OSS 120B MoE: active 5.1B이라 추론 속도는 빠르지만, 총 가중치 ~65GB + KV 캐시로 긴 컨텍스트 사용 시 128GB에서 메모리 압박 가능성 있음 (커뮤니티 보고 편차 있음).
- Qwen 계열 한자 혼입 위험: Qwen3-14B 4-bit에서는 10건 테스트 결과 0건. 단, Qwen3.5-27B 4-bit에서는 중국어 성어 혼입 확인됨 — 모델 크기/양자화에 따라 달라질 수 있음.
- GPT-OSS 120B는 한국어 네이티브가 아님 → 삼단 구조(Qwen 번역 래퍼) 유지 필요.
- 추론 모델은 사실 기반 지식 질의에서 웹 검색 없이 가정법으로 답변하는 한계 있음 → 웹 검색 통합으로 해결.
- 추론 모델에 한국어 검색 결과를 직접 주입하면 무시할 수 있음. 검색 결과가 한국어일 경우 반드시 영어로 번역 후 주입.
- LM Studio CLI(`lms`)는 PATH에 자동 등록 안 됨. 전체 경로: `/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms`
- mlx-lm은 gemma4 아키텍처를 아직 미지원 (0.31.1 기준). Gemma 4 실행에는 mlx-vlm 필요.
- Gemma 4 모델 다운로드 시 HuggingFace 토큰 없으면 rate limit 발생. `HF_TOKEN` 환경변수 설정 권장.

## Improvement Ideas

- Qwen 번역 프롬프트 튜닝 (용어집 확장, few-shot 예시 추가)
- 파이프라인 결과 파일 저장 옵션
- Hunyuan-MT 7B (WMT25 번역 대회 1위) MLX 변환 후 번역 모델 교체 검토
- 스트리밍 출력 개선 (번역 단계도 실시간 출력)
- 터미널 TUI 인터페이스 (Textual 기반) — #8
