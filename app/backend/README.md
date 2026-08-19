# ChainEye (체인아이) — Backend API

Bitcoin 자금세탁 탐지 도구의 백엔드 API. 2026 금융 AI Challenge.

## 실행 (Run)

저장소 루트에서 실행하세요:

```
python -m pip install -r deploy/requirements.txt
python -m uvicorn app.backend.main:app --port 8000
```

- CORS 허용 오리진 기본값: `http://localhost:5173`, `http://localhost:3000`
  (`CHAINEYE_CORS_ORIGINS`의 쉼표 구분 목록으로 변경 가능)
- Swagger 문서: http://localhost:8000/docs

## 모델 연동 (MODEL / MOCK 모드)

기동 시 `app/ml/inference.py` 를 import 시도합니다.

- **MODEL 모드**: import + `inference.load()` 성공 시. `/score`, `/trace` 는
  `inference.score_tx`, `inference.trace_tx` 로 위임됩니다.
- **MOCK 모드**: import/load 실패 시(모델 미완성). 내장 mock provider가 동일한
  스키마의 결정론적 샘플 데이터를 반환하며 `/health` 의 `modelLoaded=false`, `mode="mock"`.
- MODEL 모드 요청 중 추론 오류가 발생하면 503을 반환한다. 실제 모델 장애를 임의의
  MOCK 점수로 위장하지 않는다.

어느 모드로 떴는지는 시작 로그에 출력됩니다.

## 엔드포인트

| Method | Path      | 설명 |
|--------|-----------|------|
| GET    | /health   | 상태, 모델 로드 여부, 활성 모드 |
| GET    | /model-info | 실제 모델의 데이터셋, 평가 지표, 검증 방식, 판정 임계값 |
| POST   | /score    | 거래 위험 점수(0-100) + 라벨 + topFactors |
| POST   | /trace    | 자금흐름 그래프(nodes/edges), hops 기본 2 |
| POST   | /explain  | topFactors 설명(스코어 factors 재사용) |
| POST   | /report   | 한글 컴플라이언스 보고서 (Claude LLM → 템플릿 폴백) |

### 요청/응답 예시

```
GET /health
-> {"status":"ok","modelLoaded":false,"mode":"mock"}

GET /model-info
-> {"active":true, "activeModel":"LightGBM with graph-neighbor aggregates",
    "featureCount":165, "localFeatureCount":93, "neighborAggregateFeatureCount":72,
    "illicitF1":0.805076, "prAuc":0.799495,
    "localOnlyF1":0.743989, "graphF1Lift":0.061087,
    "decisionThreshold":0.523810, "validationProtocol":"rolling temporal validation and untouched future test", ...}

POST /score  {"txId":"232629023"}
-> {"txId":"232629023","riskScore":100,"label":"illicit",
    "topFactors":[{"feature":"mixer_exposure_ratio","impact":0.83}, ...]}

POST /trace  {"txId":"232629023","hops":2}
-> {"nodes":[{"id":"232629023","risk":100,"focus":true,"illicit":true}, ...],
    "edges":[...],"paths":[["232629023", ...]]}

POST /explain {"txId":"232629023"}
-> {"txId":"232629023","topFactors":[{"feature":"...","impact":0.83}, ...]}

POST /report  {"txId":"232629023","score":100,"label":"illicit",
               "topFactors":[...],
               "graphStats":{"nodeCount":9,"edgeCount":8,"highRiskCount":2}}
-> {"report":"# 자금세탁 위험 분석 보고서 ..."}
```

## 보고서 생성 (LLM → 템플릿 폴백)

`/report` 는 두 단계로 동작합니다:

1. **1순위 — LLM(선택형 provider)**: Claude(`claude_report`) 또는 OpenAI GPT
   (`openai_report`) SDK 로 FIU/컴플라이언스 스타일의 전문 한글 보고서를 생성합니다.
   제공된 수치·라벨·그래프 통계만을 근거로 삼도록(환각 금지) 프롬프트가 구성되며,
   템플릿과 동일한 4개 섹션 구조(위험 요약 / 핵심 판단 근거 / 자금흐름 관찰 / 권고
   조치)를 따릅니다. 프롬프트는 두 provider 가 동일한 것을 공유합니다.
2. **폴백 — 템플릿**: LLM 경로가 `None` 을 반환하면 `report_builder.build_report()`
   의 결정론적 템플릿으로 폴백합니다. 템플릿 경로는 항상 동작합니다.

> **API 키는 필수가 아닙니다.** 키가 전혀 없어도(심사/오프라인 환경) `/report` 는
> 100% 템플릿으로 동작합니다 — 외부 의존성·비용·네트워크 0, 결정론적. 키가 있을
> 때만 고급 LLM 보고서로 자동 업그레이드됩니다.

모든 LLM 경로는 **우아하게 실패(graceful degradation)** 합니다. 다음 경우 예외를
던지지 않고 `None` 을 반환하여 자동으로 템플릿으로 폴백합니다:

- 해당 SDK 패키지 미설치(ImportError) — `anthropic` / `openai`
- 해당 API 키 환경변수 미설정 — `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
- API 호출 중 임의의 오류(네트워크/인증/속도제한 등)

어느 경로가 실행됐는지는 로그에 출력됩니다
(`/report served via LLM (Claude).`, `/report served via LLM (OpenAI).`,
또는 `/report served via template fallback.`).
응답 스키마(`{"report": str}`)는 경로와 무관하게 동일합니다.

### 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CHAINEYE_REPORT_PROVIDER` | `claude` | `claude` / `openai` / `auto` / `template`. 알 수 없는 값은 안전하게 템플릿 처리. |
| `ANTHROPIC_API_KEY`     | (없음)         | Anthropic API 키. **미설정 시 Claude 경로 비활성 → 템플릿 폴백.** |
| `CHAINEYE_REPORT_MODEL` | `claude-opus-5` | Claude 보고서 모델 ID. |
| `OPENAI_API_KEY`        | (없음)         | OpenAI API 키. **미설정 시 OpenAI 경로 비활성 → 템플릿 폴백.** |
| `CHAINEYE_OPENAI_MODEL` | `gpt-5.6-luna` | OpenAI Responses API 보고서 모델 ID. |
| `CHAINEYE_CORS_ORIGINS` | 로컬 2개 오리진 | 쉼표로 구분한 별도 프론트엔드 허용 오리진. |

PowerShell 예시:

```powershell
# Claude 사용 (기본)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:CHAINEYE_REPORT_MODEL = "claude-opus-5"   # (선택)

# OpenAI 사용
$env:CHAINEYE_REPORT_PROVIDER = "openai"
$env:OPENAI_API_KEY = "sk-..."
$env:CHAINEYE_OPENAI_MODEL = "gpt-5.6-luna"       # (선택)
```

의존성: LLM 경로를 쓰려면 해당 SDK(`anthropic` 또는 `openai`)가 venv 에 설치되어
있어야 합니다. 미설치·미설정이어도 서버는 정상 기동하며 템플릿으로 폴백합니다.

## 프론트엔드 정적 서빙 (단일 URL 배포)

프로덕션 프론트엔드 빌드(`app/frontend/dist`)가 존재하면 백엔드가 이를 루트 경로
`/` 에 자동으로 마운트하여 API 와 UI 를 하나의 URL(단일 오리진)로 서빙합니다.

- dist 경로는 `main.py` 위치 기준(`../frontend/dist`)으로 계산합니다(하드코딩 없음).
- **dist 가 없으면(개발 모드) 조용히 건너뜁니다** — 이때는 Vite dev 서버(5173)를 사용.
- 정적 catch-all 은 모든 API 라우트 **뒤에** 마운트되므로 `/health`, `/score`,
  `/trace`, `/report`, `/explain` 는 정적 파일보다 우선합니다.

빌드 후(예: `npm run build`) 백엔드만 실행하면 `http://localhost:8000/` 에서 UI 가,
동일 오리진의 `/health` 등에서 API 가 함께 제공됩니다.

## 파일 구성

- `main.py` — FastAPI 앱, 엔드포인트, Pydantic 모델, 모델 연동/폴백, 정적 서빙.
- `mock_provider.py` — 모델 미완성 시 결정론적 샘플 데이터 제공.
- `claude_report.py` — Claude API(LLM) 기반 한글 보고서 생성기(우아한 폴백).
- `openai_report.py` — OpenAI(GPT) 기반 한글 보고서 생성기(우아한 폴백, 프롬프트 공유).
- `report_builder.py` — 템플릿 기반 한글 보고서 생성기(LLM 폴백 및 오프라인용).
