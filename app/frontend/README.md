# ChainEye (체인아이) — 프론트엔드

2026 금융 AI Challenge 출품작. 가상자산(비트코인) 자금세탁 탐지·추적 도구의 웹 UI.
Vite + React (JavaScript) + Cytoscape.js 기반 단일 페이지 앱.

## 화면 구성
- **입력바**: Elliptic 데이터셋의 숫자형 트랜잭션 ID(txId) 입력 + `분석` 버튼 + 예시 칩
- **위험도 패널(좌)**: 0~100 위험 점수 게이지 + 라벨(안전/주의/위험) + 핵심 위험 근거
- **자금 흐름 그래프(중앙)**: Cytoscape.js 방향 그래프. 대상/고위험 노드는 빨강, 일반 노드는 회색
- **AI 조사 리포트(우)**: 자동 생성된 한국어 조사 리포트(마크다운-유사 렌더링)

## 실행 방법
```bash
cd app/frontend
npm ci
npm run dev        # 개발 서버 (기본 http://localhost:5173) — 기본 독립 MOCK 데모
```
프로덕션 빌드:
```bash
npm run build      # dist/ 생성 (기본 same-origin + 실제 API)
npm run preview    # 빌드 결과 미리보기
npm test           # Node 내장 테스트 러너
```

## 환경변수(Vite) — MOCK 모드 / 실제 백엔드 / 배포 origin

설정은 **빌드타임 환경변수**로 주입되며, 로직은 `src/config.js` 에 있습니다.
`.env.development`(dev 기본) 와 `.env.production`(build 기본) 파일이 이미 제공됩니다.

| 변수 | 의미 | dev 기본값 | build(프로덕션) 기본값 |
|------|------|-----------|----------------------|
| `VITE_USE_MOCK` | `'true'` 이면 백엔드 없이 mock 샘플 데이터. 그 외/미설정은 실제 API 호출 | `true` | `false` |
| `VITE_API_BASE` | FastAPI 백엔드 베이스 URL. **빈 값('')이면 동일 출처(same-origin)** 상대경로 요청 | `http://localhost:8000` | `''` (same-origin) |

동작 규칙:
- `API_BASE = import.meta.env.VITE_API_BASE ?? ''` — 빈 문자열이면 `POST /score` 가
  `'/score'` 로 나가 **배포된 origin(백엔드가 dist/ 를 서빙하는 그 호스트)** 로 해석됩니다.
- `USE_MOCK = (import.meta.env.VITE_USE_MOCK ?? 'false') === 'true'` — 기본값 **false(실제 API)**.

### 단일 출처(single-origin) 프로덕션 배포
`npm run build` 결과물 `dist/` 를 백엔드(FastAPI)가 정적 파일로 서빙하면,
프론트와 API 가 같은 도메인/포트를 공유합니다. `VITE_API_BASE` 가 비어 있으므로
`/score`, `/trace`, `/report` 요청이 자동으로 그 배포 origin 으로 향합니다. (CORS 불필요)

### 시나리오별 실행
```bash
# 1) 독립 MOCK 데모 (백엔드 불필요) — dev 기본값
npm run dev

# 2) 로컬에서 실제 백엔드로 개발 (별도 origin, CORS 사용)
#    먼저 저장소 루트에서 백엔드 실행: python -m uvicorn app.backend.main:app --port 8000
VITE_USE_MOCK=false VITE_API_BASE=http://localhost:8000 npm run dev

# 3) 프로덕션 빌드 (same-origin, 실제 API) — build 기본값
npm run build

# 4) 배포 대상이 특정 API 호스트라면 빌드 시 지정
VITE_API_BASE=https://api.example.com npm run build
```

## 백엔드 API 계약 (프론트가 호출하는 형태)
- `GET /health` — 활성 공급자 `mode`(`model`/`mock`)와 모델 로딩 상태 확인
- `POST /score` — body `{"txId":"232629023"}` (숫자 문자열만 허용)
  → `{"txId":"...","riskScore":0-100,"label":"illicit|licit","topFactors":[{"feature":"...","impact":0.12}]}`
- `POST /trace` — body `{"txId":"...","hops":2}`
  → `{"nodes":[{"id":"...","risk":0-100,"focus":true|false,"illicit":true|false}],"edges":[...],"paths":[[...]]}`
- `POST /report` — body `{"txId":"...","score":...,"label":"illicit|licit","topFactors":[...],"graphStats":{...}}`
  → `{"report":"<korean text>"}`

## 주요 파일
```
src/
  config.js              # API_BASE / USE_MOCK / 예시 txId
  api.js                 # USE_MOCK 분기 + fetch 래퍼
  mockData.js            # 샘플 응답 생성기 (결정적 유사난수)
  riskUtils.js           # 점수 -> 라벨/색상
  App.jsx                # 3패널 오케스트레이션
  components/
    InputBar.jsx
    RiskPanel.jsx        # SVG 게이지
    GraphPanel.jsx       # Cytoscape 그래프
    ReportPanel.jsx      # 경량 마크다운 렌더러
```
