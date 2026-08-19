---
title: ChainEye
emoji: 🔗
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ChainEye (체인아이) — 가상자산 자금세탁 탐지·추적 AI

2026 금융 AI Challenge 출품작. 비트코인 트랜잭션 그래프에서 자금세탁 의심 흐름을
탐지·추적하고, 수사·컴플라이언스 담당자를 위한 근거 기반 리포트를 자동 생성하는 웹서비스.

## 구성
- **모델**: 거래 자체 특성 93개와 그래프 이웃 집계 특성 72개를 함께 쓰는 LightGBM(불법 F1 0.805 / PR-AUC 0.799) + SHAP(XAI). 그래프 특성 ablation에서 F1이 0.744 → 0.805로 상승했으며, 성능이 낮은 end-to-end GCN은 연구 후보로 분리
- **백엔드**: FastAPI (`/score`, `/trace`, `/explain`, `/report`, `/health`, `/model-info`) — `app/backend`
- **프론트엔드**: React + Cytoscape.js 자금흐름 그래프 — `app/frontend`
- **데이터**: Elliptic Data Set (203,769 tx / 234,355 edges / 166 features / 49 timesteps)

## 실행 (Docker)
```bash
docker build -t chaineye .
docker run -p 7860:7860 chaineye
# http://localhost:7860
```

입력값은 블록체인 해시가 아니라 Elliptic 데이터셋의 숫자형 `txId`다. 화면의 예시
버튼(`232629023`, `232438397`, `230425980`)으로 즉시 확인할 수 있다.

## 개발 검증

```bash
python -m pip install -r deploy/requirements-dev.txt
python -m pytest
python -m ruff check .
python -m pip_audit -r deploy/requirements.txt
python app/ml/verify_inference.py

cd app/frontend
npm ci
npm test
npm run build
npm audit --audit-level=moderate
```

## LLM 리포트 (선택)
키가 없으면 결정론적 템플릿 보고서로 100% 동작한다. 고급 보고서를 켜려면
Space Secret 또는 환경변수로:
- `ANTHROPIC_API_KEY` (기본 provider, 모델 `CHAINEYE_REPORT_MODEL`=claude-opus-5)
- 또는 `CHAINEYE_REPORT_PROVIDER=openai` + `OPENAI_API_KEY`
- 외부 호출을 항상 끄려면 `CHAINEYE_REPORT_PROVIDER=template`

문서: `docs/01_기획서.md`, `docs/02_기능명세서.md`, `docs/03_배포가이드.md`, `docs/04_심사시연가이드.md`, `docs/05_제출패키지.md`
