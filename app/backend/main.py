"""
ChainEye (체인아이) — Bitcoin money-laundering detection backend API.
2026 금융 AI Challenge.

FastAPI service exposing scoring, graph-trace, explanation and report endpoints.

Model integration is LOOSELY COUPLED: at startup we try to import the ML module
(app/ml/inference.py). If it imports and loads, /score and /trace delegate to it.
Synthetic fallback is opt-in only (CHAINEYE_ALLOW_MOCK_BACKEND=1); production
fails closed instead of presenting fabricated data as a live model response.

Run (from app/backend):
    C:\\Users\\DELL\\fsec-ai-challenge-2026\\.venv\\Scripts\\python.exe -m uvicorn main:app --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import claude_report
import mock_provider
import openai_report
import report_builder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chaineye")

# --------------------------------------------------------------------------- #
# Model integration (loose coupling)
# --------------------------------------------------------------------------- #

_ML_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ml")
)

# Runtime state describing which provider is active.
STATE = {
    "model_loaded": False,   # True only when the real ML module loaded OK
    "mode": "unavailable",   # "model" | "mock" | "unavailable"
    "inference": None,       # the imported inference module (if any)
}

_MODEL_LOAD_LOCK = threading.Lock()
_MODEL_LOAD_THREAD: Optional[threading.Thread] = None


def _try_load_model() -> None:
    """Attempt to import and load the teammate's ML inference module.

    On any failure, remain in MOCK mode. The API must never fail to start
    just because the ML side isn't ready.
    """
    if _ML_DIR not in sys.path:
        sys.path.insert(0, _ML_DIR)
    try:
        import inference  # type: ignore

        # load() is expected; tolerate a module that lacks it.
        loader = getattr(inference, "load", None)
        if callable(loader):
            loader()
        STATE["inference"] = inference
        STATE["model_loaded"] = True
        STATE["mode"] = "model"
        logger.info("ML module loaded successfully -> running in MODEL mode.")
    except Exception as exc:  # noqa: BLE001 - intentional broad fallback
        STATE["inference"] = None
        STATE["model_loaded"] = False
        STATE["mode"] = (
            "mock"
            if os.environ.get("CHAINEYE_ALLOW_MOCK_BACKEND", "0") == "1"
            else "unavailable"
        )
        logger.warning(
            "ML module unavailable (%s: %s) -> running in %s mode.",
            type(exc).__name__,
            exc,
            STATE["mode"].upper(),
        )


def _start_model_loader() -> None:
    """Load the model without delaying the web server's port binding."""
    global _MODEL_LOAD_THREAD

    with _MODEL_LOAD_LOCK:
        if STATE["model_loaded"]:
            return
        if _MODEL_LOAD_THREAD is not None and _MODEL_LOAD_THREAD.is_alive():
            return

        _MODEL_LOAD_THREAD = threading.Thread(
            target=_try_load_model,
            name="chaineye-model-loader",
            daemon=True,
        )
        _MODEL_LOAD_THREAD.start()
        logger.info("ML model loading started in the background.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _start_model_loader()
    logger.info("ChainEye backend started in %s mode.", STATE["mode"].upper())
    yield
    logger.info("ChainEye backend shutting down.")


# --------------------------------------------------------------------------- #
# App + CORS
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="ChainEye (체인아이) API",
    description="Bitcoin money-laundering detection backend — 2026 금융 AI Challenge",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    status: str = "ok"
    modelLoaded: bool
    mode: Literal["model", "mock", "unavailable"]


class TopFactor(BaseModel):
    feature: str
    impact: float


class ScoreRequest(BaseModel):
    txId: str = Field(..., description="Bitcoin transaction id to score")


class ScoreResponse(BaseModel):
    txId: str
    riskScore: Optional[int] = Field(None, ge=0, le=100)
    decisionThreshold: int = Field(..., ge=0, le=100)
    label: Literal["illicit", "licit", "unknown"]
    topFactors: List[TopFactor]


class TraceRequest(BaseModel):
    txId: str
    hops: int = Field(2, ge=1, le=4, description="Number of hops to expand")


class GraphNode(BaseModel):
    id: str
    risk: Optional[int] = Field(None, ge=0, le=100)
    focus: bool
    # Model-positive flag based on the validation-selected decision threshold.
    # Defaults keep backward compatibility with older providers.
    modelPositive: bool = False
    scored: bool = True


class GraphEdge(BaseModel):
    source: str
    target: str


class TraceResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    # Directed model-positive review candidates from the focus node. Each path
    # is an ordered list of txId strings and is not a proven fund-flow path.
    candidatePaths: List[List[str]] = Field(default_factory=list)
    decisionThreshold: int = Field(50, ge=0, le=100)


class ExplainRequest(BaseModel):
    txId: str


class ExplainResponse(BaseModel):
    txId: str
    topFactors: List[TopFactor]


class GraphStats(BaseModel):
    nodeCount: int = Field(0, ge=0)
    illicitNeighbors: int = Field(0, ge=0)
    # 프론트엔드가 전송하는 필드 (하위호환 유지: 없으면 기본값)
    edgeCount: int = Field(0, ge=0)
    highRiskCount: int = Field(0, ge=0)
    hops: int = Field(0, ge=0)


class ReportRequest(BaseModel):
    txId: str
    score: int = Field(..., ge=0, le=100)
    decisionThreshold: int = Field(50, ge=0, le=100)
    # 판단 불가(unknown)에는 보고서를 만들지 않는다. 프론트엔드뿐 아니라
    # API 계약에서도 생성 억제를 강제해 우회 호출을 막는다.
    label: Literal["illicit", "licit"]
    topFactors: List[TopFactor] = Field(default_factory=list)
    graphStats: GraphStats = Field(default_factory=GraphStats)


class ReportResponse(BaseModel):
    report: str
    generator: Literal["claude", "openai", "template"]


# --------------------------------------------------------------------------- #
# Provider dispatch helpers
# --------------------------------------------------------------------------- #

def _provider_score(tx_id: str) -> dict:
    """Delegate to the model; allow synthetic fallback only when explicit."""
    if STATE["model_loaded"] and STATE["inference"] is not None:
        try:
            return STATE["inference"].score_tx(tx_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("inference.score_tx failed (%s) -> mock fallback.", exc)
    if os.environ.get("CHAINEYE_ALLOW_MOCK_BACKEND", "0") == "1":
        return mock_provider.score_tx(tx_id)
    raise HTTPException(
        status_code=503,
        detail="Model unavailable; synthetic fallback is disabled.",
    )


def _provider_trace(tx_id: str, hops: int) -> dict:
    if STATE["model_loaded"] and STATE["inference"] is not None:
        try:
            return STATE["inference"].trace_tx(tx_id, hops)
        except Exception as exc:  # noqa: BLE001
            logger.error("inference.trace_tx failed (%s) -> mock fallback.", exc)
    if os.environ.get("CHAINEYE_ALLOW_MOCK_BACKEND", "0") == "1":
        return mock_provider.trace_tx(tx_id, hops)
    raise HTTPException(
        status_code=503,
        detail="Model unavailable; synthetic fallback is disabled.",
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    allow_mock = os.environ.get("CHAINEYE_ALLOW_MOCK_BACKEND", "0") == "1"
    mode = "model" if STATE["model_loaded"] else ("mock" if allow_mock else "unavailable")
    return HealthResponse(status="ok", modelLoaded=STATE["model_loaded"], mode=mode)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    result = _provider_score(req.txId)
    return ScoreResponse(**result)


@app.post("/trace", response_model=TraceResponse)
def trace(req: TraceRequest) -> TraceResponse:
    result = _provider_trace(req.txId, req.hops)
    return TraceResponse(**result)


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    # Reuse the scoring provider's factors (SHAP-style contributions).
    result = _provider_score(req.txId)
    return ExplainResponse(
        txId=req.txId,
        topFactors=result.get("topFactors", []),
    )


@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest) -> ReportResponse:
    top_factors = [f.model_dump() for f in req.topFactors]
    graph_stats = req.graphStats.model_dump()

    # 1순위: LLM 경로 (provider 선택 가능). 모든 LLM 경로는 우아하게 실패하도록
    # 설계됨 — 패키지 미설치 / API 키 미설정 / API 오류 시 None 을 반환한다.
    # provider 는 CHAINEYE_REPORT_PROVIDER 로 선택한다:
    #   "claude" (기본) | "openai" | "auto"(claude 실패 시 openai 시도)
    # 어떤 경우에도 최종적으로는 결정론적 템플릿으로 폴백하므로, API 키가 전혀
    # 없어도(심사/오프라인 환경) 항상 정상 동작한다.
    provider = os.environ.get("CHAINEYE_REPORT_PROVIDER", "claude").strip().lower()

    text = None
    used = None
    if provider == "openai":
        text = openai_report.generate_report_llm(
            tx_id=req.txId, score=req.score, label=req.label,
            decision_threshold=req.decisionThreshold,
            top_factors=top_factors, graph_stats=graph_stats,
        )
        used = "openai"
    elif provider == "auto":
        text = claude_report.generate_report_llm(
            tx_id=req.txId, score=req.score, label=req.label,
            decision_threshold=req.decisionThreshold,
            top_factors=top_factors, graph_stats=graph_stats,
        )
        used = "claude"
        if text is None:
            text = openai_report.generate_report_llm(
                tx_id=req.txId, score=req.score, label=req.label,
                decision_threshold=req.decisionThreshold,
                top_factors=top_factors, graph_stats=graph_stats,
            )
            used = "openai"
    else:  # "claude" 및 알 수 없는 값은 Claude 로 처리
        text = claude_report.generate_report_llm(
            tx_id=req.txId, score=req.score, label=req.label,
            decision_threshold=req.decisionThreshold,
            top_factors=top_factors, graph_stats=graph_stats,
        )
        used = "claude"

    if text is not None:
        logger.info("/report served via LLM (%s).", used)
    else:
        # 폴백: 결정론적 템플릿 경로(항상 동작).
        logger.info("/report served via template fallback.")
        text = report_builder.build_report(
            tx_id=req.txId,
            score=req.score,
            label=req.label,
            decision_threshold=req.decisionThreshold,
            top_factors=top_factors,
            graph_stats=graph_stats,
        )
        used = "template"

    return ReportResponse(report=text, generator=used)


# --------------------------------------------------------------------------- #
# Static frontend (single-origin deploy)
# --------------------------------------------------------------------------- #
# 프로덕션 프론트엔드 빌드가 존재하면 루트("/")에 마운트해 단일 URL로 서빙한다.
# API 라우트(/score /trace /report /explain /health)가 정적 catch-all 보다
# 우선하도록 반드시 모든 라우트 정의 이후(맨 마지막)에 마운트한다.
# dist 경로는 하드코딩하지 않고 main.py 위치 기준(../frontend/dist)으로 계산한다.
# dist 가 없으면(개발 모드) 조용히 건너뛴다.
_FRONTEND_DIST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)

if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
    logger.info("Serving frontend static build from %s", _FRONTEND_DIST)
else:
    logger.info("Frontend dist not found (%s) -> dev mode, static serving skipped.", _FRONTEND_DIST)
