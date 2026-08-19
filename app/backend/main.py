"""
ChainEye (체인아이) — Bitcoin money-laundering detection backend API.
2026 금융 AI Challenge.

FastAPI service exposing scoring, graph-trace, explanation and report endpoints.

Model integration is LOOSELY COUPLED: at startup we try to import the teammate's
ML module (app/ml/inference.py). If it imports and loads, /score and /trace
delegate to it (MODEL mode). Otherwise we fall back to a built-in MOCK provider
(MOCK mode) so the API always responds.

Run from the repository root:
    python -m uvicorn app.backend.main:app --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:  # package import (recommended: uvicorn app.backend.main:app)
    from . import claude_report, mock_provider, openai_report, report_builder
except ImportError:  # direct execution from app/backend, kept for compatibility
    import claude_report  # type: ignore
    import mock_provider  # type: ignore
    import openai_report  # type: ignore
    import report_builder  # type: ignore

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
    "mode": "mock",          # "model" | "mock"
    "inference": None,       # the imported inference module (if any)
}


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
        STATE["mode"] = "mock"
        logger.warning(
            "ML module unavailable (%s: %s) -> running in MOCK mode.",
            type(exc).__name__,
            exc,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load_model()
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
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "CHAINEYE_CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    modelLoaded: bool
    mode: Literal["model", "mock"]


class ModelInfoResponse(BaseModel):
    active: bool
    activeModel: str
    dataset: str
    transactions: int
    trainSamples: int
    testSamples: int
    illicitF1: float
    illicitPrecision: float
    illicitRecall: float
    prAuc: float
    rocAuc: float
    decisionThreshold: float
    validationProtocol: str
    featureCount: int
    localFeatureCount: int
    neighborAggregateFeatureCount: int
    localOnlyF1: float
    graphEnhancedF1: float
    graphF1Lift: float
    graphPrAucLift: float


class TopFactor(BaseModel):
    feature: str = Field(..., min_length=1, max_length=100)
    impact: float = Field(..., ge=-1000, le=1000)


class ScoreRequest(BaseModel):
    txId: str = Field(
        ...,
        min_length=1,
        max_length=32,
        pattern=r"^\d+$",
        description="Numeric Elliptic dataset transaction id to score",
    )


class ScoreResponse(BaseModel):
    txId: str
    riskScore: int = Field(..., ge=0, le=100)
    label: Literal["illicit", "licit"]
    topFactors: List[TopFactor]


class TraceRequest(BaseModel):
    txId: str = Field(..., min_length=1, max_length=32, pattern=r"^\d+$")
    hops: int = Field(2, ge=1, le=4, description="Number of hops to expand")


class GraphNode(BaseModel):
    id: str
    risk: int = Field(..., ge=0, le=100)
    focus: bool
    # High-risk (illicit) flag: risk >= 70. Defaults keep backward compatibility
    # with any provider that hasn't been updated to emit it yet.
    illicit: bool = False


class GraphEdge(BaseModel):
    source: str
    target: str


class TraceResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    # Suspicious directed laundering chains from the focus node to high-risk
    # nodes. Each path is an ordered list of txId strings. Optional so older
    # providers still validate.
    paths: List[List[str]] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    txId: str = Field(..., min_length=1, max_length=32, pattern=r"^\d+$")


class ExplainResponse(BaseModel):
    txId: str
    topFactors: List[TopFactor]


class GraphStats(BaseModel):
    nodeCount: int = Field(0, ge=0, le=100_000)
    # 구 클라이언트가 명시적으로 보낼 때만 사용한다. 기본값 0으로 두면 최신
    # highRiskCount를 덮어쓰므로 None을 유지한 채 직렬화에서 제외한다.
    illicitNeighbors: Optional[int] = Field(None, ge=0, le=100_000)
    # 프론트엔드가 전송하는 현재 필드
    edgeCount: int = Field(0, ge=0, le=1_000_000)
    highRiskCount: int = Field(0, ge=0, le=100_000)
    hops: int = Field(0, ge=0, le=4)


class ReportRequest(BaseModel):
    txId: str = Field(..., min_length=1, max_length=32, pattern=r"^\d+$")
    score: int = Field(..., ge=0, le=100)
    label: Literal["illicit", "licit"]
    topFactors: List[TopFactor] = Field(default_factory=list, max_length=10)
    graphStats: GraphStats = Field(default_factory=GraphStats)


class ReportResponse(BaseModel):
    report: str


# --------------------------------------------------------------------------- #
# Provider dispatch helpers
# --------------------------------------------------------------------------- #

def _provider_score(tx_id: str) -> dict:
    """Delegate to the active provider without fabricating request fallbacks."""
    if STATE["model_loaded"] and STATE["inference"] is not None:
        try:
            contains_tx = getattr(STATE["inference"], "contains_tx", None)
            if callable(contains_tx) and not contains_tx(tx_id):
                raise HTTPException(
                    status_code=404,
                    detail="해당 txId는 모델 데이터셋에 존재하지 않습니다.",
                )
            return STATE["inference"].score_tx(tx_id)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("inference.score_tx failed for txId=%s", tx_id)
            raise HTTPException(
                status_code=503,
                detail="모델 추론에 실패했습니다. 잠시 후 다시 시도해 주세요.",
            ) from exc
    return mock_provider.score_tx(tx_id)


def _provider_trace(tx_id: str, hops: int) -> dict:
    if STATE["model_loaded"] and STATE["inference"] is not None:
        try:
            return STATE["inference"].trace_tx(tx_id, hops)
        except Exception as exc:  # noqa: BLE001
            logger.exception("inference.trace_tx failed for txId=%s", tx_id)
            raise HTTPException(
                status_code=503,
                detail="그래프 추적에 실패했습니다. 잠시 후 다시 시도해 주세요.",
            ) from exc
    return mock_provider.trace_tx(tx_id, hops)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        modelLoaded=STATE["model_loaded"],
        mode=STATE["mode"],
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Expose the bundled model's reproducible evaluation evidence."""
    metrics_path = os.path.join(_ML_DIR, "metrics.json")
    ablation_path = os.path.join(_ML_DIR, "ablation_metrics.json")
    try:
        with open(metrics_path, encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        with open(ablation_path, encoding="utf-8") as ablation_file:
            ablation = json.load(ablation_file)
        local_only = ablation["feature_sets"]["transaction_local_only"][
            "untouched_test"
        ]
        graph_lift = ablation["untouched_test_difference_all_minus_local"]
        return ModelInfoResponse(
            active=bool(STATE["model_loaded"]),
            activeModel="LightGBM with graph-neighbor aggregates",
            dataset="Elliptic Bitcoin",
            transactions=203_769,
            trainSamples=int(metrics["n_train"]),
            testSamples=int(metrics["n_test"]),
            illicitF1=float(metrics["illicit_f1"]),
            illicitPrecision=float(metrics["illicit_precision"]),
            illicitRecall=float(metrics["illicit_recall"]),
            prAuc=float(metrics["pr_auc"]),
            rocAuc=float(metrics["roc_auc"]),
            decisionThreshold=float(metrics["threshold"]),
            validationProtocol=str(metrics["evaluation_protocol"]),
            featureCount=165,
            localFeatureCount=93,
            neighborAggregateFeatureCount=72,
            localOnlyF1=float(local_only["illicit_f1"]),
            graphEnhancedF1=float(metrics["illicit_f1"]),
            graphF1Lift=float(graph_lift["illicit_f1"]),
            graphPrAucLift=float(graph_lift["pr_auc"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.exception("Unable to read model evaluation evidence")
        raise HTTPException(
            status_code=503,
            detail="모델 검증 정보를 불러올 수 없습니다.",
        ) from exc


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
    graph_stats = req.graphStats.model_dump(exclude_none=True)

    # 1순위: LLM 경로 (provider 선택 가능). 모든 LLM 경로는 우아하게 실패하도록
    # 설계됨 — 패키지 미설치 / API 키 미설정 / API 오류 시 None 을 반환한다.
    # provider 는 CHAINEYE_REPORT_PROVIDER 로 선택한다:
    #   "claude" (기본) | "openai" | "auto"(claude 실패 시 openai 시도)
    # 어떤 경우에도 최종적으로는 결정론적 템플릿으로 폴백하므로, API 키가 전혀
    # 없어도(심사/오프라인 환경) 항상 정상 동작한다.
    provider = os.environ.get("CHAINEYE_REPORT_PROVIDER", "claude").strip().lower()
    if provider not in {"claude", "openai", "auto", "template"}:
        logger.warning(
            "Unknown CHAINEYE_REPORT_PROVIDER=%r; using template fallback.", provider
        )
        provider = "template"

    text = None
    used = None
    if provider == "openai":
        text = openai_report.generate_report_llm(
            tx_id=req.txId, score=req.score, label=req.label,
            top_factors=top_factors, graph_stats=graph_stats,
        )
        used = "OpenAI"
    elif provider == "auto":
        text = claude_report.generate_report_llm(
            tx_id=req.txId, score=req.score, label=req.label,
            top_factors=top_factors, graph_stats=graph_stats,
        )
        used = "Claude"
        if text is None:
            text = openai_report.generate_report_llm(
                tx_id=req.txId, score=req.score, label=req.label,
                top_factors=top_factors, graph_stats=graph_stats,
            )
            used = "OpenAI"
    elif provider == "claude":
        text = claude_report.generate_report_llm(
            tx_id=req.txId, score=req.score, label=req.label,
            top_factors=top_factors, graph_stats=graph_stats,
        )
        used = "Claude"

    if text is not None:
        logger.info("/report served via LLM (%s).", used)
    else:
        # 폴백: 결정론적 템플릿 경로(항상 동작).
        logger.info("/report served via template fallback.")
        text = report_builder.build_report(
            tx_id=req.txId,
            score=req.score,
            label=req.label,
            top_factors=top_factors,
            graph_stats=graph_stats,
        )

    return ReportResponse(report=text)


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
