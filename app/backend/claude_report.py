"""
Claude API 기반 한글 자금세탁 위험 분석 보고서 생성기 (ChainEye / 체인아이).

`report_builder.py` 의 결정론적 템플릿을 대체(또는 보강)하는 LLM 경로입니다.
Anthropic 공식 Python SDK(`anthropic`)를 사용해 FIU/컴플라이언스 스타일의
전문 보고서를 생성합니다.

설계 원칙 — **반드시 우아하게 실패해야 함(graceful degradation)**:
- `anthropic` 패키지가 없거나(ImportError),
- `ANTHROPIC_API_KEY` 환경변수가 없거나,
- API 호출이 예외를 던지면
→ 예외를 전파하지 않고 `None` 을 반환합니다. 호출측(main.py)은 `None` 을 받으면
   `report_builder.build_report()` 템플릿 경로로 폴백합니다.

환경변수:
- `ANTHROPIC_API_KEY`      — API 키(없으면 LLM 경로 비활성, None 반환).
- `CHAINEYE_REPORT_MODEL`  — 사용할 모델 ID. 기본값 "claude-opus-5".
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("chaineye.claude_report")

# anthropic 패키지는 선택적 의존성이다. 없어도 import 시점에 절대 죽지 않는다.
try:
    import anthropic  # type: ignore

    _ANTHROPIC_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - ImportError 및 그 외 모두 흡수
    anthropic = None  # type: ignore
    _ANTHROPIC_AVAILABLE = False
    logger.info("anthropic SDK unavailable (%s) -> LLM report path disabled.", exc)


_DEFAULT_MODEL = "claude-opus-5"
_MAX_TOKENS = 2048
_TIMEOUT_SECONDS = 20.0


# report_builder 와 동일한 한글 피처 라벨(프롬프트 근거 제공용).
_FEATURE_KO: Dict[str, str] = {
    "in_degree": "유입 거래 연결 수(in-degree)",
    "out_degree": "유출 거래 연결 수(out-degree)",
    "total_btc_received": "누적 수신 BTC 규모",
    "total_btc_sent": "누적 송신 BTC 규모",
    "unique_counterparties": "고유 거래상대방 수",
    "mixer_exposure_ratio": "믹서(자금세탁 서비스) 노출 비율",
    "peel_chain_depth": "필체인(peel chain) 분할 깊이",
    "fan_out_ratio": "다중 분산 송금(fan-out) 비율",
    "avg_hop_interval_sec": "홉 간 평균 이체 간격",
    "darkmarket_proximity": "다크마켓 근접도",
    "exchange_deposit_ratio": "거래소 입금 비율",
    "fresh_address_ratio": "신규 주소 사용 비율",
}


def _grade(score: int) -> str:
    if score >= 70:
        return "위험"
    if score >= 40:
        return "주의"
    return "안전"


def _feature_ko(feature: str) -> str:
    if feature in _FEATURE_KO:
        return _FEATURE_KO[feature]
    if feature.startswith("feat_"):
        try:
            index = int(feature.removeprefix("feat_"))
        except ValueError:
            return feature
        if 0 <= index <= 92:
            return f"거래 자체 특성 {index + 1} ({feature})"
        if 93 <= index <= 164:
            return f"연결 이웃 집계 특성 {index - 92} ({feature})"
    return feature


def _format_factors(top_factors: List[Dict[str, Any]]) -> str:
    if not top_factors:
        return "- (제공된 판단 근거 없음)"
    lines = []
    for f in top_factors[:5]:
        feature = str(f.get("feature", "알수없음"))
        impact = float(f.get("impact", 0.0))
        direction = "위험 상승" if impact >= 0 else "위험 하락"
        lines.append(f"- {_feature_ko(feature)} (feature={feature}): 기여도 {impact:+.3f} ({direction})")
    return "\n".join(lines)


def _build_prompt(
    tx_id: str,
    score: int,
    label: str,
    top_factors: List[Dict[str, Any]],
    graph_stats: Dict[str, Any],
) -> tuple[str, str]:
    """강한 한글 system/user 프롬프트를 구성해 (system, user) 튜플로 반환."""
    grade = _grade(int(score))
    label_ko = "불법(illicit) 의심" if str(label).lower() == "illicit" else "정상(licit) 추정"

    node_count = int(graph_stats.get("nodeCount", 0))
    edge_count = int(graph_stats.get("edgeCount", 0))
    if "illicitNeighbors" in graph_stats:
        connected_high_risk = int(graph_stats["illicitNeighbors"])
    else:
        connected_high_risk = int(graph_stats.get("highRiskCount", 0))
        if connected_high_risk > 0 and int(score) >= 70:
            connected_high_risk = max(0, connected_high_risk - 1)

    system = (
        "당신은 대한민국 금융정보분석원(FIU) 및 금융회사 자금세탁방지(AML) 부서를 위한 "
        "블록체인 자금세탁 위험 분석 전문가입니다. 비트코인 거래에 대한 정량 분석 결과만을 "
        "근거로 전문적이고 간결한 한글 컴플라이언스 보고서를 작성합니다.\n\n"
        "엄격한 규칙:\n"
        "1) 아래에 제공된 수치·라벨·그래프 통계만을 근거로 삼으십시오. 제공되지 않은 사실, "
        "수치, 거래소명, 인물, 지갑 주소를 절대 지어내지 마십시오(환각 금지).\n"
        "2) 보고서는 반드시 다음 4개 섹션 구조를 사용하십시오: "
        "위험 요약, 핵심 판단 근거, 자금흐름 관찰, 권고 조치.\n"
        "3) 전문적이고 객관적인 어조를 유지하되 간결하게 작성하십시오. 과장하지 마십시오.\n"
        "4) 최종 판단과 보고 여부는 담당 분석관의 검토가 필요함을 명시하십시오.\n"
        "5) 오직 보고서 본문만 출력하고 서두 인사말이나 메타 설명은 넣지 마십시오.\n"
        "6) 번역투, 상투적인 결론 문구, 기계적인 병렬 나열, 과도한 괄호·영어 병기를 피하고 "
        "짧고 직접적인 능동문을 우선하십시오. 사실과 수치의 의미는 바꾸지 마십시오.\n"
        "7) 보이지 않는 유니코드 문자나 탐지 회피용 표현을 사용하지 마십시오. 이 보고서가 "
        "자동 생성 자료라는 고지는 유지하십시오."
    )

    user = (
        "다음 분석 데이터를 근거로 자금세탁 위험 분석 보고서를 작성하십시오.\n\n"
        f"- 대상 거래 ID: {tx_id}\n"
        f"- 위험 점수: {int(score)} / 100 (등급: {grade})\n"
        f"- 모델 판정: {label_ko}\n"
        "- 핵심 판단 근거(topFactors, 기여도 부호는 위험 방향성):\n"
        f"{_format_factors(top_factors)}\n"
        "- 자금흐름 그래프 통계:\n"
        f"  · 연결 노드 수: {node_count}개\n"
        f"  · 자금 이동(엣지) 수: {edge_count}건\n"
        f"  · 추적 범위 내 고위험 연결 노드 수: {connected_high_risk}개\n\n"
        "위 데이터만 사용하여 위험 요약/핵심 판단 근거/자금흐름 관찰/권고 조치 "
        "4개 섹션으로 보고서를 작성하십시오. 권고 조치는 위험 등급에 맞게 제시하십시오"
        "(위험: 의심거래보고(STR) 검토 등 / 주의: 지속 모니터링 / 안전: 정기 점검)."
    )
    return system, user


def generate_report_llm(
    tx_id: str,
    score: int,
    label: str,
    top_factors: List[Dict[str, Any]],
    graph_stats: Dict[str, Any],
) -> Optional[str]:
    """Claude API 로 한글 보고서를 생성해 문자열로 반환한다.

    아래의 어떤 경우에도 예외를 전파하지 않고 None 을 반환한다:
    - anthropic 패키지 미설치
    - ANTHROPIC_API_KEY 미설정
    - API 호출 중 임의의 예외

    None 이 반환되면 호출측은 template 경로로 폴백해야 한다.
    """
    if not _ANTHROPIC_AVAILABLE or anthropic is None:
        logger.info("LLM report skipped: anthropic SDK not available.")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("LLM report skipped: ANTHROPIC_API_KEY not set.")
        return None

    model = os.environ.get("CHAINEYE_REPORT_MODEL", _DEFAULT_MODEL)

    try:
        system, user = _build_prompt(tx_id, score, label, top_factors, graph_stats)
        client = anthropic.Anthropic(api_key=api_key, timeout=_TIMEOUT_SECONDS)
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # content 는 블록 리스트. text 블록만 이어붙인다.
        parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        text = "".join(parts).strip()
        if not text:
            logger.warning("LLM report returned empty content -> falling back.")
            return None
        logger.info("LLM report generated via model=%s.", model)
        return text
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 폴백을 위해 흡수
        logger.warning(
            "LLM report generation failed (%s: %s) -> falling back to template.",
            type(exc).__name__,
            exc,
        )
        return None
