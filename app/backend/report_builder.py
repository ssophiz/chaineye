"""
Template-based Korean report generator for ChainEye (체인아이).

No external / paid LLM API is used here — this is the MVP compliance-report
generator. It renders a structured, professional FIU/compliance-style report
purely from the inputs.

The optional Claude/OpenAI paths live in separate modules. This renderer stays
as the deterministic fallback for offline, rate-limited, and template-only use.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

# Human-readable Korean labels for known feature names.
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
    """Map a 0-100 score to a Korean risk grade band."""
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


def _factor_sentence(factor: Dict[str, Any]) -> str:
    feature = str(factor.get("feature", "알수없음"))
    impact = float(factor.get("impact", 0.0))
    name = _feature_ko(feature)
    if impact >= 0:
        direction = "위험도를 높이는 방향"
    else:
        direction = "위험도를 낮추는 방향"
    return f"- {name}: 판단 기여도 {impact:+.3f} ({direction}으로 작용)"


def build_report(
    tx_id: str,
    score: int,
    label: str,
    top_factors: List[Dict[str, Any]],
    graph_stats: Dict[str, Any],
) -> str:
    """Render the full Korean investigation/compliance report as a string."""
    grade = _grade(int(score))
    label_ko = "불법(illicit) 의심" if str(label).lower() == "illicit" else "정상(licit) 추정"

    node_count = int(graph_stats.get("nodeCount", 0))
    edge_count = int(graph_stats.get("edgeCount", 0))
    # highRiskCount는 추적 그래프 전체의 고위험 노드 수다. 초점 거래가 고위험이면
    # 연결 노드 통계에서는 자신을 한 건 제외한다. 구 스키마 illicitNeighbors는 이미
    # 초점 거래가 제외된 값이므로 그대로 사용한다.
    if "illicitNeighbors" in graph_stats:
        connected_high_risk = int(graph_stats["illicitNeighbors"])
    else:
        connected_high_risk = int(graph_stats.get("highRiskCount", 0))
        if connected_high_risk > 0 and int(score) >= 70:
            connected_high_risk = max(0, connected_high_risk - 1)

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    # 한 줄 요지
    if grade == "위험":
        summary_line = (
            "해당 거래는 자금세탁 위험이 높은 것으로 평가되며 즉시 심화 검토가 필요합니다."
        )
    elif grade == "주의":
        summary_line = (
            "해당 거래는 일부 위험 신호가 관측되어 추가 모니터링이 권고됩니다."
        )
    else:
        summary_line = (
            "해당 거래에서는 유의미한 자금세탁 위험 신호가 관측되지 않았습니다."
        )

    # 핵심 판단 근거 (상위 3~5개)
    if top_factors:
        factor_lines = [_factor_sentence(f) for f in top_factors[:5]]
        factors_block = "\n".join(factor_lines)
    else:
        factors_block = "- 제공된 판단 근거(topFactors)가 없습니다."

    # 자금흐름 관찰
    neighbor_count = max(node_count - 1, 0)
    if neighbor_count > 0:
        neighbor_ratio = (connected_high_risk / neighbor_count) * 100.0
    else:
        neighbor_ratio = 0.0
    flow_block = f"- 추적 그래프 내 연결 노드 수: 총 {node_count:,}개"
    if edge_count > 0:
        flow_block += f" / 자금 이동(엣지) {edge_count:,}건"
    flow_block += (
        f"\n- 추적 범위 내 고위험 연결 노드 수: {connected_high_risk:,}개 "
        f"(전체의 약 {neighbor_ratio:.1f}%)"
    )
    if connected_high_risk > 0:
        flow_block += (
            "\n- 고위험 연결 노드가 포함되어 있어 자금의 유입·유출 경로를 추가로 확인해야 합니다."
        )
    else:
        flow_block += "\n- 현재 추적 범위에서는 별도의 고위험 연결 노드가 확인되지 않았습니다."

    # 권고 조치 (등급별)
    if grade == "위험":
        action_block = (
            "- 의심거래보고(STR) 작성 및 보고 여부를 우선 검토하십시오.\n"
            "- 해당 주소/거래에 대한 계좌·지갑 모니터링을 강화하십시오.\n"
            "- 추적 범위의 고위험 연결 노드를 포함해 자금흐름을 수사 참고자료로 정리하십시오.\n"
            "- 필요 시 거래소 KYC 정보 및 트래블룰(Travel Rule) 대상 여부를 확인하십시오."
        )
    elif grade == "주의":
        action_block = (
            "- 거래 패턴을 일정 기간 지속 모니터링하고 임계치 초과 시 재평가하십시오.\n"
            "- 추가 거래 발생 시 상대방 주소의 위험도를 함께 점검하십시오.\n"
            "- 필요 시 고객확인(EDD, 강화된 고객확인) 절차를 검토하십시오."
        )
    else:
        action_block = (
            "- 별도의 즉각 조치는 불필요하나 정기 모니터링 대상에는 유지하십시오.\n"
            "- 향후 위험 신호 변화 시 재평가하십시오."
        )

    report = f"""# 자금세탁 위험 분석 보고서

- **대상 거래 ID:** `{tx_id}`
- **위험 점수:** {int(score)} / 100
- **모델 판정:** {label_ko}
- **생성 일시:** {generated_at}

## 위험 요약

**{grade} 등급, {int(score)}점.**
{summary_line}

## 핵심 판단 근거

{factors_block}

## 자금흐름 관찰

{flow_block}

## 권고 조치

{action_block}

> 본 문서는 체인아이의 자동 분석 초안입니다. 최종 판단과 보고 여부는 담당 분석관이 검토해야 합니다."""

    return report
