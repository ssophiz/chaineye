"""
OpenAI(GPT) 기반 한글 자금세탁 위험 분석 보고서 생성기 (ChainEye / 체인아이).

`claude_report.py` 의 OpenAI 대응 버전입니다. 동일한 한글 프롬프트를 재사용하며
(단일 출처: `claude_report._build_prompt`), OpenAI 공식 Python SDK 의 Chat
Responses API 로 보고서를 생성합니다.

주의: "Codex"(구 code-davinci 계열)는 이미 폐기되었습니다. 현행 OpenAI GPT 채팅
모델을 사용하며, 모델 ID 는 `CHAINEYE_OPENAI_MODEL` 로 지정합니다.

설계 원칙 — **반드시 우아하게 실패해야 함(graceful degradation)**:
- `openai` 패키지가 없거나(ImportError),
- `OPENAI_API_KEY` 환경변수가 없거나,
- API 호출이 예외를 던지면
→ 예외를 전파하지 않고 `None` 을 반환합니다. 호출측(main.py)은 `None` 을 받으면
   템플릿 경로로 폴백합니다.

환경변수:
- `OPENAI_API_KEY`         — API 키(없으면 OpenAI 경로 비활성, None 반환).
- `CHAINEYE_OPENAI_MODEL`  — 사용할 모델 ID. 기본값 "gpt-5.6-luna".
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

# 프롬프트는 claude_report 와 100% 공유한다(근거·섹션구조·환각금지 규칙 단일 출처).
try:
    from .claude_report import _build_prompt
except ImportError:  # direct execution from app/backend
    from claude_report import _build_prompt  # type: ignore

logger = logging.getLogger("chaineye.openai_report")

# openai 패키지는 선택적 의존성이다. 없어도 import 시점에 절대 죽지 않는다.
try:
    from openai import OpenAI  # type: ignore

    _OPENAI_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - ImportError 및 그 외 모두 흡수
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False
    logger.info("openai SDK unavailable (%s) -> OpenAI report path disabled.", exc)


_DEFAULT_MODEL = "gpt-5.6-luna"
_MAX_TOKENS = 2048
_TIMEOUT_SECONDS = 20.0


def generate_report_llm(
    tx_id: str,
    score: int,
    label: str,
    top_factors: List[Dict[str, Any]],
    graph_stats: Dict[str, Any],
) -> Optional[str]:
    """OpenAI Responses API 로 한글 보고서를 생성해 문자열로 반환한다.

    아래의 어떤 경우에도 예외를 전파하지 않고 None 을 반환한다:
    - openai 패키지 미설치
    - OPENAI_API_KEY 미설정
    - API 호출 중 임의의 예외

    None 이 반환되면 호출측은 template 경로로 폴백해야 한다.
    """
    if not _OPENAI_AVAILABLE or OpenAI is None:
        logger.info("OpenAI report skipped: openai SDK not available.")
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.info("OpenAI report skipped: OPENAI_API_KEY not set.")
        return None

    model = os.environ.get("CHAINEYE_OPENAI_MODEL", _DEFAULT_MODEL)

    try:
        system, user = _build_prompt(tx_id, score, label, top_factors, graph_stats)
        client = OpenAI(api_key=api_key, timeout=_TIMEOUT_SECONDS)
        response = client.responses.create(
            model=model,
            max_output_tokens=_MAX_TOKENS,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = (response.output_text or "").strip()
        if not text:
            logger.warning("OpenAI report returned empty content -> falling back.")
            return None
        logger.info("OpenAI report generated via model=%s.", model)
        return text
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 폴백을 위해 흡수
        logger.warning(
            "OpenAI report generation failed (%s: %s) -> falling back to template.",
            type(exc).__name__,
            exc,
        )
        return None
