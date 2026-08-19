#!/usr/bin/env python3
"""Exercise the production container through its public HTTP surface."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def request_json(base_url: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def wait_for_model(base_url: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "service did not respond"
    while time.monotonic() < deadline:
        try:
            health = request_json(base_url, "/health")
            if (
                isinstance(health, dict)
                and health.get("status") == "ok"
                and health.get("mode") == "model"
                and health.get("modelLoaded") is True
            ):
                return health
            last_error = f"unexpected health response: {health!r}"
        except (OSError, ValueError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise AssertionError(f"model did not become ready within {timeout_seconds}s: {last_error}")


def check_frontend(base_url: str) -> None:
    with urllib.request.urlopen(f"{base_url}/", timeout=10) as response:
        body = response.read().decode("utf-8")
        content_type = response.headers.get_content_type()
    assert content_type == "text/html", f"unexpected frontend content type: {content_type}"
    assert '<div id="root">' in body, "frontend shell is missing its root element"


def check_api(base_url: str) -> None:
    model_info = request_json(base_url, "/model-info")
    assert isinstance(model_info, dict), "model-info was not a JSON object"
    assert model_info.get("active") is True, "model-info did not report an active model"
    assert model_info.get("activeModel") == "LightGBM with graph-neighbor aggregates"
    assert model_info.get("featureCount") == 165
    assert model_info.get("localFeatureCount") == 93
    assert model_info.get("neighborAggregateFeatureCount") == 72
    assert model_info.get("graphF1Lift", 0) > 0.06
    assert isinstance(model_info.get("transactions"), int) and model_info["transactions"] > 0
    for metric in ("illicitF1", "illicitPrecision", "illicitRecall", "prAuc", "rocAuc"):
        assert isinstance(model_info.get(metric), (int, float)) and 0 <= model_info[metric] <= 1, (
            f"invalid {metric}: {model_info.get(metric)!r}"
        )

    score = request_json(base_url, "/score", {"txId": "232629023"})
    assert isinstance(score, dict), "score was not a JSON object"
    assert score.get("txId") == "232629023", f"unexpected txId: {score.get('txId')!r}"
    assert score.get("label") == "illicit", f"unexpected known-transaction label: {score!r}"
    assert isinstance(score.get("riskScore"), int) and 0 <= score["riskScore"] <= 100
    assert isinstance(score.get("topFactors"), list), "score omitted topFactors"

    report = request_json(
        base_url,
        "/report",
        {
            "txId": score["txId"],
            "score": score["riskScore"],
            "label": score["label"],
            "topFactors": score["topFactors"],
            "graphStats": {"nodeCount": 9, "edgeCount": 8, "highRiskCount": 2, "hops": 2},
        },
    )
    assert isinstance(report, dict) and isinstance(report.get("report"), str)
    assert report["report"].strip(), "report endpoint returned an empty report"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=150)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    base_url = args.base_url.rstrip("/")
    wait_for_model(base_url, args.timeout_seconds)
    check_frontend(base_url)
    check_api(base_url)
    print("Docker smoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, ValueError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"Docker smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
