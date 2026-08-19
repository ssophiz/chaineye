"""Contract smoke test for the FastAPI layer.

Run from the repository root:
    python app/backend/verify_api.py
"""

from __future__ import annotations

import os
import sys
import time

from fastapi.testclient import TestClient


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# This contract test intentionally exercises the keyless deterministic path.
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)

import main  # noqa: E402


def main_test() -> None:
    with TestClient(main.app) as client:
        deadline = time.monotonic() + 30
        while True:
            health = client.get("/health")
            assert health.status_code == 200, health.text
            if health.json().get("modelLoaded"):
                break
            if time.monotonic() >= deadline:
                raise AssertionError(f"model did not load in time: {health.text}")
            time.sleep(0.1)
        assert health.json() == {
            "status": "ok", "modelLoaded": True, "mode": "model"
        }

        known = client.post("/score", json={"txId": "68995268"})
        assert known.status_code == 200, known.text
        known_body = known.json()
        assert known_body["label"] in {"illicit", "licit"}
        assert isinstance(known_body["riskScore"], int)
        assert known_body["label"] == "illicit"
        assert known_body["riskScore"] >= known_body["decisionThreshold"]

        report = client.post(
            "/report",
            json={
                "txId": known_body["txId"],
                "score": known_body["riskScore"],
                "decisionThreshold": known_body["decisionThreshold"],
                "label": known_body["label"],
                "topFactors": known_body["topFactors"],
                "graphStats": {
                    "nodeCount": 3,
                    "edgeCount": 2,
                    "highRiskCount": 2,
                    "hops": 2,
                },
            },
        )
        assert report.status_code == 200, report.text
        assert report.json()["generator"] == "template"
        report_text = report.json()["report"]
        assert "검토 임계값" in report_text
        assert "불법 거래나 자금세탁을 확정" in report_text
        assert "자금 혼합·경유 가능성" not in report_text

        suppressed = client.post(
            "/report",
            json={
                "txId": "999999999999",
                "score": 0,
                "decisionThreshold": known_body["decisionThreshold"],
                "label": "unknown",
                "topFactors": [],
                "graphStats": {},
            },
        )
        assert suppressed.status_code == 422, suppressed.text

        missing = client.post("/score", json={"txId": "999999999999"})
        assert missing.status_code == 200, missing.text
        assert missing.json() == {
            "txId": "999999999999",
            "riskScore": None,
            "decisionThreshold": known_body["decisionThreshold"],
            "label": "unknown",
            "topFactors": [],
        }

        trace = client.post(
            "/trace", json={"txId": "999999999999", "hops": 2}
        )
        assert trace.status_code == 200, trace.text
        focus = trace.json()["nodes"][0]
        assert trace.json()["decisionThreshold"] == known_body["decisionThreshold"]
        assert focus["risk"] is None
        assert focus["scored"] is False
        assert focus["modelPositive"] is False

        # A model-load failure must fail closed instead of returning synthetic
        # data under a live-looking API response.
        saved_loaded = main.STATE["model_loaded"]
        saved_inference = main.STATE["inference"]
        try:
            main.STATE["model_loaded"] = False
            main.STATE["inference"] = None
            unavailable = client.post("/score", json={"txId": "68995268"})
            assert unavailable.status_code == 503, unavailable.text
        finally:
            main.STATE["model_loaded"] = saved_loaded
            main.STATE["inference"] = saved_inference

    print("API CONTRACT ASSERTIONS PASSED")


if __name__ == "__main__":
    main_test()
