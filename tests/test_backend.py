from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.backend import claude_report, main, mock_provider, openai_report, report_builder


@pytest.fixture(autouse=True)
def mock_runtime_state():
    previous = main.STATE.copy()
    main.STATE.update({"model_loaded": False, "mode": "mock", "inference": None})
    yield
    main.STATE.clear()
    main.STATE.update(previous)


@pytest.fixture
def client():
    # Instantiating without a context manager intentionally avoids loading the
    # heavyweight real model for contract-level API tests.
    return TestClient(main.app)


def test_health_exposes_active_mode(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "modelLoaded": False, "mode": "mock"}


def test_model_info_exposes_reproducible_evaluation_evidence(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "Elliptic Bitcoin"
    assert body["transactions"] == 203_769
    assert body["activeModel"] == "LightGBM with graph-neighbor aggregates"
    assert body["testSamples"] == 16_670
    assert body["illicitF1"] > 0.8
    assert 0 < body["decisionThreshold"] < 1
    assert "temporal" in body["validationProtocol"]
    assert body["localFeatureCount"] == 93
    assert body["neighborAggregateFeatureCount"] == 72
    assert body["featureCount"] == 165
    assert body["localOnlyF1"] == pytest.approx(0.743988684582744)
    assert body["graphEnhancedF1"] == pytest.approx(body["illicitF1"])
    assert body["graphF1Lift"] == pytest.approx(0.06108745754923561)
    assert body["graphPrAucLift"] > 0


def test_llm_default_model_ids_are_current_and_explicit():
    assert claude_report._DEFAULT_MODEL == "claude-opus-5"
    assert openai_report._DEFAULT_MODEL == "gpt-5.6-luna"


def test_score_and_trace_contracts_in_mock_mode(client):
    score = client.post("/score", json={"txId": "232629023"})
    trace = client.post("/trace", json={"txId": "232629023", "hops": 2})

    assert score.status_code == 200
    assert 0 <= score.json()["riskScore"] <= 100
    assert trace.status_code == 200
    assert len(trace.json()["nodes"]) > 1
    assert all(path[0] == "232629023" for path in trace.json()["paths"])


@pytest.mark.parametrize("tx_id", ["", "abc", "12-34", "1" * 33])
def test_numeric_transaction_id_is_enforced(client, tx_id):
    response = client.post("/score", json={"txId": tx_id})
    assert response.status_code == 422


def test_unknown_model_transaction_is_not_replaced_with_mock_data():
    class FakeInference:
        @staticmethod
        def contains_tx(_tx_id):
            return False

    main.STATE.update({"model_loaded": True, "mode": "model", "inference": FakeInference()})

    with pytest.raises(HTTPException) as exc_info:
        main._provider_score("999999999")
    assert exc_info.value.status_code == 404


def test_model_runtime_failure_is_not_replaced_with_mock_data():
    class FakeInference:
        @staticmethod
        def contains_tx(_tx_id):
            return True

        @staticmethod
        def score_tx(_tx_id):
            raise RuntimeError("broken model")

    main.STATE.update({"model_loaded": True, "mode": "model", "inference": FakeInference()})

    with pytest.raises(HTTPException) as exc_info:
        main._provider_score("232629023")
    assert exc_info.value.status_code == 503


def test_report_template_provider_needs_no_external_key(client):
    payload = {
        "txId": "232629023",
        "score": 82,
        "label": "illicit",
        "topFactors": [{"feature": "feat_1", "impact": 0.42}],
        "graphStats": {"nodeCount": 4, "edgeCount": 3, "highRiskCount": 2, "hops": 2},
    }
    with patch.dict("os.environ", {"CHAINEYE_REPORT_PROVIDER": "template"}):
        response = client.post("/report", json=payload)

    assert response.status_code == 200
    assert "232629023" in response.json()["report"]
    assert "위험" in response.json()["report"]
    assert "추적 범위 내 고위험 연결 노드 수: 1개" in response.json()["report"]


def test_explicit_zero_legacy_neighbor_count_takes_precedence():
    report = report_builder.build_report(
        tx_id="232629023",
        score=50,
        label="illicit",
        top_factors=[],
        graph_stats={
            "nodeCount": 6,
            "edgeCount": 5,
            "illicitNeighbors": 0,
            "highRiskCount": 5,
        },
    )
    assert "추적 범위 내 고위험 연결 노드 수: 0개" in report


def test_report_uses_human_readable_elliptic_feature_names():
    report = report_builder.build_report(
        tx_id="232629023",
        score=82,
        label="illicit",
        top_factors=[{"feature": "feat_1", "impact": 0.42}],
        graph_stats={"nodeCount": 4, "edgeCount": 3, "highRiskCount": 2},
    )
    assert "거래 자체 특성 2 (feat_1)" in report
    assert "추적 범위 내 고위험 연결 노드 수: 1개" in report
    assert "직접 연결" not in report


def test_mock_provider_is_deterministic_and_marks_high_risk_nodes():
    first = mock_provider.trace_tx("232629023", hops=4)
    second = mock_provider.trace_tx("232629023", hops=4)
    assert first == second
    assert all(node["illicit"] == (node["risk"] >= 70) for node in first["nodes"])
    assert all(2 <= len(path) <= 5 for path in first["paths"])
