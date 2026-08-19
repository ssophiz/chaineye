"""
Built-in MOCK provider for ChainEye (체인아이).

Used when the teammate's ML module (app/ml/inference.py) is not importable or
fails to load. Returns realistic, deterministic sample data with the SAME shapes
as inference.score_tx / inference.trace_tx so the API always responds.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

# Candidate feature names that resemble a real Bitcoin AML feature set
# (Elliptic-style local + aggregate features).
_FEATURES: List[str] = [
    "in_degree",
    "out_degree",
    "total_btc_received",
    "total_btc_sent",
    "unique_counterparties",
    "mixer_exposure_ratio",
    "peel_chain_depth",
    "fan_out_ratio",
    "avg_hop_interval_sec",
    "darkmarket_proximity",
    "exchange_deposit_ratio",
    "fresh_address_ratio",
]


def _seed(tx_id: str) -> int:
    """Deterministic integer seed derived from the tx id."""
    digest = hashlib.sha256(tx_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _rng_sequence(seed: int, count: int) -> List[float]:
    """Tiny deterministic LCG producing floats in [0, 1)."""
    vals: List[float] = []
    state = seed & 0xFFFFFFFF
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        vals.append(state / 0x7FFFFFFF)
    return vals


def score_tx(tx_id: str) -> Dict[str, Any]:
    """Mock replacement for inference.score_tx."""
    seed = _seed(tx_id)
    rng = _rng_sequence(seed, len(_FEATURES) + 1)

    risk_score = int(round(rng[0] * 100))
    label = "illicit" if risk_score >= 50 else "licit"

    # Build signed impacts for a subset of features, sorted by magnitude.
    factors = []
    for feat, r in zip(_FEATURES, rng[1:]):
        # center around 0, scale to [-1, 1]-ish
        impact = round((r - 0.5) * 2.0, 4)
        factors.append({"feature": feat, "impact": impact})

    factors.sort(key=lambda f: abs(f["impact"]), reverse=True)
    top_factors = factors[:5]

    return {
        "txId": tx_id,
        "riskScore": risk_score,
        "label": label,
        "topFactors": top_factors,
    }


HIGH_RISK_THRESHOLD = 70   # risk >= this is high-risk / illicit
MAX_TRACE_PATHS = 8        # cap on suspicious laundering paths returned


def trace_tx(tx_id: str, hops: int = 2) -> Dict[str, Any]:
    """Mock replacement for inference.trace_tx.

    Builds a small deterministic BFS-like graph radiating from the focus tx.
    Mirrors the real inference shape: each node carries an "illicit" flag and
    the result includes a top-level "paths" list of suspicious directed chains.
    """
    hops = max(1, min(int(hops), 4))

    nodes: List[Dict[str, Any]] = [
        {"id": tx_id, "risk": _node_risk(tx_id), "focus": True}
    ]
    edges: List[Dict[str, str]] = []
    seen = {tx_id}

    frontier = [tx_id]
    counter = 0
    for hop in range(hops):
        next_frontier: List[str] = []
        # fan-out shrinks with depth: 3 at hop0, 2 afterwards
        fan = 3 if hop == 0 else 2
        for parent in frontier:
            p_seed = _seed(parent)
            branch = _rng_sequence(p_seed + hop, fan)
            for b in branch:
                counter += 1
                child = f"{tx_id[:6]}_h{hop + 1}_{counter}"
                if child in seen:
                    continue
                seen.add(child)
                nodes.append(
                    {"id": child, "risk": int(round(b * 100)), "focus": False}
                )
                edges.append({"source": parent, "target": child})
                next_frontier.append(child)
        frontier = next_frontier

    # add the illicit flag to every node (risk >= 70)
    for n in nodes:
        n["illicit"] = n["risk"] >= HIGH_RISK_THRESHOLD

    paths = _suspicious_paths(tx_id, nodes, edges, hops)
    return {"nodes": nodes, "edges": edges, "paths": paths}


def _suspicious_paths(
    focus: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, str]],
    hops: int,
) -> List[List[str]]:
    """Directed walks from focus ending at a high-risk node, len 2..hops+1."""
    risk_map = {n["id"]: n["risk"] for n in nodes}
    dadj: Dict[str, List[str]] = {}
    for e in edges:
        dadj.setdefault(e["source"], []).append(e["target"])

    max_len_nodes = max(int(hops), 1) + 1
    paths: List[List[str]] = []
    seen_paths = set()

    def walk(node: str, path: List[str]) -> None:
        if len(paths) >= MAX_TRACE_PATHS or len(path) >= max_len_nodes:
            return
        for nxt in dadj.get(node, []):
            if nxt in path:
                continue
            new_path = path + [nxt]
            if risk_map.get(nxt, 0) >= HIGH_RISK_THRESHOLD:
                key = tuple(new_path)
                if key not in seen_paths:
                    seen_paths.add(key)
                    paths.append(list(new_path))
                    if len(paths) >= MAX_TRACE_PATHS:
                        return
            walk(nxt, new_path)
            if len(paths) >= MAX_TRACE_PATHS:
                return

    walk(focus, [focus])
    return paths


def _node_risk(node_id: str) -> int:
    return int(round(_rng_sequence(_seed(node_id), 1)[0] * 100))
