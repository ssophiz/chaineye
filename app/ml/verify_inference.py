"""Smoke test for inference.py: verifies load/score_tx/trace_tx shapes."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference

inference.load()
print("load() OK")

SAMPLES = {
    "illicit": "232629023",
    "licit": "232438397",
    "unknown": "230425980",   # unlabeled but has features -> still scored
    "missing": "999999999999",  # not in features -> safe default
}

for kind, tx in SAMPLES.items():
    assert inference.contains_tx(tx) is (kind != "missing")
    r = inference.score_tx(tx)
    assert set(r.keys()) == {"txId", "riskScore", "label", "topFactors"}, r.keys()
    assert isinstance(r["txId"], str)
    assert isinstance(r["riskScore"], int) and 0 <= r["riskScore"] <= 100
    assert r["label"] in ("illicit", "licit")
    assert isinstance(r["topFactors"], list) and len(r["topFactors"]) <= 6
    for f in r["topFactors"]:
        assert set(f.keys()) == {"feature", "impact"}
        assert isinstance(f["feature"], str) and isinstance(f["impact"], float)
    print(f"\nscore_tx[{kind}] {tx}: risk={r['riskScore']} label={r['label']} "
          f"nFactors={len(r['topFactors'])}")
    print("  ", json.dumps(r["topFactors"][:3], ensure_ascii=False))

for kind, tx in [("illicit", "232629023"), ("licit", "232438397")]:
    tr = inference.trace_tx(tx, hops=2)
    assert set(tr.keys()) == {"nodes", "edges", "paths"}
    assert len(tr["nodes"]) <= 60
    foci = [n for n in tr["nodes"] if n["focus"]]
    assert len(foci) == 1 and foci[0]["id"] == tx
    for n in tr["nodes"]:
        assert set(n.keys()) == {"id", "risk", "focus", "illicit"}
        assert 0 <= n["risk"] <= 100
        assert n["illicit"] is (n["risk"] >= inference.HIGH_RISK_THRESHOLD)
    ids = {n["id"] for n in tr["nodes"]}
    for e in tr["edges"]:
        assert set(e.keys()) == {"source", "target"}
        assert e["source"] in ids and e["target"] in ids
    for path in tr["paths"]:
        assert 2 <= len(path) <= 3
        assert path[0] == tx
        assert all(node_id in ids for node_id in path)
    print(f"\ntrace_tx[{kind}] {tx}: {len(tr['nodes'])} nodes, {len(tr['edges'])} edges, "
          f"focus_risk={foci[0]['risk']}")

print("\nALL ASSERTIONS PASSED")
