"""Unit and integration tests. Run with:  python -m pytest -q"""
import numpy as np
import pytest

from bcfl.attacks import SCENARIOS
from bcfl.crypto import KeyPair, canonical_json, verify
from bcfl.data import make_partitions
from bcfl.framework import recompute_aggregation, run_federated, screen_by_norm, weighted_fedavg
from bcfl.ledger import merkle_root
from bcfl.model import N_PARAMS


@pytest.fixture(scope="module")
def iid():
    return make_partitions(0, "iid")


def test_signature_roundtrip():
    k = KeyPair.generate()
    msg = canonical_json({"b": 1, "a": [1, 2]})
    sig = k.sign(msg)
    assert verify(k.public_key, msg, sig)
    assert not verify(k.public_key, msg + b"x", sig)
    assert not verify(KeyPair.generate().public_key, msg, sig)


def test_canonical_json_is_order_independent():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_merkle_root_changes_with_any_leaf():
    leaves = [f"{i:064x}" for i in range(5)]
    r = merkle_root(leaves)
    leaves[3] = f"{99:064x}"
    assert merkle_root(leaves) != r


def test_partitions(iid):
    clients, X_te, y_te = iid
    assert [c.n for c in clients] == [101, 202, 152]
    assert len(y_te) == 114
    assert clients[0].X.shape[1] + 1 == N_PARAMS
    nonid, _, _ = make_partitions(0, "noniid")
    assert sum(c.n for c in nonid) == 455


def test_screening_rule():
    d = {"A": np.ones(3), "B": np.ones(3) * 1.1, "C": np.ones(3) * 10}
    acc, rej = screen_by_norm(d, kappa=3)
    assert set(acc) == {"A", "B"} and rej == ["C"]


def test_weighted_fedavg():
    w = np.zeros(2)
    out = weighted_fedavg(w, {"A": np.array([1.0, 0]), "B": np.array([0, 1.0])}, {"A": 1, "B": 3})
    assert np.allclose(out, [0.25, 0.75])


def test_ledger_does_not_change_model(iid):
    clients, X, y = iid
    a = run_federated(clients, X, y, 0, "fedavg", rounds=5)
    b = run_federated(clients, X, y, 0, "screen", rounds=5)
    assert np.allclose(a.w, b.w)


def test_chain_valid_and_aggregation_recomputable(iid):
    clients, X, y = iid
    r = run_federated(clients, X, y, 0, "screen", rounds=4, track_history=False)
    for rep in r.network.replicas:
        assert rep.audit(r.network.base_directory)[0]
    assert len({rep.tip for rep in r.network.replicas}) == 1
    for t in range(1, 5):
        assert recompute_aggregation(r.network, t)[0]


def test_nothing_sensitive_on_chain(iid):
    clients, X, y = iid
    r = run_federated(clients, X, y, 0, "screen", rounds=2, track_history=False)
    on_chain = b"".join(canonical_json(tx) for tx in r.network.replicas[0].all_txs())
    assert len(on_chain) < 20_000  # only digests and metadata


@pytest.mark.parametrize("sid,desc,mech,fn", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_protocol_attack_detected(iid, sid, desc, mech, fn):
    clients, X, y = iid
    r = fn(clients, X, y, 0)
    assert r["detected"], r
    assert r["on_ledger"], r


def test_signflip_poisoning_blocked_only_with_screening(iid):
    from bcfl.model import evaluate
    clients, X, y = iid
    audit = run_federated(clients, X, y, 0, "audit", attack="signflip", track_history=False)
    screen = run_federated(clients, X, y, 0, "screen", attack="signflip", track_history=False)
    assert evaluate(audit.w, X, y)["acc"] < 0.5
    assert evaluate(screen.w, X, y)["acc"] > 0.9
