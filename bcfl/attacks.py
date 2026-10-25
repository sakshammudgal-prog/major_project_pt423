"""Protocol-level attack scenarios S1-S8 (Table VI of the paper).

Each scenario runs a short framework training run, injects the misbehaviour and returns
whether it was rejected/detected, the mechanism that caught it and whether the rejection is
visible on the ledger.
"""
from __future__ import annotations

import copy
import uuid

import numpy as np

from .crypto import KeyPair, canonical_json, sha256_hex
from .framework import RoundContext, run_federated
from .ledger import GENESIS_PREV, compare_replicas, merkle_root, tx_hash
from .offchain import serialize_update

ATTACK_ROUND = 5
RUN_ROUNDS = 8
LEDGER_ATTACK_ROUNDS = 10


def _rejected_on_chain(net, reason_prefix: str) -> bool:
    return any(tx["type"] == "rejected" and tx["body"]["reason"].startswith(reason_prefix)
               for tx in net.replicas[0].all_txs())


def _hosp(ctx: RoundContext, name: str):
    return next(h for h in ctx.hospitals if h.name == name)


def _run(clients, X_te, y_te, seed, hooks, rounds=RUN_ROUNDS):
    return run_federated(clients, X_te, y_te, seed, mode="audit", rounds=rounds,
                         hooks=hooks, track_history=False)


# ---------------------------------------------------------------- S1 - S5 (contract rules)
def s1_unregistered(clients, X_te, y_te, seed):
    out = {}

    def hook(ctx):
        if ctx.round_no != ATTACK_ROUND:
            return
        outsider = KeyPair.generate()
        digest = ctx.network.store.put(serialize_update(np.ones(31)))
        from .ledger import make_tx
        tx = make_tx("submit", {"round": ctx.round_no, "client": "Hospital X", "digest": digest,
                                "n_samples": 100, "nonce": uuid.uuid4().hex}, "Hospital X", outsider)
        out["result"] = ctx.network.contract.submit(tx)

    r = _run(clients, X_te, y_te, seed, {"before_submit": hook})
    ok, reason = out["result"]
    return {"detected": (not ok) and reason.startswith("R1"), "reason": reason,
            "on_ledger": _rejected_on_chain(r.network, "R1")}


def s2_impersonation(clients, X_te, y_te, seed):
    out = {}

    def hook(ctx):
        if ctx.round_no != ATTACK_ROUND:
            return
        a = _hosp(ctx, "Hospital A")
        digest = ctx.network.store.put(serialize_update(-10 * np.ones(31)))
        tx = a.make_submit(ctx.round_no, digest, key=KeyPair.generate())  # signs as A, wrong key
        out["result"] = ctx.network.contract.submit(tx)

    r = _run(clients, X_te, y_te, seed, {"before_submit": hook})
    ok, reason = out["result"]
    return {"detected": (not ok) and reason.startswith("R2"), "reason": reason,
            "on_ledger": _rejected_on_chain(r.network, "R2")}


def s3_replay(clients, X_te, y_te, seed):
    out = {}

    def hook(ctx):
        if ctx.round_no != ATTACK_ROUND:
            return
        old = next(tx for tx in ctx.network.replicas[0].all_txs()
                   if tx["type"] == "submit" and tx["signer"] == "Hospital A"
                   and tx["body"]["round"] == ATTACK_ROUND - 1)
        out["result"] = ctx.network.contract.submit(copy.deepcopy(old))

    r = _run(clients, X_te, y_te, seed, {"before_submit": hook})
    ok, reason = out["result"]
    return {"detected": not ok, "reason": reason, "on_ledger": _rejected_on_chain(r.network, "R4")}


def s4_duplicate(clients, X_te, y_te, seed):
    out = {}

    def hook(ctx):
        if ctx.round_no != ATTACK_ROUND:
            return
        a = _hosp(ctx, "Hospital A")
        digest = ctx.network.store.put(serialize_update(np.zeros(31)))
        out["result"] = ctx.network.contract.submit(a.make_submit(ctx.round_no, digest))

    r = _run(clients, X_te, y_te, seed, {"after_submit": hook})
    ok, reason = out["result"]
    return {"detected": (not ok) and reason.startswith("R5"), "reason": reason,
            "on_ledger": _rejected_on_chain(r.network, "R5")}


def s5_malformed(clients, X_te, y_te, seed):
    out = {}

    def hook(ctx):
        if ctx.round_no != ATTACK_ROUND:
            return
        a = _hosp(ctx, "Hospital A")
        out["result"] = ctx.network.contract.submit(
            a.make_submit(ctx.round_no, ctx.digests["Hospital A"], n_samples=-50))

    r = _run(clients, X_te, y_te, seed, {"before_submit": hook})
    ok, reason = out["result"]
    return {"detected": (not ok) and reason.startswith("R6"), "reason": reason,
            "on_ledger": _rejected_on_chain(r.network, "R6")}


# ---------------------------------------------------------------- S6 off-chain tampering
def s6_offchain_tamper(clients, X_te, y_te, seed):
    from .offchain import deserialize_update

    def hook(ctx):
        if ctx.round_no != ATTACK_ROUND:
            return
        d = ctx.digests["Hospital B"]
        store = ctx.network.store
        store.tamper(d, serialize_update(10.0 * deserialize_update(store.get(d))))

    r = _run(clients, X_te, y_te, seed, {"after_submit": hook})
    rec = r.rejections[ATTACK_ROUND - 1]
    on_ledger = any(tx["type"] == "validation" and tx["body"]["client"] == "Hospital B"
                    and tx["body"]["round"] == ATTACK_ROUND and tx["body"]["status"] == "rejected"
                    for tx in r.network.replicas[0].all_txs())
    return {"detected": "Hospital B" in rec["digest_rejected"] and "Hospital B" not in rec["accepted"],
            "reason": "off-chain digest mismatch", "on_ledger": on_ledger}


# ---------------------------------------------------------------- S7 / S8 ledger rewriting
def _find_submit(replica, round_no, signer):
    for b in replica.chain:
        for i, tx in enumerate(b.txs):
            if tx["type"] == "submit" and tx["body"]["round"] == round_no and tx["signer"] == signer:
                return b, i
    raise LookupError


def s7_edit_no_rehash(clients, X_te, y_te, seed):
    r = _run(clients, X_te, y_te, seed, {}, rounds=LEDGER_ATTACK_ROUNDS)
    net = r.network
    victim = net.replicas[1]
    blk, i = _find_submit(victim, ATTACK_ROUND, "Hospital C")
    blk.txs[i]["body"]["digest"] = sha256_hex(b"forged update")
    ok, msg = victim.audit(net.base_directory)
    return {"detected": not ok, "reason": msg, "on_ledger": True,
            "located_block": blk.height, "audit_points_to": msg.split(":")[0]}


def s8_edit_full_rehash(clients, X_te, y_te, seed):
    r = _run(clients, X_te, y_te, seed, {}, rounds=LEDGER_ATTACK_ROUNDS)
    net = r.network
    victim = net.replicas[1]
    blk, i = _find_submit(victim, ATTACK_ROUND, "Hospital C")
    blk.txs[i]["body"]["digest"] = sha256_hex(b"forged update")
    # attacker recomputes the Merkle root and re-links every later block
    prev = GENESIS_PREV
    for b in victim.chain:
        b.prev_hash = prev
        b.merkle_root = merkle_root([tx_hash(t) for t in b.txs])
        prev = b.hash
    cmp = compare_replicas(net.replicas)
    ok, msg = victim.audit(net.base_directory)
    by_vote = victim.owner in cmp["dissenting"] and cmp["has_majority"]
    by_sig = (not ok) and "signature" in msg
    return {"detected": by_vote or by_sig, "reason": f"replica vote={by_vote}; signature={by_sig}",
            "on_ledger": True, "by_vote": by_vote, "by_signature": by_sig}


SCENARIOS = [
    ("S1", "Unregistered client submits an update", "registry check", s1_unregistered),
    ("S2", "Attacker signs as Hospital A with own key", "ECDSA signature check", s2_impersonation),
    ("S3", "Replay of Hospital A's previous-round tx", "round binding", s3_replay),
    ("S4", "Duplicate submission in the same round", "one-per-round rule", s4_duplicate),
    ("S5", "Malformed submission (negative count)", "input validation", s5_malformed),
    ("S6", "Stored update modified after hashing", "off-chain hash binding", s6_offchain_tamper),
    ("S7", "One replica edited, no re-hashing", "hash-chain / Merkle audit", s7_edit_no_rehash),
    ("S8", "One replica edited, full re-hashing", "replica comparison + signature", s8_edit_full_rehash),
]
