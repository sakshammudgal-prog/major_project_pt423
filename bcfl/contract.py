"""Coordination smart contract: deterministic rules R1-R6 for submit transactions.

The contract keeps the participant registry and round state derived from the ledger.
Rejected submissions are not admitted; instead a consortium-signed ``rejected`` record with
the reason is queued for the next block so that failed attempts are auditable.
"""
from __future__ import annotations

import re
import time

from .crypto import KeyPair, load_public_key, sha256_hex, verify
from .ledger import make_tx, tx_payload_bytes

HEX256 = re.compile(r"^[0-9a-f]{64}$")


class Contract:
    def __init__(self, consortium_id: str, consortium_key: KeyPair):
        self.cid = consortium_id
        self.ckey = consortium_key
        self.registry: dict[str, bytes] = {}        # client id -> public key bytes
        self.open_round: int | None = None
        self.global_digest: str | None = None
        self.submitted: dict[int, dict[str, dict]] = {}  # round -> client -> submit tx
        self.seen_nonces: set[tuple[str, str]] = set()
        self.pending: list[dict] = []                # txs waiting for the next block
        self.check_time = 0.0                        # cumulative time spent in rule checks

    # -- consortium operations ------------------------------------------------------
    def register(self, client_id: str, public_key: bytes) -> dict:
        tx = make_tx("register", {"client_id": client_id, "public_key": public_key.hex(),
                                  "pk_hash": sha256_hex(public_key)}, self.cid, self.ckey)
        self.registry[client_id] = public_key
        self.pending.append(tx)
        return tx

    def open(self, round_no: int, global_digest: str) -> dict:
        self.open_round = round_no
        self.global_digest = global_digest
        self.submitted.setdefault(round_no, {})
        tx = make_tx("open_round", {"round": round_no, "global_digest": global_digest},
                     self.cid, self.ckey)
        self.pending.append(tx)
        return tx

    def record_validation(self, round_no: int, client: str, accepted: bool, reason: str) -> dict:
        tx = make_tx("validation", {"round": round_no, "client": client,
                                    "status": "accepted" if accepted else "rejected",
                                    "reason": reason}, self.cid, self.ckey)
        self.pending.append(tx)
        return tx

    def close(self, round_no: int, new_digest: str, aggregated: list[str]) -> dict:
        tx = make_tx("close_round", {"round": round_no, "global_digest": new_digest,
                                     "aggregated": sorted(aggregated)}, self.cid, self.ckey)
        self.open_round = None
        self.pending.append(tx)
        return tx

    def drain(self) -> list[dict]:
        txs, self.pending = self.pending, []
        return txs

    # -- submit processing (R1-R6) --------------------------------------------------
    def _check(self, tx: dict) -> str | None:
        signer = tx.get("signer")
        body = tx.get("body", {}) or {}
        # R1: signer must be registered
        if signer not in self.registry:
            return "R1: signer not registered"
        # R2: signature verifies under the signer's registered key
        if not verify(load_public_key(self.registry[signer]), tx_payload_bytes(tx), tx.get("sig", "")):
            return "R2: invalid signature"
        # R3: client field equals signer
        if body.get("client") != signer:
            return "R3: client field does not match signer"
        # R4: transaction names the currently open round
        if self.open_round is None or body.get("round") != self.open_round:
            return "R4: not for the open round"
        # R5: at most one submission per client per round (and no nonce reuse)
        if signer in self.submitted.get(self.open_round, {}):
            return "R5: duplicate submission in round"
        if (signer, str(body.get("nonce"))) in self.seen_nonces:
            return "R5: nonce already used"
        # R6: input validity
        n = body.get("n_samples")
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            return "R6: sample count must be a positive integer"
        if not isinstance(body.get("digest"), str) or not HEX256.match(body["digest"]):
            return "R6: digest is not a 256-bit hex string"
        return None

    def submit(self, tx: dict) -> tuple[bool, str]:
        t0 = time.perf_counter()
        if tx.get("type") != "submit":
            reason = "not a submit transaction"
        else:
            reason = self._check(tx)
        self.check_time += time.perf_counter() - t0
        body = tx.get("body", {}) or {}
        if reason is None:
            self.submitted[self.open_round][tx["signer"]] = tx
            self.seen_nonces.add((tx["signer"], str(body.get("nonce"))))
            self.pending.append(tx)
            return True, "accepted"
        rej = make_tx("rejected", {"round": body.get("round"),
                                   "claimed_client": body.get("client", tx.get("signer")),
                                   "reason": reason}, self.cid, self.ckey)
        self.pending.append(rej)
        return False, reason

    def accepted_submissions(self, round_no: int) -> dict[str, dict]:
        return dict(self.submitted.get(round_no, {}))
