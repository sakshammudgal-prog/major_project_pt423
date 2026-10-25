"""Permissioned, replicated ledger: signed transactions, Merkle-rooted, hash-chained blocks.

Every transaction is a canonical JSON object ``{"type", "body", "signer", "sig"}`` where
``sig`` is an ECDSA P-256/SHA-256 signature over the canonical JSON of
``{"type", "body", "signer"}``. Each block stores the Merkle root of its transactions and the
hash of its predecessor. Every replica independently checks the hash link, the Merkle root
and every signature before appending a block (Section IV-B of the paper).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

from .crypto import KeyPair, canonical_json, load_public_key, sha256_hex, verify

GENESIS_PREV = "0" * 64


# --------------------------------------------------------------------------- transactions
def make_tx(tx_type: str, body: dict, signer: str, key: KeyPair) -> dict:
    payload = {"type": tx_type, "body": body, "signer": signer}
    return {**payload, "sig": key.sign(canonical_json(payload))}


def tx_payload_bytes(tx: dict) -> bytes:
    return canonical_json({"type": tx["type"], "body": tx["body"], "signer": tx["signer"]})


def tx_hash(tx: dict) -> str:
    return sha256_hex(canonical_json(tx))


def tx_size(tx: dict) -> int:
    return len(canonical_json(tx))


# --------------------------------------------------------------------------- merkle
def merkle_root(leaf_hashes: list[str]) -> str:
    if not leaf_hashes:
        return sha256_hex(b"")
    level = [bytes.fromhex(h) for h in leaf_hashes]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [bytes.fromhex(sha256_hex(level[i] + level[i + 1])) for i in range(0, len(level), 2)]
    return level[0].hex()


# --------------------------------------------------------------------------- blocks
@dataclass
class Block:
    height: int
    prev_hash: str
    merkle_root: str
    timestamp: float
    txs: list[dict] = field(default_factory=list)

    def header(self) -> dict:
        return {"height": self.height, "prev_hash": self.prev_hash,
                "merkle_root": self.merkle_root, "timestamp": self.timestamp}

    @property
    def hash(self) -> str:
        return sha256_hex(canonical_json(self.header()))

    def to_dict(self) -> dict:
        return {**self.header(), "txs": self.txs}

    def size(self) -> int:
        return len(canonical_json(self.to_dict()))


def build_block(height: int, prev_hash: str, txs: list[dict]) -> Block:
    return Block(height, prev_hash, merkle_root([tx_hash(t) for t in txs]), time.time(), list(txs))


# --------------------------------------------------------------------------- identities
class Directory:
    """Public-key directory used by replicas to verify signatures.

    The consortium key is known to all members out of band; hospital keys are learned from
    ``register`` transactions on the chain itself.
    """

    def __init__(self, consortium_id: str, consortium_pub: bytes):
        self.keys: dict[str, bytes] = {consortium_id: consortium_pub}
        self.consortium_id = consortium_id

    def copy(self) -> "Directory":
        d = Directory(self.consortium_id, self.keys[self.consortium_id])
        d.keys = dict(self.keys)
        return d


class BlockValidationError(Exception):
    pass


# --------------------------------------------------------------------------- replica
class Replica:
    """A full copy of the ledger held by one hospital."""

    def __init__(self, owner: str, directory: Directory):
        self.owner = owner
        self.directory = directory.copy()
        self.chain: list[Block] = []

    # -- validation -------------------------------------------------------------------
    def _verify_tx_sig(self, tx: dict, directory: Directory) -> bool:
        if tx["type"] == "register":
            # register is signed by the consortium; it introduces the hospital key
            pub = directory.keys.get(tx["signer"])
        else:
            pub = directory.keys.get(tx["signer"])
        if pub is None:
            return False
        return verify(load_public_key(pub), tx_payload_bytes(tx), tx["sig"])

    def validate_block(self, block: Block, prev: Block | None, directory: Directory,
                       check_signatures: bool = True) -> None:
        expected_prev = prev.hash if prev else GENESIS_PREV
        expected_height = prev.height + 1 if prev else 0
        if block.prev_hash != expected_prev:
            raise BlockValidationError(f"block {block.height}: broken hash link")
        if block.height != expected_height:
            raise BlockValidationError(f"block {block.height}: wrong height")
        if block.merkle_root != merkle_root([tx_hash(t) for t in block.txs]):
            raise BlockValidationError(f"block {block.height}: Merkle root mismatch")
        for tx in block.txs:
            if check_signatures and not self._verify_tx_sig(tx, directory):
                raise BlockValidationError(f"block {block.height}: invalid signature on {tx['type']} tx")
            if tx["type"] == "register":
                directory.keys[tx["body"]["client_id"]] = bytes.fromhex(tx["body"]["public_key"])

    def append(self, block: Block) -> float:
        """Validate then append; returns validation time in seconds."""
        t0 = time.perf_counter()
        prev = self.chain[-1] if self.chain else None
        self.validate_block(block, prev, self.directory)
        self.chain.append(copy.deepcopy(block))
        return time.perf_counter() - t0

    # -- audit ------------------------------------------------------------------------
    def audit(self, base_directory: Directory, check_signatures: bool = True) -> tuple[bool, str]:
        """Re-validate the whole chain from genesis. Returns (ok, message).
        With check_signatures=False only hash links and Merkle roots are checked."""
        directory = base_directory.copy()
        prev = None
        for blk in self.chain:
            try:
                self.validate_block(blk, prev, directory, check_signatures)
            except BlockValidationError as e:
                return False, str(e)
            prev = blk
        return True, "chain valid"

    @property
    def tip(self) -> str:
        return self.chain[-1].hash if self.chain else GENESIS_PREV

    def size_bytes(self) -> int:
        return sum(b.size() for b in self.chain)

    def all_txs(self):
        for b in self.chain:
            yield from b.txs


def compare_replicas(replicas: list[Replica]) -> dict:
    """Majority vote over chain tips; returns the honest tip and the dissenting owners."""
    tips: dict[str, list[str]] = {}
    for r in replicas:
        tips.setdefault(r.tip, []).append(r.owner)
    majority_tip, owners = max(tips.items(), key=lambda kv: len(kv[1]))
    dissent = [o for t, os in tips.items() if t != majority_tip for o in os]
    return {"majority_tip": majority_tip, "majority": owners, "dissenting": dissent,
            "has_majority": len(owners) > len(replicas) / 2}
