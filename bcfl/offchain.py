"""Content-addressed off-chain store for serialized model updates."""
from __future__ import annotations

import io

import numpy as np

from .crypto import sha256_hex


def serialize_update(delta: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(delta, dtype=np.float64), allow_pickle=False)
    return buf.getvalue()


def deserialize_update(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


class OffChainStore:
    """Blobs indexed by their SHA-256 digest. The store itself is untrusted: anyone with
    write access could replace a blob, which is why the aggregator re-hashes on read."""

    def __init__(self):
        self._blobs: dict[str, bytes] = {}

    def put(self, data: bytes) -> str:
        digest = sha256_hex(data)
        self._blobs[digest] = data
        return digest

    def get(self, digest: str) -> bytes | None:
        return self._blobs.get(digest)

    # used only by attack scenario S6 (tampering with stored updates)
    def tamper(self, digest: str, new_data: bytes) -> None:
        self._blobs[digest] = new_data
