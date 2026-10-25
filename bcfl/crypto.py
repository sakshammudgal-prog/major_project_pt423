"""Cryptographic primitives: SHA-256 digests, canonical JSON, ECDSA P-256 signatures."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> bytes:
    """Deterministic JSON encoding: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass
class KeyPair:
    private_key: ec.EllipticCurvePrivateKey

    @classmethod
    def generate(cls) -> "KeyPair":
        return cls(ec.generate_private_key(ec.SECP256R1()))

    @property
    def public_key(self) -> ec.EllipticCurvePublicKey:
        return self.private_key.public_key()

    def public_bytes(self) -> bytes:
        return public_key_bytes(self.public_key)

    def sign(self, message: bytes) -> str:
        return self.private_key.sign(message, ec.ECDSA(hashes.SHA256())).hex()


def public_key_bytes(pk: ec.EllipticCurvePublicKey) -> bytes:
    return pk.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint)


def load_public_key(data: bytes) -> ec.EllipticCurvePublicKey:
    return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), data)


def verify(public_key: ec.EllipticCurvePublicKey, message: bytes, signature_hex: str) -> bool:
    try:
        public_key.verify(bytes.fromhex(signature_hex), message, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError):
        return False
