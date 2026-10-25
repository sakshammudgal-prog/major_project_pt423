"""The blockchain-enabled FL framework: hospitals, network (ledger + contract + replicas),
aggregator, and the round protocol of Section IV-E. Also runs plain FedAvg for comparison."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .contract import Contract
from .crypto import KeyPair, sha256_hex
from .data import ClientData
from .ledger import Directory, Replica, build_block, make_tx, tx_size
from .model import evaluate, init_params, sgd_train
from .offchain import OffChainStore, deserialize_update, serialize_update

CONSORTIUM = "Consortium"
KAPPA = 3.0
ROUNDS = 30
LOCAL_EPOCHS = 5


def model_digest(w: np.ndarray) -> str:
    return sha256_hex(serialize_update(w))


# --------------------------------------------------------------------------- hospital
class Hospital:
    def __init__(self, data: ClientData, seed: int, attack: str | None = None):
        self.name = data.name
        self.data = data
        self.key = KeyPair.generate()
        self.rng = np.random.default_rng(seed)
        self.attack = attack

    def local_update(self, w: np.ndarray, epochs: int) -> np.ndarray:
        y = self.data.y
        if self.attack == "labelflip":
            y = 1 - y
        w_new = sgd_train(w, self.data.X, y, epochs, self.rng)
        delta = w_new - w
        if self.attack == "signflip":
            delta = -5.0 * delta
        elif self.attack == "noise":
            delta = self.rng.normal(0.0, 1.0, size=delta.shape)
        return delta

    def make_submit(self, round_no: int, digest: str, n_samples: int | None = None,
                    nonce: str | None = None, client: str | None = None,
                    key: KeyPair | None = None) -> dict:
        body = {"round": round_no, "client": client or self.name, "digest": digest,
                "n_samples": self.data.n if n_samples is None else n_samples,
                "nonce": nonce or uuid.uuid4().hex}
        return make_tx("submit", body, self.name, key or self.key)


# --------------------------------------------------------------------------- network
class Network:
    """Consortium key, contract, off-chain store and one ledger replica per hospital."""

    def __init__(self, hospitals: list[Hospital]):
        self.ckey = KeyPair.generate()
        self.contract = Contract(CONSORTIUM, self.ckey)
        self.store = OffChainStore()
        self.base_directory = Directory(CONSORTIUM, self.ckey.public_bytes())
        self.replicas = [Replica(h.name, self.base_directory) for h in hospitals]
        self.block_times: list[float] = []          # wall time of whole commit (all replicas)
        self.replica_validation: list[float] = []   # per-replica validation time
        self.block_sizes: list[int] = []
        for h in hospitals:
            self.contract.register(h.name, h.key.public_bytes())
        self.commit()                                # genesis block with registrations

    def commit(self) -> float:
        txs = self.contract.drain()
        prev = self.replicas[0].chain[-1] if self.replicas[0].chain else None
        height = prev.height + 1 if prev else 0
        prev_hash = prev.hash if prev else "0" * 64
        t0 = time.perf_counter()
        block = build_block(height, prev_hash, txs)
        for r in self.replicas:
            self.replica_validation.append(r.append(block))
        dt = time.perf_counter() - t0
        self.block_times.append(dt)
        self.block_sizes.append(block.size())
        return dt


# --------------------------------------------------------------------------- aggregator
@dataclass
class AggregationResult:
    w: np.ndarray
    accepted: list[str]
    digest_rejected: list[str]
    screen_rejected: list[str]
    norms: dict[str, float]


def screen_by_norm(deltas: dict[str, np.ndarray], kappa: float | None = None) -> tuple[list[str], list[str]]:
    """Eq. (1): accept k iff ||d_k|| <= kappa * median_j ||d_j||."""
    kappa = KAPPA if kappa is None else kappa
    if not deltas:
        return [], []
    norms = {k: float(np.linalg.norm(v)) for k, v in deltas.items()}
    med = float(np.median(list(norms.values())))
    acc = [k for k, n in norms.items() if n <= kappa * med]
    rej = [k for k in norms if k not in acc]
    return acc, rej


def weighted_fedavg(w: np.ndarray, deltas: dict[str, np.ndarray], counts: dict[str, int]) -> np.ndarray:
    """Eq. (2): w_{t+1} = w_t + sum_k (n_k / n_A) d_k over accepted clients."""
    if not deltas:
        return w.copy()
    n_a = sum(counts[k] for k in deltas)
    return w + sum((counts[k] / n_a) * d for k, d in deltas.items())


class Aggregator:
    def __init__(self, network: Network, screening: bool):
        self.net = network
        self.screening = screening

    def aggregate(self, round_no: int, w: np.ndarray) -> AggregationResult:
        subs = self.net.contract.accepted_submissions(round_no)
        deltas, counts, bad = {}, {}, []
        for client, tx in subs.items():
            blob = self.net.store.get(tx["body"]["digest"])
            if blob is None or sha256_hex(blob) != tx["body"]["digest"]:
                bad.append(client)
                self.net.contract.record_validation(round_no, client, False, "off-chain digest mismatch")
                continue
            deltas[client] = deserialize_update(blob)
            counts[client] = tx["body"]["n_samples"]   # sample count taken from the chain
        norms = {k: float(np.linalg.norm(v)) for k, v in deltas.items()}
        if self.screening:
            acc, rej = screen_by_norm(deltas)
        else:
            acc, rej = list(deltas), []
        for k in acc:
            self.net.contract.record_validation(round_no, k, True, "ok")
        for k in rej:
            self.net.contract.record_validation(round_no, k, False, f"norm screening (kappa={KAPPA})")
        new_w = weighted_fedavg(w, {k: deltas[k] for k in acc}, counts)
        return AggregationResult(new_w, acc, bad, rej, norms)


# --------------------------------------------------------------------------- runs
@dataclass
class RunResult:
    mode: str
    w: np.ndarray
    history: list[dict] = field(default_factory=list)       # per-round test metrics
    timings: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    rejections: list[dict] = field(default_factory=list)    # per-round screening outcome
    network: Network | None = None
    hospitals: list[Hospital] | None = None


Hook = Callable[["RoundContext"], None]


@dataclass
class RoundContext:
    round_no: int
    w: np.ndarray
    network: Network
    hospitals: list[Hospital]
    submissions: dict[str, tuple[bool, str]] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)


def run_federated(clients: list[ClientData], X_test, y_test, seed: int, mode: str = "screen",
                  rounds: int = ROUNDS, epochs: int = LOCAL_EPOCHS, attack: str | None = None,
                  attacker: str = "Hospital C", hooks: dict[str, Hook] | None = None,
                  track_history: bool = True) -> RunResult:
    """mode: 'fedavg' (trusted coordinator), 'audit' (ledger, no screening),
    'screen' (ledger + norm screening)."""
    hooks = hooks or {}
    hospitals = [Hospital(c, seed * 100 + i, attack if c.name == attacker else None)
                 for i, c in enumerate(clients)]
    w = init_params()
    res = RunResult(mode, w, hospitals=hospitals)
    net = Network(hospitals) if mode != "fedavg" else None
    agg = Aggregator(net, screening=(mode == "screen")) if net else None
    res.network = net
    T = res.timings

    for t in range(1, rounds + 1):
        t_round = time.perf_counter()
        if mode == "fedavg":
            t0 = time.perf_counter()
            deltas = {h.name: h.local_update(w, epochs) for h in hospitals}
            T["train"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            w = weighted_fedavg(w, deltas, {h.name: h.data.n for h in hospitals})
            T["aggregate"].append(time.perf_counter() - t0)
        else:
            ctx = RoundContext(t, w, net, hospitals)
            t0 = time.perf_counter()
            net.contract.open(t, net.store.put(serialize_update(w)))
            book = time.perf_counter() - t0

            t0 = time.perf_counter()
            deltas = {h.name: h.local_update(w, epochs) for h in hospitals}
            T["train"].append(time.perf_counter() - t0)

            hash_sign, txs = 0.0, {}
            for h in hospitals:
                t0 = time.perf_counter()
                blob = serialize_update(deltas[h.name])
                digest = net.store.put(blob)
                txs[h.name] = h.make_submit(t, digest)
                hash_sign += time.perf_counter() - t0
                ctx.digests[h.name] = digest
            T["hash_sign"].append(hash_sign)

            if "before_submit" in hooks:
                hooks["before_submit"](ctx)
            check0 = net.contract.check_time
            for h in hospitals:
                ctx.submissions[h.name] = net.contract.submit(txs[h.name])
            if "after_submit" in hooks:
                hooks["after_submit"](ctx)
            T["contract"].append(net.contract.check_time - check0)

            t0 = time.perf_counter()
            ar = agg.aggregate(t, w)
            T["screen_verify"].append(time.perf_counter() - t0)
            w = ar.w
            res.rejections.append({"round": t, "screen_rejected": ar.screen_rejected,
                                   "digest_rejected": ar.digest_rejected, "accepted": ar.accepted,
                                   "norms": ar.norms})

            t0 = time.perf_counter()
            net.contract.close(t, net.store.put(serialize_update(w)), ar.accepted)
            book += time.perf_counter() - t0
            T["bookkeeping"].append(book)
            T["block_commit"].append(net.commit())
            if "after_commit" in hooks:
                hooks["after_commit"](ctx)
        T["round"].append(time.perf_counter() - t_round)
        if track_history:
            res.history.append(evaluate(w, X_test, y_test))
    res.w = w
    return res


def run_centralized(clients: list[ClientData], X_test, y_test, seed: int, epochs: int = 150):
    X = np.vstack([c.X for c in clients])
    y = np.concatenate([c.y for c in clients])
    w = sgd_train(init_params(), X, y, epochs, np.random.default_rng(seed))
    return w, evaluate(w, X_test, y_test)


def recompute_aggregation(network: Network, round_no: int) -> tuple[bool, str]:
    """Any hospital can repeat round ``round_no`` from the chain and the off-chain store and
    check it against the digest written in close_round."""
    chain_txs = list(network.replicas[0].all_txs())
    close = next(tx for tx in chain_txs if tx["type"] == "close_round" and tx["body"]["round"] == round_no)
    opened = next(tx for tx in chain_txs if tx["type"] == "open_round" and tx["body"]["round"] == round_no)
    w_prev = deserialize_update(network.store.get(opened["body"]["global_digest"]))
    subs = {tx["signer"]: tx for tx in chain_txs
            if tx["type"] == "submit" and tx["body"]["round"] == round_no}
    acc = close["body"]["aggregated"]
    deltas = {k: deserialize_update(network.store.get(subs[k]["body"]["digest"])) for k in acc}
    counts = {k: subs[k]["body"]["n_samples"] for k in acc}
    w_new = weighted_fedavg(w_prev, deltas, counts)
    return model_digest(w_new) == close["body"]["global_digest"], model_digest(w_new)


def submit_tx_size(network: Network) -> int:
    for tx in network.replicas[0].all_txs():
        if tx["type"] == "submit":
            return tx_size(tx)
    return 0
