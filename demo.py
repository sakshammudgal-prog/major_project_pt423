"""A narrated walk-through of one training run, for presentations.

    python demo.py
"""
import numpy as np

from bcfl.attacks import s2_impersonation, s6_offchain_tamper, s8_edit_full_rehash
from bcfl.data import make_partitions
from bcfl.framework import recompute_aggregation, run_federated
from bcfl.model import evaluate


def line(t=""):
    print(f"\n=== {t} " + "=" * max(0, 60 - len(t)))


clients, X_te, y_te = make_partitions(0, "iid")
line("Hospitals and their private data (never shared)")
for c in clients:
    print(f"  {c.name}: {c.n} patients, {int(c.y.sum())} malignant")

line("Training 30 rounds with ledger + screening")
r = run_federated(clients, X_te, y_te, seed=0, mode="screen")
print(f"  final test accuracy: {100 * r.history[-1]['acc']:.2f}%")
net = r.network
chain = net.replicas[0].chain
print(f"  ledger: {len(chain)} blocks, {net.replicas[0].size_bytes():,} bytes, "
      f"replica tips agree: {len({rep.tip for rep in net.replicas}) == 1}")

line("What one round looks like on the chain (round 1)")
for tx in chain[1].txs:
    b = tx["body"]
    detail = {k: (v[:12] + "…" if isinstance(v, str) and len(v) > 20 else v) for k, v in b.items()}
    print(f"  {tx['type']:<12} signed by {tx['signer']:<11} {detail}")

line("Any hospital can recompute an aggregation from the chain")
ok, _ = recompute_aggregation(net, 17)
print(f"  round 17 recomputed and matches close_round digest: {ok}")

line("Attacks")
print("  impersonation      ->", s2_impersonation(clients, X_te, y_te, 0)["reason"])
print("  tampered update    ->", s6_offchain_tamper(clients, X_te, y_te, 0)["reason"])
print("  rewritten replica  ->", s8_edit_full_rehash(clients, X_te, y_te, 0)["reason"])

line("Poisoning by Hospital C (sign-flip x5)")
for mode in ["audit", "screen"]:
    p = run_federated(clients, X_te, y_te, 0, mode, attack="signflip", track_history=False)
    rej = sum("Hospital C" in x["screen_rejected"] for x in p.rejections)
    print(f"  {mode:<7} accuracy {100 * evaluate(p.w, X_te, y_te)['acc']:6.2f}%   "
          f"attacker updates rejected: {rej}/30")
print("\nThe ledger makes the attack attributable; only screening stops it.")
