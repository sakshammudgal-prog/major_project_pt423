"""Reproduce every table and figure of the paper.

    python run_experiments.py all            # everything (a few minutes)
    python run_experiments.py performance    # Table III + Fig. 2
    python run_experiments.py overhead       # Table IV + Fig. 3
    python run_experiments.py scaling        # Table V
    python run_experiments.py security       # Table VI
    python run_experiments.py poisoning      # Table VII + Fig. 4
    python run_experiments.py partitions     # Table II

Options: --seeds N (default 10), --out DIR (default results/)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bcfl.attacks import SCENARIOS
from bcfl.crypto import KeyPair
from bcfl.data import make_partitions
from bcfl.framework import (Hospital, run_centralized, run_federated, screen_by_norm,
                            submit_tx_size)
from bcfl.ledger import tx_size
from bcfl.offchain import serialize_update

EVAL_SEEDS = list(range(10))
CALIB_SEEDS = list(range(100, 105))
METRICS = ["acc", "prec", "rec", "f1", "auc"]


def pct(x):
    return 100.0 * x


def save(out: Path, name: str, obj):
    (out / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))


# ------------------------------------------------------------------------------- Table II
def exp_partitions(out, seeds):
    rows = {}
    for setting in ["iid", "noniid"]:
        clients, _, _ = make_partitions(0, setting)
        rows[setting] = {c.name: [c.n, int(c.y.sum())] for c in clients}
    lines = ["| Client | IID (cases / malignant) | Non-IID (cases / malignant) |", "|---|---|---|"]
    for name in rows["iid"]:
        a, b = rows["iid"][name], rows["noniid"][name]
        lines.append(f"| {name} | {a[0]} / {a[1]} | {b[0]} / {b[1]} |")
    save(out, "table2_partitions", rows)
    return "\n".join(lines)


# ------------------------------------------------------------------------------- Table III
def exp_performance(out, seeds):
    central = {}
    for s in seeds:
        clients, X_te, y_te = make_partitions(s, "iid")
        central[s] = run_centralized(clients, X_te, y_te, s)[1]
    res = {"iid": {}, "noniid": {}}
    curves = {"iid": {}, "noniid": {}}
    honest_rej = {}
    for setting in ["iid", "noniid"]:
        acc = {m: [] for m in ["Centralized", "FL (FedAvg)", "BC-FL audit", "BC-FL screen"]}
        curve = {"FL (FedAvg)": [], "BC-FL screen": []}
        rej, tot = 0, 0
        for s in seeds:
            clients, X_te, y_te = make_partitions(s, setting)
            acc["Centralized"].append(central[s])  # same training set, independent of partition
            for label, mode in [("FL (FedAvg)", "fedavg"), ("BC-FL audit", "audit"),
                                ("BC-FL screen", "screen")]:
                r = run_federated(clients, X_te, y_te, s, mode)
                acc[label].append(r.history[-1])
                if label in curve:
                    curve[label].append([h["acc"] for h in r.history])
                if mode == "screen":
                    rej += sum(len(x["screen_rejected"]) for x in r.rejections)
                    tot += sum(len(x["norms"]) for x in r.rejections)
        res[setting] = {m: {k: (float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals])))
                            for k in METRICS} for m, vals in acc.items()}
        curves[setting] = {k: np.mean(v, axis=0).tolist() for k, v in curve.items()}
        curves[setting]["Centralized"] = res[setting]["Centralized"]["acc"][0]
        honest_rej[setting] = rej / tot
    save(out, "table3_performance", {"metrics": res, "honest_rejection_rate": honest_rej})
    save(out, "fig2_curves", curves)
    lines = ["| Method | Acc. | Prec. | Rec. | F1 | AUC |", "|---|---|---|---|---|---|"]
    for setting, title in [("iid", "IID partition"), ("noniid", "Non-IID partition (Dirichlet α = 0.5)")]:
        lines.append(f"| *{title}* | | | | | |")
        for m, v in res[setting].items():
            lines.append(f"| {m} | {pct(v['acc'][0]):.2f}±{pct(v['acc'][1]):.2f} | {pct(v['prec'][0]):.2f} | "
                         f"{pct(v['rec'][0]):.2f} | {pct(v['f1'][0]):.2f}±{pct(v['f1'][1]):.2f} | {pct(v['auc'][0]):.2f} |")
    lines.append("")
    lines.append(f"Honest-update rejection by screening: IID {pct(honest_rej['iid']):.2f}%, "
                 f"non-IID {pct(honest_rej['noniid']):.2f}%")
    return "\n".join(lines)


# ------------------------------------------------------------------------------- Table IV
def exp_overhead(out, seeds):
    comps = {k: [] for k in ["train", "hash_sign", "contract", "screen_verify", "block_commit",
                             "bookkeeping", "round"]}
    fl_round, val, blocks, ledger, audits, hash_audits, subs = [], [], [], [], [], [], []
    for s in seeds:
        clients, X_te, y_te = make_partitions(s, "iid")
        # warm-up so first-call costs do not leak into the measurement
        run_federated(clients, X_te, y_te, s, "fedavg", rounds=2, track_history=False)
        fl = run_federated(clients, X_te, y_te, s, "fedavg", track_history=False)
        fl_round.append(np.mean(fl.timings["round"]))
        r = run_federated(clients, X_te, y_te, s, "screen", track_history=False)
        for k in comps:
            comps[k].append(np.mean(r.timings[k]))
        net = r.network
        val.append(np.mean(net.replica_validation[1 * 3:]))     # skip genesis block
        blocks.append(np.mean(net.block_sizes[1:]))
        ledger.append(net.replicas[0].size_bytes())
        t0 = time.perf_counter()
        ok, _ = net.replicas[0].audit(net.base_directory)
        audits.append(time.perf_counter() - t0)
        assert ok
        t0 = time.perf_counter()
        ok, _ = net.replicas[0].audit(net.base_directory, check_signatures=False)
        hash_audits.append(time.perf_counter() - t0)
        assert ok
        subs.append(submit_tx_size(net))
    upd = len(serialize_update(np.zeros(31)))
    ms = lambda xs: 1000 * float(np.mean(xs))
    table = {
        "Round time, plain FL (ms)": ms(fl_round),
        "Round time, framework (ms)": ms(comps["round"]),
        "Overhead per round (ms)": ms(comps["round"]) - ms(fl_round),
        "  of which block commit, 3 replicas (ms)": ms(comps["block_commit"]),
        "  of which client hash + sign (ms)": ms(comps["hash_sign"]),
        "  of which contract checks (ms)": ms(comps["contract"]),
        "  of which screening + digest verify (ms)": ms(comps["screen_verify"]),
        "Block validation per replica (ms)": ms(val),
        "Signed submit transaction (B)": float(np.mean(subs)),
        "Serialized model update, off-chain (B)": upd,
        "Mean block size per round (B)": float(np.mean(blocks)),
        "Ledger after 30 rounds (B)": float(np.mean(ledger)),
        "Hash-chain + Merkle audit of the chain (ms)": ms(hash_audits),
        "Full audit incl. every signature (ms)": ms(audits),
    }
    breakdown = {k: ms(v) for k, v in comps.items()}
    breakdown["fl_round"] = ms(fl_round)
    save(out, "table4_overhead", table)
    save(out, "fig3_breakdown", breakdown)
    lines = ["| Quantity | Value |", "|---|---|"]
    lines += [f"| {k} | {v:,.2f} |" if isinstance(v, float) and v < 1000 else f"| {k} | {v:,.0f} |"
              for k, v in table.items()]
    return "\n".join(lines)


# ------------------------------------------------------------------------------- Table V
def exp_scaling(out, seeds):
    from bcfl.framework import Hospital
    from bcfl.data import ClientData
    h = Hospital(ClientData("Hospital A", np.zeros((1, 30)), np.zeros(1)), 0)
    rows = []
    for p in [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]:
        delta = np.random.default_rng(0).normal(size=p)
        reps = 20 if p <= 10 ** 5 else 5
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            blob = serialize_update(delta)
            from bcfl.crypto import sha256_hex
            tx = h.make_submit(1, sha256_hex(blob), n_samples=100)
            times.append(time.perf_counter() - t0)
        rows.append({"params": p, "offchain_bytes": len(blob), "onchain_tx_bytes": tx_size(tx),
                     "hash_sign_ms": 1000 * float(np.median(times))})
    save(out, "table5_scaling", rows)
    lines = ["| Parameters | Off-chain size | On-chain tx | Hash + sign |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| 10^{int(np.log10(r['params']))} | {r['offchain_bytes'] / 1e6:.2f} MB | "
                     f"{r['onchain_tx_bytes']} B | {r['hash_sign_ms']:.2f} ms |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------- Table VI
def exp_security(out, seeds):
    res = []
    for sid, desc, mech, fn in SCENARIOS:
        det, led, reasons = 0, 0, set()
        for s in seeds:
            clients, X_te, y_te = make_partitions(s, "iid")
            r = fn(clients, X_te, y_te, s)
            det += bool(r["detected"])
            led += bool(r["on_ledger"])
            reasons.add(r["reason"])
        res.append({"id": sid, "scenario": desc, "mechanism": mech, "detected": det,
                    "trials": len(seeds), "recorded": led, "reasons": sorted(reasons)})
    save(out, "table6_security", res)
    lines = ["| ID | Scenario | Mechanism | Detected | Recorded |", "|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['id']} | {r['scenario']} | {r['mechanism']} | {r['detected']}/{r['trials']} | "
                     f"{r['recorded']}/{r['trials']} |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------- Table VII
def exp_poisoning(out, seeds):
    res = {}
    for setting in ["iid", "noniid"]:
        res[setting] = {}
        for attack, label in [("signflip", "Sign-flip ×5"), ("noise", "Gaussian noise"),
                              ("labelflip", "Label flip")]:
            accs = {"fedavg": [], "audit": [], "screen": []}
            rej, tot, honest_rej, honest_tot = 0, 0, 0, 0
            for s in seeds:
                clients, X_te, y_te = make_partitions(s, setting)
                for mode in accs:
                    r = run_federated(clients, X_te, y_te, s, mode, attack=attack, track_history=False)
                    from bcfl.model import evaluate
                    accs[mode].append(evaluate(r.w, X_te, y_te)["acc"])
                    if mode == "screen":
                        for x in r.rejections:
                            rej += "Hospital C" in x["screen_rejected"]
                            tot += 1
                            honest_rej += sum(1 for k in x["screen_rejected"] if k != "Hospital C")
                            honest_tot += sum(1 for k in x["norms"] if k != "Hospital C")
            res[setting][label] = {
                "FL": pct(np.mean(accs["fedavg"])), "FL_std": pct(np.std(accs["fedavg"])),
                "Audit": pct(np.mean(accs["audit"])), "Audit_std": pct(np.std(accs["audit"])),
                "Screen": pct(np.mean(accs["screen"])), "Screen_std": pct(np.std(accs["screen"])),
                "Rejected": pct(rej / tot), "HonestRejected": pct(honest_rej / honest_tot)}
    save(out, "table7_poisoning", res)
    lines = ["| Attack | FL | Audit | Screen | Attacker rejected (%) | Honest rejected (%) |",
             "|---|---|---|---|---|---|"]
    for setting, title in [("iid", "IID partition"), ("noniid", "Non-IID partition")]:
        lines.append(f"| *{title}* | | | | | |")
        for label, v in res[setting].items():
            lines.append(f"| {label} | {v['FL']:.2f} | {v['Audit']:.2f} | {v['Screen']:.2f}±{v['Screen_std']:.2f} | "
                         f"{v['Rejected']:.0f} | {v['HonestRejected']:.2f} |")
    return "\n".join(lines)


EXPERIMENTS = {
    "partitions": ("Table II - client partitions (seed 0)", exp_partitions),
    "performance": ("Table III - test performance", exp_performance),
    "overhead": ("Table IV - ledger cost", exp_overhead),
    "scaling": ("Table V - hash + sign cost vs model size", exp_scaling),
    "security": ("Table VI - protocol-level attacks", exp_security),
    "poisoning": ("Table VII - accuracy under poisoning by Hospital C", exp_poisoning),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=["all", *EXPERIMENTS])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = EVAL_SEEDS[:a.seeds]
    todo = list(EXPERIMENTS) if a.which == "all" else [a.which]
    report = [f"# Results ({len(seeds)} seeds)\n"]
    for key in todo:
        title, fn = EXPERIMENTS[key]
        t0 = time.time()
        print(f"== {title}", flush=True)
        md = fn(out, seeds)
        print(md, f"\n({time.time() - t0:.1f}s)\n", flush=True)
        report.append(f"## {title}\n\n{md}\n")
    if a.which == "all":
        (out / "RESULTS.md").write_text("\n".join(report))
        from make_figures import make_all
        make_all(out)


if __name__ == "__main__":
    main()
