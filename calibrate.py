"""Choose the screening threshold kappa on calibration seeds 100-104 (never used for
evaluation). Prints, for each kappa, the honest-update rejection rate with no attack and the
attacker rejection rate under the scaled sign-flip attack, in both partitions."""
from __future__ import annotations

import numpy as np

import bcfl.framework as fw
from bcfl.data import make_partitions

CALIB_SEEDS = range(100, 105)


def rates(kappa, setting, attack):
    fw.KAPPA = kappa
    rej_att = rej_hon = n_att = n_hon = 0
    for s in CALIB_SEEDS:
        cl, Xt, yt = make_partitions(s, setting)
        r = fw.run_federated(cl, Xt, yt, s, "screen", attack=attack, track_history=False)
        for x in r.rejections:
            for k in x["norms"]:
                bad = attack is not None and k == "Hospital C"
                rej = k in x["screen_rejected"]
                if bad:
                    n_att += 1; rej_att += rej
                else:
                    n_hon += 1; rej_hon += rej
    return (rej_hon / n_hon), (rej_att / n_att if n_att else float("nan"))


if __name__ == "__main__":
    print(f"{'kappa':>6} | {'setting':>7} | honest rej (no attack) | attacker rej (sign-flip)")
    for kappa in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        for setting in ["iid", "noniid"]:
            h, _ = rates(kappa, setting, None)
            _, a = rates(kappa, setting, "signflip")
            print(f"{kappa:>6} | {setting:>7} | {100 * h:21.2f}% | {100 * a:23.1f}%")
