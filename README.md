# Auditable Federated Learning on a Permissioned Blockchain

Reference implementation for *"Auditable Federated Learning for Multi-Hospital Diagnosis Using
a Permissioned Blockchain: Design and Evaluation of Accountability, Overhead and Robustness"*
(Mudgal, Tyagi, Solanki, Mandal, Garg — IILM University).

Three simulated hospitals train a breast-cancer classifier with federated averaging. A
replicated, permissioned ledger records who took part, signed SHA-256 digests of every model
update and the outcome of every check. Patient data and model parameters never go on-chain.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                 # 18 tests: crypto, ledger, contract, all 8 attacks
python demo.py                      # narrated walk-through (≈5 s)
python run_experiments.py all       # every table + figure, 10 seeds (≈2 min)
python calibrate.py                 # choose kappa on calibration seeds 100-104
```

Results land in `results/` (`RESULTS.md`, one JSON per table, `fig2/3/4*.png`).
Run one experiment with e.g. `python run_experiments.py poisoning --seeds 3`.

## Layout

| File | Paper section | What it does |
|---|---|---|
| `bcfl/crypto.py` | IV-B | SHA-256, canonical JSON, ECDSA P-256 keys/signatures |
| `bcfl/ledger.py` | IV-B, Table I | signed txs, Merkle root, hash-chained blocks, replicas, audit, replica vote |
| `bcfl/contract.py` | IV-C | registry, round state, rules **R1–R6**, `rejected` / `validation` records |
| `bcfl/offchain.py` | IV-A | content-addressed store for serialized updates |
| `bcfl/data.py` | V-A | WDBC, stratified 20 % test split, IID 2:4:3, Dirichlet α = 0.5, federated standardisation |
| `bcfl/model.py` | V-B | 31-parameter L2 logistic regression, mini-batch SGD, metrics |
| `bcfl/framework.py` | IV-D, IV-E | hospitals, network, aggregator, norm screening (Eq. 1), weighted FedAvg (Eq. 2), round protocol, recomputation |
| `bcfl/attacks.py` | VI-C | protocol attacks **S1–S8** |
| `run_experiments.py` | VI | Tables II–VII |
| `make_figures.py` | VI | Figs. 2–4 |

### Round protocol (`framework.run_federated`)
1. Consortium stores `w_t` off-chain and writes `open_round(t, digest)`.
2. Each hospital trains *E* = 5 local epochs, stores its update off-chain, signs a `submit`.
3. Contract applies R1–R6; failures become signed `rejected` records.
4. Aggregator re-hashes each accepted update, screens by `‖Δk‖ ≤ 3 · median ‖Δj‖`.
5. Weighted FedAvg with on-chain sample counts; `close_round(t, new digest, accepted list)`.
6. Block built; every replica checks hash link, Merkle root and every signature, then appends.

`recompute_aggregation(network, t)` lets any hospital repeat round *t* from the chain and confirm
the published digest.

## Reproduced results (10 seeds, this code) vs. the paper

| Quantity | Paper | This code |
|---|---|---|
| IID accuracy, FL = BC-FL audit = BC-FL screen | 97.37 ± 1.24 | 97.46 ± 1.27 |
| Centralized accuracy | 97.19 ± 1.61 | 97.19 ± 1.61 |
| Non-IID accuracy (all three FL variants) | 96.84 ± 1.48 | 97.19 ± 1.66 |
| Protocol attacks S1–S8 detected | 80 / 80 | 80 / 80 |
| Overhead per round | 4.65 ms | ≈ 5.2 ms |
| Block commit (3 replicas) / per-replica validation | 3.45 / 1.13 ms | 3.64 / 1.19 ms |
| Ledger after 30 rounds | 85,191 B | ≈ 84,700 B |
| Sign-flip ×5, IID: audit → screen | 8.33 → 96.93 % | 8.33 → 97.02 % |
| Sign-flip ×5, non-IID: audit → screen | 44.47 → 96.49 % | 43.60 → 96.58 % |
| Label flip, IID: attacker rejected | 0 % | 0 % |

Exact timings depend on the machine. Where the numbers differ, it is for these reasons:

* **Non-IID split.** The Dirichlet draws are not the ones used for the paper (seed 0 here gives
  A = 191/1, B = 133/82, C = 131/87). Non-IID label-flip results vary a lot across seeds
  (± 20 points), as the paper also notes.
* **Off-chain update size** is 376 B because `np.save` adds a 128-byte header to the
  248 B of raw float64 values; the signed submit tx is ≈ 366 B. Both stay constant with model size.
* **Audit time.** A hash-chain + Merkle audit takes ≈ 1.7 ms (comparable to the paper's 2.05 ms);
  re-verifying every signature as well takes ≈ 34 ms. Both are reported.
* **Gaussian noise** is σ = 1 per coordinate. The paper does not state σ; change it in
  `Hospital.local_update`.

## Assumptions and limits (same as the paper)
Single process, single ordering role, no network transport, so timings are lower bounds.
Honest registration, an honest replica majority and secret private keys are assumed.
The aggregator is checked after the fact by recomputation, not proven correct. There is no
differential privacy or secure aggregation.

## Next steps (paper's future work)
* Port `contract.py` to Hyperledger Fabric chaincode (rules R1–R6 map one-to-one).
* Replace `model.py` with a CNN on chest X-rays. The ledger code does not change, because only digests go on-chain.
* Add Krum / trimmed-mean in `Aggregator.aggregate`, and DP noise in `Hospital.local_update`.
