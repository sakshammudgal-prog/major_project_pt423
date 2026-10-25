# Results (10 seeds)

## Table II - client partitions (seed 0)

| Client | IID (cases / malignant) | Non-IID (cases / malignant) |
|---|---|---|
| Hospital A | 101 / 42 | 191 / 1 |
| Hospital B | 202 / 69 | 133 / 82 |
| Hospital C | 152 / 59 | 131 / 87 |

## Table III - test performance

| Method | Acc. | Prec. | Rec. | F1 | AUC |
|---|---|---|---|---|---|
| *IID partition* | | | | | |
| Centralized | 97.19±1.61 | 97.83 | 94.52 | 96.10±2.30 | 99.41 |
| FL (FedAvg) | 97.46±1.27 | 98.76 | 94.29 | 96.45±1.82 | 99.42 |
| BC-FL audit | 97.46±1.27 | 98.76 | 94.29 | 96.45±1.82 | 99.42 |
| BC-FL screen | 97.46±1.27 | 98.76 | 94.29 | 96.45±1.82 | 99.42 |
| *Non-IID partition (Dirichlet α = 0.5)* | | | | | |
| Centralized | 97.19±1.61 | 97.83 | 94.52 | 96.10±2.30 | 99.41 |
| FL (FedAvg) | 97.19±1.66 | 99.01 | 93.33 | 96.02±2.46 | 99.39 |
| BC-FL audit | 97.19±1.66 | 99.01 | 93.33 | 96.02±2.46 | 99.39 |
| BC-FL screen | 97.19±1.66 | 99.01 | 93.33 | 96.02±2.46 | 99.39 |

Honest-update rejection by screening: IID 0.00%, non-IID 0.00%

## Table IV - ledger cost

| Quantity | Value |
|---|---|
| Round time, plain FL (ms) | 2.95 |
| Round time, framework (ms) | 8.18 |
| Overhead per round (ms) | 5.23 |
|   of which block commit, 3 replicas (ms) | 3.64 |
|   of which client hash + sign (ms) | 0.31 |
|   of which contract checks (ms) | 0.47 |
|   of which screening + digest verify (ms) | 0.44 |
| Block validation per replica (ms) | 1.19 |
| Signed submit transaction (B) | 366.20 |
| Serialized model update, off-chain (B) | 376 |
| Mean block size per round (B) | 2,777 |
| Ledger after 30 rounds (B) | 84,678 |
| Hash-chain + Merkle audit of the chain (ms) | 1.69 |
| Full audit incl. every signature (ms) | 33.90 |

## Table V - hash + sign cost vs model size

| Parameters | Off-chain size | On-chain tx | Hash + sign |
|---|---|---|---|
| 10^3 | 0.01 MB | 364 B | 0.05 ms |
| 10^4 | 0.08 MB | 366 B | 0.13 ms |
| 10^5 | 0.80 MB | 368 B | 0.81 ms |
| 10^6 | 8.00 MB | 366 B | 7.52 ms |
| 10^7 | 80.00 MB | 366 B | 134.39 ms |

## Table VI - protocol-level attacks

| ID | Scenario | Mechanism | Detected | Recorded |
|---|---|---|---|---|
| S1 | Unregistered client submits an update | registry check | 10/10 | 10/10 |
| S2 | Attacker signs as Hospital A with own key | ECDSA signature check | 10/10 | 10/10 |
| S3 | Replay of Hospital A's previous-round tx | round binding | 10/10 | 10/10 |
| S4 | Duplicate submission in the same round | one-per-round rule | 10/10 | 10/10 |
| S5 | Malformed submission (negative count) | input validation | 10/10 | 10/10 |
| S6 | Stored update modified after hashing | off-chain hash binding | 10/10 | 10/10 |
| S7 | One replica edited, no re-hashing | hash-chain / Merkle audit | 10/10 | 10/10 |
| S8 | One replica edited, full re-hashing | replica comparison + signature | 10/10 | 10/10 |

## Table VII - accuracy under poisoning by Hospital C

| Attack | FL | Audit | Screen | Attacker rejected (%) | Honest rejected (%) |
|---|---|---|---|---|---|
| *IID partition* | | | | | |
| Sign-flip ×5 | 8.33 | 8.33 | 97.02±1.48 | 99 | 0.00 |
| Gaussian noise | 91.84 | 91.84 | 97.02±1.48 | 100 | 0.00 |
| Label flip | 92.89 | 92.89 | 92.89±1.98 | 0 | 0.00 |
| *Non-IID partition* | | | | | |
| Sign-flip ×5 | 43.60 | 43.60 | 96.58±1.49 | 82 | 0.00 |
| Gaussian noise | 94.39 | 94.39 | 96.67±1.56 | 100 | 0.00 |
| Label flip | 80.96 | 80.96 | 81.75±21.75 | 38 | 0.00 |
