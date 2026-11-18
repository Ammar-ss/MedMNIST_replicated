# Full results

Everything on this page comes from `results/runs.json` (all 90 `result.json` files, exactly as the training
script wrote them) and the CSVs next to it. Test AUC and ACC are what `medmnist.Evaluator` reports on the test
split, for the epoch with the best validation AUC (the authors' rule). **Paper** is the ResNet-50 (224) row for 2D
and the ResNet-50+3D row for 3D in Tables 3 and 4 of MedMNIST v2. Run *k* always used seed *k*.

1. [Summary](#1-summary)
2. [Test AUC, every run](#2-test-auc-every-run)
3. [Test accuracy, every run](#3-test-accuracy-every-run)
4. [Float32 check on BreastMNIST](#4-float32-check-on-breastmnist)
5. [ChestMNIST: the best epoch comes early](#5-chestmnist-the-best-epoch-comes-early)
6. [Hardware and time](#6-hardware-and-time)
7. [How I read the misses](#7-how-i-read-the-misses)

## 1. Summary

Mean ± SD over 5 runs. **Diff.** is mine minus the paper. ⚠️ = more than one AUC point below the paper.

| Dataset | AUC (mean ± SD) | Paper | Diff. | ACC (mean ± SD) | Paper | Diff. |
|---|---|---|---|---|---|---|
| *2D, ResNet-50 (224)* | | | | | | |
| ChestMNIST | 0.7713 ± 0.0017 | 0.773 | −0.0017 | 0.9474 ± 0.0004 | 0.948 | −0.0006 |
| PathMNIST | 0.9896 ± 0.0025 | 0.989 | +0.0006 | 0.8953 ± 0.0129 | 0.892 | +0.0033 |
| DermaMNIST | 0.9110 ± 0.0039 | 0.912 | −0.0010 | 0.7322 ± 0.0233 | 0.731 | +0.0012 |
| OCTMNIST | 0.9604 ± 0.0017 | 0.958 | +0.0024 | 0.7680 ± 0.0078 | 0.776 | −0.0080 |
| PneumoniaMNIST | 0.9562 ± 0.0114 | 0.962 | −0.0058 | 0.8388 ± 0.0271 | 0.884 | −0.0452 |
| RetinaMNIST | 0.7075 ± 0.0127 | 0.716 | −0.0085 | 0.5110 ± 0.0236 | 0.511 | +0.0000 |
| BreastMNIST ⚠️ | 0.8343 ± 0.0356 | 0.866 | −0.0317 | 0.8167 ± 0.0407 | 0.842 | −0.0253 |
| BloodMNIST | 0.9975 ± 0.0001 | 0.997 | +0.0005 | 0.9550 ± 0.0026 | 0.950 | +0.0050 |
| TissueMNIST | 0.9275 ± 0.0018 | 0.932 | −0.0045 | 0.6700 ± 0.0050 | 0.680 | −0.0100 |
| OrganAMNIST | 0.9963 ± 0.0017 | 0.998 | −0.0017 | 0.9376 ± 0.0050 | 0.947 | −0.0094 |
| OrganCMNIST | 0.9937 ± 0.0006 | 0.993 | +0.0007 | 0.9189 ± 0.0025 | 0.911 | +0.0079 |
| OrganSMNIST | 0.9757 ± 0.0014 | 0.975 | +0.0007 | 0.7843 ± 0.0069 | 0.785 | −0.0007 |
| *3D, ResNet-50+3D* | | | | | | |
| OrganMNIST3D | 0.9930 ± 0.0024 | 0.994 | −0.0010 | 0.8761 ± 0.0211 | 0.883 | −0.0069 |
| NoduleMNIST3D | 0.8660 ± 0.0261 | 0.875 | −0.0090 | 0.8393 ± 0.0276 | 0.847 | −0.0077 |
| FractureMNIST3D | 0.7441 ± 0.0202 | 0.725 | +0.0191 | 0.5425 ± 0.0478 | 0.494 | +0.0485 |
| AdrenalMNIST3D | 0.8438 ± 0.0277 | 0.828 | +0.0158 | 0.7980 ± 0.0162 | 0.745 | +0.0530 |
| VesselMNIST3D ⚠️ | 0.8791 ± 0.0298 | 0.907 | −0.0279 | 0.8764 ± 0.0437 | 0.918 | −0.0416 |
| SynapseMNIST3D ⚠️ | 0.8399 ± 0.0312 | 0.851 | −0.0111 | 0.7892 ± 0.0286 | 0.795 | −0.0058 |

Counted on the means: 13 datasets land within one AUC point of the paper either way, 2 more than a point above it
(FractureMNIST3D +1.91, AdrenalMNIST3D +1.58) and 3 more than a point below it (the ⚠️ ones). Only a loss counts
against the bar, so 15 of 18 pass it.

## 2. Test AUC, every run

**95% interval** is mean ± t · SD / √5 with t = 2.776 (4 degrees of freedom). **% of paper** is the difference
divided by the paper's value. SD is the sample standard deviation (n − 1).

| Dataset | Train / test | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean ± SD | 95% interval | Paper | Diff. | % of paper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ChestMNIST | 78,468 / 22,433 | 0.7718 | 0.7699 | 0.7738 | 0.7714 | 0.7696 | 0.7713 ± 0.0017 | 0.7692 to 0.7734 | 0.773 | −0.0017 | −0.22% |
| PathMNIST | 89,996 / 7,180 | 0.9894 | 0.9869 | 0.9875 | 0.9931 | 0.9910 | 0.9896 ± 0.0025 | 0.9864 to 0.9927 | 0.989 | +0.0006 | +0.06% |
| DermaMNIST | 7,007 / 2,005 | 0.9155 | 0.9091 | 0.9094 | 0.9146 | 0.9064 | 0.9110 ± 0.0039 | 0.9062 to 0.9158 | 0.912 | −0.0010 | −0.11% |
| OCTMNIST | 97,477 / 1,000 | 0.9610 | 0.9576 | 0.9609 | 0.9602 | 0.9623 | 0.9604 ± 0.0017 | 0.9582 to 0.9626 | 0.958 | +0.0024 | +0.25% |
| PneumoniaMNIST | 4,708 / 624 | 0.9623 | 0.9452 | 0.9569 | 0.9451 | 0.9717 | 0.9562 ± 0.0114 | 0.9421 to 0.9704 | 0.962 | −0.0058 | −0.60% |
| RetinaMNIST | 1,080 / 400 | 0.7102 | 0.7177 | 0.7197 | 0.7013 | 0.6888 | 0.7075 ± 0.0127 | 0.6917 to 0.7233 | 0.716 | −0.0085 | −1.18% |
| BreastMNIST ⚠️ | 546 / 156 | 0.8657 | 0.8412 | 0.8277 | 0.8601 | 0.7766 | 0.8343 ± 0.0356 | 0.7901 to 0.8785 | 0.866 | −0.0317 | −3.67% |
| BloodMNIST | 11,959 / 3,421 | 0.9975 | 0.9977 | 0.9975 | 0.9974 | 0.9975 | 0.9975 ± 0.0001 | 0.9974 to 0.9977 | 0.997 | +0.0005 | +0.05% |
| TissueMNIST | 165,466 / 47,280 | 0.9275 | 0.9304 | 0.9273 | 0.9255 | 0.9270 | 0.9275 ± 0.0018 | 0.9253 to 0.9298 | 0.932 | −0.0045 | −0.48% |
| OrganAMNIST | 34,561 / 17,778 | 0.9974 | 0.9979 | 0.9950 | 0.9941 | 0.9971 | 0.9963 ± 0.0017 | 0.9942 to 0.9984 | 0.998 | −0.0017 | −0.17% |
| OrganCMNIST | 12,975 / 8,216 | 0.9944 | 0.9941 | 0.9928 | 0.9934 | 0.9937 | 0.9937 ± 0.0006 | 0.9929 to 0.9945 | 0.993 | +0.0007 | +0.07% |
| OrganSMNIST | 13,932 / 8,827 | 0.9759 | 0.9771 | 0.9738 | 0.9749 | 0.9769 | 0.9757 ± 0.0014 | 0.9740 to 0.9774 | 0.975 | +0.0007 | +0.07% |
| OrganMNIST3D | 971 / 610 | 0.9934 | 0.9955 | 0.9894 | 0.9922 | 0.9946 | 0.9930 ± 0.0024 | 0.9901 to 0.9960 | 0.994 | −0.0010 | −0.10% |
| NoduleMNIST3D | 1,158 / 310 | 0.8984 | 0.8831 | 0.8559 | 0.8621 | 0.8303 | 0.8660 ± 0.0261 | 0.8335 to 0.8984 | 0.875 | −0.0090 | −1.03% |
| FractureMNIST3D | 1,027 / 240 | 0.7449 | 0.7200 | 0.7276 | 0.7650 | 0.7628 | 0.7441 ± 0.0202 | 0.7189 to 0.7692 | 0.725 | +0.0191 | +2.63% |
| AdrenalMNIST3D | 1,188 / 298 | 0.8259 | 0.8545 | 0.8714 | 0.8048 | 0.8625 | 0.8438 ± 0.0277 | 0.8094 to 0.8782 | 0.828 | +0.0158 | +1.91% |
| VesselMNIST3D ⚠️ | 1,335 / 382 | 0.8962 | 0.8609 | 0.8653 | 0.9229 | 0.8503 | 0.8791 ± 0.0298 | 0.8421 to 0.9162 | 0.907 | −0.0279 | −3.07% |
| SynapseMNIST3D ⚠️ | 1,230 / 352 | 0.7945 | 0.8708 | 0.8247 | 0.8444 | 0.8650 | 0.8399 ± 0.0312 | 0.8011 to 0.8786 | 0.851 | −0.0111 | −1.31% |

The paper's value is inside my 95% interval for all three datasets marked ⚠️.

## 3. Test accuracy, every run

| Dataset | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean ± SD | Paper | Diff. |
|---|---|---|---|---|---|---|---|---|
| ChestMNIST | 0.9479 | 0.9476 | 0.9473 | 0.9471 | 0.9470 | 0.9474 ± 0.0004 | 0.948 | −0.0006 |
| PathMNIST | 0.9102 | 0.8833 | 0.8997 | 0.9031 | 0.8804 | 0.8953 ± 0.0129 | 0.892 | +0.0033 |
| DermaMNIST | 0.7556 | 0.7387 | 0.7287 | 0.7436 | 0.6943 | 0.7322 ± 0.0233 | 0.731 | +0.0012 |
| OCTMNIST | 0.7670 | 0.7570 | 0.7760 | 0.7650 | 0.7750 | 0.7680 ± 0.0078 | 0.776 | −0.0080 |
| PneumoniaMNIST | 0.8526 | 0.8157 | 0.8269 | 0.8189 | 0.8798 | 0.8388 ± 0.0271 | 0.884 | −0.0452 |
| RetinaMNIST | 0.4750 | 0.5200 | 0.5000 | 0.5275 | 0.5325 | 0.5110 ± 0.0236 | 0.511 | +0.0000 |
| BreastMNIST | 0.8590 | 0.8269 | 0.8141 | 0.8333 | 0.7500 | 0.8167 ± 0.0407 | 0.842 | −0.0253 |
| BloodMNIST | 0.9544 | 0.9576 | 0.9573 | 0.9512 | 0.9544 | 0.9550 ± 0.0026 | 0.950 | +0.0050 |
| TissueMNIST | 0.6703 | 0.6776 | 0.6694 | 0.6635 | 0.6691 | 0.6700 ± 0.0050 | 0.680 | −0.0100 |
| OrganAMNIST | 0.9372 | 0.9450 | 0.9358 | 0.9312 | 0.9387 | 0.9376 ± 0.0050 | 0.947 | −0.0094 |
| OrganCMNIST | 0.9227 | 0.9198 | 0.9159 | 0.9183 | 0.9180 | 0.9189 ± 0.0025 | 0.911 | +0.0079 |
| OrganSMNIST | 0.7926 | 0.7832 | 0.7761 | 0.7798 | 0.7898 | 0.7843 ± 0.0069 | 0.785 | −0.0007 |
| OrganMNIST3D | 0.8803 | 0.8984 | 0.8410 | 0.8820 | 0.8787 | 0.8761 ± 0.0211 | 0.883 | −0.0069 |
| NoduleMNIST3D | 0.8484 | 0.8548 | 0.8290 | 0.8677 | 0.7968 | 0.8393 ± 0.0276 | 0.847 | −0.0077 |
| FractureMNIST3D | 0.5875 | 0.5250 | 0.4833 | 0.5958 | 0.5208 | 0.5425 ± 0.0478 | 0.494 | +0.0485 |
| AdrenalMNIST3D | 0.7886 | 0.7852 | 0.8188 | 0.7852 | 0.8121 | 0.7980 ± 0.0162 | 0.745 | +0.0530 |
| VesselMNIST3D | 0.8953 | 0.8901 | 0.8822 | 0.9136 | 0.8010 | 0.8764 ± 0.0437 | 0.918 | −0.0416 |
| SynapseMNIST3D | 0.7614 | 0.8011 | 0.7557 | 0.8153 | 0.8125 | 0.7892 ± 0.0286 | 0.795 | −0.0058 |

Accuracy moves more than AUC between seeds, especially on the small sets (SD up to 4.8 points on FractureMNIST3D).
More than one point short: PneumoniaMNIST (−4.5), VesselMNIST3D (−4.2) and BreastMNIST (−2.5). TissueMNIST is
exactly at −1.0. FractureMNIST3D (+4.9) and AdrenalMNIST3D (+5.3) are well above the paper. PneumoniaMNIST is the
odd one, because its AUC is only 0.6 points short. I haven't looked into it yet.

## 4. Float32 check on BreastMNIST

Mixed precision is the only one of my changes that touches the arithmetic, so I took the dataset that misses the
paper by the most and ran all 5 seeds again in plain float32 (same script, just without `--amp`). These 5 runs are
not counted anywhere else.

| Run (seed) | AUC, mixed precision | AUC, float32 | Change | ACC, mixed precision | ACC, float32 |
|---|---|---|---|---|---|
| 1 | 0.8657 | 0.8431 | −0.0226 | 0.8590 | 0.8397 |
| 2 | 0.8412 | 0.8672 | +0.0260 | 0.8269 | 0.8397 |
| 3 | 0.8277 | 0.7999 | −0.0278 | 0.8141 | 0.7308 |
| 4 | 0.8601 | 0.8682 | +0.0081 | 0.8333 | 0.8141 |
| 5 | 0.7766 | 0.8329 | +0.0563 | 0.7500 | 0.7628 |
| **Mean ± SD** | 0.8343 ± 0.0356 | 0.8423 ± 0.0282 | +0.0080 | 0.8167 ± 0.0407 | 0.7974 ± 0.0487 |
| **Paper** | 0.866 | | | 0.842 | |

Float32 comes out 0.8 points higher on average, still 2.4 below the paper, and single seeds move both ways by up
to 5.6 points. Accuracy goes the other way. So mixed precision doesn't explain the gap; the seed matters much more
than the number format on a test set of 156 images. Raw numbers: `results/float32_breastmnist.csv`.

## 5. ChestMNIST: the best epoch comes early

Per-epoch log of ChestMNIST run 5 (`results/chestmnist_run5_history.json`, all 100 epochs). Epochs count from 1
here; `result.json` counts from 0, so its `best_epoch: 12` is epoch 13.

| Epoch | Train loss | Val AUC | Val ACC |
|---|---|---|---|
| 1 | 0.1782 | 0.6848 | 0.9492 |
| 5 | 0.1627 | 0.7549 | 0.9494 |
| 10 | 0.1560 | 0.7618 | 0.9492 |
| 13 (best) | 0.1490 | 0.7742 | 0.9486 |
| 20 | 0.0826 | 0.7166 | 0.9413 |
| 25 | 0.0244 | 0.6925 | 0.9313 |
| 50 | 0.0050 | 0.6966 | 0.9340 |
| 51 | 0.0030 | 0.6968 | 0.9375 |
| 75 | 0.0002 | 0.6940 | 0.9369 |
| 100 | 0.0003 | 0.6957 | 0.9362 |

The validation AUC peaks at epoch 13 and then falls about 7.8 points while the training loss heads to zero. The
learning-rate drops at epochs 50 and 75 don't bring it back. The reported test AUC comes from the epoch-13 weights,
because the script keeps the best validation epoch. This is the right panel of the figure in the README.

## 6. Hardware and time

| Dataset | Train images | Time per run on one T4 (rough) | GPUs used |
|---|---|---|---|
| ChestMNIST | 78,468 | 8 hours | RTX 3090, T4 |
| PathMNIST | 89,996 | 7.9 hours | T4 |
| DermaMNIST | 7,007 | 40 minutes | T4 |
| OCTMNIST | 97,477 | 8.6 hours | T4 |
| PneumoniaMNIST | 4,708 | 30 minutes | T4 |
| RetinaMNIST | 1,080 | 5 minutes | T4 |
| BreastMNIST | 546 | 5 minutes | T4 |
| BloodMNIST | 11,959 | 1.1 hours | T4 |
| TissueMNIST | 165,466 | 15 hours | T4 |
| OrganAMNIST | 34,561 | 3.3 hours | T4 |
| OrganCMNIST | 12,975 | 1.3 hours | T4 |
| OrganSMNIST | 13,932 | 1.4 hours | T4 |
| OrganMNIST3D | 971 | 30 minutes | T4 |
| NoduleMNIST3D | 1,158 | 35 minutes | T4 |
| FractureMNIST3D | 1,027 | 30 minutes | T4 |
| AdrenalMNIST3D | 1,188 | 35 minutes | T4 |
| VesselMNIST3D | 1,335 | 35 minutes | T4 |
| SynapseMNIST3D | 1,230 | 35 minutes | T4 |

Times per run are rough, from my notebooks. Two runs train at a time on Kaggle's two T4s, so a session gets through
about twice as many runs as these times suggest.

- ChestMNIST runs 1 to 4 ran on an RTX 3090 on a server I later lost access to, before I had copied their run
  folders. Their numbers come from the training log, and `runs.json` and `all_runs.csv` both say so.
- All 90 runs used mixed precision (`--amp`).
- `train_hours` in `all_runs.csv` is missing for runs that were split across sessions (`runs.json` only has
  `hours_in_finishing_session` for those) and for the four 3090 runs, so I don't add the hours up.

## 7. How I read the misses

- **Large datasets reproduce.** All eight 2D datasets with more than 10,000 training images are within 0.5 AUC
  points of the paper, with an SD of 0.25 points or less over the 5 seeds.
- **The misses are on small test sets.** BreastMNIST (156 test images), VesselMNIST3D (382) and SynapseMNIST3D
  (352) miss by 3.2, 2.8 and 1.1 points. My 5 runs on them have SDs of 3.6, 3.0 and 3.1 points, bigger than the
  bar, and the paper's value is inside the 95% interval each time.
- **The noise goes both ways.** FractureMNIST3D and AdrenalMNIST3D, which are just as small, came out 1.9 and 1.6
  points above the paper.
- **Relative reading.** If "under 1%" means 1% of the paper's value instead of one AUC point, RetinaMNIST (−1.18%)
  and NoduleMNIST3D (−1.03%) miss as well, narrowly.
- **Not my patches.** The float32 rerun (section 4) rules out mixed precision for BreastMNIST, and none of the
  other changes touch what the model learns (see the README and `scripts/patches_vs_released.diff`).
