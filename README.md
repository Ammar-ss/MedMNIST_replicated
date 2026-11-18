# MedMNIST v2 baselines, rerun from scratch

I reran the ResNet-50 baselines from **MedMNIST v2** ([Yang et al., *Scientific Data* 2023](https://doi.org/10.1038/s41597-022-01721-8))
with the authors' own training scripts from [MedMNIST/experiments](https://github.com/MedMNIST/experiments)
(commit `70b6b3a`). That's ResNet-50 at 224 px on all 12 2D datasets and ResNet-50+3D on all 6 3D datasets,
5 runs each with seeds 1 to 5. So 90 training runs, almost all of them on free Kaggle T4s.

I wanted solid baselines before trying anything of my own on MedMNIST, and you can only trust a baseline you've
run yourself. Most of the work turned out to be fitting 100-epoch runs into 12-hour Kaggle sessions, not the
training itself.

**[Full results](RESULTS.md)** · **[Notebooks](notebooks/)** · **[Patched scripts](scripts/)** · **[Raw numbers](results/)**

## TL;DR

- **15 of 18 datasets land within one AUC point of the paper.** ChestMNIST, the one I started with, gives
  0.7713 ± 0.0017 against the reported 0.773.
- The 3 that miss (BreastMNIST, VesselMNIST3D, SynapseMNIST3D) all have small test sets, and my 5 runs on each
  spread more than the gap. The paper's number is inside my 95% interval every time.
- The big 2D datasets reproduce really tightly. All eight with more than 10k training images are within half a
  point of the paper, and their 5 runs barely move (SD of 0.25 points or less).
- I reran BreastMNIST in float32 to check whether my mixed-precision change caused the miss. It didn't.
- ChestMNIST's validation AUC peaks at epoch 13 out of 100 and then drops almost 8 points while the training
  loss goes to zero. The reported number only holds up because the script keeps the best epoch.

## Results

Test AUC and accuracy, mean ± SD over 5 runs. **Paper** is the ResNet-50 (224) row for 2D and the ResNet-50+3D
row for 3D (Tables 3 and 4 of the paper). **Diff.** is mine minus the paper. ⚠️ means more than one AUC point
below the paper.

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

Every single run, the 95% intervals, the float32 check and the rest are in **[RESULTS.md](RESULTS.md)**.

![How far each dataset lands from the paper, and ChestMNIST's training curve](assets/results_overview.png)

*Left: my test AUC minus the paper's, per dataset (big dot = mean of 5 runs, bar = 95% interval, small dots =
single runs, dashed line = one point below). Right: ChestMNIST run 5, validation AUC and training loss per epoch.*

## What I found

**The big datasets are boring, in a good way.** PathMNIST, ChestMNIST, OCTMNIST, BloodMNIST, TissueMNIST and the
three Organ 2D sets all land within 0.5 AUC points of the paper, and the SD over their 5 seeds is a quarter of a
point or less. If your code is broken, these are where you'd see it, and they look fine.

**The small datasets are noisy.** On BreastMNIST, RetinaMNIST and five of the six 3D sets, my 5 seeds spread by
1.3 to 3.6 AUC points (SD). That's more than the one-point bar itself, so a miss on these says more about the size
of the test set than about the code. BreastMNIST's test set has 156 images. Two small 3D sets went the other way
and came out *above* the paper: FractureMNIST3D by 1.9 points and AdrenalMNIST3D by 1.6.

**Depends how you read "1%".** As one AUC point, 3 datasets miss. As 1% of the paper's value, RetinaMNIST
(−1.18%) and NoduleMNIST3D (−1.03%) also miss, just barely.

**Accuracy moves a lot more than AUC.** PneumoniaMNIST is 4.5 points short on accuracy while its AUC is only 0.6
short. VesselMNIST3D is 4.2 short and BreastMNIST 2.5 short, TissueMNIST sits right on −1.0, and FractureMNIST3D
and AdrenalMNIST3D are about 5 points *above*. I haven't dug into PneumoniaMNIST yet. I judge the bar on AUC since
that's the main number the benchmark compares, but the accuracy numbers are all in the tables.

**It wasn't the mixed precision.** Float16 is the only change I made that touches the maths, so I reran all 5
BreastMNIST runs in float32. They average 0.8423 ± 0.0282, against 0.8343 with mixed precision and 0.866 in the
paper, and the run-by-run changes go both ways (−2.8 to +5.6 points). So float16 isn't what separates my
BreastMNIST runs from the paper.

**ChestMNIST peaks early and then memorises.** In run 5 the validation AUC is best at epoch 13 (0.774) and is
about 7.8 points lower by epoch 100, while the training loss goes to basically zero. The script keeps the best
validation epoch, so the test number is fine, but none of the last 87 epochs ever beats epoch 13 on validation.

## What's in here

```
medmnist-v2-replication/
├── README.md                        you are here
├── RESULTS.md                       every table: each run, intervals, float32 check, ChestMNIST curve, timing
├── requirements.txt
├── notebooks/                       one Kaggle notebook per dataset, 01_chestmnist ... 18_synapsemnist3d
├── scripts/
│   ├── MedMNIST2D/train_and_eval_pytorch_resumable.py   authors' 2D script + my patches
│   ├── MedMNIST3D/train_and_eval_pytorch_resumable.py   authors' 3D script + my patches
│   ├── patches_vs_released.diff     every change against the released scripts
│   └── LICENSE                      the authors' Apache 2.0 licence
├── results/
│   ├── summary.csv                  one row per dataset (sizes, every run, mean, SD, paper, diff, AUC and ACC)
│   ├── all_runs.csv                 one row per run (AUC, ACC, GPU, hours, which session it ran in)
│   ├── runs.json                    all 90 result.json files, exactly as the script wrote them
│   ├── float32_breastmnist.csv      the float32 rerun
│   ├── chestmnist_run5_history.json loss and validation AUC/ACC for every epoch of ChestMNIST run 5
│   └── performance_table_filled.xlsx  the same numbers in spreadsheet form (AUC sheet, ACC sheet, every run)
└── assets/
    └── results_overview.png / .pdf  the figure above
```

## Running it

### On Kaggle (this is what I did)

1. Upload the notebook for the dataset you want (say `notebooks/07_breastmnist.ipynb`) to Kaggle.
2. Set the accelerator to **GPU T4 x2**, turn **internet on**, and hit **Save & Run All**.
3. That's it for most datasets. The notebook clones the authors' repo at the right commit, writes in my patched
   scripts, downloads the data and trains two runs at a time (one per GPU). It stops cleanly before Kaggle's
   12-hour limit.
4. ChestMNIST, PathMNIST and OCTMNIST take 8 to 9 hours per run, and TissueMNIST about 15, so those need more
   than one session. Start the next session with the previous output attached through **Add Input** and run it
   again: finished runs get skipped and stopped runs resume from their last checkpoint.

Each run ends up in its own folder with a `result.json`, the logs and the test predictions, and the notebook
writes a `summary.csv` and a `results_only.zip` (no weights) at the end. Rough times per dataset are in
[RESULTS.md](RESULTS.md#6-hardware-and-time).

### Without Kaggle

The patched scripts import `models.py` and `utils.py` from the authors' repo, so they have to sit inside it. I
only ever ran them through the notebooks, but these are the same commands the notebooks call:

```bash
git clone https://github.com/MedMNIST/experiments.git
cd experiments
git checkout 70b6b3a7ad7afddff1df2a3b735235830fbdb142
pip install medmnist==3.0.2 acsconv==0.1.1 tensorboardX

# 2D, e.g. BreastMNIST run 1
cp /path/to/this/repo/scripts/MedMNIST2D/train_and_eval_pytorch_resumable.py MedMNIST2D/
cd MedMNIST2D
python train_and_eval_pytorch_resumable.py --data_flag breastmnist --model_flag resnet50 --resize --as_rgb \
    --size 28 --batch_size 128 --num_epochs 100 --gpu_ids 0 --output_root ./output --run run1 --seed 1 \
    --amp --skip_logging_evals --download

# 3D, e.g. SynapseMNIST3D run 1 (add --shape_transform for adrenalmnist3d and vesselmnist3d, like the paper)
cp /path/to/this/repo/scripts/MedMNIST3D/train_and_eval_pytorch_resumable.py ../MedMNIST3D/
cd ../MedMNIST3D
python train_and_eval_pytorch_resumable.py --data_flag synapsemnist3d --model_flag resnet50 --conv Conv3d \
    --as_rgb --size 28 --batch_size 32 --num_epochs 100 --gpu_ids 0 --output_root ./output --run run1 --seed 1 \
    --amp --skip_logging_evals --download
```

Drop `--amp` to train in float32. Run *k* used seed *k* for all 90 runs.

## What I changed in the authors' scripts

Nothing about what the model learns. The model, data, loss, optimiser, learning-rate schedule, epochs, batch size
and the rule for picking the best epoch are all as released. The changes are only there to get long runs through
12-hour sessions:

- **mixed precision** (float16), for speed;
- **a checkpoint every ten minutes**, so a run can stop cleanly before the session ends and resume in the next one;
- **no extra pass over the training and test sets after every epoch.** The released script only does that to log
  to TensorBoard. The validation pass that picks the best epoch still runs every epoch;
- **two guards against float16 overflow**, added after one PathMNIST run ended in a NaN loss. They do nothing
  unless a value overflows, and the runs before that used the script without them;
- **a one-line fix** that lets ACSConv 0.1.1 convert batch-norm layers on current PyTorch.

Every change is marked `# PATCH` in `scripts/`, and `scripts/patches_vs_released.diff` shows all of them against
the released files.

## Stuff that went wrong along the way

- One PathMNIST run hit a float16 overflow and its loss turned into NaN partway through. That's where the overflow
  guards came from.
- ACSConv 0.1.1 crashes when it converts batch-norm layers on recent PyTorch versions. That's the one-line fix in
  the 3D script.
- I lost access to the server that ran four of the ChestMNIST runs before I had copied their folders, so for those
  four I only have the numbers from the training log. `all_runs.csv` and `runs.json` both say so.
- The free Kaggle quota was the real limit. TissueMNIST alone is about 75 GPU-hours for 5 runs, so I packed several
  datasets into each session and spread the long runs over several sessions. The notebooks here are the cleaned-up,
  one-dataset versions of that code.

## Credit

The datasets and the original training scripts belong to the MedMNIST team. If you use any of this, please cite
their paper:

```bibtex
@article{medmnistv2,
  title   = {MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author  = {Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and
             Pfister, Hanspeter and Ni, Bingbing},
  journal = {Scientific Data},
  volume  = {10},
  pages   = {41},
  year    = {2023},
  doi     = {10.1038/s41597-022-01721-8}
}
```

The 3D models are converted with [ACSConv](https://github.com/M3DV/ACSConv) (Yang et al., *IEEE JBHI* 2021).

Their experiments repository is released under the Apache License 2.0 (a copy is in `scripts/LICENSE`). The
scripts here and inside the notebooks are modified versions of theirs, with every change marked.
