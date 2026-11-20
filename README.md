# MedMNIST baselines, rerun

Rerun of the ResNet-50 baselines from MedMNIST v2 ([Yang et al., Scientific Data 2023](https://doi.org/10.1038/s41597-022-01721-8)) using the authors' training scripts from [MedMNIST/experiments](https://github.com/MedMNIST/experiments) (commit `70b6b3a`). ResNet-50 (224) on the 12 2D datasets and ResNet-50+3D on the 6 3D datasets, 5 runs each with seeds 1 to 5, so 90 runs in total. Almost all of them ran on free Kaggle T4s.

Full tables are in [RESULTS.md](RESULTS.md).

## Results

Test AUC and ACC, mean ± SD over 5 runs. Paper is the ResNet-50 (224) row for 2D and the ResNet-50+3D row for 3D (Tables 3 and 4 of the paper). Diff. is mine minus the paper. * means more than one AUC point below the paper.

| Dataset | AUC | Paper | Diff. | ACC | Paper | Diff. |
|---|---|---|---|---|---|---|
| ChestMNIST | 0.7713 ± 0.0017 | 0.773 | -0.0017 | 0.9474 ± 0.0004 | 0.948 | -0.0006 |
| PathMNIST | 0.9896 ± 0.0025 | 0.989 | +0.0006 | 0.8953 ± 0.0129 | 0.892 | +0.0033 |
| DermaMNIST | 0.9110 ± 0.0039 | 0.912 | -0.0010 | 0.7322 ± 0.0233 | 0.731 | +0.0012 |
| OCTMNIST | 0.9604 ± 0.0017 | 0.958 | +0.0024 | 0.7680 ± 0.0078 | 0.776 | -0.0080 |
| PneumoniaMNIST | 0.9562 ± 0.0114 | 0.962 | -0.0058 | 0.8388 ± 0.0271 | 0.884 | -0.0452 |
| RetinaMNIST | 0.7075 ± 0.0127 | 0.716 | -0.0085 | 0.5110 ± 0.0236 | 0.511 | +0.0000 |
| BreastMNIST * | 0.8343 ± 0.0356 | 0.866 | -0.0317 | 0.8167 ± 0.0407 | 0.842 | -0.0253 |
| BloodMNIST | 0.9975 ± 0.0001 | 0.997 | +0.0005 | 0.9550 ± 0.0026 | 0.950 | +0.0050 |
| TissueMNIST | 0.9275 ± 0.0018 | 0.932 | -0.0045 | 0.6700 ± 0.0050 | 0.680 | -0.0100 |
| OrganAMNIST | 0.9963 ± 0.0017 | 0.998 | -0.0017 | 0.9376 ± 0.0050 | 0.947 | -0.0094 |
| OrganCMNIST | 0.9937 ± 0.0006 | 0.993 | +0.0007 | 0.9189 ± 0.0025 | 0.911 | +0.0079 |
| OrganSMNIST | 0.9757 ± 0.0014 | 0.975 | +0.0007 | 0.7843 ± 0.0069 | 0.785 | -0.0007 |
| OrganMNIST3D | 0.9930 ± 0.0024 | 0.994 | -0.0010 | 0.8761 ± 0.0211 | 0.883 | -0.0069 |
| NoduleMNIST3D | 0.8660 ± 0.0261 | 0.875 | -0.0090 | 0.8393 ± 0.0276 | 0.847 | -0.0077 |
| FractureMNIST3D | 0.7441 ± 0.0202 | 0.725 | +0.0191 | 0.5425 ± 0.0478 | 0.494 | +0.0485 |
| AdrenalMNIST3D | 0.8438 ± 0.0277 | 0.828 | +0.0158 | 0.7980 ± 0.0162 | 0.745 | +0.0530 |
| VesselMNIST3D * | 0.8791 ± 0.0298 | 0.907 | -0.0279 | 0.8764 ± 0.0437 | 0.918 | -0.0416 |
| SynapseMNIST3D * | 0.8399 ± 0.0312 | 0.851 | -0.0111 | 0.7892 ± 0.0286 | 0.795 | -0.0058 |

15 of the 18 datasets are less than one AUC point below the paper. The three that miss (BreastMNIST, VesselMNIST3D, SynapseMNIST3D) all have small test sets (156 to 382 images), and for all three the paper's number is inside the 95% interval of my 5 runs. Mixed precision is the only change I made that affects the arithmetic, so I reran BreastMNIST in float32 as a check. It gave 0.8423 ± 0.0282, so that is not the cause.

![results](results_overview.png)

Left: my test AUC minus the paper's for each dataset, with the number of test images in brackets. Dots are the mean of 5 runs, error bars the 95% CI, crosses the single runs and the red dashed line is one point below the paper. Right: validation AUC and training loss of ChestMNIST run 5. The best epoch is 13, after which the model overfits. The dashed lines are the learning-rate drops at epochs 50 and 75.

## Files

- `notebooks/`: one Kaggle notebook per dataset (01_chestmnist to 18_synapsemnist3d)
- `scripts/`: the authors' 2D and 3D training scripts with my changes, plus `patches_vs_released.diff` and their license
- `results/`: every run (`runs.json`, `all_runs.csv`), the per-dataset summary, the float32 BreastMNIST rerun, the ChestMNIST run 5 history and the filled spreadsheet

## Running it

On Kaggle: upload a notebook, set the accelerator to GPU T4 x2, turn internet on and run all. The notebook clones the authors' repo at the right commit, writes the patched scripts, downloads the data and trains two runs at a time. ChestMNIST, PathMNIST, OCTMNIST and TissueMNIST need more than one 12-hour session. For those, add the previous session's output as input and run it again, and finished runs are skipped while unfinished ones resume from their checkpoint.

Without Kaggle, the scripts need to sit inside the authors' repo because they import its `models.py` and `utils.py`:

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

# 3D, e.g. SynapseMNIST3D run 1 (add --shape_transform for adrenalmnist3d and vesselmnist3d)
cp /path/to/this/repo/scripts/MedMNIST3D/train_and_eval_pytorch_resumable.py ../MedMNIST3D/
cd ../MedMNIST3D
python train_and_eval_pytorch_resumable.py --data_flag synapsemnist3d --model_flag resnet50 --conv Conv3d \
    --as_rgb --size 28 --batch_size 32 --num_epochs 100 --gpu_ids 0 --output_root ./output --run run1 --seed 1 \
    --amp --skip_logging_evals --download
```

Drop `--amp` to train in float32.

## Changes to the authors' scripts

The model, data, loss, optimizer, schedule, epochs, batch size and best-epoch selection are the same as released. I only changed how the runs execute so they fit into 12-hour Kaggle sessions:

- mixed precision (`--amp`)
- a checkpoint every 10 minutes so a run can resume in the next session
- skipping the per-epoch train and test evaluations, which the original script only logs to TensorBoard (validation still runs every epoch)
- two guards against float16 overflow, added after one PathMNIST run hit NaN (they do nothing unless an overflow happens)
- a one-line fix so ACSConv 0.1.1 can convert batch-norm layers on current PyTorch

All changes are marked `# PATCH` in the scripts and listed in `scripts/patches_vs_released.diff`.

Four of the ChestMNIST runs were trained on a server I later lost access to, so for those I only have the numbers from the training log. This is noted in `runs.json` and `all_runs.csv`.

## Credit

The datasets and original training scripts are from the MedMNIST team (Apache 2.0, see `scripts/LICENSE`). If you use this, please cite their paper:

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

The 3D models use [ACSConv](https://github.com/M3DV/ACSConv) (Yang et al., IEEE JBHI 2021).
