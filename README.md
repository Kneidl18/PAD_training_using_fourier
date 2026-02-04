# PAD Training Using Fourier Features

This repo contains the coursework project "Image Processing – Fourier Extraction" on finger-vein presentation attack detection (PAD). The code extracts Fourier magnitude and phase features from grayscale images and evaluates a k-NN classifier under two tasks.

- Task 1: baseline real vs spoof evaluation with group-based 5-fold CV and reduced training sizes (4/5, 2/5, 1/5 of users).
- Task 2: integration of synthetic spoof samples (residual and variational) with three training setups (Steps 3–5) while keeping the same test folds.

See `report/presentation.pdf` for the slides.

**Repo Layout**
- `code/` Python implementation, configs, and dependencies.
- `report/presentation.pdf` presentation slides.

**Setup**
From `code/`:

```bash
conda env create -f environment.yml
conda activate bildverarbeitung
```

Or with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Data Configuration**
The code reads dataset roots from `code/.env`:

```bash
REAL_DATA_ROOT=...
SYNTHETIC_DATA_ROOT_RESIDUAL=...
SYNTHETIC_DATA_ROOT_VARIATIONAL=...
```

Expected structure (simplified):
- PLUS: `REAL_DATA_ROOT/PLUS/{real,spoof}/...`
- IDIAP, SCUT: `REAL_DATA_ROOT/<DATASET>/full/{train,test,dev}/{real,spoof}/...`
- Synthetic: `SYNTHETIC_DATA_ROOT_*/<DATASET or DATASET_matched>/spoof/...`

User IDs are parsed from filenames (rules in `code/src/data_utils.py`).

**Running**
From `code/`:

```bash
python -m src.main --dataset ALL --task both
```

Options:
- `--dataset {ALL,PLUS,IDIAP,SCUT}`
- `--task {1,2,both}`
- `--plots` saves CV split visuals (`cv_split_*.png`)
- `--print_corr` prints magnitude/phase correlation when using combined features

Output is printed to stdout with APCER, BPCER, and ACER for each step.

**Notes**
- Task 1 evaluates magnitude-only, phase-only, and combined features (30 radial bands each).
- Task 2 uses magnitude "thirds" selection configured per dataset in `code/src/config.py`.
