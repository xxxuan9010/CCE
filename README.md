# CCE: efficiency-based ensemble learning for imbalanced classification

This repository contains the experiment code and reported result files for the manuscript **“Balancing asymmetric misclassification costs in financial risk classification through efficiency-based ensemble learning.”**

The proposed CCE framework evaluates candidate classifiers with a context-dependent assurance-region constrained Free Disposal Hull (CAR-FDH) model and selects an efficient, diverse subset for ensemble construction. The repository preserves the dataset-specific code used in the reported experiments. It does not refactor the implementations into a single package because several datasets use experiment-specific variants.

## Repository contents

```text
.
├── DATASETS.md
├── README.md
├── requirements.txt
├── data/
│   └── README.md
└── experiments/
    ├── car-insurance/
    ├── uci-credit-card-default/
    ├── caravan-insurance/
    ├── credit-card-fraud/
    ├── breast-cancer/
    ├── diabetes-prediction/
    ├── malware-api-call-sequences/
    ├── mammography/
    ├── thyroid-cancer-recurrence/
    └── tuandromd/
```

Each experiment directory contains the Python implementation used for that dataset and a `result/` directory with the reported run-level and summary CSV files. The four financial datasets also contain the ablation and robustness entry points and their result files.

## Requirements

- Python 3.12 or a compatible recent Python 3 release
- Packages listed in `requirements.txt`
- A working MOSEK installation and licence, because the current implementation calls MOSEK through Pyomo

Create an isolated environment and install the Python dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

MOSEK requires separate licence configuration. See the official MOSEK installation documentation for the operating system in use.

## Data preparation

Raw and locally prepared dataset files are not redistributed in this repository. Download them from the original sources listed in [DATASETS.md](DATASETS.md), prepare numeric CSV files with the expected filenames and target columns, and either place each file in its experiment directory or pass an absolute path through `--data`.

The CSV files used in the experiments were numeric analysis-ready copies rather than untouched upstream files. In several cases, categorical variables were encoded numerically and the target was renamed to `class`. Researchers should follow the data-source documentation and the dimensions reported in the manuscript when reconstructing these inputs.

## Running an experiment

Run commands from the relevant experiment directory so that local imports resolve correctly. For example:

```bash
cd experiments/uci-credit-card-default
python main_repeat.py \
  --data UCI_Credit_Card.csv \
  --target default.payment.next.month \
  --kmax 10 \
  --n_runs 10 \
  --n_jobs 10
```

The repeated experiments use consecutive seeds beginning at `--seed_initial` (default: 0). The result files included in this repository contain runs with seeds 0–9.

### Dataset-specific inputs

| Experiment directory | Expected CSV filename | Target column | Main entry point |
|---|---|---|---|
| `car-insurance` | `Car Insurance Data.csv` | `class` | `main_repeat.py` |
| `uci-credit-card-default` | `UCI_Credit_Card.csv` | `default.payment.next.month` | `main_repeat.py` |
| `caravan-insurance` | `caravan-insurance-challenge.csv` | `CARAVAN` | `main_repeat.py` |
| `credit-card-fraud` | `creditcard.csv` | `Class` | `main_repeat.py` |
| `breast-cancer` | `Breast_Cancer.csv` | `class` | `main_repeat.py` |
| `diabetes-prediction` | `diabetes_prediction_dataset.csv` | `class` | `main_repeat.py` |
| `malware-api-call-sequences` | `Malware Analysis Datasets API Call Sequences.csv` | `class` | `main_repeat.py` |
| `mammography` | `Microcalcification Classification.csv` | `class` | `main_repeat.py` |
| `thyroid-cancer-recurrence` | `Differentiated Thyroid Cancer Recurrence.csv` | `class` | `main_repeat.py` |
| `tuandromd` | `TUANDROMD.csv` | `class` | `main_repeat.py` |

The financial experiment directories additionally include:

- `main_ablation.py` for component ablations;
- `main_robust.py` for CAR-FDH parameter robustness analyses.

Use `python <entry-point>.py --help` to inspect all command-line options.

## Outputs

By default, scripts write outputs to `./result`. The archived result files include:

- `results_repeat.csv`: run-level results;
- `results_summary.csv`: means and standard deviations across repeated runs;
- financial-dataset ablation and CAR-FDH grid results where applicable.

## Scope of the release

The upload excludes cluster-specific Slurm shell scripts, scheduler logs, Python bytecode caches, the source ZIP archive, and an additional malware-executable experiment that is not part of the ten datasets reported in the manuscript.

## Citation

The manuscript citation and archival DOI will be added after publication or repository archiving.
