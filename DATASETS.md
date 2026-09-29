# Dataset sources and access information

The study uses ten publicly accessible benchmark datasets. The repository does not redistribute the locally prepared CSV files. This avoids duplicating data with unclear or source-specific redistribution terms, and it keeps the code repository below GitHub's ordinary file-size limits.

| Experiment | Source | Persistent identifier or URL | Source licence/status | Expected local file |
|---|---|---|---|---|
| Car Insurance Data | Kaggle, Sagnik1511 | https://www.kaggle.com/datasets/sagnik1511/car-insurance-data | Licence shown as `Unknown` in Kaggle metadata when checked; download from the source | `Car Insurance Data.csv` |
| Default of Credit Card Clients | UCI Machine Learning Repository | https://doi.org/10.24432/C55S3H | CC BY 4.0 | `UCI_Credit_Card.csv` |
| Insurance Company Benchmark (COIL 2000) | UCI Machine Learning Repository | https://doi.org/10.24432/C5630S | CC BY 4.0 | `caravan-insurance-challenge.csv` |
| Credit Card Fraud Detection | Kaggle / ULB Machine Learning Group | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | Open Database Licence / Database Contents Licence as reported by Kaggle | `creditcard.csv` |
| Breast Cancer | Kaggle, Reihanenamdari; derived from SEER data | https://www.kaggle.com/datasets/reihanenamdari/breast-cancer | CC BY 4.0 as reported by Kaggle | `Breast_Cancer.csv` |
| Diabetes Prediction Dataset | Kaggle, iammustafatz | https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset | `Data files © Original Authors` as reported by Kaggle; download from the source | `diabetes_prediction_dataset.csv` |
| Malware Analysis Datasets: API Call Sequences | Kaggle, ang3loliveira | https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences | CC BY 4.0 as reported by Kaggle | `Malware Analysis Datasets API Call Sequences.csv` |
| Mammography / Microcalcification | OpenML dataset 310 | https://www.openml.org/d/310 | Public as reported by OpenML | `Microcalcification Classification.csv` |
| Differentiated Thyroid Cancer Recurrence | UCI Machine Learning Repository | https://doi.org/10.24432/C5632J | CC BY 4.0 | `Differentiated Thyroid Cancer Recurrence.csv` |
| TUANDROMD | UCI Machine Learning Repository | https://doi.org/10.24432/C5560H | CC BY 4.0 | `TUANDROMD.csv` |

## Analysis-ready files

The experimental CSV files in the authors' working archive are numeric, analysis-ready versions. They may differ from the upstream downloads in column naming, categorical encoding, identifier removal, and missing-value handling. In particular, several targets are named `class` in the experiment files. The expected filenames and target columns are listed in the main README.

For a permanent replication release, the recommended approach is to archive either:

1. preprocessing scripts that deterministically reconstruct every analysis-ready file from the original source; or
2. analysis-ready files only where the source licence explicitly permits redistribution, accompanied by attribution and transformation notes.

Always comply with the terms shown by the source repository at the time of download.
