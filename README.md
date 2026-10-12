# NIDS — Network Intrusion Detection System

A machine-learning based Network Intrusion Detection System trained on **CICIDS2017**.  
Detects **DDoS**, **Port Scans**, **Brute Force**, and **Botnet** traffic in network flows.

---

## Project Structure

```
CIDS NETWORKING/
├── data/
│   ├── raw/            # Original CICIDS2017 CSVs  ← place files here
│   ├── processed/      # Cleaned parquet files (auto-generated)
│   └── external/       # NSL-KDD / UNSW-NB15 for cross-dataset tests
├── notebooks/          # EDA and experiment notebooks
├── src/
│   ├── data/           # Loaders, cleaners, preprocessors
│   ├── features/       # Feature engineering, SMOTE balancing
│   ├── models/         # Training, evaluation, hyperparameter tuning
│   ├── anomaly/        # Autoencoder anomaly detector
│   ├── dashboard/      # Streamlit dashboard
│   └── explainability/ # SHAP value computation
├── models/             # Saved model artifacts (.joblib)
├── reports/            # Evaluation plots and metrics JSON
├── config/             # model_config.yaml, feature_list.yaml
└── tests/              # pytest unit + integration tests
```

## Quick Start

### 1 — Get the Data
Download `MachineLearningCSV.zip` from [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)  
Extract all CSVs into `data/raw/`.

### 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### 3 — Preprocess & Train (Binary)
```bash
python -m src.data.preprocess          # Clean + encode + scale
python -m src.models.train_binary      # Train RF + XGBoost (binary)
python -m src.models.evaluate          # Print metrics + save plots
```

### 4 — Multiclass Extension
```bash
python -m src.models.train_multiclass  # Classify each attack type
```

### 5 — Anomaly Detection
```bash
python -m src.anomaly.train_autoencoder   # Train on benign only
python -m src.anomaly.detect              # Flag high-error flows
```

### 6 — Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Dataset — CICIDS2017

| Day        | Attack Types                                  |
|------------|-----------------------------------------------|
| Monday     | Benign only (baseline)                        |
| Tuesday    | Brute Force (FTP-Patator, SSH-Patator)        |
| Wednesday  | DoS / DDoS (Slowloris, Slowhttptest, Hulk, GoldenEye) |
| Thursday   | Web Attacks + Infiltration                    |
| Friday     | Port Scan, Botnet (ARES), DDoS               |

## Evaluation Metrics

- **Primary:** Recall (attack class) — missing an attack > false alarm
- **Secondary:** F1 per class, Precision, ROC-AUC
- **Never use accuracy alone** — classes are severely imbalanced

## Citation

> Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani,  
> "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization",  
> ICISSP 2018.
