<div align="center">

# 🩺 AutoHeal

### An autonomous self-healing ML pipeline that detects concept drift and automatically retrains itself — no human in the loop.

[![Python](https://img.shields.io/badge/Python-3-blue?style=flat-square&logo=python)](https://python.org)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square&logo=plotly)](https://matplotlib.org)

</div>

---

## ⚡ What This Does

Traditional ML systems silently degrade in production as data drifts away from what they were trained on — someone has to notice the drop and manually retrain. **AutoHeal removes the human from that loop.** It continuously monitors rolling prediction accuracy, automatically detects when performance falls below a threshold, retrains the model on the fly, and redeploys it — all without intervention.

The pipeline is evaluated on the **Breast Cancer Wisconsin Diagnostic** and **Iris** datasets, with injected concept drift, and benchmarked against two baselines: no monitoring at all, and fixed-schedule retraining.

---

## 🎯 Results

| Method | Accuracy During Drift | Retrains Needed | Human Input Required |
|---|---|---|---|
| No monitoring | Low (degrades and stays degraded) | 0 | Yes |
| Scheduled retraining | Moderate | 1 (fixed schedule) | Yes |
| **AutoHeal (proposed)** | **High (auto-recovers)** | **Only as needed** | **No** |

AutoHeal detects the drift, triggers a retrain automatically, and restores accuracy back above the threshold — without anyone watching the dashboard.

![Accuracy over time](docs/images/fig1_accuracy_over_time.png)

---

## 🧠 How It Works

Train Initial Model
↓
Monitor Predictions
↓
Calculate Rolling Accuracy
↓
Performance Drop?
↓             ↓
No            Yes
↓             ↓
Continue     Detect Drift
↓
Retrain Model
↓
Deploy Updated Model
↓
Continue Monitoring

Drift is detected using a rolling-window accuracy check: once accuracy falls below threshold for several consecutive windows, a retrain is triggered automatically on fresh data, and the new model is deployed in place of the old one.

---

## 📊 More Results

### Method Comparison

![Method comparison](docs/images/fig2_method_comparison.png)

### Breast Cancer — Five Independent Runs

![Breast Cancer five runs](docs/images/fig3_bc_five_runs.png)

### Dataset Comparison — Breast Cancer vs Iris

![Dataset comparison](docs/images/fig4_dataset_comparison.png)

### Combined Summary

![Combined results](docs/images/fig5_combined_paper.png)

---

## 🏗️ Architecture

- **Model:** Random Forest Classifier (scikit-learn)
- **Drift detection:** Rolling-window accuracy monitoring with consecutive-failure threshold
- **Datasets:** Breast Cancer Wisconsin Diagnostic, Iris (with injected concept drift)
- **Retraining trigger:** Automatic, threshold-based (no manual scheduling)
- **Outputs:** CSV result summaries + PNG visualizations, generated per run

---

## ⚙️ Setup

### Prerequisites
- Python 3
- pip

### Run it

```bash
# Clone the repo
git clone https://github.com/ShreyaSindhe/AUTO-HEAL.git
cd AUTO-HEAL

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python autoheal.py
```

Each run regenerates a `results/` folder containing:
- `breast_cancer_results.csv`, `iris_results.csv`, `comparison_results.csv`
- `fig1_accuracy_over_time.png` through `fig5_combined_paper.png`

---

## 🔬 Experiments

The pipeline runs:

- Initial model training on clean data
- Concept drift simulation
- Rolling-accuracy drift detection
- Automatic retraining on drift detection
- Recovery evaluation post-retrain
- Baseline comparison (no monitoring vs. scheduled retraining vs. AutoHeal)
- Statistical analysis across multiple runs
- Automatic visualization of all results

---

## 📁 Project Structure

```
AUTO-HEAL/
├── autoheal.py                       # Main pipeline script
├── requirements.txt                  # Python dependencies
├── .gitignore
├── README.md
├── AutoHeal_Methodology_Diagram.pdf  # Methodology diagram
├── docs/
│   └── images/                       # Result plots shown in this README
└── results/                          # Generated per run (CSVs + PNGs) — not tracked in git
```

---

## 🚀 Future Work

- [ ] Additional drift detection algorithms (ADWIN, Page-Hinkley)
- [ ] MLOps integration — experiment tracking, model registry, CI/CD
- [ ] Real-time streaming data support
- [ ] Deep learning model support
- [ ] Cloud deployment
- [ ] Interactive monitoring dashboard

---

## 🛠️ Tech Stack

`Python` `scikit-learn` `Pandas` `NumPy` `Matplotlib`

---

## 👥 Contributors

| Name | GitHub |
|---|---|
| Shreya Sindhe | [@ShreyaSindhe](https://github.com/ShreyaSindhe) |
| Tharun C | [@Tharunnxx](https://github.com/Tharunnxx) |
