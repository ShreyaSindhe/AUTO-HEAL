#  AutoHeal

> An autonomous self-healing machine learning pipeline that continuously monitors model performance, detects concept drift, and automatically retrains models to maintain prediction accuracy.


## Overview

AutoHeal is a machine learning framework designed to automatically detect performance degradation caused by concept drift and recover model performance through autonomous retraining.

Unlike traditional machine learning systems that require manual intervention, AutoHeal continuously monitors prediction accuracy, detects drift using rolling performance metrics, and redeploys an updated model whenever necessary.

The project evaluates its effectiveness using the Breast Cancer Wisconsin Diagnostic and Iris datasets and compares its performance against conventional monitoring strategies.

## Features

- Automatic concept drift detection
- Autonomous model retraining
- Continuous performance monitoring
- Rolling accuracy analysis
- Random Forest classification
- Performance comparison with baseline approaches
- Classification reports
- Confusion matrix generation
- Experimental result visualization

## Workflow

```text
Train Initial Model
        │
        ▼
Monitor Predictions
        │
        ▼
Calculate Rolling Accuracy
        │
        ▼
Performance Drop?
    │             │
   No            Yes
    │             │
Continue     Detect Drift
                    │
                    ▼
           Retrain Model
                    │
                    ▼
          Deploy Updated Model
                    │
                    ▼
          Continue Monitoring
```

## Tech Stack

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

## Project Structure

```
AUTO-HEAL/
│
├── autoheal.py
├── README.md
```

## Installation

Clone the repository

```bash
git clone https://github.com/ShreyaSindhe/AUTO-HEAL.git
```

Move into the project

```bash
cd AUTO-HEAL
```

Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

Run the project

```bash
python autoheal.py
```
## Experiments

The project performs:

- Initial model training
- Concept drift simulation
- Drift detection
- Automatic retraining
- Recovery evaluation
- Baseline comparison
- Statistical analysis across multiple runs
- Visualization of results

## Results

AutoHeal demonstrates the ability to:

- Detect concept drift automatically
- Restore model accuracy after degradation
- Reduce manual intervention
- Outperform traditional monitoring strategies in maintaining prediction performance

## Future Work

- Deep learning support
- Real-time streaming data
- Additional drift detection algorithms
- MLOps integration
- Cloud deployment
- Interactive monitoring dashboard


## Author
*Shreya Sindhe*
