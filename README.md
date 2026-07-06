# AutoHeal

> An autonomous self-healing machine learning pipeline that continuously monitors model performance, detects concept drift, and automatically retrains models to maintain prediction accuracy.

## Overview

AutoHeal is a machine learning framework designed to automatically detect performance degradation caused by concept drift and recover model performance through autonomous retraining.

Unlike traditional machine learning systems that require manual intervention, AutoHeal continuously monitors prediction accuracy, detects drift using rolling performance metrics, and redeploys an updated model whenever necessary.

The project evaluates its effectiveness using the Breast Cancer Wisconsin Diagnostic and Iris datasets, and compares its performance against conventional monitoring strategies (no monitoring, and fixed-schedule retraining).

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
- Results automatically saved as CSV and PNG files

## Workflow

```text
Train Initial Model
        |
        v
Monitor Predictions
        |
        v
Calculate Rolling Accuracy
        |
        v
   Performance Drop?
    |            |
   No           Yes
    |            |
Continue    Detect Drift
                 |
                 v
          Retrain Model
                 |
                 v
       Deploy Updated Model
                 |
                 v
       Continue Monitoring
```

## Tech Stack

- Python 3
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

## Project Structure
