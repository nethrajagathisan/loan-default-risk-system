# Raw Data

This project uses the **L&T Vehicle Loan Default Prediction** dataset from Kaggle.

## Download

Download the dataset from:

https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction

Place the downloaded `train.csv` file here:

```
data/raw/train.csv
```

> Only `train.csv` is required. The original competition's `test.csv` is unlabeled and is not used in this project.

## Build the processed dataset

From the project root, run:

```bash
python src/data_processing.py
```

This generates the processed dataset used for model training.