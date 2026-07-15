"""Optuna hyperparameter search for the XGBoost default model.

Tunes on an inner train/validation split carved out of the training portion,
so the final test split (same seed as model_training.py) is never touched.
Best parameters are saved to models/xgb_best_params.joblib, which
model_training.py picks up automatically on the next run.
"""
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = "data/processed/model_df.parquet"
BEST_PARAMS_PATH = "models/xgb_best_params.joblib"
STUDY_PATH = "models/optuna_study.joblib"

TARGET = "loan_status"
TEST_SIZE = 0.18       # must match model_training.py
RANDOM_STATE = 42      # must match model_training.py
N_TRIALS = 40
TIMEOUT_SECONDS = 2400


def run_search():
    df = pd.read_parquet(DATA_PATH)
    feature_names = [c for c in df.columns if c != TARGET]
    # float32 halves memory with no meaningful accuracy loss for tree models
    X = df[feature_names].astype(np.float32)
    y = df[TARGET]
    del df

    # Outer split identical to model_training.py — the test set stays untouched
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    del X
    # Inner split for tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    del X_train
    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    def objective(trial):
        params = {
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "n_jobs": 4,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
        }
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    # catch keeps the study alive if a single trial fails (e.g. transient OOM)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, catch=(Exception,))

    best = study.best_trial
    best_params = {
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        # early stopping already found the right tree count for these params
        "n_estimators": best.user_attrs["best_iteration"] + 1,
        **best.params,
    }
    joblib.dump(best_params, BEST_PARAMS_PATH)
    joblib.dump(study, STUDY_PATH)

    print(f"[OK] {len(study.trials)} trials done. Best validation AUC: {best.value:.4f}")
    print(f"     Best params saved to {BEST_PARAMS_PATH}: {best_params}")
    print("     Now re-run: python src/model_training.py")


if __name__ == "__main__":
    run_search()
