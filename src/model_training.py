import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

DATA_PATH = "data/processed/model_df.parquet"
MODEL_PATH = "models/model_xgb.joblib"
CAL_MODEL_PATH = "models/model_xgb_calibrated.joblib"
THRESHOLD_PATH = "models/model_threshold.joblib"
FEATURES_PATH = "models/feature_names.joblib"
METADATA_PATH = "models/model_training_metadata.joblib"


# ---- XGBoost Parameters ----
# scale_pos_weight is computed from the training labels at fit time so it
# adapts to the dataset's actual default rate. If src/hyperparameter_tuning.py
# has been run, its Optuna-tuned parameters are used; otherwise these
# fallback defaults apply.
BEST_PARAMS_PATH = "models/xgb_best_params.joblib"

fallback_xgb_params = {
    "tree_method": "hist",
    #"device": "cuda",
    "objective": "binary:logistic",
    "max_depth": 4,
    "learning_rate": 0.06296939501995362,
    "gamma": 0.49470084266224107,
    "min_child_weight": 7,
    "subsample": 0.8800704280225073,
    "colsample_bytree": 0.5857661985145072,
    "reg_alpha": 0.12002226134106342,
    "reg_lambda": 2.839977470486064,
    "eval_metric": "auc",
    "random_state": 42
}


def load_xgb_params():
    try:
        params = joblib.load(BEST_PARAMS_PATH)
        print(f"Using Optuna-tuned parameters from {BEST_PARAMS_PATH}")
        params.pop("scale_pos_weight", None)  # recomputed from training labels
        return params
    except FileNotFoundError:
        print("No tuned parameters found; using fallback defaults "
              "(run src/hyperparameter_tuning.py to tune).")
        return fallback_xgb_params

TARGET = "loan_status"
CALIBRATION_METHOD = "isotonic"
TEST_SIZE = 0.18
RANDOM_STATE = 42
TARGET_BAD_RATE = 0.10
MIN_APPROVAL_RATE = 0.05

def train_and_save_model():
    # Load processed data
    df = pd.read_parquet(DATA_PATH)
    
    # Separate features and target
    feature_names = [col for col in df.columns if col != TARGET]
    X = df[feature_names]
    y = df[TARGET]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Train best XGBoost
    xgb_params = load_xgb_params()
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"scale_pos_weight (from training data): {scale_pos_weight:.3f}")
    xgb = XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight)
    xgb.fit(X_train, y_train)

    # Calibrate probabilities (FrozenEstimator replaces the removed cv="prefit")
    cal_model = CalibratedClassifierCV(FrozenEstimator(xgb), method=CALIBRATION_METHOD)
    cal_model.fit(X_train, y_train)
    cal_pred_probs = cal_model.predict_proba(X_test)[:, 1]

    # Find optimal threshold using approval/bad rate tradeoff
    results = pd.DataFrame({
        "pred_pd": cal_pred_probs,
        "actual": y_test.values
    })
    thresholds = np.linspace(0, 1, 101)
    approval_rates, bad_rates = [], []
    for t in thresholds:
        approved = results[results["pred_pd"] <= t]
        approval_rates.append(len(approved) / len(results))
        if len(approved) > 0:
            bad_rates.append(approved["actual"].mean())
        else:
            bad_rates.append(np.nan)
    bad_rates_array = np.array(bad_rates)
    thresholds_array = np.array(thresholds)
    safe_indices = np.where(
        (bad_rates_array <= TARGET_BAD_RATE) &
        (np.array(approval_rates) >= MIN_APPROVAL_RATE)
    )[0]
    if len(safe_indices) > 0:
        safe_idx = safe_indices[-1]
        safe_threshold = thresholds_array[safe_idx]
        print(f"Max safe threshold: {safe_threshold:.3f}")
        print(f"Bad rate at cutoff: {bad_rates_array[safe_idx]:.2%}")
        print(f"Approval rate at cutoff: {approval_rates[safe_idx]:.2%}")
    else:
        safe_threshold = 0.5  # fallback default
        print("No threshold meets target; fallback to 0.5.")

    # Save artifacts
    joblib.dump(xgb, MODEL_PATH)
    joblib.dump(cal_model, CAL_MODEL_PATH)
    joblib.dump(safe_threshold, THRESHOLD_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    joblib.dump({
        "target_bad_rate": TARGET_BAD_RATE,
        "min_approval_rate": MIN_APPROVAL_RATE,
        "test_size": TEST_SIZE,
        "calibration_method": CALIBRATION_METHOD
    }, METADATA_PATH)
    print("[OK] Model, threshold, and metadata saved.")

if __name__ == "__main__":
    train_and_save_model()