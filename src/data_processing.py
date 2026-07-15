import re

import pandas as pd
import numpy as np


RAW_DATA_PATH = "data/raw/train.csv"
OUTPUT_PATH = "data/processed/model_df.parquet"

# Ordinal mapping of bureau risk grades from PERFORM_CNS.SCORE.DESCRIPTION
# (A = Very Low Risk ... M = Very High Risk)
BUREAU_GRADE_MAP = {g: i for i, g in enumerate("ABCDEFGHIJKLM", start=1)}


def _normalize_columns(df):
    """Normalize raw column names: 'PERFORM_CNS.SCORE' -> 'perform_cns_score'.

    Kaggle mirrors of the LTFS dataset differ in casing and separators
    ('.', ' ', '_'), so everything is collapsed to snake_case.
    """
    df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in df.columns]
    return df


def _duration_to_months(value):
    """Parse durations like '1yrs 11mon' into total months."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)\s*yrs?\s*(\d+)\s*mon", str(value))
    if not match:
        return np.nan
    return int(match.group(1)) * 12 + int(match.group(2))


def preprocess_pipeline(input_path):
    df = pd.read_csv(input_path)
    df = _normalize_columns(df)

    ### Target — rename so downstream scripts keep a stable name

    df = df.rename(columns={"loan_default": "loan_status"})
    df["loan_status"] = df["loan_status"].astype(int)

    ### Applicant demographics

    dob = pd.to_datetime(df["date_of_birth"], format="%d-%m-%y", errors="coerce")
    disbursal = pd.to_datetime(df["disbursaldate"], format="%d-%m-%y", errors="coerce")
    df["age_years"] = (disbursal - dob).dt.days / 365.25
    # two-digit years can parse into the future (e.g. '68' -> 2068)
    df.loc[df["age_years"] < 0, "age_years"] += 100
    df = df.drop(columns=["date_of_birth", "disbursaldate"])

    emp = df["employment_type"].astype(str).str.strip().str.lower()
    df["emp_salaried"] = (emp == "salaried").astype(int)
    df["emp_self_employed"] = emp.str.startswith("self").astype(int)
    df = df.drop(columns=["employment_type"])

    ### KYC document flags

    doc_flags = ["aadhar_flag", "pan_flag", "voterid_flag", "driving_flag", "passport_flag"]
    df["kyc_docs_count"] = df[doc_flags].sum(axis=1)

    ### Bureau score (CIBIL-equivalent, 300-890 when scored)

    desc = df["perform_cns_score_description"].fillna("")
    df["no_bureau_history"] = desc.str.contains("No Bureau History", case=False).astype(int)
    df["bureau_not_scored"] = desc.str.contains("Not Scored", case=False).astype(int)
    df["bureau_risk_grade"] = desc.str.extract(r"^([A-M])-")[0].map(BUREAU_GRADE_MAP)

    df["bureau_score"] = df["perform_cns_score"].where(df["perform_cns_score"] >= 300)
    df["bureau_score_missing"] = df["bureau_score"].isna().astype(int)
    df = df.drop(columns=["perform_cns_score", "perform_cns_score_description"])

    ### Bureau account aggregates (primary + secondary tradelines)

    df["total_accts"] = df["pri_no_of_accts"] + df["sec_no_of_accts"]
    df["active_accts"] = df["pri_active_accts"] + df["sec_active_accts"]
    df["overdue_accts"] = df["pri_overdue_accts"] + df["sec_overdue_accts"]

    current_balance = (df["pri_current_balance"] + df["sec_current_balance"]).clip(lower=0)
    sanctioned_amount = (df["pri_sanctioned_amount"] + df["sec_sanctioned_amount"]).clip(lower=0)
    bureau_disbursed = (df["pri_disbursed_amount"] + df["sec_disbursed_amount"]).clip(lower=0)
    monthly_obligation = (df["primary_instal_amt"] + df["sec_instal_amt"]).clip(lower=0)

    df["current_balance_log"] = np.log1p(current_balance)
    df["sanctioned_amount_log"] = np.log1p(sanctioned_amount)
    df["bureau_disbursed_log"] = np.log1p(bureau_disbursed)
    df["monthly_obligation_log"] = np.log1p(monthly_obligation)

    df["bureau_util_pct"] = (current_balance / (sanctioned_amount + 1)).clip(upper=2) * 100
    df["active_ratio"] = df["active_accts"] / (df["total_accts"] + 1)
    df["overdue_ratio"] = df["overdue_accts"] / (df["active_accts"] + 1)

    df = df.drop(columns=[
        "pri_no_of_accts", "pri_active_accts", "pri_overdue_accts",
        "pri_current_balance", "pri_sanctioned_amount", "pri_disbursed_amount",
        "sec_no_of_accts", "sec_active_accts", "sec_overdue_accts",
        "sec_current_balance", "sec_sanctioned_amount", "sec_disbursed_amount",
        "primary_instal_amt", "sec_instal_amt",
    ])

    ### Credit history / recency

    df["avg_acct_age_months"] = df["average_acct_age"].map(_duration_to_months)
    df["credit_history_months"] = df["credit_history_length"].map(_duration_to_months)
    df = df.drop(columns=["average_acct_age", "credit_history_length"])

    df["delinq_flag"] = (df["delinquent_accts_in_last_six_months"] > 0).astype(int)
    df["new_credit_ratio"] = df["new_accts_in_last_six_months"] / (df["total_accts"] + 1)

    ### Loan terms

    df["disbursed_amount_log"] = np.log1p(df["disbursed_amount"])
    df["asset_cost_log"] = np.log1p(df["asset_cost"])
    df["instal_to_loan_ratio"] = np.expm1(df["monthly_obligation_log"]) / (df["disbursed_amount"] + 1)
    df = df.drop(columns=["disbursed_amount", "asset_cost"])

    ### High-cardinality IDs: frequency-encode geography, drop the rest

    state_freq = df["state_id"].value_counts(normalize=True)
    df["state_freq"] = df["state_id"].map(state_freq)

    manufacturer_freq = df["manufacturer_id"].value_counts(normalize=True)
    df["manufacturer_freq"] = df["manufacturer_id"].map(manufacturer_freq)

    df = df.drop(columns=[
        "uniqueid", "branch_id", "supplier_id", "manufacturer_id",
        "current_pincode_id", "employee_code_id", "state_id", "mobileno_avl_flag",
    ])

    ### Interaction terms

    df["util_score"] = df["bureau_util_pct"] * df["bureau_score"]
    df["ltv_score_ratio"] = df["ltv"] / (df["bureau_score"].fillna(300) + 1)
    df["overdue_x_delinq"] = df["overdue_accts"] * df["delinq_flag"]
    df["age_x_history"] = df["age_years"] * df["credit_history_months"]

    return df


def run_preprocessing():
    import os
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} not found. Download 'train.csv' from "
            "https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction "
            "and place it at data/raw/train.csv"
        )
    df = preprocess_pipeline(RAW_DATA_PATH)
    df.to_parquet(OUTPUT_PATH)
    print(f"[OK] Preprocessing complete. {len(df):,} rows, {df.shape[1]} columns. Saved to {OUTPUT_PATH}")
    print(f"   Default rate: {df['loan_status'].mean():.2%}")


if __name__ == "__main__":
    run_preprocessing()
