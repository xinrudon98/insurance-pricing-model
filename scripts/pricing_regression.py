import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_PATH = Path("data/sample_pricing_data.csv")
OUTPUT_DIR = Path("output")

TARGET_COL = "premium_rate"
BASE_FEATURES = ["latitude", "longitude"]
CATEGORICAL_FEATURES = ["hazard_score", "month_bound", "tiv_band", "cov_a_band"]

# Optional flag to exclude specific segment / broker group
EXCLUDE_SPECIAL_SEGMENT = False

# =========================
# HELPERS
# =========================
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date_bound"] = pd.to_datetime(df["date_bound"])
    df["month_bound"] = df["date_bound"].dt.to_period("M").astype(str)

    df["hazard_score"] = df["hazard_score"].astype(str)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["property_limit"] = pd.to_numeric(df["property_limit"], errors="coerce")
    df["cov_a_limit"] = pd.to_numeric(df["cov_a_limit"], errors="coerce")
    df["premium_rate"] = pd.to_numeric(df["premium_rate"], errors="coerce")

    limit_bins = [0, 1e6, 5e6, 10e6, np.inf]
    limit_labels = ["0-1M", "1-5M", "5-10M", ">10M"]

    df["tiv_band"] = pd.cut(df["property_limit"], bins=limit_bins, labels=limit_labels)
    df["cov_a_band"] = pd.cut(df["cov_a_limit"], bins=limit_bins, labels=limit_labels)

    df = df.dropna(subset=["premium_rate", "latitude", "longitude"])

    if EXCLUDE_SPECIAL_SEGMENT and "segment_flag" in df.columns:
        df = df[df["segment_flag"] != 1]

    return df


def build_model_matrix(df: pd.DataFrame, include_month: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL]
    x_base = df[BASE_FEATURES].copy()

    dummy_cols = ["hazard_score", "tiv_band", "cov_a_band"]
    if include_month:
        dummy_cols.append("month_bound")

    x_cat = pd.get_dummies(df[dummy_cols], drop_first=False)
    x = pd.concat([x_base, x_cat], axis=1)
    x = sm.add_constant(x).astype(float)

    return x, y


def fit_ols_model(x: pd.DataFrame, y: pd.Series):
    return sm.OLS(y, x).fit()


def identify_outliers(y_true: pd.Series, y_pred: pd.Series, z: float = 1.96) -> pd.Series:
    resid_std = np.std(y_true - y_pred)
    lower = y_pred - z * resid_std
    upper = y_pred + z * resid_std
    return (y_true < lower) | (y_true > upper)


def save_model_summary(model, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: pd.Series, outliers_mask: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label="All Data")
    plt.scatter(y_true[outliers_mask], y_pred[outliers_mask], label="Outliers")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Actual Premium Rate")
    plt.ylabel("Predicted Premium Rate")
    plt.title("Actual vs Predicted Premium Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    print("Loading pricing dataset...")
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_PATH)
    df = prepare_data(df)

    print(f"Rows after preparation: {len(df)}")

    # Model 1: Full model
    x_full, y_full = build_model_matrix(df, include_month=True)
    model_full = fit_ols_model(x_full, y_full)
    y_pred_full = model_full.predict(x_full)
    outliers_full = identify_outliers(y_full, y_pred_full)

    print("Full model R-squared:", round(model_full.rsquared, 4))
    save_model_summary(model_full, OUTPUT_DIR / "primary_model_summary.txt")
    plot_actual_vs_predicted(
        y_full,
        y_pred_full,
        outliers_full,
        OUTPUT_DIR / "primary_actual_vs_predicted.png"
    )

    outliers_df = df.loc[outliers_full].copy()
    outliers_df["predicted_premium_rate"] = y_pred_full[outliers_full]
    outliers_df["residual"] = y_full[outliers_full] - y_pred_full[outliers_full]
    outliers_df.to_csv(OUTPUT_DIR / "primary_outliers.csv", index=False)

    # Model 2: No monthly effect
    x_nomonth, y_nomonth = build_model_matrix(df, include_month=False)
    model_nomonth = fit_ols_model(x_nomonth, y_nomonth)

    print("No-month model R-squared:", round(model_nomonth.rsquared, 4))
    save_model_summary(model_nomonth, OUTPUT_DIR / "supplementary_model_summary.txt")

    print("Modeling completed.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
