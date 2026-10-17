
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "fruit_store_sales.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def train_model():
    # 1. Check whether the dataset exists
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"CSV file not found: {DATA_FILE}\n"
            "Place fruit_store_sales.csv inside the data folder."
        )

    # 2. Read and prepare the dataset
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])

    required = [
        "date", "fruit", "category", "unit_price",
        "units_sold", "promotion", "store"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required).sort_values("date").copy()
    df["month"] = df["date"].dt.month

    features = [
        "fruit", "category", "unit_price",
        "promotion", "month", "store"
    ]
    target = "units_sold"

    # 3. Chronological train/test split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    if train_df.empty or test_df.empty:
        raise ValueError("Add more CSV rows for training and testing.")

    # 4. Encode text columns for machine learning
    categorical = ["fruit", "category", "store"]
    numerical = ["unit_price", "promotion", "month"]

    preprocessing = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        ("numerical", "passthrough", numerical)
    ])

    # 5. Build the ML pipeline
    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                min_samples_leaf=2
            )
        )
    ])

    # 6. Train the model
    model.fit(train_df[features], train_df[target])

    # 7. Evaluate on later records
    predictions = model.predict(test_df[features])

    mae = mean_absolute_error(test_df[target], predictions)
    rmse = mean_squared_error(
        test_df[target], predictions
    ) ** 0.5
    r2 = (
        r2_score(test_df[target], predictions)
        if len(test_df) > 1 else float("nan")
    )

    # 8. Save model and evaluation report
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {"pipeline": model, "features": features},
        MODEL_DIR / "units_sold_model.joblib"
    )

    report = (
        "FRUIT STORE ML MODEL EVALUATION\n"
        "--------------------------------\n"
        f"Dataset rows: {len(df)}\n"
        f"Training rows: {len(train_df)}\n"
        f"Testing rows: {len(test_df)}\n"
        f"MAE: {mae:.2f} units\n"
        f"RMSE: {rmse:.2f} units\n"
        f"R2 score: {r2:.3f}\n"
    )

    (REPORT_DIR / "model_metrics.txt").write_text(
        report, encoding="utf-8"
    )

    print(report)
    print("Model saved in the models folder.")
    print("Evaluation saved in the reports folder.")


if __name__ == "__main__":
    train_model()
