"""Train the custom linear regression model and save it for the web app.

Run once:  python train_model.py
Needs california_housing_test.csv in the same folder.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

CSV_PATH = "california_housing_test.csv"


class MyLR:
    """Multiple linear regression using the normal equation."""

    def __init__(self):
        self.m = None  # coefficients
        self.b = None  # intercept

    def fit(self, X, y):
        X = np.insert(np.asarray(X, dtype=float), 0, 1, axis=1)
        y = np.asarray(y, dtype=float)
        # (X^T X)^-1 X^T y  (pinv is safer than inv if columns are correlated)
        betas = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.b = betas[0]
        self.m = betas[1:]

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.m + self.b


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MyLR()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("Test R2 :", round(r2_score(y_test, pred), 4))
    print("Test MAE:", round(mean_absolute_error(y_test, pred), 2))

    # Save everything the app needs: coefficients + feature info for the input form
    saved = {
        "features": list(X.columns),
        "coef": model.m.tolist(),
        "intercept": float(model.b),
        "feature_min": X.min().tolist(),
        "feature_max": X.max().tolist(),
        "feature_median": X.median().tolist(),
        "target": df.columns[-1],
    }
    with open("model.json", "w") as f:
        json.dump(saved, f, indent=2)
    print("Saved model.json")
