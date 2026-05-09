import pandas as pd
import numpy as np
import mlflow
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train():
    # โหลดข้อมูล
    X_train = pd.read_csv('data/X_train.csv', index_col=0)
    y_train = pd.read_csv('data/y_train.csv', index_col=0)
    X_test = pd.read_csv('data/X_test.csv', index_col=0)
    y_test = pd.read_csv('data/y_test.csv', index_col=0)

    # ตั้งค่า MLflow
    mlflow.set_tracking_uri("file://" + os.path.join(os.getcwd(), "mlruns"))
    mlflow.set_experiment("XAU_Forecasting")

    with mlflow.start_run(run_name="XGB_Weekly_Run"):
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)

        # คำนวณ Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Logging
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.xgboost.log_model(model, "model")
        
        print(f"Training finished. RMSE: {rmse:.4f}")

if __name__ == "__main__":
    train()