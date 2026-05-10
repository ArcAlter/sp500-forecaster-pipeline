from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(title="S&P 500 Prediction API")

# กำหนด Tracking URI ให้ตรงกับที่เก็บ log (ถ้าใช้ DagsHub หรือ Remote Server ให้ใส่ URL ตรงนี้)
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

MODEL_NAME = "sp500_xgboost_model"
# ดึงโมเดลเวอร์ชันล่าสุด (Latest)
model_uri = f"models:/{MODEL_NAME}/latest"
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/predict")
def predict(open_price: float, high: float, low: float, close: float, volume: float):
    # เตรียมข้อมูลให้เหมือนตอนเทรน
    data = pd.DataFrame([[open_price, high, low, close, volume]], 
                        columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    prediction = model.predict(data)
    return {
        "model_version": "latest",
        "predicted_open": float(prediction[0][0]),
        "predicted_high": float(prediction[0][1])
    }