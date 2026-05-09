import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_data():
    df = pd.read_csv('data/raw_data.csv', index_col=0)
    
    # --- ป้องกัน Data Leakage ---
    # เราจะใช้ข้อมูลของวันนี้ (T) เพื่อทายราคาของวันพรุ่งนี้ (T+1)
    df['target_open'] = df['Open'].shift(-1)
    df['target_close'] = df['Close'].shift(-1)
    
    # ลบแถวสุดท้ายที่เป็น NaN เพราะไม่มีเฉลยของวันพรุ่งนี้
    df.dropna(inplace=True)
    
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df[['target_open', 'target_close']]
    
    # Split ข้อมูลแบบไม่ Shuffle เพราะเป็น Time-series (สำคัญมาก!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Save ข้อมูลที่ผ่านการ Process แล้ว
    X_train.to_csv('data/X_train.csv'); X_test.to_csv('data/X_test.csv')
    y_train.to_csv('data/y_train.csv'); y_test.to_csv('data/y_test.csv')
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_data()