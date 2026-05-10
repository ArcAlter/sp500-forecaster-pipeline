import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_data():
    df = pd.read_csv('data/raw_data.csv', index_col=0)
    
    df['target_next_close'] = df['Close'].shift(-1)
    
    # 2. เพิ่ม Lag Features: ราคาปิดของ 1, 2, 3 วันก่อนหน้า
    df['close_lag1'] = df['Close'].shift(1)
    df['close_lag2'] = df['Close'].shift(2)
    
    # 3. เพิ่ม Moving Average: ค่าเฉลี่ย 5 วัน และ 20 วัน
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    
    # ลบแถวสุดท้ายที่เป็น NaN เพราะไม่มีเฉลยของวันพรุ่งนี้
    df.dropna(inplace=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'close_lag1', 'close_lag2', 'ma5', 'ma20']
    X = df[features]
    y = df['target_next_close']
    
    # Split ข้อมูลแบบไม่ Shuffle เพราะเป็น Time-series (สำคัญมาก!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Save ข้อมูลที่ผ่านการ Process แล้ว
    X_train.to_csv('data/X_train.csv'); X_test.to_csv('data/X_test.csv')
    y_train.to_csv('data/y_train.csv'); y_test.to_csv('data/y_test.csv')
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_data()