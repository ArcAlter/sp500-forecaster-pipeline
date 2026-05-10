import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_data():
# 1. โหลดข้อมูล
    df = pd.read_csv('data/raw_data.csv', index_col=0)
    
    # --- แก้ไขตรงนี้: บังคับให้ทุกคอลัมน์เป็นตัวเลข (ถ้าเจอตัวหนังสือจะกลายเป็น NaN แล้วโดนลบทิ้ง) ---
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['Close'], inplace=True) # ลบแถวที่ Close ไม่ใช่ตัวเลขออก
    
    # 2. คำนวณ Features ตามปกติ
    df['target_next_close'] = df['Close'].shift(-1)
    df['close_lag1'] = df['Close'].shift(1)
    df['close_lag2'] = df['Close'].shift(2)
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    
    # ลบแถวที่เป็น NaN (จาก MA และ Shift)
    df.dropna(inplace=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'close_lag1', 'close_lag2', 'ma5', 'ma20']
    X = df[features]
    y = df['target_next_close']
    
    # Split ข้อมูล
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Save ผลลัพธ์
    X_train.to_csv('data/X_train.csv'); X_test.to_csv('data/X_test.csv')
    y_train.to_csv('data/y_train.csv'); y_test.to_csv('data/y_test.csv')
    print("Preprocessing complete with clean numeric data.")

if __name__ == "__main__":
    preprocess_data()