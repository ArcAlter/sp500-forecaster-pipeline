import yfinance as yf
import os
import pandas as pd

def ingest_data():
    # ticker = "^xau" # หรือ "^XAU" ตามที่คุณต้องการ
    # data = yf.download(ticker, period="max", interval="1d") # ใช้ 2 ปีเพื่อให้โมเดลมีข้อมูลเรียนรู้พอ
    
    # os.makedirs('data', exist_ok=True)
    # data.to_csv('data/raw_data.csv')
    # print("Successfully ingested data to data/raw_data.csv")
    ticker = "^XAU"
    data = yf.download(ticker, period="5y", interval="1d")
    
    # --- แก้ไขตรงนี้: ลบ Multi-index ออกให้เหลือแค่ Open, High, Low, Close ---
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/raw_data.csv')
    print("Successfully ingested data and flattened columns.")

if __name__ == "__main__":
    ingest_data()