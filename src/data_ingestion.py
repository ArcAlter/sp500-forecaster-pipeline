import yfinance as yf
import os

def ingest_data():
    ticker = "^xau" # หรือ "^XAU" ตามที่คุณต้องการ
    data = yf.download(ticker, period="max", interval="1d") # ใช้ 2 ปีเพื่อให้โมเดลมีข้อมูลเรียนรู้พอ
    
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/raw_data.csv')
    print("Successfully ingested data to data/raw_data.csv")

if __name__ == "__main__":
    ingest_data()