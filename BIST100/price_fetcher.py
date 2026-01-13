# src/price_fetcher.py
"""
BIST100 (XU100) Fiyat Verisi Çekme Modülü
Yahoo Finance API kullanarak günlük fiyat verileri çeker.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# --- AYARLAR ---
OUTPUT_DIR = "data"
PRICE_FILE = os.path.join(OUTPUT_DIR, "bist100_prices.csv")

# BIST100 Endeksi - Yahoo Finance sembolü
BIST100_SYMBOL = "XU100.IS"

# Alternatif olarak büyük BIST şirketleri de eklenebilir
MAJOR_STOCKS = {
    "XU100.IS": "BIST100_Index",
    "THYAO.IS": "Turkish_Airlines",
    "GARAN.IS": "Garanti_BBVA",
    "AKBNK.IS": "Akbank",
    "KCHOL.IS": "Koc_Holding",
    "ISCTR.IS": "Is_Bankasi",
    "TUPRS.IS": "Tupras",
    "ASELS.IS": "Aselsan",
    "SISE.IS": "Sisecam",
    "EREGL.IS": "Eregli_Demir"
}


def fetch_bist100_prices(start_date: str, end_date: str, save: bool = True) -> pd.DataFrame:
    """
    BIST100 endeks fiyatlarını çeker.
    
    Args:
        start_date: Başlangıç tarihi (YYYY-MM-DD)
        end_date: Bitiş tarihi (YYYY-MM-DD)
        save: CSV'ye kaydet (varsayılan True)
    
    Returns:
        DataFrame: Tarih, Open, High, Low, Close, Volume, Daily_Return
    """
    print(f"📈 BIST100 Fiyat Verisi Çekiliyor: {start_date} -> {end_date}")
    
    try:
        # Yahoo Finance'den veri çek
        ticker = yf.Ticker(BIST100_SYMBOL)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            print("⚠️ Veri bulunamadı!")
            return pd.DataFrame()
        
        # Index'i sütuna çevir ve temizle
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Günlük getiri hesapla
        df['Daily_Return'] = df['Close'].pct_change() * 100  # Yüzde olarak
        
        # İleri 1 günlük getiri (tahmin için)
        df['Next_Day_Return'] = df['Daily_Return'].shift(-1)
        
        # Fiyat değişim yönü (binary classification için)
        df['Price_Direction'] = (df['Daily_Return'] > 0).astype(int)
        df['Next_Day_Direction'] = (df['Next_Day_Return'] > 0).astype(int)
        
        # Volatilite (20 günlük rolling std)
        df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()
        
        # Sütunları seç ve yeniden adlandır
        result_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                        'Daily_Return', 'Next_Day_Return', 
                        'Price_Direction', 'Next_Day_Direction',
                        'Volatility_20d']].copy()
        
        print(f"✅ {len(result_df)} günlük veri çekildi.")
        print(f"   Tarih Aralığı: {result_df['Date'].min()} -> {result_df['Date'].max()}")
        print(f"   Ortalama Günlük Getiri: {result_df['Daily_Return'].mean():.3f}%")
        
        if save:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            result_df.to_csv(PRICE_FILE, index=False)
            print(f"💾 Kaydedildi: {PRICE_FILE}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return pd.DataFrame()


def fetch_multiple_stocks(start_date: str, end_date: str, 
                          symbols: dict = None, save: bool = True) -> pd.DataFrame:
    """
    Birden fazla hisse senedi için veri çeker.
    Sektörel analiz veya şirket bazlı sentiment için kullanılabilir.
    """
    if symbols is None:
        symbols = MAJOR_STOCKS
    
    print(f"📊 Çoklu Hisse Verisi Çekiliyor ({len(symbols)} sembol)...")
    
    all_data = []
    
    for symbol, name in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if not df.empty:
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                df['Symbol'] = symbol
                df['Company'] = name
                df['Daily_Return'] = df['Close'].pct_change() * 100
                all_data.append(df)
                print(f"   ✅ {name}: {len(df)} gün")
            else:
                print(f"   ⚠️ {name}: Veri yok")
                
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        
        if save:
            output_file = os.path.join(OUTPUT_DIR, "bist_stocks_prices.csv")
            result_df.to_csv(output_file, index=False)
            print(f"💾 Kaydedildi: {output_file}")
        
        return result_df
    
    return pd.DataFrame()


def load_prices(file_path: str = None) -> pd.DataFrame:
    """Kaydedilmiş fiyat verisini yükler."""
    if file_path is None:
        file_path = PRICE_FILE
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'])
        print(f"📂 Yüklendi: {file_path} ({len(df)} satır)")
        return df
    else:
        print(f"⚠️ Dosya bulunamadı: {file_path}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test: 2021-2024 arası veri çek
    df = fetch_bist100_prices("2021-01-01", "2021-05-30")
    
    if not df.empty:
        print("\n📊 Veri Özeti:")
        print(df.describe())
        print("\n🔍 Son 5 Gün:")
        print(df.tail())

