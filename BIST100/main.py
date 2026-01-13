#!/usr/bin/env python3
# src/main.py
"""
BIST100 Sentiment Analizi - Ana Pipeline
=========================================
Bu script tüm projeyi sırasıyla çalıştırır:
1. Haber verisi çekme (veya mevcut CSV kullanma)
2. BIST100 fiyat verisi çekme
3. FinBERT ile sentiment analizi
4. Sentiment-Fiyat korelasyon ve tahmin analizi

Kullanım:
    python main.py                    # Tüm pipeline (haber çekmeden)
    python main.py --fetch-news       # Haberleri de çek (uzun sürer!)
    python main.py --start 2023-01-01 --end 2024-12-31  # Tarih belirt
"""

import argparse
import os
import sys
from datetime import datetime

# --- AYARLAR ---
DATA_DIR = "data"
RESULTS_DIR = "results"
NEWS_FILE = os.path.join(DATA_DIR, "bist_financial_news_v3.csv")
PRICE_FILE = os.path.join(DATA_DIR, "bist100_prices.csv")
SENTIMENT_FILE = os.path.join(DATA_DIR, "news_with_sentiment.csv")
DAILY_SENTIMENT_FILE = os.path.join(DATA_DIR, "daily_sentiment.csv")


def print_banner():
    """Başlangıç banner'ı"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🇹🇷 BIST100 SENTIMENT ANALİZİ PROJESİ 📊                   ║
    ║   FinBERT ile Finansal Haber Analizi                         ║
    ║                                                              ║
    ║   Data Intensive Computing - Final Project                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Gerekli kütüphaneleri kontrol et"""
    print("\n🔍 Bağımlılıklar kontrol ediliyor...")
    
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'torch': 'torch',
        'transformers': 'transformers',
        'yfinance': 'yfinance',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm',
        'statsmodels': 'statsmodels'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (EKSİK)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Eksik paketler: {', '.join(missing)}")
        print("   Yüklemek için: pip install -r requirements.txt")
        return False
    
    print("   ✅ Tüm bağımlılıklar mevcut!")
    return True


def step_1_fetch_news(start_date: str, end_date: str, skip: bool = True):
    """Adım 1: Haber verisi çekme"""
    print("\n" + "=" * 60)
    print("📰 ADIM 1: HABER VERİSİ")
    print("=" * 60)
    
    if skip and os.path.exists(NEWS_FILE):
        import pandas as pd
        df = pd.read_csv(NEWS_FILE)
        print(f"✅ Mevcut haber dosyası kullanılıyor: {NEWS_FILE}")
        print(f"   Toplam {len(df)} haber mevcut.")
        return True
    
    if skip:
        print("⚠️  Haber dosyası bulunamadı!")
        print("   --fetch-news parametresi ile çalıştırın veya")
        print("   mevcut bir CSV dosyasını data/ klasörüne koyun.")
        return False
    
    print(f"🔄 Haberler çekiliyor: {start_date} -> {end_date}")
    print("   (Bu işlem uzun sürebilir...)")
    
    try:
        from news_scraper import fetch_robust_data
        fetch_robust_data(start_date, end_date)
        return True
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False


def step_2_fetch_prices(start_date: str, end_date: str):
    """Adım 2: BIST100 fiyat verisi çekme"""
    print("\n" + "=" * 60)
    print("📈 ADIM 2: BIST100 FİYAT VERİSİ")
    print("=" * 60)
    
    try:
        from price_fetcher import fetch_bist100_prices
        df = fetch_bist100_prices(start_date, end_date, save=True)
        return not df.empty
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False


def step_3_sentiment_analysis():
    """Adım 3: FinBERT ile sentiment analizi"""
    print("\n" + "=" * 60)
    print("🤖 ADIM 3: FINBERT SENTIMENT ANALİZİ")
    print("=" * 60)
    
    if not os.path.exists(NEWS_FILE):
        print(f"❌ Haber dosyası bulunamadı: {NEWS_FILE}")
        return False
    
    try:
        from sentiment_analyzer import analyze_news_file, get_daily_sentiment
        import pandas as pd
        
        # Sentiment analizi yap
        df = analyze_news_file(NEWS_FILE, SENTIMENT_FILE)
        
        if df.empty:
            return False
        
        # Günlük sentiment hesapla
        daily_sentiment = get_daily_sentiment(df)
        daily_sentiment.to_csv(DAILY_SENTIMENT_FILE, index=False)
        print(f"💾 Günlük sentiment kaydedildi: {DAILY_SENTIMENT_FILE}")
        
        return True
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_4_price_sentiment_analysis():
    """Adım 4: Sentiment-Fiyat analizi"""
    print("\n" + "=" * 60)
    print("📊 ADIM 4: SENTIMENT-FİYAT ANALİZİ")
    print("=" * 60)
    
    try:
        from sentiment_price_analysis import run_full_analysis
        analyzer = run_full_analysis()
        return analyzer is not None
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_results_summary():
    """Sonuçların özetini göster"""
    print("\n" + "=" * 60)
    print("📋 SONUÇ ÖZETİ")
    print("=" * 60)
    
    # Oluşturulan dosyalar
    files = {
        PRICE_FILE: "BIST100 Fiyat Verisi",
        SENTIMENT_FILE: "Sentiment Analizi Sonuçları",
        DAILY_SENTIMENT_FILE: "Günlük Sentiment",
        os.path.join(RESULTS_DIR, "correlation_matrix.png"): "Korelasyon Matrisi",
        os.path.join(RESULTS_DIR, "confusion_matrix.png"): "Confusion Matrix",
        os.path.join(RESULTS_DIR, "feature_importance.png"): "Özellik Önemleri",
        os.path.join(RESULTS_DIR, "event_study.png"): "Event Study",
        os.path.join(RESULTS_DIR, "time_series.png"): "Zaman Serisi",
        os.path.join(RESULTS_DIR, "analysis_report.txt"): "Analiz Raporu"
    }
    
    print("\n📁 Oluşturulan Dosyalar:")
    for filepath, description in files.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} bytes"
            print(f"   ✅ {description}: {filepath} ({size_str})")
        else:
            print(f"   ⬜ {description}: Henüz oluşturulmadı")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='BIST100 Sentiment Analizi Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--fetch-news', action='store_true',
                       help='Haberleri internetten çek (uzun sürer)')
    parser.add_argument('--start', type=str, default='2021-01-01',
                       help='Başlangıç tarihi (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='Bitiş tarihi (YYYY-MM-DD)')
    parser.add_argument('--skip-sentiment', action='store_true',
                       help='Sentiment analizini atla (mevcut dosya varsa)')
    parser.add_argument('--only-analysis', action='store_true',
                       help='Sadece analiz yap (veri çekme)')
    
    args = parser.parse_args()
    
    # Banner
    print_banner()
    
    # Tarih validasyonu
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        print(f"📅 Tarih Aralığı: {args.start} -> {args.end}")
    except ValueError as e:
        print(f"❌ Geçersiz tarih formatı: {e}")
        sys.exit(1)
    
    # Dizinleri oluştur
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Bağımlılık kontrolü
    if not check_dependencies():
        print("\n❌ Eksik bağımlılıklar var. Önce yükleyin.")
        sys.exit(1)
    
    # Pipeline adımları
    success = True
    
    if not args.only_analysis:
        # Adım 1: Haber verisi
        if not step_1_fetch_news(args.start, args.end, skip=not args.fetch_news):
            print("\n⚠️  Haber verisi bulunamadı. Devam edilemiyor.")
            success = False
        
        # Adım 2: Fiyat verisi
        if success:
            if not step_2_fetch_prices(args.start, args.end):
                print("\n⚠️  Fiyat verisi çekilemedi.")
                success = False
    
    # Adım 3: Sentiment analizi
    if success and not args.skip_sentiment:
        # Mevcut sentiment dosyası var mı kontrol et
        if os.path.exists(SENTIMENT_FILE) and args.skip_sentiment:
            print(f"\n✅ Mevcut sentiment dosyası kullanılıyor: {SENTIMENT_FILE}")
        else:
            if not step_3_sentiment_analysis():
                print("\n⚠️  Sentiment analizi başarısız.")
                success = False
    
    # Adım 4: Fiyat-Sentiment analizi
    if success:
        if not step_4_price_sentiment_analysis():
            print("\n⚠️  Analiz tamamlanamadı.")
            success = False
    
    # Sonuç özeti
    show_results_summary()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ PIPELINE BAŞARIYLA TAMAMLANDI!")
        print("=" * 60)
        print("\n📂 Sonuçlar için 'results/' klasörüne bakın.")
        print("📊 Detaylı rapor: results/analysis_report.txt")
    else:
        print("\n" + "=" * 60)
        print("⚠️  PIPELINE KISMI OLARAK TAMAMLANDI")
        print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

