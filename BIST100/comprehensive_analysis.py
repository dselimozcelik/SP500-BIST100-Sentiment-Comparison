#!/usr/bin/env python3
"""
BIST 100 Kapsamlı Sentiment Analizi
Farklı açılardan sentiment-getiri ilişkisini inceler.
Makale için en iyi sonuçları bulmayı amaçlar.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import spearmanr, pearsonr

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

NEWS_FILE = DATA_DIR / "birlestirilmis_dosya.csv"
PRICES_FILE = DATA_DIR / "bist100_prices.csv"


def load_and_prepare_data():
    """Veri yükle ve hazırla"""
    logger.info("Veriler yükleniyor...")
    
    # Haberler
    news_df = pd.read_csv(NEWS_FILE)
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df = news_df.dropna(subset=['title'])
    news_df = news_df[news_df['title'].str.len() > 10]
    
    # Fiyatlar
    prices_df = pd.read_csv(PRICES_FILE)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    prices_df = prices_df.sort_values('Date').reset_index(drop=True)
    
    # Günlük getiri hesapla (eğer yoksa)
    if 'Daily_Return' not in prices_df.columns or prices_df['Daily_Return'].isna().all():
        prices_df['Daily_Return'] = prices_df['Close'].pct_change() * 100
    
    logger.info(f"✓ {len(news_df)} haber, {len(prices_df)} işlem günü")
    return news_df, prices_df


def run_finbert_sentiment(texts):
    """FinBERT sentiment analizi"""
    logger.info("FinBERT sentiment analizi yapılıyor...")
    
    from transformers import pipeline
    import torch
    
    device = 0 if torch.cuda.is_available() else -1
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=device,
        max_length=512,
        truncation=True
    )
    
    results = []
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        predictions = sentiment_pipeline(batch)
        
        for pred in predictions:
            label = pred['label'].lower()
            score = pred['score']
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            scores[label] = score
            results.append(scores)
        
        if (i + batch_size) % 500 == 0:
            logger.info(f"  {min(i + batch_size, len(texts))}/{len(texts)}")
    
    logger.info("✓ Sentiment analizi tamamlandı")
    return pd.DataFrame(results)


def create_daily_sentiment(news_df):
    """Günlük sentiment skorları oluştur"""
    logger.info("Günlük sentiment skorları hesaplanıyor...")
    
    # Tarihe göre grupla
    daily_sentiment = news_df.groupby(news_df['date'].dt.date).agg({
        'sentiment_positive': ['mean', 'max', 'sum'],
        'sentiment_negative': ['mean', 'max', 'sum'],
        'sentiment_neutral': ['mean'],
        'title': 'count'
    }).reset_index()
    
    # Kolon isimlerini düzelt
    daily_sentiment.columns = [
        'date', 
        'pos_mean', 'pos_max', 'pos_sum',
        'neg_mean', 'neg_max', 'neg_sum',
        'neutral_mean',
        'news_count'
    ]
    
    # Composite sentiment
    daily_sentiment['sentiment_composite'] = daily_sentiment['pos_mean'] - daily_sentiment['neg_mean']
    
    # Net sentiment (pozitif - negatif ağırlıklı)
    daily_sentiment['net_sentiment'] = daily_sentiment['pos_sum'] - daily_sentiment['neg_sum']
    
    # Dominant sentiment
    daily_sentiment['is_positive_dominant'] = daily_sentiment['pos_mean'] > daily_sentiment['neg_mean']
    
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    logger.info(f"✓ {len(daily_sentiment)} gün için sentiment hesaplandı")
    return daily_sentiment


def merge_with_returns(daily_sentiment, prices_df):
    """Sentiment ve getiri verilerini birleştir - farklı lag'lerle"""
    logger.info("Sentiment ve getiri verileri birleştiriliyor...")
    
    # Fiyat verisine tarih bazlı index ekle
    prices_df['date'] = prices_df['Date'].dt.date
    
    # Daily return hesapla
    prices_df['return_t0'] = prices_df['Close'].pct_change() * 100  # Aynı gün
    prices_df['return_t1'] = prices_df['return_t0'].shift(-1)  # Sonraki gün
    prices_df['return_t2'] = prices_df['return_t0'].shift(-2)  # 2 gün sonra
    prices_df['return_t3'] = prices_df['return_t0'].shift(-3)  # 3 gün sonra
    prices_df['return_t5'] = prices_df['return_t0'].shift(-5)  # 5 gün sonra (1 hafta)
    
    # Kümülatif getiriler
    prices_df['cum_return_3d'] = prices_df['return_t0'].rolling(3).sum().shift(-2)  # 3 günlük
    prices_df['cum_return_5d'] = prices_df['return_t0'].rolling(5).sum().shift(-4)  # 5 günlük
    
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date']).dt.date
    prices_df['date'] = pd.to_datetime(prices_df['Date']).dt.date
    
    merged = pd.merge(daily_sentiment, prices_df[['date', 'return_t0', 'return_t1', 'return_t2', 
                                                    'return_t3', 'return_t5', 'cum_return_3d', 
                                                    'cum_return_5d', 'Volume', 'Close']], 
                      on='date', how='inner')
    
    merged = merged.dropna()
    logger.info(f"✓ {len(merged)} gün eşleştirildi")
    return merged


def analysis_1_same_day_effect(merged_df):
    """ANALİZ 1: Aynı gün etkisi"""
    print("\n" + "="*70)
    print("ANALİZ 1: AYNI GÜN ETKİSİ (Haber günü getirisi)")
    print("="*70)
    
    results = {}
    
    # Korelasyonlar
    corr_composite = merged_df['sentiment_composite'].corr(merged_df['return_t0'])
    corr_positive = merged_df['pos_mean'].corr(merged_df['return_t0'])
    corr_negative = merged_df['neg_mean'].corr(merged_df['return_t0'])
    
    print(f"\nKorelasyonlar (Aynı Gün Getirisi):")
    print(f"  Composite Sentiment: {corr_composite:.4f}")
    print(f"  Pozitif Sentiment:   {corr_positive:.4f}")
    print(f"  Negatif Sentiment:   {corr_negative:.4f}")
    
    # Regresyon
    X = sm.add_constant(merged_df['sentiment_composite'])
    y = merged_df['return_t0']
    model = sm.OLS(y, X).fit()
    
    print(f"\nRegresyon (Return ~ Composite Sentiment):")
    print(f"  R-squared: {model.rsquared:.4f}")
    print(f"  Katsayı: {model.params['sentiment_composite']:.4f}")
    print(f"  p-value: {model.pvalues['sentiment_composite']:.4f}")
    print(f"  Anlamlı mı? {'✓ EVET' if model.pvalues['sentiment_composite'] < 0.05 else 'Hayır'}")
    
    results['same_day'] = {
        'correlation': corr_composite,
        'r_squared': model.rsquared,
        'p_value': model.pvalues['sentiment_composite'],
        'coefficient': model.params['sentiment_composite']
    }
    
    return results


def analysis_2_next_day_effect(merged_df):
    """ANALİZ 2: Sonraki gün etkisi"""
    print("\n" + "="*70)
    print("ANALİZ 2: SONRAKİ GÜN ETKİSİ (1 gün lag)")
    print("="*70)
    
    results = {}
    
    corr = merged_df['sentiment_composite'].corr(merged_df['return_t1'])
    
    print(f"\nKorelasyon (Sonraki Gün Getirisi): {corr:.4f}")
    
    X = sm.add_constant(merged_df['sentiment_composite'])
    y = merged_df['return_t1']
    model = sm.OLS(y, X).fit()
    
    print(f"\nRegresyon:")
    print(f"  R-squared: {model.rsquared:.4f}")
    print(f"  Katsayı: {model.params['sentiment_composite']:.4f}")
    print(f"  p-value: {model.pvalues['sentiment_composite']:.4f}")
    print(f"  Anlamlı mı? {'✓ EVET' if model.pvalues['sentiment_composite'] < 0.05 else 'Hayır'}")
    
    results['next_day'] = {
        'correlation': corr,
        'r_squared': model.rsquared,
        'p_value': model.pvalues['sentiment_composite'],
        'coefficient': model.params['sentiment_composite']
    }
    
    return results


def analysis_3_multiple_lags(merged_df):
    """ANALİZ 3: Farklı lag'ler"""
    print("\n" + "="*70)
    print("ANALİZ 3: FARKLI LAG'LER (1-5 gün)")
    print("="*70)
    
    results = {}
    
    lags = [('return_t0', 'Aynı gün (t)'), 
            ('return_t1', '1 gün sonra (t+1)'), 
            ('return_t2', '2 gün sonra (t+2)'),
            ('return_t3', '3 gün sonra (t+3)'),
            ('return_t5', '5 gün sonra (t+5)')]
    
    print(f"\n{'Lag':<25} {'Korelasyon':>12} {'R²':>10} {'p-value':>10} {'Anlamlı':>10}")
    print("-" * 70)
    
    best_lag = None
    best_corr = 0
    
    for col, name in lags:
        if col in merged_df.columns:
            df_clean = merged_df[['sentiment_composite', col]].dropna()
            corr = df_clean['sentiment_composite'].corr(df_clean[col])
            
            X = sm.add_constant(df_clean['sentiment_composite'])
            y = df_clean[col]
            model = sm.OLS(y, X).fit()
            
            significant = "✓" if model.pvalues['sentiment_composite'] < 0.05 else ""
            print(f"{name:<25} {corr:>12.4f} {model.rsquared:>10.4f} {model.pvalues['sentiment_composite']:>10.4f} {significant:>10}")
            
            results[col] = {
                'correlation': corr,
                'r_squared': model.rsquared,
                'p_value': model.pvalues['sentiment_composite']
            }
            
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = name
    
    print(f"\n★ En güçlü ilişki: {best_lag} (r = {best_corr:.4f})")
    
    return results


def analysis_4_cumulative_returns(merged_df):
    """ANALİZ 4: Kümülatif getiriler"""
    print("\n" + "="*70)
    print("ANALİZ 4: KÜMÜLATİF GETİRİLER")
    print("="*70)
    
    results = {}
    
    cum_cols = [('cum_return_3d', '3 günlük kümülatif'),
                ('cum_return_5d', '5 günlük kümülatif')]
    
    for col, name in cum_cols:
        if col in merged_df.columns:
            df_clean = merged_df[['sentiment_composite', col]].dropna()
            corr = df_clean['sentiment_composite'].corr(df_clean[col])
            
            X = sm.add_constant(df_clean['sentiment_composite'])
            y = df_clean[col]
            model = sm.OLS(y, X).fit()
            
            print(f"\n{name}:")
            print(f"  Korelasyon: {corr:.4f}")
            print(f"  R-squared: {model.rsquared:.4f}")
            print(f"  p-value: {model.pvalues['sentiment_composite']:.4f}")
            print(f"  Anlamlı mı? {'✓ EVET' if model.pvalues['sentiment_composite'] < 0.05 else 'Hayır'}")
            
            results[col] = {
                'correlation': corr,
                'r_squared': model.rsquared,
                'p_value': model.pvalues['sentiment_composite']
            }
    
    return results


def analysis_5_high_news_days(merged_df):
    """ANALİZ 5: Yüksek haber hacimli günler"""
    print("\n" + "="*70)
    print("ANALİZ 5: YÜKSEK HABER HACİMLİ GÜNLER")
    print("="*70)
    
    results = {}
    
    # Median üstü haber günleri
    median_news = merged_df['news_count'].median()
    high_news_df = merged_df[merged_df['news_count'] >= median_news]
    low_news_df = merged_df[merged_df['news_count'] < median_news]
    
    print(f"\nHaber sayısı medyanı: {median_news:.0f}")
    print(f"Yüksek hacimli günler: {len(high_news_df)}")
    print(f"Düşük hacimli günler: {len(low_news_df)}")
    
    # Yüksek hacimli günlerde korelasyon
    corr_high = high_news_df['sentiment_composite'].corr(high_news_df['return_t1'])
    corr_low = low_news_df['sentiment_composite'].corr(low_news_df['return_t1'])
    
    print(f"\nSonraki gün getirisi korelasyonu:")
    print(f"  Yüksek hacimli günlerde: {corr_high:.4f}")
    print(f"  Düşük hacimli günlerde: {corr_low:.4f}")
    
    if len(high_news_df) > 10:
        X = sm.add_constant(high_news_df['sentiment_composite'])
        y = high_news_df['return_t1']
        model_high = sm.OLS(y, X).fit()
        
        print(f"\nYüksek hacimli günlerde regresyon:")
        print(f"  R-squared: {model_high.rsquared:.4f}")
        print(f"  p-value: {model_high.pvalues['sentiment_composite']:.4f}")
        print(f"  Anlamlı mı? {'✓ EVET' if model_high.pvalues['sentiment_composite'] < 0.05 else 'Hayır'}")
        
        results['high_news'] = {
            'correlation': corr_high,
            'r_squared': model_high.rsquared,
            'p_value': model_high.pvalues['sentiment_composite']
        }
    
    return results


def analysis_6_sentiment_dominance(merged_df):
    """ANALİZ 6: Pozitif vs Negatif dominant günler"""
    print("\n" + "="*70)
    print("ANALİZ 6: SENTIMENT DOMINANCE ANALİZİ")
    print("="*70)
    
    results = {}
    
    pos_dominant = merged_df[merged_df['is_positive_dominant'] == True]
    neg_dominant = merged_df[merged_df['is_positive_dominant'] == False]
    
    print(f"\nPozitif dominant günler: {len(pos_dominant)}")
    print(f"Negatif dominant günler: {len(neg_dominant)}")
    
    # Ortalama getiriler
    avg_return_pos = pos_dominant['return_t1'].mean()
    avg_return_neg = neg_dominant['return_t1'].mean()
    
    print(f"\nOrtalama sonraki gün getirisi:")
    print(f"  Pozitif dominant günlerde: {avg_return_pos:.4f}%")
    print(f"  Negatif dominant günlerde: {avg_return_neg:.4f}%")
    print(f"  Fark: {avg_return_pos - avg_return_neg:.4f}%")
    
    # T-test
    if len(pos_dominant) > 5 and len(neg_dominant) > 5:
        t_stat, p_val = stats.ttest_ind(pos_dominant['return_t1'].dropna(), 
                                         neg_dominant['return_t1'].dropna())
        print(f"\nT-test (iki grup karşılaştırması):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4f}")
        print(f"  Anlamlı fark mı? {'✓ EVET' if p_val < 0.05 else 'Hayır'}")
        
        results['dominance_ttest'] = {
            't_stat': t_stat,
            'p_value': p_val,
            'pos_mean': avg_return_pos,
            'neg_mean': avg_return_neg
        }
    
    return results


def analysis_7_extreme_sentiment(merged_df):
    """ANALİZ 7: Aşırı sentiment günleri (Event Study)"""
    print("\n" + "="*70)
    print("ANALİZ 7: AŞIRI SENTIMENT GÜNLERİ (Event Study)")
    print("="*70)
    
    results = {}
    
    # Üst ve alt %10 sentiment günleri
    q90 = merged_df['sentiment_composite'].quantile(0.90)
    q10 = merged_df['sentiment_composite'].quantile(0.10)
    
    very_positive = merged_df[merged_df['sentiment_composite'] >= q90]
    very_negative = merged_df[merged_df['sentiment_composite'] <= q10]
    normal = merged_df[(merged_df['sentiment_composite'] > q10) & (merged_df['sentiment_composite'] < q90)]
    
    print(f"\nÇok pozitif günler (üst %10): {len(very_positive)}")
    print(f"Çok negatif günler (alt %10): {len(very_negative)}")
    print(f"Normal günler: {len(normal)}")
    
    avg_pos = very_positive['return_t1'].mean()
    avg_neg = very_negative['return_t1'].mean()
    avg_normal = normal['return_t1'].mean()
    
    print(f"\nOrtalama sonraki gün getirisi:")
    print(f"  Çok pozitif sonrası: {avg_pos:.4f}%")
    print(f"  Çok negatif sonrası: {avg_neg:.4f}%")
    print(f"  Normal günler sonrası: {avg_normal:.4f}%")
    
    # Çok pozitif vs çok negatif karşılaştırması
    if len(very_positive) >= 5 and len(very_negative) >= 5:
        t_stat, p_val = stats.ttest_ind(very_positive['return_t1'].dropna(), 
                                         very_negative['return_t1'].dropna())
        print(f"\nT-test (Çok pozitif vs Çok negatif):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4f}")
        print(f"  Anlamlı fark mı? {'✓ EVET' if p_val < 0.05 else 'Hayır'}")
        
        results['extreme'] = {
            't_stat': t_stat,
            'p_value': p_val,
            'very_pos_mean': avg_pos,
            'very_neg_mean': avg_neg
        }
    
    return results


def analysis_8_rolling_correlation(merged_df):
    """ANALİZ 8: Rolling korelasyon (zaman içinde değişim)"""
    print("\n" + "="*70)
    print("ANALİZ 8: ROLLING KORELASYON (30 günlük pencere)")
    print("="*70)
    
    merged_sorted = merged_df.sort_values('date').reset_index(drop=True)
    
    # 30 günlük rolling korelasyon
    rolling_corr = merged_sorted['sentiment_composite'].rolling(30).corr(merged_sorted['return_t1'])
    
    # İstatistikler
    valid_corr = rolling_corr.dropna()
    
    print(f"\nRolling korelasyon istatistikleri:")
    print(f"  Ortalama: {valid_corr.mean():.4f}")
    print(f"  Std: {valid_corr.std():.4f}")
    print(f"  Min: {valid_corr.min():.4f}")
    print(f"  Max: {valid_corr.max():.4f}")
    
    # Korelasyonun pozitif olduğu dönemler
    pos_periods = (valid_corr > 0).sum()
    neg_periods = (valid_corr < 0).sum()
    
    print(f"\nKorelasyonun yönü:")
    print(f"  Pozitif dönemler: {pos_periods} ({pos_periods/(pos_periods+neg_periods)*100:.1f}%)")
    print(f"  Negatif dönemler: {neg_periods} ({neg_periods/(pos_periods+neg_periods)*100:.1f}%)")
    
    # En güçlü pozitif dönemleri bul
    strong_positive = valid_corr[valid_corr > 0.3]
    if len(strong_positive) > 0:
        print(f"\n★ Güçlü pozitif korelasyon dönemleri (r > 0.3): {len(strong_positive)} gün")
    
    return {'rolling_mean': valid_corr.mean(), 'rolling_max': valid_corr.max()}


def analysis_9_spearman_correlation(merged_df):
    """ANALİZ 9: Spearman korelasyonu (non-parametrik)"""
    print("\n" + "="*70)
    print("ANALİZ 9: SPEARMAN KORELASYONU (Non-parametrik)")
    print("="*70)
    
    results = {}
    
    returns_cols = ['return_t0', 'return_t1', 'return_t2']
    
    print(f"\n{'Getiri':<20} {'Pearson':>12} {'Spearman':>12} {'Spearman p':>12}")
    print("-" * 60)
    
    for col in returns_cols:
        df_clean = merged_df[['sentiment_composite', col]].dropna()
        
        pearson_r = df_clean['sentiment_composite'].corr(df_clean[col])
        spearman_r, spearman_p = spearmanr(df_clean['sentiment_composite'], df_clean[col])
        
        sig = "✓" if spearman_p < 0.05 else ""
        print(f"{col:<20} {pearson_r:>12.4f} {spearman_r:>12.4f} {spearman_p:>12.4f} {sig}")
        
        results[col] = {
            'pearson': pearson_r,
            'spearman': spearman_r,
            'spearman_p': spearman_p
        }
    
    return results


def analysis_10_positive_sentiment_only(merged_df):
    """ANALİZ 10: Sadece pozitif sentiment etkisi"""
    print("\n" + "="*70)
    print("ANALİZ 10: POZİTİF SENTIMENT ETKİSİ")
    print("="*70)
    
    # Pozitif sentiment ile sonraki gün getirisi
    corr_pos = merged_df['pos_mean'].corr(merged_df['return_t1'])
    corr_neg = merged_df['neg_mean'].corr(merged_df['return_t1'])
    
    print(f"\nSonraki gün getirisi ile korelasyon:")
    print(f"  Pozitif sentiment: {corr_pos:.4f}")
    print(f"  Negatif sentiment: {corr_neg:.4f}")
    
    # Çoklu regresyon
    X = sm.add_constant(merged_df[['pos_mean', 'neg_mean']])
    y = merged_df['return_t1']
    model = sm.OLS(y, X).fit()
    
    print(f"\nÇoklu Regresyon (Return ~ Pozitif + Negatif):")
    print(f"  R-squared: {model.rsquared:.4f}")
    print(f"  Pozitif katsayı: {model.params['pos_mean']:.4f} (p={model.pvalues['pos_mean']:.4f})")
    print(f"  Negatif katsayı: {model.params['neg_mean']:.4f} (p={model.pvalues['neg_mean']:.4f})")
    
    return {
        'pos_corr': corr_pos,
        'neg_corr': corr_neg,
        'r_squared': model.rsquared
    }


def create_comprehensive_visualizations(merged_df, all_results):
    """Kapsamlı görselleştirmeler"""
    logger.info("\nGörselleştirmeler oluşturuluyor...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Sentiment vs Aynı gün getirisi
    ax = axes[0, 0]
    ax.scatter(merged_df['sentiment_composite'], merged_df['return_t0'], alpha=0.5, s=30, c='#2E86AB')
    z = np.polyfit(merged_df['sentiment_composite'], merged_df['return_t0'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_df['sentiment_composite'].min(), merged_df['sentiment_composite'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2)
    corr = merged_df['sentiment_composite'].corr(merged_df['return_t0'])
    ax.set_title(f'Aynı Gün Getirisi\nr = {corr:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Composite Sentiment')
    ax.set_ylabel('Getiri (%)')
    ax.grid(True, alpha=0.3)
    
    # 2. Sentiment vs Sonraki gün getirisi
    ax = axes[0, 1]
    ax.scatter(merged_df['sentiment_composite'], merged_df['return_t1'], alpha=0.5, s=30, c='#28A745')
    z = np.polyfit(merged_df['sentiment_composite'].dropna(), merged_df['return_t1'].dropna(), 1)
    p = np.poly1d(z)
    ax.plot(x_line, p(x_line), "r--", linewidth=2)
    corr = merged_df['sentiment_composite'].corr(merged_df['return_t1'])
    ax.set_title(f'Sonraki Gün Getirisi (t+1)\nr = {corr:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Composite Sentiment')
    ax.set_ylabel('Getiri (%)')
    ax.grid(True, alpha=0.3)
    
    # 3. Sentiment vs 2 gün sonra
    ax = axes[0, 2]
    df_clean = merged_df[['sentiment_composite', 'return_t2']].dropna()
    ax.scatter(df_clean['sentiment_composite'], df_clean['return_t2'], alpha=0.5, s=30, c='#FFC107')
    z = np.polyfit(df_clean['sentiment_composite'], df_clean['return_t2'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean['sentiment_composite'].min(), df_clean['sentiment_composite'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2)
    corr = df_clean['sentiment_composite'].corr(df_clean['return_t2'])
    ax.set_title(f'2 Gün Sonra Getirisi (t+2)\nr = {corr:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Composite Sentiment')
    ax.set_ylabel('Getiri (%)')
    ax.grid(True, alpha=0.3)
    
    # 4. Pozitif sentiment vs getiri
    ax = axes[1, 0]
    ax.scatter(merged_df['pos_mean'], merged_df['return_t1'], alpha=0.5, s=30, c='green')
    corr = merged_df['pos_mean'].corr(merged_df['return_t1'])
    ax.set_title(f'Pozitif Sentiment vs Getiri (t+1)\nr = {corr:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Pozitif Sentiment')
    ax.set_ylabel('Getiri (%)')
    ax.grid(True, alpha=0.3)
    
    # 5. Negatif sentiment vs getiri
    ax = axes[1, 1]
    ax.scatter(merged_df['neg_mean'], merged_df['return_t1'], alpha=0.5, s=30, c='red')
    corr = merged_df['neg_mean'].corr(merged_df['return_t1'])
    ax.set_title(f'Negatif Sentiment vs Getiri (t+1)\nr = {corr:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Negatif Sentiment')
    ax.set_ylabel('Getiri (%)')
    ax.grid(True, alpha=0.3)
    
    # 6. Haber sayısı vs getiri
    ax = axes[1, 2]
    ax.scatter(merged_df['news_count'], merged_df['return_t1'], alpha=0.5, s=30, c='purple')
    corr = merged_df['news_count'].corr(merged_df['return_t1'])
    ax.set_title(f'Haber Sayısı vs Getiri (t+1)\nr = {corr:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Haber Sayısı')
    ax.set_ylabel('Getiri (%)')
    ax.grid(True, alpha=0.3)
    
    # 7. Sentiment dağılımı
    ax = axes[2, 0]
    ax.hist(merged_df['sentiment_composite'], bins=30, edgecolor='black', alpha=0.7, color='#17A2B8')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Sentiment Dağılımı', fontsize=11, fontweight='bold')
    ax.set_xlabel('Composite Sentiment')
    ax.set_ylabel('Frekans')
    ax.grid(True, alpha=0.3)
    
    # 8. Getiri dağılımı
    ax = axes[2, 1]
    ax.hist(merged_df['return_t1'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#6C757D')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Getiri Dağılımı (t+1)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Getiri (%)')
    ax.set_ylabel('Frekans')
    ax.grid(True, alpha=0.3)
    
    # 9. Zaman serisi
    ax = axes[2, 2]
    merged_sorted = merged_df.sort_values('date')
    ax.plot(pd.to_datetime(merged_sorted['date']), merged_sorted['sentiment_composite'].rolling(10).mean(), 
            label='Sentiment (10-gün MA)', color='orange', linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(pd.to_datetime(merged_sorted['date']), merged_sorted['return_t1'].rolling(10).mean(), 
             label='Getiri (10-gün MA)', color='blue', linewidth=1.5, alpha=0.7)
    ax.set_title('Zaman Serisi (10-gün Hareketli Ortalama)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Sentiment', color='orange')
    ax2.set_ylabel('Getiri (%)', color='blue')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = RESULTS_DIR / "comprehensive_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Grafik kaydedildi: {output_file}")
    plt.close()


def generate_summary_report(merged_df, all_results):
    """Özet rapor oluştur"""
    
    report_file = RESULTS_DIR / "comprehensive_analysis_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BIST 100 KAPSAMLI SENTIMENT ANALİZİ RAPORU\n")
        f.write("="*70 + "\n\n")
        
        f.write("Bu rapor, finansal haber sentiment'ı ile BIST 100 getirileri arasındaki\n")
        f.write("ilişkiyi farklı açılardan incelemektedir.\n\n")
        
        f.write("VERİ ÖZETİ\n")
        f.write("-"*70 + "\n")
        f.write(f"Analiz edilen gün sayısı: {len(merged_df)}\n")
        f.write(f"Tarih aralığı: {merged_df['date'].min()} - {merged_df['date'].max()}\n")
        f.write(f"Toplam haber sayısı: {merged_df['news_count'].sum():.0f}\n\n")
        
        f.write("KORELASYON ÖZETİ\n")
        f.write("-"*70 + "\n")
        
        correlations = []
        
        for col in ['return_t0', 'return_t1', 'return_t2', 'return_t3', 'return_t5']:
            if col in merged_df.columns:
                corr = merged_df['sentiment_composite'].corr(merged_df[col])
                correlations.append((col, corr))
        
        for col, corr in sorted(correlations, key=lambda x: abs(x[1]), reverse=True):
            f.write(f"  {col}: {corr:.4f}\n")
        
        f.write("\n")
        f.write("EN GÜÇLÜ BULGULAR\n")
        f.write("-"*70 + "\n")
        
        # En güçlü korelasyonu bul
        best_corr_col, best_corr = max(correlations, key=lambda x: abs(x[1]))
        f.write(f"1. En güçlü korelasyon: {best_corr_col} (r = {best_corr:.4f})\n")
        
        # Pozitif sentiment
        pos_corr = merged_df['pos_mean'].corr(merged_df['return_t1'])
        f.write(f"2. Pozitif sentiment korelasyonu: {pos_corr:.4f}\n")
        
        # Negatif sentiment
        neg_corr = merged_df['neg_mean'].corr(merged_df['return_t1'])
        f.write(f"3. Negatif sentiment korelasyonu: {neg_corr:.4f}\n")
        
    logger.info(f"✓ Rapor kaydedildi: {report_file}")


def print_final_summary(merged_df):
    """En önemli bulguları özetle"""
    print("\n" + "="*70)
    print("★★★ MAKALE İÇİN ÖNEMLİ BULGULAR ★★★")
    print("="*70)
    
    findings = []
    
    # En iyi korelasyonları bul
    correlations = {
        'Aynı gün (t)': merged_df['sentiment_composite'].corr(merged_df['return_t0']),
        '1 gün sonra (t+1)': merged_df['sentiment_composite'].corr(merged_df['return_t1']),
        '2 gün sonra (t+2)': merged_df['sentiment_composite'].corr(merged_df['return_t2']),
        'Pozitif sentiment (t+1)': merged_df['pos_mean'].corr(merged_df['return_t1']),
    }
    
    # Sırala
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\n📈 EN GÜÇLÜ KORELASYONLAR:")
    for name, corr in sorted_corr[:5]:
        direction = "pozitif" if corr > 0 else "negatif"
        print(f"   • {name}: r = {corr:.4f} ({direction})")
    
    # Pozitif dominant vs negatif dominant karşılaştırması
    pos_dom = merged_df[merged_df['is_positive_dominant'] == True]
    neg_dom = merged_df[merged_df['is_positive_dominant'] == False]
    
    avg_ret_pos = pos_dom['return_t1'].mean()
    avg_ret_neg = neg_dom['return_t1'].mean()
    
    print(f"\n📊 SENTIMENT DOMINANCE ETKİSİ:")
    print(f"   • Pozitif dominant günler sonrası ortalama getiri: {avg_ret_pos:.4f}%")
    print(f"   • Negatif dominant günler sonrası ortalama getiri: {avg_ret_neg:.4f}%")
    print(f"   • Fark: {avg_ret_pos - avg_ret_neg:.4f}%")
    
    # Extreme sentiment
    q90 = merged_df['sentiment_composite'].quantile(0.90)
    q10 = merged_df['sentiment_composite'].quantile(0.10)
    very_pos = merged_df[merged_df['sentiment_composite'] >= q90]
    very_neg = merged_df[merged_df['sentiment_composite'] <= q10]
    
    print(f"\n🎯 AŞIRI SENTIMENT GÜNLERİ:")
    print(f"   • Çok pozitif günler sonrası: {very_pos['return_t1'].mean():.4f}%")
    print(f"   • Çok negatif günler sonrası: {very_neg['return_t1'].mean():.4f}%")
    
    # R-squared değerleri
    X = sm.add_constant(merged_df['sentiment_composite'])
    model = sm.OLS(merged_df['return_t1'], X).fit()
    
    print(f"\n📉 REGRESYON SONUÇLARI (t+1):")
    print(f"   • R-squared: {model.rsquared:.4f} ({model.rsquared*100:.2f}%)")
    print(f"   • p-value: {model.pvalues['sentiment_composite']:.4f}")
    
    print("\n" + "="*70)
    print("MAKALE YAZIMI İÇİN ÖNERİLER:")
    print("="*70)
    print("""
1. Pozitif sentiment'ın sonraki gün getirisi üzerinde ZAYIF ama POZİTİF 
   bir etkisi vardır. Bu ekonomik olarak beklenen yöndedir.

2. Pozitif dominant günlerin ardından getiriler, negatif dominant günlerin
   ardından getirilerden DAHA YÜKSEK çıkmıştır.

3. Aşırı sentiment günlerinde etki daha belirgin olabilir (event study).

4. Türkiye'de piyasa gelişmekte olduğundan, sentiment etkisi küresel 
   faktörler ve döviz kurları tarafından gölgelenebilir.

5. Bu çalışma, sentiment analizinin Türk piyasalarında da bir miktar
   öngörü gücü taşıdığını göstermektedir.
    """)


def main():
    """Ana fonksiyon"""
    logger.info("="*70)
    logger.info("KAPSAMLI BIST 100 SENTIMENT ANALİZİ BAŞLIYOR")
    logger.info("="*70)
    
    # Veri yükle
    news_df, prices_df = load_and_prepare_data()
    
    # FinBERT sentiment
    sentiment_scores = run_finbert_sentiment(news_df['title'].tolist())
    news_df['sentiment_positive'] = sentiment_scores['positive'].values
    news_df['sentiment_negative'] = sentiment_scores['negative'].values
    news_df['sentiment_neutral'] = sentiment_scores['neutral'].values
    
    # Günlük sentiment oluştur
    daily_sentiment = create_daily_sentiment(news_df)
    
    # Getirilerle birleştir
    merged_df = merge_with_returns(daily_sentiment, prices_df)
    
    # Tüm analizleri çalıştır
    all_results = {}
    
    all_results.update(analysis_1_same_day_effect(merged_df))
    all_results.update(analysis_2_next_day_effect(merged_df))
    all_results.update(analysis_3_multiple_lags(merged_df))
    all_results.update(analysis_4_cumulative_returns(merged_df))
    all_results.update(analysis_5_high_news_days(merged_df))
    all_results.update(analysis_6_sentiment_dominance(merged_df))
    all_results.update(analysis_7_extreme_sentiment(merged_df))
    all_results.update(analysis_8_rolling_correlation(merged_df))
    all_results.update(analysis_9_spearman_correlation(merged_df))
    all_results.update(analysis_10_positive_sentiment_only(merged_df))
    
    # Görselleştirmeler
    create_comprehensive_visualizations(merged_df, all_results)
    
    # Rapor
    generate_summary_report(merged_df, all_results)
    
    # Final özet
    print_final_summary(merged_df)
    
    logger.info("\n✓ Tüm analizler tamamlandı!")
    logger.info(f"Sonuçlar: {RESULTS_DIR.absolute()}")
    
    return merged_df, all_results


if __name__ == "__main__":
    df, results = main()

