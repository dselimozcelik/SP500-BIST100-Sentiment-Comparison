# src/sentiment_price_analysis.py
"""
Sentiment ve Fiyat Verilerini Birleştirip Analiz Eden Modül
Korelasyon, regresyon ve tahmin analizi yapar.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# --- AYARLAR ---
DATA_DIR = "data"
RESULTS_DIR = "results"

# Matplotlib stil
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


class SentimentPriceAnalyzer:
    """
    Sentiment ve fiyat verilerini birleştirip analiz eden sınıf.
    """
    
    def __init__(self):
        self.merged_data = None
        self.model_results = {}
        
        # Sonuç dizinini oluştur
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def load_and_merge_data(self, 
                            sentiment_file: str = None,
                            price_file: str = None) -> pd.DataFrame:
        """
        Sentiment ve fiyat verilerini yükleyip birleştirir.
        """
        if sentiment_file is None:
            sentiment_file = os.path.join(DATA_DIR, "daily_sentiment.csv")
        if price_file is None:
            price_file = os.path.join(DATA_DIR, "bist100_prices.csv")
        
        print("📂 Veriler yükleniyor...")
        
        # Sentiment verisi
        if os.path.exists(sentiment_file):
            sentiment_df = pd.read_csv(sentiment_file)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
            print(f"   ✅ Sentiment: {len(sentiment_df)} gün")
        else:
            print(f"   ❌ Sentiment dosyası bulunamadı: {sentiment_file}")
            return pd.DataFrame()
        
        # Fiyat verisi
        if os.path.exists(price_file):
            price_df = pd.read_csv(price_file)
            price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
            print(f"   ✅ Fiyat: {len(price_df)} gün")
        else:
            print(f"   ❌ Fiyat dosyası bulunamadı: {price_file}")
            return pd.DataFrame()
        
        # Birleştir (inner join - sadece her iki veri setinde de olan günler)
        merged = pd.merge(
            sentiment_df, 
            price_df,
            left_on='date',
            right_on='Date',
            how='inner'
        )
        
        # Gereksiz sütunları kaldır
        if 'Date' in merged.columns:
            merged = merged.drop(columns=['Date'])
        
        print(f"\n🔗 Birleştirildi: {len(merged)} ortak gün")
        
        # Lag değişkenleri ekle (geçmiş sentiment'in etkisi)
        merged = merged.sort_values('date')
        merged['sentiment_lag1'] = merged['avg_sentiment'].shift(1)
        merged['sentiment_lag2'] = merged['avg_sentiment'].shift(2)
        merged['sentiment_lag3'] = merged['avg_sentiment'].shift(3)
        
        # Kümülatif sentiment (son 5 gün)
        merged['sentiment_ma5'] = merged['avg_sentiment'].rolling(window=5).mean()
        
        # Sentiment momentum (değişim)
        merged['sentiment_change'] = merged['avg_sentiment'].diff()
        
        # NaN'ları temizle
        merged = merged.dropna()
        
        self.merged_data = merged
        print(f"   Son veri: {len(merged)} satır (lag sonrası)")
        
        return merged
    
    def correlation_analysis(self) -> dict:
        """
        Sentiment ve fiyat arasındaki korelasyonu analiz eder.
        """
        if self.merged_data is None:
            print("❌ Önce load_and_merge_data() çalıştırın!")
            return {}
        
        print("\n" + "=" * 60)
        print("📊 KORELASYON ANALİZİ")
        print("=" * 60)
        
        df = self.merged_data
        
        # Korelasyon matrisi
        corr_cols = ['avg_sentiment', 'sentiment_lag1', 'sentiment_ma5',
                     'Daily_Return', 'Next_Day_Return', 'Close', 'Volume']
        
        # Mevcut sütunları filtrele
        corr_cols = [c for c in corr_cols if c in df.columns]
        
        corr_matrix = df[corr_cols].corr()
        
        # Görselleştir
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0,
                    fmt='.3f', ax=ax, vmin=-1, vmax=1)
        ax.set_title('Sentiment vs Fiyat Korelasyon Matrisi', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'correlation_matrix.png'), dpi=150)
        plt.close()
        
        # Önemli korelasyonlar
        results = {
            'sentiment_vs_same_day_return': df['avg_sentiment'].corr(df['Daily_Return']),
            'sentiment_vs_next_day_return': df['avg_sentiment'].corr(df['Next_Day_Return']),
            'lagged_sentiment_vs_return': df['sentiment_lag1'].corr(df['Daily_Return']),
            'sentiment_ma5_vs_return': df['sentiment_ma5'].corr(df['Daily_Return'])
        }
        
        print("\n🔍 Önemli Korelasyonlar:")
        for name, value in results.items():
            significance = "***" if abs(value) > 0.1 else "**" if abs(value) > 0.05 else "*" if abs(value) > 0.02 else ""
            print(f"   {name}: {value:.4f} {significance}")
        
        # İstatistiksel anlamlılık testi
        print("\n📐 İstatistiksel Anlamlılık (Pearson):")
        stat, p_value = stats.pearsonr(df['avg_sentiment'], df['Next_Day_Return'])
        print(f"   Sentiment -> Ertesi Gün Getiri: r={stat:.4f}, p={p_value:.4f}")
        print(f"   Anlamlı mı? {'Evet ✅' if p_value < 0.05 else 'Hayır ❌'} (p < 0.05)")
        
        self.model_results['correlation'] = results
        return results
    
    def granger_causality_test(self, max_lag: int = 5) -> dict:
        """
        Granger nedensellik testi - sentiment fiyatı tahmin ediyor mu?
        """
        if self.merged_data is None:
            return {}
        
        print("\n" + "=" * 60)
        print("📊 GRANGER NEDENSELLİK TESTİ")
        print("=" * 60)
        
        from statsmodels.tsa.stattools import grangercausalitytests
        
        df = self.merged_data[['avg_sentiment', 'Daily_Return']].dropna()
        
        print(f"\nH0: Sentiment, Getiriyi Granger-nedenlemez")
        print(f"Test edilen lag sayısı: 1-{max_lag}")
        
        try:
            results = grangercausalitytests(df[['Daily_Return', 'avg_sentiment']], 
                                           maxlag=max_lag, verbose=False)
            
            print("\n   Lag | F-test p-değeri | Sonuç")
            print("   " + "-" * 40)
            
            granger_results = {}
            for lag in range(1, max_lag + 1):
                p_value = results[lag][0]['ssr_ftest'][1]
                significant = "✅ Anlamlı" if p_value < 0.05 else "❌ Anlamsız"
                print(f"   {lag:3d} | {p_value:.4f}          | {significant}")
                granger_results[f'lag_{lag}'] = p_value
            
            self.model_results['granger'] = granger_results
            return granger_results
            
        except Exception as e:
            print(f"❌ Granger testi hatası: {e}")
            return {}
    
    def train_prediction_model(self, target: str = 'Next_Day_Direction') -> dict:
        """
        Sentiment'ten fiyat yönü tahmini için ML modeli eğitir.
        
        Args:
            target: 'Next_Day_Direction' (binary) veya 'Next_Day_Return' (continuous)
        """
        if self.merged_data is None:
            return {}
        
        print("\n" + "=" * 60)
        print("🤖 TAHMİN MODELİ EĞİTİMİ")
        print("=" * 60)
        
        df = self.merged_data.dropna()
        
        # Özellikler
        feature_cols = ['avg_sentiment', 'sentiment_lag1', 'sentiment_lag2',
                        'sentiment_ma5', 'sentiment_change', 'news_count',
                        'sentiment_std', 'Volatility_20d']
        
        # Mevcut sütunları filtrele
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[feature_cols]
        y = df[target]
        
        # Veriyi böl (zaman serisi için shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Zaman sırasını koru!
        )
        
        # Normalizasyon
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\n📊 Veri Boyutları:")
        print(f"   Eğitim: {len(X_train)} | Test: {len(X_test)}")
        print(f"   Özellikler: {feature_cols}")
        
        # Binary classification (yön tahmini)
        if 'Direction' in target:
            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            results = {}
            best_model = None
            best_acc = 0
            
            print("\n🎯 Model Performansları:")
            print("   " + "-" * 50)
            
            for name, model in models.items():
                # Eğit
                model.fit(X_train_scaled, y_train)
                
                # Tahmin
                y_pred = model.predict(X_test_scaled)
                
                # Metrikler
                acc = accuracy_score(y_test, y_pred)
                
                # Cross-validation (zaman serisi için özel split)
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv)
                
                results[name] = {
                    'accuracy': acc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"   {name}:")
                print(f"      Test Accuracy: {acc:.4f}")
                print(f"      CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
                if acc > best_acc:
                    best_acc = acc
                    best_model = (name, model)
            
            # En iyi model detayları
            print(f"\n🏆 En İyi Model: {best_model[0]} (Accuracy: {best_acc:.4f})")
            
            # Confusion matrix
            y_pred_best = best_model[1].predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred_best)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Düşüş', 'Yükseliş'],
                       yticklabels=['Düşüş', 'Yükseliş'])
            ax.set_xlabel('Tahmin')
            ax.set_ylabel('Gerçek')
            ax.set_title(f'Confusion Matrix - {best_model[0]}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
            plt.close()
            
            # Feature importance (Random Forest için)
            if isinstance(best_model[1], (RandomForestClassifier, GradientBoostingClassifier)):
                importances = best_model[1].feature_importances_
                feat_imp = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("\n📊 Özellik Önemleri:")
                for _, row in feat_imp.iterrows():
                    print(f"   {row['feature']}: {row['importance']:.4f}")
                
                # Görselleştir
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feat_imp, x='importance', y='feature', palette='viridis', ax=ax)
                ax.set_title('Özellik Önemleri (Feature Importance)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Önem')
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
                plt.close()
            
            # Classification report
            print("\n📋 Sınıflandırma Raporu:")
            print(classification_report(y_test, y_pred_best, 
                                       target_names=['Düşüş', 'Yükseliş']))
            
            self.model_results['prediction'] = results
            return results
        
        else:
            # Regression (sürekli değer tahmini)
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n📊 Regresyon Sonuçları:")
            print(f"   MSE: {mse:.4f}")
            print(f"   R²: {r2:.4f}")
            
            return {'mse': mse, 'r2': r2}
    
    def event_study(self, threshold: float = 0.3) -> pd.DataFrame:
        """
        Aşırı pozitif/negatif sentiment günlerinde ne oluyor?
        Event study yaklaşımı.
        """
        if self.merged_data is None:
            return pd.DataFrame()
        
        print("\n" + "=" * 60)
        print("📊 EVENT STUDY ANALİZİ")
        print("=" * 60)
        
        df = self.merged_data
        
        # Aşırı pozitif günler
        positive_events = df[df['avg_sentiment'] > threshold]
        # Aşırı negatif günler
        negative_events = df[df['avg_sentiment'] < -threshold]
        
        print(f"\n🔍 Threshold: ±{threshold}")
        print(f"   Aşırı pozitif günler: {len(positive_events)}")
        print(f"   Aşırı negatif günler: {len(negative_events)}")
        
        if len(positive_events) > 0:
            print(f"\n   📈 Pozitif Günlerde Ortalama Ertesi Gün Getiri: "
                  f"{positive_events['Next_Day_Return'].mean():.4f}%")
        
        if len(negative_events) > 0:
            print(f"   📉 Negatif Günlerde Ortalama Ertesi Gün Getiri: "
                  f"{negative_events['Next_Day_Return'].mean():.4f}%")
        
        # Normal günler
        normal_days = df[(df['avg_sentiment'] >= -threshold) & 
                         (df['avg_sentiment'] <= threshold)]
        print(f"   ➖ Normal Günlerde Ortalama Ertesi Gün Getiri: "
              f"{normal_days['Next_Day_Return'].mean():.4f}%")
        
        # Görselleştir
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Sol: Sentiment dağılımı
        ax1 = axes[0]
        ax1.hist(df['avg_sentiment'], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        ax1.axvline(threshold, color='green', linestyle='--', label=f'Pozitif threshold ({threshold})')
        ax1.axvline(-threshold, color='red', linestyle='--', label=f'Negatif threshold ({-threshold})')
        ax1.set_xlabel('Günlük Sentiment Skoru')
        ax1.set_ylabel('Gün Sayısı')
        ax1.set_title('Sentiment Dağılımı', fontweight='bold')
        ax1.legend()
        
        # Sağ: Scatter plot
        ax2 = axes[1]
        colors = ['red' if s < -threshold else 'green' if s > threshold else 'gray' 
                  for s in df['avg_sentiment']]
        ax2.scatter(df['avg_sentiment'], df['Next_Day_Return'], c=colors, alpha=0.6, s=30)
        ax2.set_xlabel('Sentiment Skoru')
        ax2.set_ylabel('Ertesi Gün Getiri (%)')
        ax2.set_title('Sentiment vs Ertesi Gün Getiri', fontweight='bold')
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'event_study.png'), dpi=150)
        plt.close()
        
        return df
    
    def time_series_plot(self):
        """
        Sentiment ve fiyat zaman serisi grafiği.
        """
        if self.merged_data is None:
            return
        
        df = self.merged_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # 1. Fiyat grafiği
        ax1 = axes[0]
        ax1.plot(df['date'], df['Close'], color='navy', linewidth=1.2)
        ax1.set_ylabel('BIST100 Kapanış', fontsize=11)
        ax1.set_title('BIST100 ve Sentiment Zaman Serisi', fontsize=14, fontweight='bold')
        ax1.fill_between(df['date'], df['Close'], alpha=0.3, color='navy')
        
        # 2. Sentiment grafiği
        ax2 = axes[1]
        ax2.bar(df['date'], df['avg_sentiment'], 
               color=['green' if x > 0 else 'red' for x in df['avg_sentiment']], 
               alpha=0.7, width=1)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Günlük Sentiment', fontsize=11)
        ax2.fill_between(df['date'], df['sentiment_ma5'], alpha=0.3, color='blue', label='5-gün MA')
        ax2.legend()
        
        # 3. Haber sayısı
        ax3 = axes[2]
        ax3.bar(df['date'], df['news_count'], color='purple', alpha=0.6, width=1)
        ax3.set_ylabel('Haber Sayısı', fontsize=11)
        ax3.set_xlabel('Tarih', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'time_series.png'), dpi=150)
        plt.close()
        
        print(f"📊 Zaman serisi grafiği kaydedildi: {RESULTS_DIR}/time_series.png")
    
    def generate_report(self) -> str:
        """
        Analiz sonuçlarının özetini oluşturur.
        """
        report = []
        report.append("=" * 60)
        report.append("📊 BIST100 SENTIMENT ANALİZİ RAPORU")
        report.append("=" * 60)
        
        if self.merged_data is not None:
            df = self.merged_data
            report.append(f"\n📅 Veri Aralığı: {df['date'].min()} - {df['date'].max()}")
            report.append(f"📈 Toplam Gün Sayısı: {len(df)}")
            report.append(f"📰 Toplam Haber Sayısı: {df['news_count'].sum():.0f}")
        
        if 'correlation' in self.model_results:
            report.append("\n--- KORELASYON SONUÇLARI ---")
            for k, v in self.model_results['correlation'].items():
                report.append(f"   {k}: {v:.4f}")
        
        if 'prediction' in self.model_results:
            report.append("\n--- TAHMİN MODELİ SONUÇLARI ---")
            for model, metrics in self.model_results['prediction'].items():
                report.append(f"   {model}: Accuracy={metrics['accuracy']:.4f}")
        
        report_text = "\n".join(report)
        
        # Dosyaya kaydet
        report_file = os.path.join(RESULTS_DIR, "analysis_report.txt")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n💾 Rapor kaydedildi: {report_file}")
        
        return report_text


def run_full_analysis():
    """
    Tam analiz pipeline'ı çalıştırır.
    """
    analyzer = SentimentPriceAnalyzer()
    
    # 1. Verileri yükle ve birleştir
    analyzer.load_and_merge_data()
    
    if analyzer.merged_data is None or len(analyzer.merged_data) < 10:
        print("❌ Yeterli veri yok! Önce haber ve fiyat verilerini çekin.")
        return None
    
    # 2. Korelasyon analizi
    analyzer.correlation_analysis()
    
    # 3. Granger nedensellik
    analyzer.granger_causality_test()
    
    # 4. Tahmin modeli
    analyzer.train_prediction_model()
    
    # 5. Event study
    analyzer.event_study()
    
    # 6. Zaman serisi grafiği
    analyzer.time_series_plot()
    
    # 7. Rapor oluştur
    analyzer.generate_report()
    
    return analyzer


if __name__ == "__main__":
    run_full_analysis()

