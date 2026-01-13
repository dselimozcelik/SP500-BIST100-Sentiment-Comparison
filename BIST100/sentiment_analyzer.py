# src/sentiment_analyzer.py
"""
FinBERT ile Finansal Haber Sentiment Analizi Modülü
ProsusAI/finbert modelini kullanır - finansal metinler için özel eğitilmiş BERT.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# --- AYARLAR ---
MODEL_NAME = "ProsusAI/finbert"  # Finansal sentiment için en iyi model
DATA_DIR = "data"
NEWS_FILE = os.path.join(DATA_DIR, "bist_financial_news_v3.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "news_with_sentiment.csv")

# Sentiment etiketleri
LABELS = ['negative', 'neutral', 'positive']


class FinBERTSentimentAnalyzer:
    """
    FinBERT tabanlı finansal sentiment analizi sınıfı.
    """
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        """
        Args:
            model_name: Hugging Face model adı
            device: 'cuda' veya 'cpu' (None ise otomatik seçer)
        """
        print(f"🤖 FinBERT Modeli Yükleniyor: {model_name}")
        
        # Device seçimi
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"   Device: {self.device}")
        
        # Model ve tokenizer yükle
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model başarıyla yüklendi!")
    
    def analyze_single(self, text: str) -> dict:
        """
        Tek bir metin için sentiment analizi yapar.
        
        Args:
            text: Analiz edilecek metin (haber başlığı veya içerik)
        
        Returns:
            dict: {
                'sentiment': 'positive'/'negative'/'neutral',
                'confidence': float (0-1),
                'positive_prob': float,
                'negative_prob': float,
                'neutral_prob': float,
                'sentiment_score': float (-1 to +1)
            }
        """
        if not text or pd.isna(text):
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'positive_prob': 0.33,
                'negative_prob': 0.33,
                'neutral_prob': 0.34,
                'sentiment_score': 0.0
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # İnference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
        
        # Softmax ile olasılıklara çevir
        probs = softmax(logits)
        
        # En yüksek olasılıklı sınıf
        predicted_class = np.argmax(probs)
        sentiment = LABELS[predicted_class]
        confidence = float(probs[predicted_class])
        
        # Sentiment skoru: -1 (çok negatif) ile +1 (çok pozitif) arası
        # positive_prob - negative_prob formülü
        sentiment_score = float(probs[2] - probs[0])  # positive - negative
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_prob': float(probs[2]),
            'negative_prob': float(probs[0]),
            'neutral_prob': float(probs[1]),
            'sentiment_score': sentiment_score
        }
    
    def analyze_batch(self, texts: list, batch_size: int = 16) -> list:
        """
        Batch halinde sentiment analizi yapar (daha hızlı).
        
        Args:
            texts: Metin listesi
            batch_size: Her batch'teki metin sayısı
        
        Returns:
            list: Her metin için sentiment sonucu
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analizi"):
            batch_texts = texts[i:i + batch_size]
            
            # Boş/NaN metinleri filtrele
            valid_texts = []
            valid_indices = []
            for j, text in enumerate(batch_texts):
                if text and not pd.isna(text):
                    valid_texts.append(str(text))
                    valid_indices.append(j)
            
            # Boş batch kontrolü
            if not valid_texts:
                for _ in batch_texts:
                    results.append({
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'positive_prob': 0.33,
                        'negative_prob': 0.33,
                        'neutral_prob': 0.34,
                        'sentiment_score': 0.0
                    })
                continue
            
            # Tokenize
            inputs = self.tokenizer(
                valid_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # İnference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()
            
            # Her metin için sonuç oluştur
            batch_results = [None] * len(batch_texts)
            
            for k, idx in enumerate(valid_indices):
                probs = softmax(logits[k])
                predicted_class = np.argmax(probs)
                
                batch_results[idx] = {
                    'sentiment': LABELS[predicted_class],
                    'confidence': float(probs[predicted_class]),
                    'positive_prob': float(probs[2]),
                    'negative_prob': float(probs[0]),
                    'neutral_prob': float(probs[1]),
                    'sentiment_score': float(probs[2] - probs[0])
                }
            
            # Boş olanlar için default
            for j in range(len(batch_results)):
                if batch_results[j] is None:
                    batch_results[j] = {
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'positive_prob': 0.33,
                        'negative_prob': 0.33,
                        'neutral_prob': 0.34,
                        'sentiment_score': 0.0
                    }
            
            results.extend(batch_results)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'title',
                          batch_size: int = 16) -> pd.DataFrame:
        """
        DataFrame'deki haberlere sentiment analizi uygular.
        
        Args:
            df: Haber verisi içeren DataFrame
            text_column: Analiz edilecek sütun adı
            batch_size: Batch boyutu
        
        Returns:
            DataFrame: Sentiment sütunları eklenmiş DataFrame
        """
        print(f"\n📊 {len(df)} haber için sentiment analizi başlıyor...")
        
        # Metinleri al
        texts = df[text_column].tolist()
        
        # Batch analiz
        results = self.analyze_batch(texts, batch_size)
        
        # Sonuçları DataFrame'e ekle
        df_result = df.copy()
        df_result['sentiment'] = [r['sentiment'] for r in results]
        df_result['sentiment_confidence'] = [r['confidence'] for r in results]
        df_result['positive_prob'] = [r['positive_prob'] for r in results]
        df_result['negative_prob'] = [r['negative_prob'] for r in results]
        df_result['neutral_prob'] = [r['neutral_prob'] for r in results]
        df_result['sentiment_score'] = [r['sentiment_score'] for r in results]
        
        # İstatistikler
        print("\n📈 Sentiment Dağılımı:")
        print(df_result['sentiment'].value_counts())
        print(f"\nOrtalama Sentiment Skoru: {df_result['sentiment_score'].mean():.3f}")
        
        return df_result


def analyze_news_file(input_file: str = NEWS_FILE, output_file: str = OUTPUT_FILE,
                      text_column: str = 'title') -> pd.DataFrame:
    """
    CSV dosyasındaki haberleri analiz eder ve sonucu kaydeder.
    """
    print(f"📂 Haber dosyası okunuyor: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ Dosya bulunamadı: {input_file}")
        return pd.DataFrame()
    
    # Veriyi yükle
    df = pd.read_csv(input_file)
    print(f"   {len(df)} haber bulundu.")
    
    # Analyzer oluştur
    analyzer = FinBERTSentimentAnalyzer()
    
    # Analiz et
    df_result = analyzer.analyze_dataframe(df, text_column=text_column)
    
    # Kaydet
    df_result.to_csv(output_file, index=False)
    print(f"\n💾 Sonuç kaydedildi: {output_file}")
    
    return df_result


def get_daily_sentiment(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Günlük ortalama sentiment hesaplar.
    Aynı gündeki tüm haberlerin sentiment'ini birleştirir.
    
    Args:
        df: Sentiment analizi yapılmış haber DataFrame'i
        date_column: Tarih sütunu adı
    
    Returns:
        DataFrame: Günlük sentiment özeti
    """
    print("\n📅 Günlük sentiment hesaplanıyor...")
    
    # Tarih formatını düzelt
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column]).dt.date
    
    # Günlük agregasyon
    daily = df_copy.groupby(date_column).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'positive_prob': 'mean',
        'negative_prob': 'mean',
        'neutral_prob': 'mean',
        'title': 'count'  # Haber sayısı
    }).reset_index()
    
    # Sütun adlarını düzelt
    daily.columns = [date_column, 'avg_sentiment', 'sentiment_std', 'sentiment_count',
                     'avg_positive', 'avg_negative', 'avg_neutral', 'news_count']
    
    # Sentiment kategorisi (günlük dominant sentiment)
    def categorize(score):
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    daily['daily_sentiment_category'] = daily['avg_sentiment'].apply(categorize)
    
    print(f"   {len(daily)} gün için sentiment hesaplandı.")
    
    return daily


if __name__ == "__main__":
    # Test çalıştırması
    print("=" * 60)
    print("FinBERT Sentiment Analizi Test")
    print("=" * 60)
    
    # Mevcut haber dosyasını analiz et
    df = analyze_news_file()
    
    if not df.empty:
        # Günlük sentiment hesapla
        daily_sentiment = get_daily_sentiment(df)
        
        # Günlük sentiment kaydet
        daily_output = os.path.join(DATA_DIR, "daily_sentiment.csv")
        daily_sentiment.to_csv(daily_output, index=False)
        print(f"💾 Günlük sentiment kaydedildi: {daily_output}")
        
        print("\n📊 Günlük Sentiment Örneği:")
        print(daily_sentiment.head(10))

