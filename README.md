# Weekend News Sentiment and Market Returns: SP500 vs BIST100 Comparative Analysis

A quantitative comparative analysis investigating the relationship between weekend financial news sentiment and Monday stock market returns across developed (S&P 500) and emerging (BIST 100) markets using FinBERT sentiment analysis and OLS regression.

## 📊 Project Overview

This study examines whether news sentiment during market closure periods (weekends) has predictive power for subsequent market openings in both **developed** and **emerging markets**. We analyze both S&P 500 (USA) and BIST 100 (Turkey) index data from 2021-2024, processing thousands of financial news articles to test if weekend sentiment correlates with Monday gap returns.

**Research Question:** Does weekend news sentiment predict Monday gap returns differently in developed vs. emerging markets?

**Hypothesis:** Financial news sentiment has a different magnitude of impact in emerging markets (BIST 100) compared to developed markets (S&P 500).

## 🗂️ Data Sources

### S&P 500 Analysis

- **Financial News**: Multi-source dataset from HuggingFace ([`Brianferrell787/financial-news-multisource`](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource))
  - S&P 500 daily headlines
  - CNBC financial news
  - Yahoo Finance articles
  
- **Market Data**: S&P 500 index (^GSPC) via Yahoo Finance API

### BIST 100 Analysis

- **Financial News**: Collected via separate web scraping pipeline (repository available separately)
- **Market Data**: BIST 100 index data

> **Note:** The BIST 100 news collection code is maintained in a separate repository. This repository focuses on the sentiment analysis and regression methodology that can be applied to both markets.

## 🔧 Methodology

### Data Processing Pipeline

```
News Data → Weekend Filter → FinBERT Sentiment → Returns Calculation → OLS Regression
```

**Weekend Window Definition:** Friday 21:00 UTC - Sunday 23:59 UTC

### Key Components

This repository contains the analysis pipeline that can be applied to both markets:

1. **News Loading** (`news_loader.py`): Download and preprocess financial news (configured for S&P 500 from HuggingFace)
2. **Weekend Filtering** (`weekend_filter.py`): Extract articles within weekend window
3. **Sentiment Analysis** (`sentiment.py`): Apply FinBERT model for sentiment scoring
4. **Returns Calculation** (`returns.py`): Calculate Monday gap returns
5. **Statistical Analysis** (`analysis.py`): OLS regression and visualization

### Sentiment Model

**FinBERT** ([`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)) - A BERT model fine-tuned on financial texts that classifies sentiment as positive, negative, or neutral.

**Composite Sentiment Score** = `mean(positive) - mean(negative)`

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-compatible GPU for faster sentiment analysis

### Setup

```bash
# Clone the repository
git clone https://github.com/dselimozcelik/SP500-BIST100-Sentiment-Comparison.git
cd SP500-BIST100-Sentiment-Comparison

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for S&P 500 dataset access)
python hf_login.py
```

See [`DATASET_ACCESS_INSTRUCTIONS.md`](DATASET_ACCESS_INSTRUCTIONS.md) for detailed dataset access setup.

## 📈 Usage

### Running the Analysis Pipeline

The pipeline can be run for either market by configuring the appropriate data sources:

```bash
# Step 1: Load financial news
python news_loader.py

# Step 2: Filter to weekend articles only
python weekend_filter.py

# Step 3: Perform sentiment analysis with FinBERT
python sentiment.py

# Step 4: Calculate Monday gap returns
python returns.py

# Step 5: Run OLS regression and generate visualizations
python analysis.py
```

### Exploring Results

```bash
# View processed data
python view_data.py

# Check dataset structure
python explore_dataset.py
```

## 📊 Results

All analysis outputs are saved to the `results/` directory:

| File | Description |
|------|-------------|
| `correlation_heatmap.png` | Correlation matrix between sentiment and returns |
| `sentiment_returns_scatter.png` | Scatter plots showing relationships |
| `time_series.png` | Time series of sentiment and returns |
| `residual_diagnostics.png` | Regression diagnostic plots |
| `analysis_summary.txt` | Statistical summary and key findings |
| `regression_results.txt` | Detailed OLS regression output |
| `descriptive_statistics.csv` | Dataset summary statistics |

### Comparative Analysis

The methodology allows for direct comparison between:
- **S&P 500** (Developed Market): High liquidity, mature investor base
- **BIST 100** (Emerging Market): Higher volatility, developing market characteristics

**Expected Findings:**
- Stronger sentiment-return relationship in emerging markets due to less efficient information processing
- Higher volatility in BIST 100 gap returns
- Different sentiment profiles between markets

## 📁 Project Structure

```
SP500-BIST100-Sentiment-Comparison/
├── news_loader.py           # Step 1: Load news (configured for S&P 500)
├── weekend_filter.py        # Step 2: Filter weekend articles
├── sentiment.py             # Step 3: FinBERT sentiment analysis
├── returns.py               # Step 4: Calculate gap returns
├── analysis.py              # Step 5: Regression analysis
├── view_data.py             # Data exploration utility
├── requirements.txt         # Python dependencies
├── data/                    # Processed data files (ignored in git)
├── results/                 # Analysis outputs (ignored in git)
└── README.md                # This file
```

> **Note:** BIST 100 news collection scripts are maintained in a separate repository due to different data collection requirements.

# 🇹🇷 BIST 100 Implementation Details 

This repository now includes the specific implementation for the **BIST 100** leg of the analysis, which uses a custom web scraping pipeline and streamlined execution.

## 📁 BIST 100 Project Structure

The BIST 100 specific files are located in the `src/` directory and root:

```
├── news_scraper.py                  # Custom BIST 100 News Scraper
├── main.py                      # Main pipeline orchestrator for BIST 100
├── price_fetcher.py             # BIST data fetcher via yfinance
├── sentiment_analyzer.py        # FinBERT model wrapper
├── sentiment_price_analysis.py  # Statistical & ML analysis (Core Logic)
├── comprehensive_analysis.py    # Detailed reporting script
├── create_final_figures.py      # Publication-ready figure generation
├── positive_findings_analysis.py# Focused positive findings analysis
└── data_quality_check.py        # Data validation utility
```

##  Running the BIST 100 Analysis

Unlike the S&P 500 pipeline which runs in steps, the BIST 100 analysis is orchestrated by a single script:

### 1. Full Pipeline (Scrape + Prcocess + Analyze)
```bash
# Run the entire pipeline
python src/main.py

# Force new data collection (Scraping)
python src/main.py --fetch-news
```

### 2. Analytical Reports
To generate the specific BIST 100 reports and figures:

```bash
# Generate comprehensive statistical report
python src/comprehensive_analysis.py

# Create publication-ready figures (saved to 'makale/')
python src/create_final_figures.py
```

##  BIST 100 Specific Findings

*   **Same-Day Effect:** We observed a statistically significant positive correlation (r = 0.078) between news sentiment and same-day BIST 100 returns.
*   **Reversal Effect:** A significant negative correlation (r = -0.081) on the following day, suggesting market overreaction and correction.
*   **Significance:** These findings are statistically significant (p < 0.05) and consistent with behavioral finance theories in emerging markets.


## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.x |
| Data Processing | pandas, numpy |
| Sentiment Analysis | transformers, torch (FinBERT) |
| Market Data | yfinance |
| Statistical Analysis | statsmodels, scipy |
| Visualization | matplotlib, seaborn |
| Data Storage | Apache Parquet |

## 📝 Requirements

```
pandas
numpy
datasets
transformers
torch
yfinance
matplotlib
seaborn
scipy
statsmodels
```

See [`requirements.txt`](requirements.txt) for exact versions.

## 🔍 Future Work

- Extend analysis to additional emerging markets
- Alternative sentiment models (FinBERT-tone, SentimentLM)
- Intraday sentiment decay analysis
- Event studies for high-impact weekends
- Non-linear machine learning approaches (Random Forest, XGBoost)
- Cross-market sentiment spillover effects

## 📄 License

This project is for academic research purposes.

## 🙏 Acknowledgments

- **FinBERT**: ProsusAI/finbert sentiment model
- **S&P 500 Dataset**: Brianferrell787/financial-news-multisource (HuggingFace)
- **Market Data**: Yahoo Finance

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: The `data/` and `results/` directories are excluded from version control due to size. Run the pipeline to regenerate these files locally.
