#!/usr/bin/env python3
"""
Step 6: Analysis and Regression
Perform OLS regression analysis of Monday gap returns vs weekend sentiment.
Generate visualizations and statistical summaries.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = Path("data/sentiment_returns.parquet")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data():
    """Load merged sentiment and returns data"""
    logger.info(f"Loading data from {INPUT_FILE}...")

    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        logger.error("Please run returns.py first")
        return None

    df = pd.read_parquet(INPUT_FILE)
    logger.info(f"✓ Loaded {len(df)} weekend observations")

    return df


def descriptive_statistics(df):
    """Generate descriptive statistics"""
    logger.info("\n" + "="*60)
    logger.info("DESCRIPTIVE STATISTICS")
    logger.info("="*60)

    stats_df = df[['gap_return_pct', 'sentiment_composite',
                   'sentiment_positive_mean', 'sentiment_negative_mean',
                   'sentiment_neutral_mean', 'article_count']].describe()

    print(stats_df.to_string())

    # Save to CSV
    stats_output = OUTPUT_DIR / "descriptive_statistics.csv"
    stats_df.to_csv(stats_output)
    logger.info(f"\n✓ Saved descriptive statistics to {stats_output}")


def correlation_analysis(df):
    """Perform correlation analysis"""
    logger.info("\n" + "="*60)
    logger.info("CORRELATION ANALYSIS")
    logger.info("="*60)

    # Select relevant columns
    corr_cols = ['gap_return_pct', 'sentiment_composite',
                 'sentiment_positive_mean', 'sentiment_negative_mean',
                 'sentiment_neutral_mean', 'article_count']

    corr_matrix = df[corr_cols].corr()

    # Print correlations with returns
    logger.info("\nCorrelations with Gap Return (%):")
    returns_corr = corr_matrix['gap_return_pct'].sort_values(ascending=False)
    for var, corr in returns_corr.items():
        if var != 'gap_return_pct':
            logger.info(f"  {var:30s}: {corr:7.4f}")

    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, fmt='.3f')
    plt.title('Correlation Matrix: Sentiment and Returns', fontsize=16, fontweight='bold')
    plt.tight_layout()

    heatmap_output = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved correlation heatmap to {heatmap_output}")
    plt.close()


def regression_analysis(df):
    """Perform OLS regression analysis"""
    logger.info("\n" + "="*60)
    logger.info("REGRESSION ANALYSIS")
    logger.info("="*60)

    results = {}

    # Model 1: Simple regression - Composite sentiment
    logger.info("\nModel 1: Gap Return ~ Composite Sentiment")
    X1 = sm.add_constant(df['sentiment_composite'])
    y = df['gap_return_pct']
    model1 = sm.OLS(y, X1).fit()
    results['model1'] = model1
    print(model1.summary())

    # Model 2: Positive and Negative sentiment separately
    logger.info("\n" + "-"*60)
    logger.info("Model 2: Gap Return ~ Positive + Negative Sentiment")
    X2 = sm.add_constant(df[['sentiment_positive_mean', 'sentiment_negative_mean']])
    model2 = sm.OLS(y, X2).fit()
    results['model2'] = model2
    print(model2.summary())

    # Model 3: With article count control
    logger.info("\n" + "-"*60)
    logger.info("Model 3: Gap Return ~ Composite Sentiment + Article Count")
    X3 = sm.add_constant(df[['sentiment_composite', 'article_count']])
    model3 = sm.OLS(y, X3).fit()
    results['model3'] = model3
    print(model3.summary())

    # Model 4: Full model with all sentiment components
    logger.info("\n" + "-"*60)
    logger.info("Model 4: Gap Return ~ Positive + Negative + Neutral + Article Count")
    X4 = sm.add_constant(df[['sentiment_positive_mean', 'sentiment_negative_mean',
                              'sentiment_neutral_mean', 'article_count']])
    model4 = sm.OLS(y, X4).fit()
    results['model4'] = model4
    print(model4.summary())

    # Test for heteroskedasticity (Breusch-Pagan test)
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC TESTS")
    logger.info("="*60)
    for i, (name, model) in enumerate(results.items(), 1):
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        logger.info(f"\nModel {i} - Breusch-Pagan test for heteroskedasticity:")
        logger.info(f"  LM Statistic: {bp_test[0]:.4f}")
        logger.info(f"  p-value: {bp_test[1]:.4f}")
        if bp_test[1] < 0.05:
            logger.info("  ⚠ Evidence of heteroskedasticity (p < 0.05)")
        else:
            logger.info("  ✓ No evidence of heteroskedasticity (p >= 0.05)")

    # Save regression results
    results_output = OUTPUT_DIR / "regression_results.txt"
    with open(results_output, 'w') as f:
        for i, (name, model) in enumerate(results.items(), 1):
            f.write(f"{'='*60}\n")
            f.write(f"MODEL {i}\n")
            f.write(f"{'='*60}\n\n")
            f.write(model.summary().as_text())
            f.write("\n\n")

    logger.info(f"\n✓ Saved regression results to {results_output}")

    return results


def create_visualizations(df, results):
    """Create visualization plots"""
    logger.info("\n" + "="*60)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*60)

    # 1. Scatter plot: Sentiment vs Returns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Composite Sentiment
    ax1 = axes[0, 0]
    ax1.scatter(df['sentiment_composite'], df['gap_return_pct'], alpha=0.6, s=50)

    # Add regression line
    X = df['sentiment_composite']
    y = df['gap_return_pct']
    z = np.polyfit(X, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(X.min(), X.max(), 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')

    ax1.set_xlabel('Composite Sentiment (Positive - Negative)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Monday Gap Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Sentiment vs Returns\nCorrelation: {df["sentiment_composite"].corr(df["gap_return_pct"]):.4f}',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Positive Sentiment
    ax2 = axes[0, 1]
    ax2.scatter(df['sentiment_positive_mean'], df['gap_return_pct'], alpha=0.6, s=50, color='green')
    X2 = df['sentiment_positive_mean']
    z2 = np.polyfit(X2, y, 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(X2.min(), X2.max(), 100)
    ax2.plot(x_line2, p2(x_line2), "r--", linewidth=2)
    ax2.set_xlabel('Positive Sentiment', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Monday Gap Return (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Positive Sentiment vs Returns\nCorrelation: {df["sentiment_positive_mean"].corr(df["gap_return_pct"]):.4f}',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Negative Sentiment
    ax3 = axes[1, 0]
    ax3.scatter(df['sentiment_negative_mean'], df['gap_return_pct'], alpha=0.6, s=50, color='red')
    X3 = df['sentiment_negative_mean']
    z3 = np.polyfit(X3, y, 1)
    p3 = np.poly1d(z3)
    x_line3 = np.linspace(X3.min(), X3.max(), 100)
    ax3.plot(x_line3, p3(x_line3), "r--", linewidth=2)
    ax3.set_xlabel('Negative Sentiment', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Monday Gap Return (%)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Negative Sentiment vs Returns\nCorrelation: {df["sentiment_negative_mean"].corr(df["gap_return_pct"]):.4f}',
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Article Count
    ax4 = axes[1, 1]
    ax4.scatter(df['article_count'], df['gap_return_pct'], alpha=0.6, s=50, color='purple')
    X4 = df['article_count']
    z4 = np.polyfit(X4, y, 1)
    p4 = np.poly1d(z4)
    x_line4 = np.linspace(X4.min(), X4.max(), 100)
    ax4.plot(x_line4, p4(x_line4), "r--", linewidth=2)
    ax4.set_xlabel('Article Count', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Monday Gap Return (%)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Article Count vs Returns\nCorrelation: {df["article_count"].corr(df["gap_return_pct"]):.4f}',
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_output = OUTPUT_DIR / "sentiment_returns_scatter.png"
    plt.savefig(scatter_output, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved scatter plots to {scatter_output}")
    plt.close()

    # 2. Time series plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Plot returns over time
    ax1.plot(df['weekend_date'], df['gap_return_pct'], marker='o', linewidth=1, markersize=3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Monday Gap Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('S&P 500 Monday Gap Returns Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot sentiment over time
    ax2.plot(df['weekend_date'], df['sentiment_composite'], marker='o', linewidth=1, markersize=3, color='orange')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Weekend Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Composite Sentiment', fontsize=12, fontweight='bold')
    ax2.set_title('Weekend News Sentiment Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    timeseries_output = OUTPUT_DIR / "time_series.png"
    plt.savefig(timeseries_output, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved time series plot to {timeseries_output}")
    plt.close()

    # 3. Residual plots for Model 1
    model1 = results['model1']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(model1.fittedvalues, model1.resid, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Fitted Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax1.set_title('Residuals vs Fitted', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    ax2 = axes[0, 1]
    stats.probplot(model1.resid, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Histogram of residuals
    ax3 = axes[1, 0]
    ax3.hist(model1.resid, bins=20, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Scale-Location plot
    ax4 = axes[1, 1]
    standardized_resid = np.sqrt(np.abs(model1.resid / np.std(model1.resid)))
    ax4.scatter(model1.fittedvalues, standardized_resid, alpha=0.6)
    ax4.set_xlabel('Fitted Values', fontsize=12, fontweight='bold')
    ax4.set_ylabel('√|Standardized Residuals|', fontsize=12, fontweight='bold')
    ax4.set_title('Scale-Location Plot', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    residual_output = OUTPUT_DIR / "residual_diagnostics.png"
    plt.savefig(residual_output, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved residual diagnostics to {residual_output}")
    plt.close()


def generate_summary_report(df, results):
    """Generate a comprehensive summary report"""
    logger.info("\n" + "="*60)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("="*60)

    report_output = OUTPUT_DIR / "analysis_summary.txt"

    with open(report_output, 'w') as f:
        f.write("="*60 + "\n")
        f.write("WEEKEND NEWS SENTIMENT AND MONDAY RETURNS ANALYSIS\n")
        f.write("S&P 500 - 2021-2024\n")
        f.write("="*60 + "\n\n")

        f.write("DATA SUMMARY\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Observations: {len(df)}\n")
        f.write(f"Date Range: {df['weekend_date'].min()} to {df['weekend_date'].max()}\n")
        f.write(f"Total Weekend Articles Analyzed: {df['article_count'].sum():.0f}\n")
        f.write(f"Average Articles per Weekend: {df['article_count'].mean():.1f}\n\n")

        f.write("RETURNS SUMMARY\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean Monday Gap Return: {df['gap_return_pct'].mean():.3f}%\n")
        f.write(f"Median Monday Gap Return: {df['gap_return_pct'].median():.3f}%\n")
        f.write(f"Std Dev: {df['gap_return_pct'].std():.3f}%\n")
        f.write(f"Min: {df['gap_return_pct'].min():.3f}%\n")
        f.write(f"Max: {df['gap_return_pct'].max():.3f}%\n")
        f.write(f"Positive Returns: {(df['gap_return_pct'] > 0).sum()} ({(df['gap_return_pct'] > 0).mean()*100:.1f}%)\n")
        f.write(f"Negative Returns: {(df['gap_return_pct'] < 0).sum()} ({(df['gap_return_pct'] < 0).mean()*100:.1f}%)\n\n")

        f.write("SENTIMENT SUMMARY\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean Composite Sentiment: {df['sentiment_composite'].mean():.3f}\n")
        f.write(f"Mean Positive Score: {df['sentiment_positive_mean'].mean():.3f}\n")
        f.write(f"Mean Negative Score: {df['sentiment_negative_mean'].mean():.3f}\n")
        f.write(f"Mean Neutral Score: {df['sentiment_neutral_mean'].mean():.3f}\n\n")

        f.write("CORRELATION ANALYSIS\n")
        f.write("-"*60 + "\n")
        f.write(f"Composite Sentiment - Returns: {df['sentiment_composite'].corr(df['gap_return_pct']):.4f}\n")
        f.write(f"Positive Sentiment - Returns: {df['sentiment_positive_mean'].corr(df['gap_return_pct']):.4f}\n")
        f.write(f"Negative Sentiment - Returns: {df['sentiment_negative_mean'].corr(df['gap_return_pct']):.4f}\n")
        f.write(f"Article Count - Returns: {df['article_count'].corr(df['gap_return_pct']):.4f}\n\n")

        f.write("REGRESSION RESULTS SUMMARY\n")
        f.write("-"*60 + "\n")

        model1 = results['model1']
        f.write("\nModel 1: Gap Return ~ Composite Sentiment\n")
        f.write(f"  R-squared: {model1.rsquared:.4f}\n")
        f.write(f"  Sentiment Coefficient: {model1.params['sentiment_composite']:.4f}\n")
        f.write(f"  Sentiment p-value: {model1.pvalues['sentiment_composite']:.4f}\n")
        f.write(f"  Significant at 5%? {'Yes' if model1.pvalues['sentiment_composite'] < 0.05 else 'No'}\n")

        model2 = results['model2']
        f.write("\nModel 2: Gap Return ~ Positive + Negative Sentiment\n")
        f.write(f"  R-squared: {model2.rsquared:.4f}\n")
        f.write(f"  Positive Coefficient: {model2.params['sentiment_positive_mean']:.4f} (p={model2.pvalues['sentiment_positive_mean']:.4f})\n")
        f.write(f"  Negative Coefficient: {model2.params['sentiment_negative_mean']:.4f} (p={model2.pvalues['sentiment_negative_mean']:.4f})\n")

        f.write("\n" + "="*60 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*60 + "\n\n")

        # Interpretation based on results
        if model1.pvalues['sentiment_composite'] < 0.05:
            f.write("✓ Weekend news sentiment has a statistically significant relationship with Monday gap returns.\n")
        else:
            f.write("✗ Weekend news sentiment does NOT have a statistically significant relationship with Monday gap returns (p >= 0.05).\n")

        correlation = df['sentiment_composite'].corr(df['gap_return_pct'])
        if abs(correlation) < 0.1:
            strength = "very weak"
        elif abs(correlation) < 0.3:
            strength = "weak"
        elif abs(correlation) < 0.5:
            strength = "moderate"
        else:
            strength = "strong"

        direction = "positive" if correlation > 0 else "negative"
        f.write(f"\nThe correlation between sentiment and returns is {strength} and {direction} ({correlation:.4f}).\n")

        if model1.rsquared < 0.05:
            f.write(f"\nThe low R-squared ({model1.rsquared:.4f}) indicates that weekend sentiment explains less than 5% of the variation in Monday gap returns.\n")
            f.write("This suggests that other factors (e.g., global events, monetary policy, earnings announcements) may be more important drivers of Monday returns.\n")

    logger.info(f"✓ Saved summary report to {report_output}")

    # Print key findings to console
    logger.info("\n" + "="*60)
    logger.info("KEY FINDINGS")
    logger.info("="*60)

    model1 = results['model1']
    sentiment_coef = model1.params['sentiment_composite']
    sentiment_pval = model1.pvalues['sentiment_composite']

    logger.info(f"\n1. Correlation: {df['sentiment_composite'].corr(df['gap_return_pct']):.4f}")
    logger.info(f"2. Regression Coefficient: {sentiment_coef:.4f}")
    logger.info(f"3. P-value: {sentiment_pval:.4f}")
    logger.info(f"4. R-squared: {model1.rsquared:.4f}")
    logger.info(f"5. Statistical Significance: {'Yes (p < 0.05)' if sentiment_pval < 0.05 else 'No (p >= 0.05)'}")

    logger.info(f"\nInterpretation:")
    logger.info(f"A 1-unit increase in composite sentiment is associated with a {sentiment_coef:.4f}% change in Monday gap returns.")

    if sentiment_pval < 0.05:
        logger.info(f"This relationship is statistically significant at the 5% level.")
    else:
        logger.info(f"This relationship is NOT statistically significant at the 5% level.")
        logger.info(f"We cannot conclude that weekend news sentiment predicts Monday returns in this dataset.")


def main():
    """Main analysis pipeline"""
    logger.info("="*60)
    logger.info("STEP 6: ANALYSIS AND REGRESSION")
    logger.info("="*60)

    # Load data
    df = load_data()
    if df is None:
        return

    # Descriptive statistics
    descriptive_statistics(df)

    # Correlation analysis
    correlation_analysis(df)

    # Regression analysis
    results = regression_analysis(df)

    # Create visualizations
    create_visualizations(df, results)

    # Generate summary report
    generate_summary_report(df, results)

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"All results saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
