# Dataset Quality Checker

A Python package for evaluating and maintaining dataset quality by identifying common data issues such as missing values, duplicates, outliers, and class imbalances. It also supports advanced assessments like schema validation, semantic consistency checks, hypothesis testing, time-series analysis, and data drift detection—making it a comprehensive tool for ensuring data reliability.

---

## 🔧 Features

### 🔍 Basic Checks
- **Missing Values**: Detect columns with missing data and calculate missing percentages.
- **Duplicate Rows**: Identify and return duplicate rows.
- **Outliers**: Detect numeric outliers using Z-score, IQR, or Isolation Forest.
- **Class Imbalance**: Analyze class distribution in categorical columns.
- **Empty Columns**: Identify columns with no non-null values.

### 🚀 Advanced Checks
- **Schema Validation**: Validate against a predefined schema with type, format, and range constraints.
- **Data Type Consistency**: Identify columns with inconsistent or mixed data types.
- **Correlation Analysis**: Flag highly correlated numeric features.
- **Unique Values**: Detect columns with only a single unique value.
- **Rare Categories**: Identify infrequent categories based on a frequency threshold.
- **Multicollinearity**: Detect using Variance Inflation Factor (VIF).
- **Unexpected Values**: Flag invalid entries in categorical columns.
- **Naming Convention Check**: Verify adherence to column naming patterns.
- **Cross-Column Rules**: Validate inter-column dependencies and logical rules.
- **Conditional Probability Deviations**: Identify deviations in expected conditional distributions.
- **Bootstrap Sampling**: Estimate confidence intervals for numeric columns using bootstrapping.

### 📊 Statistical Testing
- **Hypothesis Testing**: Perform t-tests, chi-squared tests, or ANOVA.
- **Uniformity Test**: Check if data follows a uniform distribution.
- **Confidence Intervals**: Compute for numeric columns.
- **Outlier Impact**: Assess how outliers affect key statistics.

### 📈 Time-Series Analysis
- **Time Gaps**: Identify missing or unordered timestamps.
- **Trend Detection**: Spot anomalies or trends in numeric time-series data.
- **Seasonality**: Detect recurring seasonal patterns.
- **Change Point Detection**: Identify structural breaks or shifts.
- **Stationarity Test**: Use ADF test to check for stationarity.
- **Forecast Metrics**: Evaluate time-series model accuracy.

### 📝 Text and NLP Analysis
- **Text Length Check**: Detect excessively long text entries.
- **Text Similarity**: Find highly similar text pairs.
- **Sentiment Analysis**: Assign sentiment scores.
- **NER**: Extract entities like names, dates, organizations.
- **Keyword Extraction**: Use RAKE for identifying key terms.
- **Spelling Correction**: Auto-correct spelling in text columns.
- **Topic Modeling**: Extract dominant topics via LDA.
- **Tokenization**: Break text into words or sentences.
- **Language Detection**: Detect text language.
- **Stopword Analysis**: Count common stopwords.
- **N-gram Analysis**: Identify frequent n-gram patterns.

### 📊 Data Drift and Anomaly Detection
- **Data Drift**: Compare distributions to a baseline dataset.
- **Anomalies**: Use Isolation Forest to identify numeric anomalies.
- **Rare Events**: Detect unusual numeric events using statistical thresholds.

### 🌐 Semantic & Cross-Column Validation
- **Semantic Checks**: Validate data against expected semantic mappings.
- **Dependency Validation**: Enforce and validate inter-column logic.
- **Conditional Probabilities**: Detect inconsistencies in dependent probabilities.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/navdeep-G/data-quality-checker.git
cd dataset-quality-checker
pip install -r requirements.txt
```

## Usage
To use the dataset quality checker, import the DatasetQualityChecker class:
```
from src.checker import DatasetQualityChecker
import pandas as pd
df = pd.read_csv("path/to/your/data.csv")
checker = DatasetQualityChecker(df)
report = checker.generate_report()
print(report)
```
## Contributing
Please see [CONTRIBUTING.md](https://github.com/navdeep-G/data-quality-checker/blob/main/CONTRIBUTING.md) for guidelines.
