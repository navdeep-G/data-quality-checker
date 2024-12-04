# Dataset Quality Checker

A Python tool for assessing and ensuring data quality, including detecting missing values, duplicates, outliers, class imbalances, and more. This tool is designed to handle common data quality issues and support advanced quality checks such as schema validation, semantic consistency, and data drift detection.

---

## Features

### Basic Checks
- **Missing Values**: Identify columns with missing data and calculate the percentage of missing values.
- **Duplicate Rows**: Detect and return duplicate rows in the dataset.
- **Outliers**: Detect numeric outliers using Z-scores.
- **Class Imbalance**: Analyze class distribution for categorical columns.

### Advanced Checks
- **Schema Validation**: Validate data against a predefined schema for type, format, and range constraints.
- **Data Type Consistency**: Identify columns with mixed or inconsistent data types.
- **Correlation Analysis**: Flag highly correlated numeric features to avoid redundancy.
- **Unique Values**: Identify columns with only a single unique value.
- **Rare Categories**: Detect rare categories in categorical columns based on a frequency threshold.
- **Multicollinearity**: Use Variance Inflation Factor (VIF) to detect multicollinearity.
- **Unexpected Values**: Identify unexpected or invalid values in a categorical column.
- **Column Naming Convention**: Check if column names follow a specific naming pattern.

### Time-Series and Text Analysis
- **Time-Series Gaps**: Detect missing or unordered timestamps in time-series data.
- **Temporal Trends**: Analyze anomalies or deviations in numeric data over time.
- **Seasonality Detection**: Identify seasonal patterns in time-series data.
- **Text Length**: Check for excessively long text entries in a column.
- **Text Similarity**: Identify pairs of text entries with high similarity using a similarity threshold.

### Data Drift and Anomaly Detection
- **Data Drift**: Compare the distribution of a column to a baseline dataset to detect drift.
- **Anomalies**: Use Isolation Forest to detect anomalies in numeric data.
- **Rare Events**: Identify rare numeric events using statistical thresholds.

### Semantic and Cross-Column Validation
- **Semantic Consistency**: Validate values in a column against a predefined semantic mapping.
- **Cross-Column Dependency**: Verify that cross-column rules and dependencies are satisfied.
- **Conditional Probabilities**: Check if conditional probabilities deviate from expected distributions.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/username/dataset-quality-checker.git
cd dataset-quality-checker
pip install -r requirements.txt
```

## Usage
To use the dataset quality checker, import the DatasetQualityChecker class:

python
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
