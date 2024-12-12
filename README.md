# Dataset Quality Checker

A Python package designed to evaluate and maintain data quality by identifying common data issues like missing values, duplicates, outliers, and class imbalances. It also includes advanced assessments such as schema validation, semantic consistency checks, and data drift detection, making it a versatile tool for ensuring the reliability of your datasets.

---

## Features

### 🔍 **Basic Checks**
- **Missing Values**: Identify columns with missing data and calculate the percentage of missing values.
- **Duplicate Rows**: Detect and return duplicate rows in the dataset.
- **Outliers**: Detect numeric outliers using Z-scores.
- **Class Imbalance**: Analyze class distribution for categorical columns.

### 🚀 **Advanced Checks**
- **Schema Validation**: Ensure the dataset conforms to a predefined schema with type, format, and range constraints.
- **Data Type Consistency**: Identify columns with mixed or inconsistent data types.
- **Correlation Analysis**: Flag highly correlated numeric features to reduce redundancy.
- **Unique Values**: Detect columns with only a single unique value.
- **Rare Categories**: Identify rare categories in categorical columns using a frequency threshold.
- **Multicollinearity**: Detect multicollinearity using the Variance Inflation Factor (VIF).
- **Unexpected Values**: Flag invalid or unexpected values in categorical columns.
- **Column Naming Convention**: Verify column names adhere to a specified naming pattern.

### 📈 **Time-Series and Text Analysis**
- **Time-Series Gaps**: Identify missing or unordered timestamps in time-series data.
- **Temporal Trends**: Detect anomalies or deviations in numeric data over time.
- **Seasonality Detection**: Identify seasonal patterns in time-series data.
- **Text Length**: Flag excessively long text entries in a column.
- **Text Similarity**: Identify pairs of highly similar text entries based on a similarity threshold.

### 📊 **Data Drift and Anomaly Detection**
- **Data Drift**: Compare column distributions to a baseline dataset for drift detection.
- **Anomalies**: Use Isolation Forest to identify anomalies in numeric data.
- **Rare Events**: Detect rare numeric events using statistical thresholds.

### 🌐 **Semantic and Cross-Column Validation**
- **Semantic Consistency**: Validate column values against a predefined semantic mapping.
- **Cross-Column Dependency**: Ensure that cross-column rules and dependencies are satisfied.
- **Conditional Probabilities**: Detect deviations in expected conditional probabilities.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/username/dataset-quality-checker.git
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
