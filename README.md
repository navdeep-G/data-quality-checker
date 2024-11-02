# Dataset Quality Checker

A Python tool for checking common data quality issues in datasets, including missing values, duplicates, outliers, and class imbalance.

## Features
- Check for missing values and their percentages.
- Identify duplicate rows.
- Detect outliers based on Z-scores.
- Analyze class imbalance in categorical columns.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/username/dataset-quality-checker.git
cd dataset-quality-checker
pip install -r requirements.txt
```

## Usage
To use the dataset quality checker, import the `DatasetQualityChecker` class:

```python
from src.checker import DatasetQualityChecker
import pandas as pd

df = pd.read_csv("path/to/your/data.csv")
checker = DatasetQualityChecker(df)
report = checker.generate_report()
print(report)
```

## Contributing
Please see `CONTRIBUTING.md` for guidelines.
