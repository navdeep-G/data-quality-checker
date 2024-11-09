import pandas as pd

class DatasetQualityChecker:
    def __init__(self, data):
        self.data = data

    def check_missing_values(self):
        """Check for missing values and return the percentage of missing data for each column."""
        missing = self.data.isnull().mean() * 100
        return missing[missing > 0]

    def check_duplicates(self):
        """Check for duplicate rows in the dataset."""
        return self.data.duplicated().sum()

    def check_outliers(self, threshold=3):
        """Detect outliers using the Z-score method."""
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        outliers = ((numeric_data - numeric_data.mean()).abs() > (threshold * numeric_data.std())).sum()
        return outliers[outliers > 0]

    def check_imbalance(self, column):
        """Check for class imbalance in a categorical column."""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        counts = self.data[column].value_counts(normalize=True) * 100
        return counts

    def generate_report(self):
        """Generate a summary report of data quality issues."""
        report = {
            "missing_values": self.check_missing_values(),
            "duplicates": self.check_duplicates(),
            "outliers": self.check_outliers(),
        }
        return report

if __name__ == "__main__":
    df = pd.read_csv("../data/sample_data.csv")
    checker = DatasetQualityChecker(df)
    report = checker.generate_report()
    print("Dataset Quality Report:\n", report)
