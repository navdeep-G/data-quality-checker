import pandas as pd
from scipy.stats import ks_2samp

class DatasetQualityChecker:
    def __init__(self, data):
        self.data = data

    def check_missing_values(self):
        """Check for missing values and return the percentage of missing data for each column."""
        missing = self.data.isnull().mean() * 100
        return missing[missing > 0]

    def check_duplicates(data, subset=None):
        duplicates = data.duplicated(subset=subset)
        if duplicates.any():
            print(f"Found {duplicates.sum()} duplicates.")
            return data[duplicates]
        else:
            print("No duplicates found.")

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
    
    def validate_schema(data, schema_file='schema_config.json'):
        """
        Generic schema validation function for any dataset and schema.

        Args:
            data (pd.DataFrame): DataFrame to validate.
            schema_file (str): Path to the schema configuration file.

        Returns:
            bool: True if validation passes, otherwise False.
        """
        # Load schema
        with open(schema_file, 'r') as file:
            schema = json.load(file)

        validation_passed = True

        # Iterate over schema columns to validate
        for column, properties in schema.get("columns", {}).items():
            # Check if required column exists
            if properties.get("required") and column not in data.columns:
                print(f"Missing required column: '{column}'")
                validation_passed = False
                continue

            # If column is optional and missing, skip further checks
            if column not in data.columns:
                continue

            # Check data type
            expected_type = properties.get("type")
            if expected_type:
                if expected_type == "int" and not pd.api.types.is_integer_dtype(data[column]):
                    print(f"Column '{column}' should be integer.")
                    validation_passed = False
                elif expected_type == "float" and not pd.api.types.is_float_dtype(data[column]):
                    print(f"Column '{column}' should be float.")
                    validation_passed = False
                elif expected_type == "string" and not pd.api.types.is_string_dtype(data[column]):
                    print(f"Column '{column}' should be string.")
                    validation_passed = False
                elif expected_type == "datetime":
                    try:
                        pd.to_datetime(data[column], format=properties.get("format"))
                    except ValueError:
                        print(f"Column '{column}' has invalid datetime format.")
                        validation_passed = False

            # Check min/max constraints
            if "min" in properties and (data[column] < properties["min"]).any():
                print(f"Column '{column}' contains values below the minimum of {properties['min']}.")
                validation_passed = False
            if "max" in properties and (data[column] > properties["max"]).any():
                print(f"Column '{column}' contains values above the maximum of {properties['max']}.")
                validation_passed = False

        return validation_passed

    def check_data_type_consistency(self):
        """Check for inconsistent data types within each column."""
        inconsistent_columns = {}
        for column in self.data.columns:
            unique_types = self.data[column].map(type).nunique()
            if unique_types > 1:
                inconsistent_columns[column] = unique_types
        return inconsistent_columns

    def check_correlation(self, threshold=0.9):
        """
        Check for highly correlated numeric features.

        Args:
            threshold (float): Correlation value above which features are flagged.

        Returns:
            list of tuples: Pairs of correlated features.
        """
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_data.corr()
        correlated_features = [
            (col1, col2) for col1 in correlation_matrix.columns for col2 in correlation_matrix.columns
            if col1 != col2 and abs(correlation_matrix[col1][col2]) > threshold
        ]
        return correlated_features

    def check_unique_values(self):
        """Identify columns with only one unique value."""
        single_value_columns = [col for col in self.data.columns if self.data[col].nunique() == 1]
        return single_value_columns

    def check_text_length(self, column, max_length=255):
        """
        Check text data for excessive length.

        Args:
            column (str): Name of the text column to check.
            max_length (int): Maximum allowed length for text.

        Returns:
            pd.Series: Rows with text length exceeding the maximum.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' is not of string type.")

        return self.data[self.data[column].str.len() > max_length]

    def check_unexpected_values(self, column, expected_values):
        """
        Identify unexpected values in a categorical column.

        Args:
            column (str): Name of the categorical column to check.
            expected_values (list): List of expected values.

        Returns:
            pd.Series: Rows with unexpected values.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        unexpected = ~self.data[column].isin(expected_values)
        return self.data[unexpected]

    def check_time_series_gaps(self, timestamp_column):
        """
        Check for missing or unordered timestamps in time-series data.

        Args:
            timestamp_column (str): Name of the timestamp column.

        Returns:
            dict: Information on missing or unordered timestamps.
        """
        if timestamp_column not in self.data.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found.")
        self.data[timestamp_column] = pd.to_datetime(self.data[timestamp_column])
        gaps = self.data[timestamp_column].diff().dt.total_seconds().dropna()
        unordered = (gaps < 0).sum()
        return {
            "missing_gaps": (gaps == 0).sum(),
            "unordered_timestamps": unordered
        }

    import re

    def check_pattern_consistency(self, column, regex):
        """
        Check if values in a column match a specific pattern.

        Args:
            column (str): Name of the column to check.
            regex (str): Regular expression for the expected pattern.

        Returns:
            pd.DataFrame: Rows where the pattern does not match.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        pattern = re.compile(regex)
        invalid_rows = self.data[~self.data[column].astype(str).apply(lambda x: bool(pattern.match(x)))]
        return invalid_rows

    def check_data_drift(self, baseline_data, column):
        """
        Check if the distribution of a column has shifted compared to baseline data.

        Args:
            baseline_data (pd.DataFrame): Baseline dataset for comparison.
            column (str): Name of the column to check for drift.

        Returns:
            float: p-value indicating if the distributions are statistically different.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the current dataset.")
        if column not in baseline_data.columns:
            raise ValueError(f"Column '{column}' does not exist in the baseline dataset.")

        current_values = self.data[column].dropna()
        baseline_values = baseline_data[column].dropna()

        _, p_value = ks_2samp(current_values, baseline_values)
        return p_value

    def check_rare_categories(self, column, threshold=1):
        """
        Identify rare categories in a column.

        Args:
            column (str): Name of the column to analyze.
            threshold (int): Minimum count below which a category is considered rare.

        Returns:
            list: Categories that are considered rare.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        value_counts = self.data[column].value_counts()
        rare_categories = value_counts[value_counts < threshold].index.tolist()
        return rare_categories


if __name__ == "__main__":
    df = pd.read_csv("../data/sample_data.csv")
    checker = DatasetQualityChecker(df)
    report = checker.generate_report()
    print("Dataset Quality Report:\n", report)
