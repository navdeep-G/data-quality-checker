import pandas as pd

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


if __name__ == "__main__":
    df = pd.read_csv("../data/sample_data.csv")
    checker = DatasetQualityChecker(df)
    report = checker.generate_report()
    print("Dataset Quality Report:\n", report)
