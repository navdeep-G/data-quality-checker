import numpy as np
from scipy.stats import ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter
from langdetect import detect
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import phonenumbers
import gensim.downloader as api
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, kstest, chisquare


### 1. DataQualityChecker Class (20 methods)
class DataQualityChecker:
    """
    A comprehensive class for assessing and addressing data quality issues.

    Attributes:
        data (pd.DataFrame): The dataset to analyze.
    """

    def __init__(self, data):
        """
        Initialize the DataQualityChecker with a pandas DataFrame.

        Args:
            data (pd.DataFrame): The dataset to analyze.

        Raises:
            ValueError: If the input is not a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data

    def generate_report(self):
        """
        Generate a summary report of data quality issues.

        Returns:
            dict: A dictionary containing missing values, duplicate records, and outlier counts.
        """
        report = {
            "missing_values": self.check_missing_values(),
            "duplicates": self.check_duplicate_records(),
            "outliers": self.check_outliers(),
        }
        return report

    def check_numeric_column_ranges(self, column_ranges):
        """
        Ensure numeric columns fall within pre-defined acceptable ranges.

        Args:
            column_ranges (dict): A dictionary where keys are column names and values are tuples (min, max)
                                  defining the acceptable range for the column.

        Returns:
            dict: A dictionary containing columns and their rows that fall outside the defined ranges.

        Raises:
            ValueError: If the specified columns do not exist or are not numeric.
        """
        if not isinstance(column_ranges, dict):
            raise ValueError(
                "column_ranges must be a dictionary with column names as keys and (min, max) tuples as values.")

        invalid_data = {}

        for column, (min_val, max_val) in column_ranges.items():
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' does not exist in the dataset.")

            if not pd.api.types.is_numeric_dtype(self.data[column]):
                raise ValueError(f"Column '{column}' is not numeric.")

            # Check for rows outside the range
            invalid_rows = self.data[(self.data[column] < min_val) | (self.data[column] > max_val)]
            if not invalid_rows.empty:
                invalid_data[column] = invalid_rows

        return invalid_data

    def check_temporal_data_consistency(self, timestamp_column, interval_columns=None):
        """
        Validate chronological order and check for overlapping time intervals.

        Args:
            timestamp_column (str): The name of the column containing timestamps.
            interval_columns (tuple, optional): A tuple of two column names representing start and end times for intervals.

        Returns:
            dict: A dictionary containing:
                - unordered_timestamps: The number of unordered timestamps.
                - overlapping_intervals: Rows with overlapping intervals (if interval_columns is provided).
        Raises:
            ValueError: If the specified columns do not exist or are invalid.
        """
        if timestamp_column not in self.data.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' does not exist in the dataset.")

        # Validate timestamp column for chronological order
        self.data[timestamp_column] = pd.to_datetime(self.data[timestamp_column], errors='coerce')
        unordered_timestamps = (self.data[timestamp_column].diff() < pd.Timedelta(0)).sum()

        result = {"unordered_timestamps": unordered_timestamps}

        # If interval columns are provided, check for overlaps
        if interval_columns:
            start_col, end_col = interval_columns
            if start_col not in self.data.columns or end_col not in self.data.columns:
                raise ValueError("Start or end column does not exist in the dataset.")

            self.data[start_col] = pd.to_datetime(self.data[start_col], errors='coerce')
            self.data[end_col] = pd.to_datetime(self.data[end_col], errors='coerce')

            overlapping_intervals = self.data[
                (self.data[end_col] > self.data[start_col].shift(-1)) &
                (self.data[start_col] < self.data[end_col].shift(-1))
                ]
            result["overlapping_intervals"] = overlapping_intervals

        return result

    def check_email_validity(self, column):
        """
        Checks if email addresses in a column are valid.

        Args:
            column (str): The column name to check.

        Returns:
            pd.Series: A boolean series indicating invalid email addresses.
        """
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return ~self.data[column].astype(str).str.match(email_regex)

    def check_phone_number_validity(self, column, country_code="+1"):
        """
        Checks if phone numbers in a column are valid for a specific country code.

        Args:
            column (str): The column name to check.
            country_code (str): The country code to use for validation.

        Returns:
            pd.Series: A boolean series indicating invalid phone numbers.
        """

        def is_valid_number(number):
            try:
                z = phonenumbers.parse(number, country_code)
                return phonenumbers.is_valid_number(z)
            except (phonenumbers.NumberParseException, phonenumbers.phonenumberutil.NumberParseException):
                return False

        return self.data[column].apply(is_valid_number)

    def check_cross_column_dependency(self, column1, column2, rule):
        """
        Check for violations of cross-column dependency rules.

        Args:
            column1 (str): The first column involved in the dependency.
            column2 (str): The second column involved in the dependency.
            rule (callable): A function that evaluates the dependency.

        Returns:
            pd.DataFrame: Rows where the dependency rule is violated.

        Raises:
            ValueError: If either column is missing or the rule is not callable.
        """
        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise ValueError("One or both specified columns do not exist in the dataset.")
        if not callable(rule):
            raise ValueError("The rule must be a callable function.")
        violations = self.data[~self.data.apply(lambda row: rule(row[column1], row[column2]), axis=1)]
        return violations

    def target_feature_relationship(self, target_column, feature_columns):
        """
        Plot the relationship between the target column and numeric features.

        Args:
            target_column (str): The target column.
            feature_columns (list): A list of feature columns.

        Raises:
            ValueError: If the target column or any feature column is missing.
        """
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' does not exist.")
        for feature in feature_columns:
            if feature not in self.data.columns:
                print(f"Skipping {feature}, as it does not exist.")
                continue
            sns.boxplot(x=self.data[target_column], y=self.data[feature])
            plt.title(f"{feature} vs {target_column}")
            plt.show()

    def check_pattern_consistency(self, column, regex):
        """
        Check if values in a column match a specific pattern.

        Args:
            column (str): The name of the column to check.
            regex (str): The regular expression defining the pattern.

        Returns:
            pd.DataFrame: Rows where the pattern does not match.

        Raises:
            ValueError: If the column is missing or the regex is invalid.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        try:
            pattern = re.compile(regex)
        except re.error:
            raise ValueError("Invalid regular expression provided.")
        invalid_rows = self.data[~self.data[column].astype(str).apply(lambda x: bool(pattern.match(x)))]
        return invalid_rows

    def check_unexpected_values(self, column, expected_values):
        """
        Identify unexpected values in a categorical column.

        Args:
            column (str): The categorical column to analyze.
            expected_values (list): List of valid expected values.

        Returns:
            pd.DataFrame: Rows with unexpected values.

        Raises:
            ValueError: If the column is missing or the expected values are not a list.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not isinstance(expected_values, list):
            raise ValueError("Expected values must be provided as a list.")
        unexpected = ~self.data[column].isin(expected_values)
        return self.data[unexpected]

    def check_data_coverage(self, reference_data, columns):
        """
        Check if the dataset sufficiently covers all unique values in specified columns of a reference dataset.

        Args:
            reference_data (pd.DataFrame): The reference dataset.
            columns (list): List of columns to compare.

        Returns:
            dict: Columns and their missing unique values.

        Raises:
            ValueError: If the columns are not present in either dataset.
        """
        missing_coverage = {}
        for column in columns:
            if column not in self.data.columns or column not in reference_data.columns:
                raise ValueError(f"Column '{column}' does not exist in one of the datasets.")
            missing_values = set(reference_data[column].unique()) - set(self.data[column].unique())
            missing_coverage[column] = missing_values
        return missing_coverage

    def detect_data_leaks(self, target_column, feature_columns):
        """
        Detect potential data leaks by checking for high correlation between features and the target column.

        Args:
            target_column (str): The target column.
            feature_columns (list): List of feature columns to analyze.

        Returns:
            dict: Features with correlation exceeding 0.8.

        Raises:
            ValueError: If the target column or feature columns are missing.
        """
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' does not exist.")
        correlations = {}
        for feature in feature_columns:
            if feature not in self.data.columns:
                raise ValueError(f"Feature column '{feature}' does not exist.")
            correlation = self.data[feature].corr(self.data[target_column])
            if abs(correlation) > 0.8:
                correlations[feature] = correlation
        return correlations

    # Additional methods follow the same template with specific functionality...
    def check_missing_values(self):
        """
        Calculate the percentage of missing values for each column.

        Returns:
            pd.Series: A series containing columns with missing value percentages.

        Raises:
            ValueError: If the dataset is empty.
        """
        if self.data.empty:
            raise ValueError("Dataset is empty.")
        missing = self.data.isnull().mean() * 100
        return missing[missing > 0]

    def check_outliers(self, method='zscore', threshold=3):
        """
        Detect outliers in numeric columns of the dataset using specified statistical or machine learning methods.

        Args:
            method (str, optional): The method to use for outlier detection.
                - 'zscore': Uses Z-score to identify outliers based on standard deviation.
                - 'iqr': Uses the Interquartile Range (IQR) to detect outliers.
                - 'isolation_forest': Applies Isolation Forest for anomaly detection.
            threshold (float, optional): The threshold for identifying outliers.
                - Relevant only for 'zscore' (default is 3).
                - Ignored for 'iqr' and 'isolation_forest'.

        Returns:
            pd.DataFrame: A DataFrame containing:
                - **Column**: Column name.
                - **Outlier_Count**: Number of outliers detected in each column.

        Raises:
            ValueError: If no numeric columns are available in the dataset.
            ValueError: If an invalid `method` is passed.

        Examples:
            >>> analyzer = DataAnalyzer(dataframe)
            >>> analyzer.check_outliers(method='zscore', threshold=3)

            Column    Outlier_Count
            -------   -------------
            col1      5
            col2      2

            >>> analyzer.check_outliers(method='iqr')

            Column    Outlier_Count
            -------   -------------
            col1      3
            col2      1

            >>> analyzer.check_outliers(method='isolation_forest')

            Column    Outlier_Count
            -------   -------------
            col1      7
            col2      4

        Notes:
            - **Z-score Method:** Outliers are defined as points where |z| > threshold.
            - **IQR Method:** Outliers are points outside the range [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
            - **Isolation Forest:** A machine learning model is used for anomaly detection.
            - Results may vary slightly between methods due to different assumptions about data distribution.
        """
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        if numeric_data.empty:
            raise ValueError("No numeric columns available for outlier detection.")

        if method == 'zscore':
            z_scores = (numeric_data - numeric_data.mean()) / numeric_data.std()
            outlier_counts = (z_scores.abs() > threshold).sum()
        elif method == 'iqr':
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_counts = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()
        elif method == 'isolation_forest':
            outlier_counts = pd.Series([
                IsolationForest(contamination=0.05).fit_predict(numeric_data[col].values.reshape(-1, 1)).sum()
                for col in numeric_data.columns
            ], index=numeric_data.columns)
        else:
            raise ValueError("Invalid method. Choose from 'zscore', 'iqr', 'isolation_forest'.")

        return pd.DataFrame({'Column': outlier_counts.index, 'Outlier_Count': outlier_counts.values})

    def check_imbalance(self, column):
        """
        Check for class imbalance in a categorical column.

        Args:
            column (str): The column to analyze for imbalance.

        Returns:
            pd.Series: A series showing the percentage distribution of each class.

        Raises:
            ValueError: If the column is missing or not categorical.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_object_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not categorical.")
        return self.data[column].value_counts(normalize=True) * 100

    def check_data_type_consistency(self):
        """
        Check for inconsistent data types within columns.

        Returns:
            dict: A dictionary of columns with inconsistent data types.

        Raises:
            ValueError: If the dataset is empty.
        """
        if self.data.empty:
            raise ValueError("Dataset is empty.")
        inconsistent = {}
        for col in self.data.columns:
            unique_types = self.data[col].map(type).nunique()
            if unique_types > 1:
                inconsistent[col] = unique_types
        return inconsistent

    def check_unique_values(self):
        """
        Identify columns with only one unique value.

        Returns:
            list: Columns with a single unique value.

        Raises:
            ValueError: If the dataset is empty.
        """
        if self.data.empty:
            raise ValueError("Dataset is empty.")
        single_value_columns = [col for col in self.data.columns if self.data[col].nunique() == 1]
        return single_value_columns

    def validate_schema(self, schema_file):
        """
        Validate the dataset schema against a provided JSON schema file.

        Args:
            schema_file (str): Path to the schema JSON file.

        Returns:
            list: Columns missing in the dataset but required by the schema.

        Raises:
            FileNotFoundError: If the schema file does not exist.
            ValueError: If the schema file is not a valid JSON.
        """
        try:
            with open(schema_file, "r") as file:
                schema = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file '{schema_file}' not found.")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in schema file.")
        missing_columns = [col for col in schema["columns"] if col not in self.data.columns]
        return missing_columns

    def check_rare_categories(self, column, threshold=1):
        """
        Identify rare categories in a column.

        Args:
            column (str): The column to analyze.
            threshold (int): The minimum count for a category to be considered common.

        Returns:
            list: Categories considered rare.

        Raises:
            ValueError: If the column is missing.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        value_counts = self.data[column].value_counts()
        rare_categories = value_counts[value_counts < threshold].index.tolist()
        return rare_categories

    def check_column_naming_convention(self, regex_pattern=r"^[a-z_]+$"):
        """
        Verify if column names adhere to a specific naming convention.

        Args:
            regex_pattern (str): The regular expression defining the naming convention.

        Returns:
            list: Columns not matching the naming convention.

        Raises:
            ValueError: If the provided regex pattern is invalid.
        """
        try:
            pattern = re.compile(regex_pattern)
        except re.error:
            raise ValueError("Invalid regular expression pattern.")
        inconsistent_columns = [col for col in self.data.columns if not pattern.match(col)]
        return inconsistent_columns

    def detect_anomalies(self, column, contamination=0.05):
        """
        Detect anomalies in a numeric column using Isolation Forest.

        Args:
            column (str): The numeric column to analyze.
            contamination (float): The proportion of anomalies in the data.

        Returns:
            pd.Series: A boolean series indicating anomalies.

        Raises:
            ValueError: If the column is missing or not numeric.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric.")
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.data['anomaly'] = isolation_forest.fit_predict(self.data[[column]])
        return self.data['anomaly'] == -1

    def check_sampling_bias(self, column, baseline_distribution):
        """
        Compare the distribution of a column with a baseline distribution.

        Args:
            column (str): The column to compare.
            baseline_distribution (dict): Expected distribution as a dictionary.

        Returns:
            dict: Deviations from the baseline distribution.

        Raises:
            ValueError: If the column is missing or the baseline distribution is not a dictionary.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not isinstance(baseline_distribution, dict):
            raise ValueError("Baseline distribution must be provided as a dictionary.")
        actual_distribution = self.data[column].value_counts(normalize=True).to_dict()
        deviations = {
            category: actual_distribution.get(category, 0) - baseline_distribution.get(category, 0)
            for category in baseline_distribution
        }
        return deviations

    def check_encoding_consistency(self, column):
        """
        Check if the encoding of a text column is consistent.

        Args:
            column (str): The column to check.

        Returns:
            bool: True if encoding is consistent, False otherwise.
        """
        try:
            self.data[column].apply(lambda x: x.encode('utf-8').decode('utf-8'))
            return True
        except UnicodeDecodeError:
            return False

    def detect_duplicates(self):
        """
        Identify duplicate rows and columns in the dataset.

        Returns:
            dict: A dictionary containing:
                - 'duplicate_rows': A DataFrame with duplicate rows.
                - 'duplicate_columns': A list of duplicate column names.
        """
        if self.data.empty:
            raise ValueError("The dataset is empty.")

        duplicate_rows = self.data[self.data.duplicated()]
        duplicate_columns = self.data.columns[self.data.T.duplicated()].tolist()

        return {
            "duplicate_rows": duplicate_rows,
            "duplicate_columns": duplicate_columns,
        }

    def detect_sparse_and_empty_columns(self, sparsity_threshold=0.9):
        """
        Identify columns that are either completely empty or have a high percentage of missing/zero values.

        Args:
            sparsity_threshold (float): The sparsity threshold (default: 90%).

        Returns:
            dict: A dictionary containing:
                - 'empty_columns': Columns that are completely empty.
                - 'sparse_columns': Columns with sparsity above the given threshold.
        """
        if self.data.empty:
            raise ValueError("The dataset is empty.")

        empty_columns = [col for col in self.data.columns if self.data[col].isnull().all()]
        sparse_columns = [
            col for col in self.data.columns
            if (self.data[col].isnull().mean() + (self.data[col] == 0).mean()) > sparsity_threshold
        ]

        return {
            "empty_columns": empty_columns,
            "sparse_columns": sparse_columns,
        }

    def validate_foreign_key(self, column, reference_column):
        """
        Validate foreign key constraints between two datasets.

        Args:
            column (str): The column in the current dataset to validate.
            reference_column (pd.Series): The reference column from another dataset.

        Returns:
            pd.DataFrame: Rows in the dataset where the foreign key is invalid.
        """
        invalid_rows = self.data[~self.data[column].isin(reference_column)]
        return invalid_rows

    def detect_string_length_outliers(self, column, min_length=1, max_length=255):
        """
        Identify text entries in a column that are too short or too long.

        Args:
            column (str): The column to analyze.
            min_length (int): Minimum acceptable string length.
            max_length (int): Maximum acceptable string length.

        Returns:
            pd.DataFrame: Rows with outlier string lengths.
        """
        outliers = self.data[
            (self.data[column].str.len() < min_length) | (self.data[column].str.len() > max_length)
            ]
        return outliers

    def detect_mixed_data_types(self, column):
        """
        Identify columns with mixed data types.

        Args:
            column (str): The column to analyze.

        Returns:
            bool: True if mixed data types are present, False otherwise.
        """
        unique_types = self.data[column].map(type).nunique()
        return unique_types > 1

    def check_invalid_date_formats(self, column, date_format='%Y-%m-%d'):
        """
        Identify invalid date formats in a column.

        Args:
            column (str): The column to check.
            date_format (str): The expected date format (default: '%Y-%m-%d').

        Returns:
            pd.DataFrame: Rows with invalid date formats.
        """
        invalid_dates = self.data[~self.data[column].apply(
            lambda x: pd.to_datetime(x, format=date_format, errors='coerce').notnull()
        )]
        return invalid_dates

    def detect_column_redundancy(self, correlation_threshold=0.95):
        """
        Detect redundant columns using both correlation and exact equality.

        Args:
            correlation_threshold (float): Threshold for detecting correlated redundancy (default: 0.95).

        Returns:
            dict: A dictionary containing:
                - 'correlated_columns': Pairs of columns with high correlation.
                - 'exact_redundant_columns': Pairs of columns with identical values.
        """
        if self.data.empty:
            raise ValueError("The dataset is empty.")

        # Detect correlated columns
        corr_matrix = self.data.corr()
        correlated_columns = [
            (col1, col2) for col1 in corr_matrix.columns for col2 in corr_matrix.columns
            if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > correlation_threshold
        ]

        # Detect exactly redundant columns
        exact_redundant_columns = [
            (col1, col2) for col1 in self.data.columns for col2 in self.data.columns
            if col1 != col2 and self.data[col1].equals(self.data[col2])
        ]

        return {
            "correlated_columns": correlated_columns,
            "exact_redundant_columns": exact_redundant_columns,
        }

    def validate_categorical_consistency(self, column, valid_categories):
        """
        Ensure all categories in a column are within a predefined set of valid categories.

        Args:
            column (str): The column to validate.
            valid_categories (set): The set of valid categories.

        Returns:
            pd.Series: Rows with invalid categories.
        """
        invalid_rows = self.data[~self.data[column].isin(valid_categories)]
        return invalid_rows

    def check_data_completeness(self, required_columns):
        """
        Check if the dataset contains all required columns.

        Args:
            required_columns (list): A list of column names that should be present in the dataset.

        Returns:
            list: Missing columns that are required but not in the dataset.
        """
        if not isinstance(required_columns, list):
            raise ValueError("Required columns must be provided as a list.")
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        return missing_columns

    def check_column_duplicates(self, column):
        """
        Check for duplicate values within a specific column.

        Args:
            column (str): The name of the column to check.

        Returns:
            pd.DataFrame: Rows with duplicate values in the specified column.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        duplicates = self.data[self.data.duplicated(subset=[column])]
        return duplicates

    def check_consistency_across_columns(self, column1, column2, rule):
        """
        Check if values in one column are consistent with another column based on a rule.

        Args:
            column1 (str): The first column.
            column2 (str): The second column.
            rule (callable): A function to define consistency between two columns.

        Returns:
            pd.DataFrame: Rows where the consistency rule is violated.
        """
        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise ValueError("One or both specified columns do not exist in the dataset.")
        if not callable(rule):
            raise ValueError("Rule must be a callable function.")
        inconsistent_rows = self.data[~self.data.apply(lambda row: rule(row[column1], row[column2]), axis=1)]
        return inconsistent_rows

    def validate_numeric_precision(self, column, precision):
        """
        Validate that numeric values in a column do not exceed a specified precision.

        Args:
            column (str): The numeric column to validate.
            precision (int): Maximum allowed number of decimal places.

        Returns:
            pd.DataFrame: Rows where numeric precision exceeds the specified limit.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric.")
        invalid_rows = self.data[
            self.data[column].apply(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0) > precision]
        return invalid_rows

    def check_null_rows(self):
        """
        Identify rows where all columns are null.

        Returns:
            pd.DataFrame: Rows with all values as null.
        """
        null_rows = self.data[self.data.isnull().all(axis=1)]
        return null_rows

    def check_partition_column_completeness(self, partition_column, required_columns):
        """
        Check if all partitions contain data for required columns.

        Args:
            partition_column (str): The column defining partitions.
            required_columns (list): List of required columns.

        Returns:
            dict: A dictionary with partitions as keys and missing columns as values.
        """
        missing_columns = {}
        for partition, group in self.data.groupby(partition_column):
            missing = [col for col in required_columns if col not in group.columns or group[col].isnull().all()]
            if missing:
                missing_columns[partition] = missing
        return missing_columns

### 2. StatisticalAnalyzer Class (6 methods)
class StatisticalAnalyzer:
    def __init__(self, data):
        self.data = data

    def plot_cdf(self, column):
        data = self.data[column].dropna()
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cdf)
        plt.title(f"CDF of {column}")
        plt.xlabel(column)
        plt.ylabel("CDF")
        plt.grid(True)
        plt.show()

    def plot_correlation_heatmap(self):
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    def check_conditional_probability(self, column1, column2, expected_probabilities):
        probs = self.data.groupby(column1)[column2].value_counts(normalize=True).to_dict()
        return {
            cond: {
                outcome: probs.get((cond, outcome), 0) - expected
                for outcome, expected in expected_outcomes.items()
            }
            for cond, expected_outcomes in expected_probabilities.items()
        }

    def detect_data_drift(self, baseline_data, column):
        current_values = self.data[column].dropna()
        baseline_values = baseline_data[column].dropna()
        _, p_value = ks_2samp(current_values, baseline_values)
        return p_value

    def check_outlier_impact(self, column, method="mean"):
        z_scores = (self.data[column] - self.data[column].mean()) / self.data[column].std()
        non_outliers = self.data[z_scores.abs() <= 3]
        return getattr(self.data[column], method)() - getattr(non_outliers[column], method)()

    def low_variance_features(self, threshold=0.01):
        variances = self.data.var()
        return variances[variances < threshold].index.tolist()


    def bootstrap_sampling_analysis(self, column, metric='mean', n_iterations=1000, confidence_level=0.95):
        """
        Use bootstrapping to estimate the variability of dataset metrics.

        Args:
            column (str): The numeric column to perform bootstrap sampling on.
            metric (str): The metric to estimate. Options: 'mean', 'median', 'std'.
            n_iterations (int): Number of bootstrap samples to draw.
            confidence_level (float): Confidence level for the confidence interval.

        Returns:
            dict: A dictionary containing:
                - 'bootstrap_estimate': The bootstrap estimate of the metric.
                - 'lower_bound': Lower bound of the confidence interval.
                - 'upper_bound': Upper bound of the confidence interval.
                - 'bootstrap_distribution': Array of bootstrap estimates.

        Raises:
            ValueError: If the column does not exist, is not numeric, or an invalid metric is specified.
        """
        # Validate inputs
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric.")

        if metric not in ['mean', 'median', 'std']:
            raise ValueError("Invalid metric. Choose from 'mean', 'median', 'std'.")

        # Define metric functions
        metric_functions = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std
        }

        # Data preparation
        data = self.data[column].dropna().values
        n_samples = len(data)

        # Bootstrap Sampling (Vectorized)
        np.random.seed(42)  # For reproducibility
        bootstrap_samples = np.random.choice(data, size=(n_iterations, n_samples), replace=True)
        bootstrap_estimates = np.apply_along_axis(metric_functions[metric], 1, bootstrap_samples)

        # Confidence Interval
        lower_bound, upper_bound = np.percentile(
            bootstrap_estimates,
            [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100]
        )
        bootstrap_estimate = metric_functions[metric](bootstrap_estimates)

        # Plot Bootstrap Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(bootstrap_estimates, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(lower_bound, color='green', linestyle='--', label='Lower Bound')
        plt.axvline(upper_bound, color='red', linestyle='--', label='Upper Bound')
        plt.axvline(bootstrap_estimate, color='blue', linestyle='--', label='Bootstrap Estimate')
        plt.title(f'Bootstrap Sampling Distribution of {metric} ({confidence_level * 100:.0f}% CI)')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        return {
            "bootstrap_estimate": bootstrap_estimate,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "bootstrap_distribution": bootstrap_estimates
        }

    def analyze_confidence_intervals(self, column, confidence_level=0.95):
        """
        Calculate and plot confidence intervals for a numeric column.

        Args:
            column (str): The numeric column to analyze.
            confidence_level (float): The confidence level for the interval (default is 0.95).

        Returns:
            dict: A dictionary containing:
                - 'mean': The mean of the column.
                - 'lower_bound': Lower bound of the confidence interval.
                - 'upper_bound': Upper bound of the confidence interval.

        Raises:
            ValueError: If the column is not numeric or does not exist.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric.")

        import scipy.stats as stats

        # Drop missing values
        data = self.data[column].dropna()

        # Calculate sample mean and standard error
        mean = data.mean()
        sem = stats.sem(data)  # Standard error of the mean

        # Calculate confidence interval
        margin_of_error = stats.t.ppf((1 + confidence_level) / 2, len(data) - 1) * sem
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        # Plotting the confidence interval
        plt.figure(figsize=(8, 4))
        plt.axvline(mean, color='blue', linestyle='--', label='Mean')
        plt.axvline(lower_bound, color='green', linestyle='--', label='Lower Bound')
        plt.axvline(upper_bound, color='red', linestyle='--', label='Upper Bound')
        plt.hist(data, bins=30, alpha=0.5, color='gray', edgecolor='black')
        plt.title(f'Confidence Interval for {column} ({confidence_level * 100:.0f}% Confidence Level)')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        return {
            "mean": mean,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    import pandas as pd
    from scipy.stats import ttest_ind, chi2_contingency, f_oneway

    def perform_hypothesis_testing(self, test_type, column1, column2=None, group_column=None):
        """
        Perform hypothesis testing using t-tests, chi-squared tests, or ANOVA.

        Args:
            test_type (str): The type of test to perform. Options: 't-test', 'chi-squared', 'anova'.
            column1 (str): The first column involved in the test (dependent or observed variable).
            column2 (str, optional): The second column for t-tests or chi-squared tests.
            group_column (str, optional): The grouping column for ANOVA.

        Returns:
            dict: A dictionary containing:
                - 'test_statistic': The test statistic.
                - 'p_value': The p-value of the test.

        Raises:
            ValueError: If the input parameters or column data types are invalid.
        """
        # Validate test_type
        valid_tests = ['t-test', 'chi-squared', 'anova']
        if test_type not in valid_tests:
            raise ValueError(f"Invalid test_type. Choose from {', '.join(valid_tests)}.")

        # Validate column1
        if column1 not in self.data.columns:
            raise ValueError(f"Column '{column1}' does not exist in the dataset.")

        # Perform specific tests
        if test_type == 't-test':
            # Validate column2
            if column2 is None or column2 not in self.data.columns:
                raise ValueError(f"Column '{column2}' must be specified and exist for a t-test.")

            # Perform independent t-test
            group1 = self.data[column1].dropna()
            group2 = self.data[column2].dropna()
            test_statistic, p_value = ttest_ind(group1, group2, equal_var=False)

        elif test_type == 'chi-squared':
            # Validate column2
            if column2 is None or column2 not in self.data.columns:
                raise ValueError(f"Column '{column2}' must be specified and exist for a chi-squared test.")

            # Perform chi-squared test
            contingency_table = pd.crosstab(self.data[column1], self.data[column2])
            test_statistic, p_value, _, _ = chi2_contingency(contingency_table)

        elif test_type == 'anova':
            # Validate group_column
            if group_column is None or group_column not in self.data.columns:
                raise ValueError(f"Group column '{group_column}' must be specified and exist for ANOVA.")

            # Perform one-way ANOVA
            groups = [group[column1].dropna().values for _, group in self.data.groupby(group_column)]
            if len(groups) < 2:
                raise ValueError("ANOVA requires at least two groups for comparison.")
            test_statistic, p_value = f_oneway(*groups)

        return {
            "test_statistic": test_statistic,
            "p_value": p_value
        }

    def check_uniform_distribution(self, column, p_value_threshold=0.05):
        """
        Test if a numeric or categorical column follows a uniform distribution.

        Args:
            column (str): The name of the column to test.
            p_value_threshold (float): The p-value threshold for rejecting the null hypothesis.

        Returns:
            dict: A dictionary containing:
                - 'is_uniform': True if the column is uniformly distributed, False otherwise.
                - 'p_value': The p-value from the chi-squared or KS test.

        Raises:
            ValueError: If the column does not exist or is not numeric/categorical.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        col_data = self.data[column].dropna()

        if col_data.empty:
            raise ValueError(f"Column '{column}' has no valid data.")

        if pd.api.types.is_numeric_dtype(col_data):
            # Normalize numeric data range to [0, 1]
            min_val, max_val = col_data.min(), col_data.max()
            scaled_data = (col_data - min_val) / (max_val - min_val)

            # Perform KS test against uniform distribution
            _, p_value = kstest(scaled_data, 'uniform')
        elif pd.api.types.is_object_dtype(col_data):
            # Calculate observed and expected frequencies for categorical data
            observed_freq = col_data.value_counts()
            expected_freq = [len(col_data) / len(observed_freq)] * len(observed_freq)

            # Perform chi-squared test
            _, p_value = chisquare(observed_freq, expected_freq)
        else:
            raise ValueError(f"Column '{column}' is not numeric or categorical.")

        return {
            "is_uniform": p_value > p_value_threshold,
            "p_value": p_value
        }

    def check_correlation(self, threshold=0.9):
        """
        Identify highly correlated numeric features.

        Args:
            threshold (float): The correlation coefficient threshold.

        Returns:
            list: Pairs of columns with correlations exceeding the threshold.

        Raises:
            ValueError: If no numeric columns are available in the dataset.
        """
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        if numeric_data.empty:
            raise ValueError("No numeric columns available for correlation check.")
        correlation_matrix = numeric_data.corr()
        correlated_features = [
            (col1, col2) for col1 in correlation_matrix.columns for col2 in correlation_matrix.columns
            if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > threshold
        ]
        return correlated_features

    def check_multicollinearity(self, threshold=10):
        """
        Check for multicollinearity using Variance Inflation Factor (VIF).

        Args:
            threshold (float): The VIF threshold above which a feature is flagged.

        Returns:
            pd.DataFrame: A DataFrame with features and their VIF values exceeding the threshold.

        Raises:
            ValueError: If the dataset has no numeric columns.
        """
        numeric_data = self.data.select_dtypes(include=["float64", "int64"]).dropna()
        if numeric_data.empty:
            raise ValueError("No numeric columns available for multicollinearity check.")
        vif_data = pd.DataFrame()
        vif_data["feature"] = numeric_data.columns
        vif_data["VIF"] = [
            variance_inflation_factor(numeric_data.values, i) for i in range(numeric_data.shape[1])
        ]
        return vif_data[vif_data["VIF"] > threshold]


### 3. TimeSeriesAnalyzer Class (3 methods)
class TimeSeriesAnalyzer:
    """
    Analyzes time series data for gaps, seasonality, and rare events.

    Attributes:
        data (pd.DataFrame): The time series data.
    """

    def __init__(self, data):
        """
        Initializes the TimeSeriesAnalyzer with the provided time series data.

        Args:
            data (pd.DataFrame): The time series data.

        Raises:
            TypeError: If the data is not a pandas DataFrame.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.data = data

    def detect_change_points(self, column, timestamp_column, method='mean_shift', threshold=1.0):
        """
        Identify structural breaks or change points in time-series data.

        Args:
            column (str): The numeric column containing time-series data.
            timestamp_column (str): The column containing timestamps.
            method (str): Method for change point detection. Options: 'mean_shift', 'cumsum'.
            threshold (float): Threshold for detecting significant change points.

        Returns:
            dict: A dictionary containing:
                - 'change_points': Indices or timestamps of detected change points.
                - 'method': Method used for detection.

        Raises:
            ValueError: If the columns do not exist, data is insufficient, or invalid method is specified.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if timestamp_column not in self.data.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' does not exist in the dataset.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        # Ensure timestamps are in datetime format
        self.data[timestamp_column] = pd.to_datetime(self.data[timestamp_column], errors='coerce')
        self.data = self.data.dropna(subset=[timestamp_column, column])
        self.data.set_index(timestamp_column, inplace=True)
        self.data.sort_index(inplace=True)

        if len(self.data) < 10:
            raise ValueError("Insufficient data for change point detection. At least 10 data points are required.")

        import ruptures as rpt

        ts_data = self.data[column].values

        # Choose the method
        if method == 'mean_shift':
            model = "l2"  # Least squares for mean shift detection
        elif method == 'cumsum':
            model = "l1"  # L1 norm for cumulative sum change detection
        else:
            raise ValueError("Invalid method. Choose from 'mean_shift' or 'cumsum'.")

        # Apply change point detection
        algo = rpt.Pelt(model=model).fit(ts_data)
        change_points = algo.predict(pen=threshold)

        # Convert indices to timestamps
        change_timestamps = self.data.index[change_points[:-1]] if change_points else []

        # Plot time-series with detected change points
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, ts_data, label='Time-Series Data')
        for cp in change_timestamps:
            plt.axvline(cp, color='red', linestyle='--', label='Change Point')
        plt.title(f"Change Point Detection using {method}")
        plt.xlabel('Timestamp')
        plt.ylabel(column)
        plt.legend()
        plt.show()

        return {
            "change_points": list(change_timestamps),
            "method": method
        }

    def forecast_accuracy_metrics(self, actual_column, predicted_column):
        """
        Evaluate forecast accuracy metrics for predictive models.

        Args:
            actual_column (str): Column containing the actual values.
            predicted_column (str): Column containing the predicted values.

        Returns:
            dict: A dictionary containing:
                - 'RMSE': Root Mean Squared Error.
                - 'MAPE': Mean Absolute Percentage Error.
                - 'MAE': Mean Absolute Error.
                - 'R2': R-squared Score.
                - 'MedianAE': Median Absolute Error.
                - 'SMAPE': Symmetric Mean Absolute Percentage Error.
                - 'Bias': Mean Bias Deviation.

        Raises:
            ValueError: If the columns do not exist or contain invalid data.
        """
        if actual_column not in self.data.columns:
            raise ValueError(f"Actual values column '{actual_column}' does not exist in the dataset.")
        if predicted_column not in self.data.columns:
            raise ValueError(f"Predicted values column '{predicted_column}' does not exist in the dataset.")

        if not pd.api.types.is_numeric_dtype(self.data[actual_column]) or not pd.api.types.is_numeric_dtype(
                self.data[predicted_column]):
            raise ValueError("Both actual and predicted columns must be numeric.")

        # Drop missing values
        valid_data = self.data[[actual_column, predicted_column]].dropna()

        actual = valid_data[actual_column]
        predicted = valid_data[predicted_column]

        from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

        # Calculate metrics
        rmse = mean_squared_error(actual, predicted, squared=False)
        mae = mean_absolute_error(actual, predicted)
        median_ae = median_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        mape = (np.abs((actual - predicted) / actual)).mean() * 100
        smape = (2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))).mean() * 100
        bias = (predicted - actual).mean()

        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(actual.reset_index(drop=True), label='Actual', marker='o')
        plt.plot(predicted.reset_index(drop=True), label='Predicted', marker='x')
        plt.title('Forecast Accuracy: Actual vs Predicted')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

        return {
            "RMSE": rmse,
            "MAE": mae,
            "MedianAE": median_ae,
            "MAPE": mape,
            "SMAPE": smape,
            "Bias": bias,
            "R2": r2
        }

    def seasonal_trend_analysis(self, column, timestamp_column, period='M', model='additive'):
        """
        Plot seasonal trends and anomalies for long-term time-series data.

        Args:
            column (str): The numeric column containing time-series data.
            timestamp_column (str): The column containing timestamps.
            period (str): Frequency of the time series ('D' for daily, 'M' for monthly, 'Y' for yearly).
            model (str): Type of decomposition model - 'additive' or 'multiplicative'.

        Returns:
            dict: A dictionary containing the decomposed components:
                - 'trend': The trend component.
                - 'seasonal': The seasonal component.
                - 'residual': The residual (anomaly) component.

        Raises:
            ValueError: If columns are invalid or data is insufficient for analysis.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if timestamp_column not in self.data.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' does not exist in the dataset.")

        # Ensure timestamps are in datetime format
        self.data[timestamp_column] = pd.to_datetime(self.data[timestamp_column], errors='coerce')
        self.data = self.data.dropna(subset=[timestamp_column, column])
        self.data.set_index(timestamp_column, inplace=True)
        self.data.sort_index(inplace=True)

        if len(self.data) < 2:
            raise ValueError("Insufficient data for seasonal analysis. At least two timestamps are required.")

        from statsmodels.tsa.seasonal import seasonal_decompose

        # Perform seasonal decomposition
        decomposition = seasonal_decompose(self.data[column], model=model, period={'D': 1, 'M': 12, 'Y': 365}[period])

        # Plot decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed')

        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')

        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonality')

        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residuals (Anomalies)')

        plt.tight_layout()
        plt.show()

        return {
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid
        }

    def detect_non_stationarity(self, column, significance_level=0.05):
        """
        Apply Augmented Dickey-Fuller (ADF) test to check time-series stationarity.

        Args:
            column (str): The time-series column to analyze.
            significance_level (float): The significance level for the ADF test (default is 0.05).

        Returns:
            dict: A dictionary containing:
                - 'adf_statistic': The ADF test statistic.
                - 'p_value': The p-value from the test.
                - 'stationary': Boolean indicating if the series is stationary.
                - 'critical_values': Critical values at different confidence levels.

        Raises:
            ValueError: If the column does not exist, is not numeric, or contains insufficient data.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric for stationarity testing.")

        from statsmodels.tsa.stattools import adfuller

        # Drop missing values
        ts_data = self.data[column].dropna()

        if len(ts_data) < 10:
            raise ValueError("Insufficient data for stationarity testing. At least 10 data points are required.")

        # Perform Augmented Dickey-Fuller test
        adf_result = adfuller(ts_data)
        adf_statistic, p_value, _, _, critical_values, _ = adf_result

        result = {
            "adf_statistic": adf_statistic,
            "p_value": p_value,
            "stationary": p_value <= significance_level,
            "critical_values": critical_values
        }

        # Plot the time series
        plt.figure(figsize=(10, 6))
        plt.plot(ts_data, label='Time Series Data')
        plt.title(f'ADF Test for Stationarity on {column}')
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.legend()
        plt.show()

        return result

    def check_time_series_gaps(self, timestamp_column):
        """
        Analyzes the time series data for gaps in the provided timestamp column.

        Args:
            timestamp_column (str): The name of the column containing timestamps.

        Returns:
            dict: A dictionary containing:
                gaps (int): The total number of missing values in the timestamp column.
                unordered (int): The number of timestamp values that are out of order.
        """

        self.data[timestamp_column] = pd.to_datetime(self.data[timestamp_column])
        gaps = self.data[timestamp_column].isnull().sum()
        unordered = (self.data[timestamp_column].diff().dt.total_seconds() < 0).sum()
        return {"gaps": gaps, "unordered": unordered}

    def time_series_decomposition(self, column, frequency):
        """
        Decomposes the time series data in the specified column into trend, seasonality, and residuals using seasonal decomposition.

        Args:
            column (str): The name of the column containing the time series data.
            frequency (int): The seasonal period of the data (e.g., 12 for monthly data).

        Returns:
            statsmodels.tsa.seasonal.seasonal_decompose: The seasonal decomposition object.
        """

        series = self.data[column].dropna()
        decomp = sm.seasonal_decompose(series, model='additive', period=frequency)
        decomp.plot()
        plt.show()
        return decomp

    def check_rare_events(self, column, z_threshold=3):
        """
        Identifies and returns rows in the data where the absolute value of the z-score
        in the specified column exceeds the given threshold.

        Args:
            column (str): The name of the column containing the data for calculating z-scores.
            z_threshold (float, optional): The threshold for identifying rare events (defaults to 3).

        Returns:
            pd.DataFrame: A DataFrame containing only the rows where the absolute value of
                the z-score in the specified column exceeds the threshold.
        """

        z_scores = abs((self.data[column] - self.data[column].mean()) / self.data[column].std())
        return self.data[z_scores > z_threshold]


### 4. NLPAnalyzer Class (14 methods)
class NLPAnalyzer:
    def __init__(self, data):
        self.data = data
        self.model = self._load_word2vec_model()

    @staticmethod
    def _load_word2vec_model():
        """
        Load Word2Vec model (cached for reuse).

        Returns:
            model: Pre-trained Word2Vec model.
        """
        print("🔄 Loading Word2Vec model (Google News 300)...")
        return api.load('word2vec-google-news-300')

    def word_embedding_similarity(self, column, word1, word2):
        """
        Calculate similarity between two words for each row in a specified text column.

        Args:
            column (str): The text column to analyze.
            word1 (str): First word for comparison.
            word2 (str): Second word for comparison.

        Returns:
            pd.Series: A Pandas Series with cosine similarity scores for each row.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")

        if not word1 or not word2:
            raise ValueError("Both word1 and word2 must be non-empty strings.")

        def calculate_similarity(row):
            try:
                if pd.isnull(row):
                    return None
                return self.model.similarity(word1, word2)
            except KeyError as e:
                return f"Word not found in vocabulary: {e}"

        return self.data[column].apply(calculate_similarity)

    def correct_spelling(self, column):
        """
        Correct spelling errors in a text column.

        Args:
            column (str): The text column to correct.

        Returns:
            pd.Series: Text column with corrected spelling.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        from textblob import TextBlob

        return self.data[column].apply(lambda x: str(TextBlob(x).correct()) if pd.notnull(x) else x)

    def extract_keywords(self, column, top_n=10):
        """
        Extract keywords from text data using RAKE (Rapid Automatic Keyword Extraction).

        Args:
            column (str): The text column to extract keywords from.
            top_n (int): Number of top keywords to return.

        Returns:
            pd.Series: Keywords extracted from each row.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        from rake_nltk import Rake

        rake = Rake()

        def extract(text):
            if pd.isnull(text):
                return []
            rake.extract_keywords_from_text(text)
            return rake.get_ranked_phrases()[:top_n]

        return self.data[column].apply(extract)

    def named_entity_frequency(self, column, entity_type='PERSON', model='spacy'):
        """
        Calculate the frequency of a specific named entity type in a text column.

        Args:
            column (str): The text column to analyze.
            entity_type (str): The entity type to count (e.g., PERSON, ORG, DATE).
            model (str): NLP model for NER ('spacy').

        Returns:
            dict: Frequency distribution of named entities of the specified type.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        import spacy
        nlp = spacy.load('en_core_web_sm')

        entity_counts = {}

        for text in self.data[column].dropna():
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == entity_type:
                    entity_counts[ent.text] = entity_counts.get(ent.text, 0) + 1

        return dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True))

    def topic_modeling(self, column, n_topics=5, n_top_words=5):
        """
        Perform topic modeling on a text column using Latent Dirichlet Allocation (LDA).

        Args:
            column (str): The text column to analyze.
            n_topics (int): Number of topics to identify.
            n_top_words (int): Number of top words per topic.

        Returns:
            list: A list of topics with top words.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        text_data = self.data[column].dropna().astype(str).tolist()
        vectorizer = CountVectorizer(stop_words='english')
        text_matrix = vectorizer.fit_transform(text_data)

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(text_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics.append(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
        return topics

    def named_entity_recognition(self, column, model='spacy', entity_types=None):
        """
        Extract named entities like names, organizations, or dates from text.

        Args:
            column (str): The name of the text column to analyze.
            model (str): NLP model for NER ('spacy' or 'nltk').
            entity_types (list): List of entity types to filter (e.g., ['PERSON', 'ORG', 'DATE']).

        Returns:
            pd.Series: A pandas Series containing dictionaries of extracted entities for each row.

        Raises:
            ValueError: If the column does not exist, is not string type, or invalid model is specified.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        if model not in ['spacy', 'nltk']:
            raise ValueError("Invalid model. Choose from 'spacy' or 'nltk'.")

        import nltk
        import re
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)

        results = []

        if model == 'spacy':
            import spacy
            nlp = spacy.load('en_core_web_sm')

            def extract_entities_spacy(text):
                if pd.isnull(text) or not text.strip():
                    return {}
                doc = nlp(text)
                entities = {ent.label_: [] for ent in doc.ents}
                for ent in doc.ents:
                    if not entity_types or ent.label_ in entity_types:
                        entities.setdefault(ent.label_, []).append(ent.text)
                return entities

            results = self.data[column].apply(extract_entities_spacy)

        elif model == 'nltk':
            from nltk import word_tokenize, pos_tag, ne_chunk
            from nltk.tree import Tree

            def extract_entities_nltk(text):
                if pd.isnull(text) or not text.strip():
                    return {}
                tokens = word_tokenize(text)
                tagged = pos_tag(tokens)
                chunked = ne_chunk(tagged)
                entities = {}
                for subtree in chunked:
                    if isinstance(subtree, Tree):
                        entity_label = subtree.label()
                        entity_text = " ".join([token for token, pos in subtree.leaves()])
                        if not entity_types or entity_label in entity_types:
                            entities.setdefault(entity_label, []).append(entity_text)
                return entities

            results = self.data[column].apply(extract_entities_nltk)

        return results

    def text_tokenization(self, column, level='word', language='english'):
        """
        Tokenize text into words or sentences for pre-processing.

        Args:
            column (str): The name of the text column to tokenize.
            level (str): Level of tokenization - 'word' or 'sentence'.
            language (str): Language of the text (default is 'english').

        Returns:
            pd.Series: A pandas Series containing tokenized words or sentences for each row.

        Raises:
            ValueError: If the column does not exist or is not of string type.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize

        nltk.download('punkt', quiet=True)  # Ensure tokenizers are downloaded

        if level == 'word':
            return self.data[column].fillna("").apply(lambda x: word_tokenize(x, language=language))
        elif level == 'sentence':
            return self.data[column].fillna("").apply(lambda x: sent_tokenize(x, language=language))
        else:
            raise ValueError("Invalid level. Choose from 'word' or 'sentence'.")

    def check_text_similarity(self, column, similarity_threshold=0.8):
        """
        Identify pairs of text entries in a column with high similarity.

        Args:
            column (str): Name of the column to check.
            similarity_threshold (float): Threshold for similarity (0 to 1).

        Returns:
            list of tuples: Pairs of similar text entries.
        """
        from difflib import SequenceMatcher

        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        text_data = self.data[column].dropna().astype(str).tolist()
        similar_pairs = []

        for i, text1 in enumerate(text_data):
            for j, text2 in enumerate(text_data):
                if i < j:  # Avoid duplicate comparisons
                    similarity = SequenceMatcher(None, text1, text2).ratio()
                    if similarity >= similarity_threshold:
                        similar_pairs.append((text1, text2, similarity))
        return similar_pairs

    def check_text_length(self, column, max_length=255):
        return self.data[self.data[column].str.len() > max_length]

    def sentiment_analysis(self, column):
        return self.data[column].apply(lambda x: TextBlob(x).sentiment)

    def detect_language(self, column):
        return self.data[column].apply(lambda x: detect(x) if pd.notnull(x) else None)

    def compute_tfidf(self, column, max_features=100):
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data[column].dropna())
        return pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    def text_similarity(self, column):
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.data[column])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return pd.DataFrame(similarity_matrix)

    def count_stopwords(self, column, language="english"):
        stop_words = set(stopwords.words(language))
        return self.data[column].apply(lambda x: sum(1 for w in str(x).split() if w.lower() in stop_words))

    def n_gram_analysis(self, column, n=2):
        vectorizer = CountVectorizer(ngram_range=(n, n))
        n_grams = vectorizer.fit_transform(self.data[column].dropna())
        return dict(sorted(Counter(vectorizer.vocabulary_).items(), key=lambda x: x[1]))

    def frequent_words(self, col):
        all_words = self.data[col].str.split().explode()
        return Counter(all_words).most_common()

    def category_feature_interaction(self, categorical_column, numeric_column):
        """
        Analyze interaction between categorical and numeric columns.
        """
        if categorical_column not in self.data.columns or numeric_column not in self.data.columns:
            raise ValueError("One or both specified columns do not exist.")
        interaction_stats = self.data.groupby(categorical_column)[numeric_column].describe()
        return interaction_stats

    def word_frequency(self, column, top_n=10):
        words = self.data[column].dropna().str.split().explode()
        return Counter(words).most_common(top_n)

    def check_common_words(self, column, top_n=10):
        """Identify the most common words in a text column."""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        vectorizer = CountVectorizer(stop_words=stopwords.words("english"))
        word_counts = vectorizer.fit_transform(self.data[column].dropna())
        word_freq = pd.DataFrame(
            word_counts.toarray(), columns=vectorizer.get_feature_names_out()
        ).sum().sort_values(ascending=False).head(top_n)
        return word_freq

    def compute_cosine_similarity(self, column):
        """
        Compute pairwise cosine similarity for text entries.
        """
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.data[column].dropna())
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return pd.DataFrame(similarity_matrix)

    def analyze_text_length(self, column, min_length=5, max_length=500):
        """
        Analyze the length of text entries.
        """
        lengths = self.data[column].str.len()
        return self.data[(lengths < min_length) | (lengths > max_length)]
