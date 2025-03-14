import holidays
import numpy as np
from scipy.stats import ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import json
import phonenumbers
import gensim.downloader as api
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, chisquare
from itertools import combinations
from scipy.stats import skew, kurtosis, kstest
from scipy.signal import find_peaks
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
from textblob import TextBlob
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from difflib import SequenceMatcher
from scipy.stats import levene, bartlett
from scipy.stats import shapiro
from sklearn.feature_selection import mutual_info_classif
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import ruptures as rpt
from scipy.fftpack import fft
import textstat
import matplotlib.pyplot as plt
import seaborn as sns


# 1. DataQualityChecker Class (20 methods)
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
            "duplicates": self.detect_duplicates(),
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

    def detect_row_level_drift(self, reference_data, key_column):
        """
        Detect row-level drift by comparing current data with reference data.

        Args:
            reference_data (pd.DataFrame): The reference dataset.
            key_column (str): The column representing unique keys.

        Returns:
            pd.DataFrame: Rows with mismatched data.
        """
        merged = self.data.merge(reference_data, on=key_column, suffixes=("_current", "_reference"))
        drifted_rows = merged[~merged.filter(like="_current").equals(merged.filter(like="_reference"))]
        return drifted_rows

    def validate_aggregation(self, raw_data, groupby_columns, aggregation_rules):
        """
        Validate the accuracy of aggregations against raw data.

        Args:
            raw_data (pd.DataFrame): The raw dataset to validate against.
            groupby_columns (list): Columns to group by for aggregation.
            aggregation_rules (dict): Dictionary defining aggregation logic (e.g., {col: "sum"}).

        Returns:
            pd.DataFrame: Aggregated rows that do not match.
        """
        aggregated = raw_data.groupby(groupby_columns).agg(aggregation_rules).reset_index()
        mismatched = aggregated[~aggregated.isin(self.data.to_dict("list")).all(axis=1)]
        return mismatched

    def validate_data_types_across_partitions(self, partition_column):
        """
        Validate data types across partitions.

        Args:
            partition_column (str): The column defining partitions.

        Returns:
            dict: A dictionary with partitions as keys and inconsistent columns as values.
        """
        type_issues = {}
        for partition, group in self.data.groupby(partition_column):
            for col in group.columns:
                types = group[col].map(type).unique()
                if len(types) > 1:
                    if partition not in type_issues:
                        type_issues[partition] = []
                    type_issues[partition].append(col)
        return type_issues

    def check_data_integrity_after_joins(self, reference_data, join_keys):
        """
        Check for data loss or duplication after joins.

        Args:
            reference_data (pd.DataFrame): The dataset to compare after the join.
            join_keys (list): The keys used for the join.

        Returns:
            dict: Rows missing or duplicated after the join.
        """
        merged = self.data.merge(reference_data, on=join_keys, how="outer", indicator=True)
        missing_rows = merged[merged["_merge"] == "left_only"]
        duplicated_rows = merged[merged["_merge"] == "both"].duplicated(subset=join_keys, keep=False)
        return {"missing_rows": missing_rows, "duplicated_rows": duplicated_rows}

    def detect_overlapping_categories(self, columns):
        """
        Detect overlapping categories in multiple categorical columns.

        Args:
            columns (list): List of column names to check for overlapping categories.

        Returns:
            dict: Overlapping categories between columns.
                  Keys are tuples of column pairs, and values are lists of overlapping categories.
        """
        if not all(col in self.data.columns for col in columns):
            raise ValueError("One or more specified columns do not exist in the dataset.")

        # Get unique values from each column
        category_sets = {col: set(self.data[col].dropna().unique()) for col in columns}
        overlaps = {}

        # Check for overlaps between all pairs of columns
        for col1, col2 in combinations(columns, 2):
            overlap = category_sets[col1].intersection(category_sets[col2])
            if overlap:
                overlaps[(col1, col2)] = list(overlap)

        return overlaps

    def validate_column_relationships(self, column_pairs, relationship_fn):
        """
        Validate logical relationships between column pairs.

        Args:
            column_pairs (list): List of tuples specifying column pairs (col1, col2).
            relationship_fn (callable): A function defining the expected relationship.

        Returns:
            pd.DataFrame: Rows where the relationship is violated.
        """
        violations = pd.DataFrame()
        for col1, col2 in column_pairs:
            if col1 not in self.data.columns or col2 not in self.data.columns:
                raise ValueError(f"Columns '{col1}' or '{col2}' do not exist.")
            invalid_rows = self.data[~self.data.apply(lambda row: relationship_fn(row[col1], row[col2]), axis=1)]
            violations = pd.concat([violations, invalid_rows])
        return violations

    def detect_multiclass_imbalance(self, column, imbalance_threshold=0.1):
        """
        Detect class imbalance in a multi-class categorical column.

        Args:
            column (str): The name of the column to analyze.
            imbalance_threshold (float): Threshold for detecting imbalance (default: 10%).

        Returns:
            dict: Classes with percentages below the imbalance threshold.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        class_distribution = self.data[column].value_counts(normalize=True)
        return class_distribution[class_distribution < imbalance_threshold].to_dict()

    def detect_inconsistent_casing(self, column):
        """
        Detect inconsistent casing in a text column.

        Args:
            column (str): The column to check.

        Returns:
            pd.Series: Unique values with inconsistent casing.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        lowercased = self.data[column].str.lower().dropna()
        duplicates = lowercased.duplicated(keep=False)
        return self.data[column][duplicates].unique()

    def detect_date_granularity_inconsistencies(self, column):
        """
        Detect inconsistent date granularity in a date column.

        Args:
            column (str): The column to check.

        Returns:
            pd.DataFrame: Rows with inconsistent date granularity.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        self.data[column] = pd.to_datetime(self.data[column], errors='coerce')
        granularities = self.data[column].dropna().dt.strftime('%Y-%m-%d').str.split('-').str.len()
        inconsistent_rows = self.data[granularities != granularities.mode()[0]]
        return inconsistent_rows

    def check_null_proportions_by_group(self, column, group_by):
        """
        Check for significant differences in null proportions across groups.

        Args:
            column (str): The column to analyze for nulls.
            group_by (str): The column to group by.

        Returns:
            pd.Series: Proportions of nulls for each group.
        """
        if column not in self.data.columns or group_by not in self.data.columns:
            raise ValueError(f"Columns '{column}' or '{group_by}' do not exist.")
        null_proportions = self.data.groupby(group_by)[column].apply(lambda x: x.isnull().mean())
        return null_proportions

    def detect_duplicates_in_subset(self, subset_columns):
        """
        Detect duplicate rows based on a subset of columns.

        Args:
            subset_columns (list): List of columns to check for duplicates.

        Returns:
            pd.DataFrame: Rows that are duplicates within the subset.
        """
        if not all(col in self.data.columns for col in subset_columns):
            raise ValueError("One or more specified columns do not exist in the dataset.")
        duplicates = self.data[self.data.duplicated(subset=subset_columns, keep=False)]
        return duplicates


# 2. StatisticalAnalyzer Class (6 methods)
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

    def detect_skewness_kurtosis(self, column):
        """
        Detect skewness and kurtosis for a given numerical column.

        Args:
            column (str): The column name to analyze.

        Returns:
            dict: A dictionary containing:
                - 'skewness': Degree of asymmetry (positive = right-skewed, negative = left-skewed).
                - 'kurtosis': Measure of tail heaviness (high = heavy tails, low = light tails).

        Raises:
            ValueError: If the column is not numeric.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric.")

        col_data = self.data[column].dropna()

        return {
            "skewness": skew(col_data),
            "kurtosis": kurtosis(col_data)
        }

    def check_normality(self, column):
        """
        Perform normality tests using Shapiro-Wilk and KS tests.

        Args:
            column (str): The column to analyze.

        Returns:
            dict: A dictionary with p-values for the normality tests.

        Raises:
            ValueError: If the column is not numeric.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric.")

        col_data = self.data[column].dropna()

        shapiro_p = shapiro(col_data)[1]  # Shapiro-Wilk test
        ks_p = kstest(col_data, 'norm')[1]  # Kolmogorov-Smirnov test

        return {
            "shapiro_p_value": shapiro_p,
            "ks_p_value": ks_p,
            "normal": shapiro_p > 0.05 and ks_p > 0.05  # Higher p-value means likely normal
        }

    def detect_multimodal_distribution(self, column, min_distance=10):
        """
        Detect if a numerical column has a multimodal distribution.

        Args:
            column (str): The column name to analyze.
            min_distance (int): Minimum distance between peaks.

        Returns:
            dict: A dictionary containing:
                - 'num_peaks': Number of peaks detected.
                - 'peak_positions': Values at the peak positions.

        Raises:
            ValueError: If the column is not numeric.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric.")

        col_data = self.data[column].dropna()
        hist_values, bin_edges = np.histogram(col_data, bins=30)

        peaks, _ = find_peaks(hist_values, distance=min_distance)

        return {
            "num_peaks": len(peaks),
            "peak_positions": bin_edges[peaks].tolist()
        }

    def measure_data_spread(self, column):
        """
        Measure the spread of numerical data using IQR and variance.

        Args:
            column (str): The column name to analyze.

        Returns:
            dict: A dictionary containing:
                - 'variance': The variance of the column.
                - 'iqr': The interquartile range (IQR).

        Raises:
            ValueError: If the column is not numeric.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric.")

        col_data = self.data[column].dropna()
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1

        return {
            "variance": np.var(col_data, ddof=1),
            "iqr": iqr
        }

    def compute_population_stability_index(self, baseline_data, column, bins=10):
        """
        Compute Population Stability Index (PSI) to detect data drift.

        Args:
            baseline_data (pd.DataFrame): Reference dataset.
            column (str): The column to analyze.
            bins (int): Number of bins for frequency calculation.

        Returns:
            float: PSI value indicating drift (Higher PSI = More drift).

        Raises:
            ValueError: If the column is not numeric or does not exist.
        """
        if column not in self.data.columns or column not in baseline_data.columns:
            raise ValueError(f"Column '{column}' must exist in both datasets.")

        if not pd.api.types.is_numeric_dtype(self.data[column]) or not pd.api.types.is_numeric_dtype(
                baseline_data[column]):
            raise ValueError(f"Column '{column}' must be numeric in both datasets.")

        current_values = self.data[column].dropna()
        baseline_values = baseline_data[column].dropna()

        # Create bins
        bin_edges = np.histogram_bin_edges(np.concatenate([current_values, baseline_values]), bins=bins)

        # Calculate distributions
        current_hist, _ = np.histogram(current_values, bins=bin_edges)
        baseline_hist, _ = np.histogram(baseline_values, bins=bin_edges)

        # Normalize
        current_hist = current_hist / sum(current_hist)
        baseline_hist = baseline_hist / sum(baseline_hist)

        # Avoid division by zero
        current_hist = np.where(current_hist == 0, 0.0001, current_hist)
        baseline_hist = np.where(baseline_hist == 0, 0.0001, baseline_hist)

        psi = np.sum((baseline_hist - current_hist) * np.log(baseline_hist / current_hist))

        return psi

    def check_homoscedasticity(self, column, group_column, test='levene'):
        """
        Checks if different groups have equal variance (homoscedasticity).

        Args:
            column (str): The numeric column to check.
            group_column (str): The categorical column defining groups.
            test (str): Statistical test to use ('levene' or 'bartlett').

        Returns:
            dict: Containing test statistic and p-value.

        Raises:
            ValueError: If the column is not numeric or group_column is not categorical.
        """
        if column not in self.data.columns or group_column not in self.data.columns:
            raise ValueError(f"Columns '{column}' or '{group_column}' do not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        if not pd.api.types.is_object_dtype(self.data[group_column]) and not pd.api.types.is_categorical_dtype(
                self.data[group_column]):
            raise ValueError(f"Column '{group_column}' must be categorical.")

        groups = [group[column].dropna() for _, group in self.data.groupby(group_column)]

        if test == 'levene':
            test_statistic, p_value = levene(*groups)
        elif test == 'bartlett':
            test_statistic, p_value = bartlett(*groups)
        else:
            raise ValueError("Invalid test. Choose 'levene' or 'bartlett'.")

        return {
            "test_statistic": test_statistic,
            "p_value": p_value,
            "equal_variance": p_value > 0.05
        }

    def check_monotonicity(self, column):
        """
        Checks whether a numeric column follows a monotonic increasing or decreasing trend.

        Args:
            column (str): The numeric column to check.

        Returns:
            dict: Containing whether the trend is increasing, decreasing, or neither.

        Raises:
            ValueError: If the column is not numeric.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        col_data = self.data[column].dropna()
        diffs = np.diff(col_data)

        return {
            "monotonic_increasing": np.all(diffs >= 0),
            "monotonic_decreasing": np.all(diffs <= 0),
            "strictly_monotonic": np.all(diffs > 0) or np.all(diffs < 0)
        }

    def check_multivariate_normality(self, columns):
        """
        Check if a set of numerical columns follows a multivariate normal distribution.

        Args:
            columns (list): List of numeric columns.

        Returns:
            dict: Containing p-values from Shapiro-Wilk test for each column.

        Raises:
            ValueError: If the columns do not exist or are not numeric.
        """
        for column in columns:
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' does not exist.")
            if not pd.api.types.is_numeric_dtype(self.data[column]):
                raise ValueError(f"Column '{column}' must be numeric.")

        results = {col: shapiro(self.data[col].dropna())[1] for col in columns}

        return {
            "p_values": results,
            "multivariate_normal": all(p > 0.05 for p in results.values())
        }

    def compute_cohens_d(self, column, group_column):
        """
        Computes Cohen’s D effect size between two groups.

        Args:
            column (str): The numeric column to analyze.
            group_column (str): The categorical column defining groups.

        Returns:
            float: Cohen’s D effect size.

        Raises:
            ValueError: If the column is not numeric or group_column is not categorical.
        """
        if column not in self.data.columns or group_column not in self.data.columns:
            raise ValueError(f"Columns '{column}' or '{group_column}' do not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        if not pd.api.types.is_object_dtype(self.data[group_column]) and not pd.api.types.is_categorical_dtype(
                self.data[group_column]):
            raise ValueError(f"Column '{group_column}' must be categorical.")

        groups = self.data.groupby(group_column)[column].apply(list)

        if len(groups) != 2:
            raise ValueError("Cohen's D requires exactly two groups.")

        group1, group2 = groups.values
        mean1, mean2 = np.mean(group1), np.mean(group2)
        pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)

        return (mean1 - mean2) / pooled_std

    def compute_mutual_information(self, column, target_column):
        """
        Compute mutual information between a feature and target.

        Args:
            column (str): The independent variable.
            target_column (str): The dependent (target) variable.

        Returns:
            float: Mutual information score.

        Raises:
            ValueError: If columns do not exist or are not categorical.
        """
        if column not in self.data.columns or target_column not in self.data.columns:
            raise ValueError(f"Columns '{column}' or '{target_column}' do not exist.")

        if not pd.api.types.is_object_dtype(self.data[column]) and not pd.api.types.is_categorical_dtype(
                self.data[column]):
            raise ValueError(f"Column '{column}' must be categorical.")

        X = self.data[column].astype("category").cat.codes.values.reshape(-1, 1)
        y = self.data[target_column]

        return mutual_info_classif(X, y)[0]


# 3. TimeSeriesAnalyzer Class (3 methods)
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

    def exponential_moving_average(self, column, span=10):
        """
        Computes the Exponential Moving Average (EMA) to smooth time-series data.

        Args:
            column (str): The numeric column containing time-series data.
            span (int): The span for the EMA.

        Returns:
            pd.Series: The computed EMA values.

        Raises:
            ValueError: If column is missing or invalid.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        ema = self.data[column].ewm(span=span, adjust=False).mean()
        return ema

    def seasonal_strength(self, column, frequency):
        """
        Measures the strength of seasonality in time series.

        Args:
            column (str): The numeric column containing time-series data.
            frequency (int): Seasonal period (e.g., 12 for monthly data).

        Returns:
            float: Strength of seasonality (0 = no seasonality, 1 = strong seasonality).

        Raises:
            ValueError: If column is missing or invalid.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        ts_data = self.data[column].dropna()
        moving_avg = ts_data.rolling(window=frequency, center=True).mean()
        residuals = ts_data - moving_avg
        strength = 1 - (residuals.var() / ts_data.var())

        return max(0, strength)

    def rolling_window_forecast(self, column, window=12):
        """
        Forecasts future values using a rolling average.

        Args:
            column (str): The numeric column containing time-series data.
            window (int): The rolling window size.

        Returns:
            pd.Series: Forecasted values.

        Raises:
            ValueError: If column is missing or invalid.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        forecast = self.data[column].rolling(window=window).mean().shift(1)
        return forecast

    def fourier_transform_analysis(self, column):
        """
        Performs Fourier Transform to analyze dominant frequencies in time-series data.

        Args:
            column (str): The numeric column containing time-series data.

        Returns:
            tuple: Frequencies and corresponding amplitudes.

        Raises:
            ValueError: If column is missing or invalid.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        ts_data = self.data[column].dropna().values
        n = len(ts_data)
        frequencies = np.fft.fftfreq(n)
        amplitudes = np.abs(fft(ts_data))

        plt.figure(figsize=(10, 5))
        plt.plot(frequencies[:n // 2], amplitudes[:n // 2])
        plt.title(f"Fourier Transform of {column}")
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.show()

        return frequencies[:n // 2], amplitudes[:n // 2]

    def detect_missing_timestamps(self, timestamp_column, freq='D'):
        """
        Detects missing time intervals in a time-series dataset.

        Args:
            timestamp_column (str): The timestamp column.
            freq (str): Frequency ('D' for daily, 'H' for hourly, etc.).

        Returns:
            list: Missing timestamps.

        Raises:
            ValueError: If column is missing or not a datetime.
        """
        if timestamp_column not in self.data.columns:
            raise ValueError(f"Column '{timestamp_column}' does not exist.")

        self.data[timestamp_column] = pd.to_datetime(self.data[timestamp_column])
        complete_range = pd.date_range(start=self.data[timestamp_column].min(), end=self.data[timestamp_column].max(),
                                       freq=freq)
        missing_timestamps = set(complete_range) - set(self.data[timestamp_column])

        return sorted(missing_timestamps)

    def autoregressive_forecast(self, column, lags=3, steps=5):
        """
        Forecasts time series using an Autoregressive (AR) model.

        Args:
            column (str): The numeric column containing time-series data.
            lags (int): Number of lags for the AR model.
            steps (int): Number of future steps to predict.

        Returns:
            pd.Series: Forecasted values.

        Raises:
            ValueError: If column is missing or invalid.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        ts_data = self.data[column].dropna()
        model = AutoReg(ts_data, lags=lags).fit()
        forecast = model.predict(start=len(ts_data), end=len(ts_data) + steps - 1)

        return forecast

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
        decomp = seasonal_decompose(series, model='additive', period=frequency)
        decomp.plot()
        plt.show()
        return decomp

    def detect_anomalies_zscore(self, column, threshold=3.0):
        """
        Detects anomalies in time series using Z-score method.

        Args:
            column (str): The numeric column containing time-series data.
            threshold (float): The Z-score threshold for anomaly detection.

        Returns:
            pd.DataFrame: Data points flagged as anomalies.

        Raises:
            ValueError: If column does not exist or is not numeric.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        col_data = self.data[column].dropna()
        z_scores = (col_data - col_data.mean()) / col_data.std()
        anomalies = self.data.loc[z_scores.abs() > threshold]

        return anomalies

    def check_serial_correlation(self, column, lags=10):
        """
        Tests for autocorrelation in a time series.

        Args:
            column (str): The time series column.
            lags (int): Number of lag observations to check.

        Returns:
            dict: Autocorrelation values for specified lags.

        Raises:
            ValueError: If the column does not exist or is not numeric.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        col_data = self.data[column].dropna()

        return {
            "autocorrelation": acf(col_data, nlags=lags).tolist()
        }

    def identify_seasonality(self, column, lags=50):
        """
        Identifies seasonality in time series using ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function).

        Args:
            column (str): The time-series column.
            lags (int): Number of lags to consider for seasonality.

        Returns:
            None: Displays ACF and PACF plots.

        Raises:
            ValueError: If the column is not numeric or does not exist.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        col_data = self.data[column].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_acf(col_data, lags=lags, ax=axes[0])
        plot_pacf(col_data, lags=lags, ax=axes[1])

        axes[0].set_title(f"Autocorrelation Function (ACF) for {column}")
        axes[1].set_title(f"Partial Autocorrelation Function (PACF) for {column}")

        plt.tight_layout()
        plt.show()

    def holt_winters_forecast(self, column, timestamp_column, periods=12, seasonal='add', trend='add'):
        """
        Forecasts time series using Holt-Winters Exponential Smoothing.

        Args:
            column (str): The time-series column.
            timestamp_column (str): The timestamp column.
            periods (int): Number of future periods to forecast.
            seasonal (str): Seasonal component ('add' or 'mul').
            trend (str): Trend component ('add' or 'mul').

        Returns:
            pd.DataFrame: A DataFrame containing actual and forecasted values.

        Raises:
            ValueError: If columns do not exist or contain invalid data.
        """
        if column not in self.data.columns or timestamp_column not in self.data.columns:
            raise ValueError(f"Columns '{column}' or '{timestamp_column}' do not exist.")

        self.data[timestamp_column] = pd.to_datetime(self.data[timestamp_column])
        self.data.set_index(timestamp_column, inplace=True)
        self.data.sort_index(inplace=True)

        ts_data = self.data[column].dropna()

        model = ExponentialSmoothing(ts_data, trend=trend, seasonal=seasonal, seasonal_periods=periods)
        fit = model.fit()

        forecast_index = pd.date_range(start=ts_data.index[-1], periods=periods, freq='M')
        forecast_values = fit.forecast(periods)

        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(ts_data.index, ts_data, label="Actual", color='blue')
        plt.plot(forecast_index, forecast_values, label="Forecast", color='red', linestyle='dashed')
        plt.title(f"Holt-Winters Forecast for {column}")
        plt.xlabel("Time")
        plt.ylabel(column)
        plt.legend()
        plt.show()

        return pd.DataFrame({"Timestamp": forecast_index, "Forecast": forecast_values})

    def detect_spikes(self, column, threshold=2.0):
        """
        Detects sudden spikes or dips in time-series data.

        Args:
            column (str): The numeric column containing time-series data.
            threshold (float): Multiple of standard deviation to flag as a spike.

        Returns:
            pd.DataFrame: Rows where a spike or dip occurs.

        Raises:
            ValueError: If column does not exist or is not numeric.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        col_data = self.data[column].dropna()
        diffs = col_data.diff().abs()

        spikes = self.data.loc[diffs > threshold * diffs.std()]
        return spikes

    def cross_correlation(self, column1, column2, max_lag=10):
        """
        Computes cross-correlation between two time series columns.

        Args:
            column1 (str): First time series column.
            column2 (str): Second time series column.
            max_lag (int): Maximum lag to compute correlation.

        Returns:
            dict: Cross-correlation values at different lags.

        Raises:
            ValueError: If columns do not exist or are not numeric.
        """
        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise ValueError(f"Columns '{column1}' and '{column2}' must exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column1]) or not pd.api.types.is_numeric_dtype(
                self.data[column2]):
            raise ValueError(f"Both columns must be numeric.")

        col1_data = self.data[column1].dropna()
        col2_data = self.data[column2].dropna()

        lags = range(-max_lag, max_lag + 1)
        correlations = [col1_data.corr(col2_data.shift(lag)) for lag in lags]

        plt.figure(figsize=(10, 5))
        plt.stem(lags, correlations, use_line_collection=True)
        plt.xlabel("Lag")
        plt.ylabel("Cross-Correlation")
        plt.title(f"Cross-Correlation between {column1} and {column2}")
        plt.axhline(y=0, color='black', linestyle='--')
        plt.grid(True)
        plt.show()

        return dict(zip(lags, correlations))

    def check_weekend_holiday_effects(self, column, timestamp_column, country='US'):
        """
        Analyzes whether weekends or holidays impact the time-series values.

        Args:
            column (str): The numeric column containing time-series data.
            timestamp_column (str): The timestamp column.
            country (str): Country code for holidays (default: 'US').

        Returns:
            dict: Average values for weekdays, weekends, and holidays.

        Raises:
            ValueError: If columns do not exist or are invalid.
        """
        if column not in self.data.columns or timestamp_column not in self.data.columns:
            raise ValueError(f"Columns '{column}' or '{timestamp_column}' do not exist.")

        self.data[timestamp_column] = pd.to_datetime(self.data[timestamp_column])
        self.data['day_of_week'] = self.data[timestamp_column].dt.dayofweek
        self.data['is_weekend'] = self.data['day_of_week'] >= 5

        country_holidays = holidays.country_holidays(country)
        self.data['is_holiday'] = self.data[timestamp_column].apply(lambda x: x in country_holidays)

        averages = {
            "weekday_avg": self.data.loc[~self.data['is_weekend'], column].mean(),
            "weekend_avg": self.data.loc[self.data['is_weekend'], column].mean(),
            "holiday_avg": self.data.loc[self.data['is_holiday'], column].mean(),
        }

        return averages

    def detect_structural_breaks(self, column, model="l2", penalty=5):
        """
        Detects structural breaks (abrupt changes in trend) in a time series.

        Args:
            column (str): The numeric column containing time-series data.
            model (str): Model for detecting change points ('l1', 'l2', 'rbf').
            penalty (int): Penalty value for detecting change points.

        Returns:
            list: Indices of detected structural breaks.

        Raises:
            ValueError: If column is missing or invalid.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        ts_data = self.data[column].dropna().values
        algo = rpt.Pelt(model=model).fit(ts_data)
        change_points = algo.predict(pen=penalty)

        plt.figure(figsize=(12, 6))
        plt.plot(ts_data, label="Time-Series Data")
        for cp in change_points[:-1]:
            plt.axvline(cp, color='red', linestyle="--", label="Structural Break")
        plt.title(f"Structural Breaks in {column}")
        plt.legend()
        plt.show()

        return change_points


# 4. NLPAnalyzer Class (14 methods)
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

        return self.data[column].apply(lambda x: str(TextBlob(x).correct()) if pd.notnull(x) else x)

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

    def check_text_length(self, column, max_length=255):
        return self.data[self.data[column].str.len() > max_length]

    def detect_language(self, column):
        """
        Detect the language of text data.

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.Series: Detected language codes.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        return self.data[column].dropna().apply(lambda x: detect(x))

    def count_stopwords(self, column, language="english"):
        stop_words = set(stopwords.words(language))
        return self.data[column].apply(lambda x: sum(1 for w in str(x).split() if w.lower() in stop_words))

    def category_feature_interaction(self, categorical_column, numeric_column):
        """
        Analyze interaction between categorical and numeric columns.
        """
        if categorical_column not in self.data.columns or numeric_column not in self.data.columns:
            raise ValueError("One or both specified columns do not exist.")
        interaction_stats = self.data.groupby(categorical_column)[numeric_column].describe()
        return interaction_stats

    def lexical_diversity(self, column, mode="row"):
        """
        Compute lexical diversity (ratio of unique words to total words).

        Args:
            column (str): The text column to analyze.
            mode (str): "row" for row-wise diversity (default), "overall" for dataset-wide diversity.

        Returns:
            float or pd.Series:
                - If mode="row": Returns a Pandas Series with lexical diversity scores for each row.
                - If mode="overall": Returns a single float representing lexical diversity for the entire dataset.

        Raises:
            ValueError: If the column does not exist or mode is invalid.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        # Tokenize and flatten words for row-wise computation
        if mode == "row":
            return self.data[column].dropna().apply(
                lambda text: len(set(text.split())) / len(text.split()) if text.strip() else 0
            )

        # Compute overall lexical diversity for the entire dataset
        elif mode == "overall":
            all_words = self.data[column].dropna().str.split().explode()
            unique_words = set(all_words)
            return len(unique_words) / len(all_words) if len(all_words) > 0 else 0

        else:
            raise ValueError(
                "Invalid mode. Choose 'row' for per-row diversity or 'overall' for dataset-wide diversity.")

    def named_entity_consistency(self, column, entity_type="ORG"):
        """
        Detect inconsistent usage of named entities.

        Args:
            column (str): The text column to analyze.
            entity_type (str): Type of named entity (e.g., 'ORG', 'PERSON', 'GPE').

        Returns:
            dict: Entities with inconsistent casing or spelling variations.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        nlp = spacy.load("en_core_web_sm")
        entity_dict = {}

        for text in self.data[column].dropna():
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == entity_type:
                    entity_dict.setdefault(ent.text.lower(), set()).add(ent.text)

        return {k: list(v) for k, v in entity_dict.items() if len(v) > 1}

    def subjectivity_analysis(self, column):
        """
        Compute subjectivity scores (0 = objective, 1 = highly subjective).

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.Series: Subjectivity scores.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        return self.data[column].dropna().apply(lambda x: TextBlob(x).sentiment.subjectivity)

    def word_length_distribution(self, column):
        """
        Compute and visualize the distribution of word lengths.

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.Series: Word length frequency distribution.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        word_lengths = self.data[column].dropna().apply(lambda x: [len(word) for word in x.split()])
        all_lengths = [length for sublist in word_lengths for length in sublist]

        plt.figure(figsize=(10, 5))
        sns.histplot(all_lengths, bins=20, kde=True)
        plt.title("Word Length Distribution")
        plt.xlabel("Word Length")
        plt.ylabel("Frequency")
        plt.show()

        return pd.Series(all_lengths).value_counts().sort_index()

    def sentence_length_distribution(self, column):
        """
        Compute and visualize sentence length distribution.

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.Series: Sentence length frequency distribution.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        nltk.download("punkt", quiet=True)

        sentence_lengths = self.data[column].dropna().apply(lambda x: [len(sent.split()) for sent in sent_tokenize(x)])
        all_lengths = [length for sublist in sentence_lengths for length in sublist]

        plt.figure(figsize=(10, 5))
        sns.histplot(all_lengths, bins=20, kde=True)
        plt.title("Sentence Length Distribution")
        plt.xlabel("Number of Words in Sentence")
        plt.ylabel("Frequency")
        plt.show()

        return pd.Series(all_lengths).value_counts().sort_index()

    def character_count_distribution(self, column):
        """
        Compute and visualize the distribution of character counts.

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.Series: Character count distribution.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        char_counts = self.data[column].dropna().apply(len)

        plt.figure(figsize=(10, 5))
        sns.histplot(char_counts, bins=20, kde=True)
        plt.title("Character Count Distribution")
        plt.xlabel("Character Count")
        plt.ylabel("Frequency")
        plt.show()

        return char_counts.value_counts().sort_index()

    from sklearn.feature_extraction.text import CountVectorizer

    def n_gram_distribution(self, column, n=2, top_n=20):
        """
        Identify most common n-grams in text data.

        Args:
            column (str): The text column to analyze.
            n (int): Size of the n-gram (2 for bigrams, 3 for trigrams).
            top_n (int): Number of top n-grams to display.

        Returns:
            dict: Most common n-grams with counts.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words="english")
        n_grams = vectorizer.fit_transform(self.data[column].dropna())

        n_gram_counts = dict(zip(vectorizer.get_feature_names_out(), n_grams.toarray().sum(axis=0)))
        sorted_n_grams = dict(sorted(n_gram_counts.items(), key=lambda item: item[1], reverse=True)[:top_n])

        return sorted_n_grams

    def pos_distribution(self, column):
        """
        Compute the distribution of parts of speech (POS) in text data.

        Args:
            column (str): The text column to analyze.

        Returns:
            dict: POS tag counts.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        # Load pre-trained spaCy NLP model
        nlp = spacy.load("en_core_web_sm")
        pos_counts = Counter()

        # Process each text entry in the column
        for text in self.data[column].dropna():
            doc = nlp(text)
            pos_counts.update([token.pos_ for token in doc])  # Count POS tags

        # Plot POS distribution
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(pos_counts.keys()), y=list(pos_counts.values()))
        plt.title("Part-of-Speech (POS) Distribution")
        plt.xlabel("POS Tag")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

        return dict(pos_counts)

    def check_text_redundancy(self, column, n=3):
        """
        Identify commonly repeated phrases in text data.

        Args:
            column (str): The text column to analyze.
            n (int): Minimum number of occurrences to consider redundancy.

        Returns:
            dict: Repeated phrases and their counts.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        phrases = self.data[column].dropna().str.split().explode()
        phrase_counts = Counter(phrases)

        return {phrase: count for phrase, count in phrase_counts.items() if count >= n}

    def named_entity_analysis(self, column, model='spacy', entity_types=None, return_frequency=False):
        """
        Perform Named Entity Recognition (NER) and optionally compute entity frequency.

        Args:
            column (str): The text column to analyze.
            model (str): NLP model for NER ('spacy' or 'nltk').
            entity_types (list, optional): List of entity types to filter (e.g., ['PERSON', 'ORG', 'DATE']).
            return_frequency (bool): If True, return entity frequency instead of per-row entity extraction.

        Returns:
            - If return_frequency=False: A pandas Series with extracted named entities for each row.
            - If return_frequency=True: A dictionary with entity frequency counts.

        Raises:
            ValueError: If the column does not exist, is not string type, or invalid model is specified.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")
        if model not in ['spacy', 'nltk']:
            raise ValueError("Invalid model. Choose from 'spacy' or 'nltk'.")

        entity_counts = Counter()
        results = []

        if model == 'spacy':
            nlp = spacy.load('en_core_web_sm')

            def extract_entities_spacy(text):
                if pd.isnull(text) or not text.strip():
                    return {}
                doc = nlp(text)
                entities = {ent.label_: [] for ent in doc.ents}
                for ent in doc.ents:
                    if not entity_types or ent.label_ in entity_types:
                        entities.setdefault(ent.label_, []).append(ent.text)
                        entity_counts[ent.text] += 1  # Count occurrences
                return entities

            results = self.data[column].apply(extract_entities_spacy)

        elif model == 'nltk':
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)

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
                            entity_counts[entity_text] += 1  # Count occurrences
                return entities

            results = self.data[column].apply(extract_entities_nltk)

        return dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)) if return_frequency else results

    def text_tokenization_analysis(self, column, level="word", n_gram=None, language="english"):
        """
        Perform text tokenization and n-gram analysis.

        Args:
            column (str): The text column to tokenize.
            level (str): "word" for word tokenization, "sentence" for sentence tokenization.
            n_gram (int): Set to an integer (e.g., 2 for bigrams) to compute n-grams.
            language (str): Language for tokenization.

        Returns:
            pd.Series or dict: Tokenized text or n-gram counts.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        nltk.download("punkt", quiet=True)

        text_data = self.data[column].dropna().astype(str)

        if n_gram:
            vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram), stop_words="english")
            n_grams = vectorizer.fit_transform(text_data)
            n_gram_counts = dict(zip(vectorizer.get_feature_names_out(), n_grams.toarray().sum(axis=0)))
            return dict(sorted(n_gram_counts.items(), key=lambda x: x[1], reverse=True))

        if level == "word":
            return text_data.apply(lambda x: word_tokenize(x, language=language))

        elif level == "sentence":
            return text_data.apply(lambda x: sent_tokenize(x, language=language))

        else:
            raise ValueError("Invalid level. Choose 'word' or 'sentence'.")

    def sentiment_analysis(self, column, return_distribution=False):
        """
        Perform sentiment analysis on a text column.

        Args:
            column (str): The text column to analyze.
            return_distribution (bool): If True, return a distribution of sentiment scores.

        Returns:
            pd.Series or dict:
                - If return_distribution=False: Returns a Series with sentiment polarity scores.
                - If return_distribution=True: Returns a histogram of sentiment scores.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        sentiments = self.data[column].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)

        if return_distribution:
            return dict(sentiments.value_counts().sort_index())

        return sentiments

    def analyze_text_complexity(self, column):
        """
        Analyze text complexity using readability scores, text length, and compression ratio.

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.DataFrame: Readability scores, text length statistics, and compression ratios.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        def compute_metrics(text):
            if pd.isnull(text):
                return None
            words = text.split()
            unique_words = set(words)
            return {
                "Text_Length": len(text),
                "Word_Count": len(words),
                "Unique_Word_Ratio": len(unique_words) / len(words) if words else 0,
                "Flesch_Reading_Ease": textstat.flesch_reading_ease(text),
                "SMOG_Index": textstat.smog_index(text),
                "Dale_Chall_Score": textstat.dale_chall_readability_score(text),
            }

        scores = self.data[column].apply(compute_metrics).dropna()
        return pd.DataFrame(scores.tolist(), index=self.data.index)

    def analyze_text_keywords(self, column, method="rake", top_n=10, exclude_stopwords=True):
        """
        Extract keywords or analyze word frequency.

        Args:
            column (str): The text column to analyze.
            method (str): Keyword extraction method ("rake" or "word_freq").
            top_n (int): Number of top results to return.
            exclude_stopwords (bool): Whether to exclude stopwords in word frequency analysis.

        Returns:
            dict: Extracted keywords or word frequency distribution.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        text_data = self.data[column].dropna().astype(str)

        if method == "rake":
            rake = Rake()
            rake.extract_keywords_from_text(" ".join(text_data))
            return {phrase: score for phrase, score in rake.get_word_degrees().items()[:top_n]}

        if method == "word_freq":
            words = text_data.str.split().explode()
            if exclude_stopwords:
                stop_words = set(stopwords.words("english"))
                words = words[~words.isin(stop_words)]
            return dict(Counter(words).most_common(top_n))

        raise ValueError("Invalid method. Choose 'rake' or 'word_freq'.")

    def analyze_text_similarity(self, column, similarity_method="word2vec", similarity_threshold=0.8, max_features=100):
        """
        Perform comprehensive text similarity analysis by combining:
        - Word2Vec similarity (pairwise comparisons)
        - High-similarity text pairs detection
        - TF-IDF vectorization for structured text comparison

        Args:
            column (str): The text column to analyze.
            similarity_method (str): Similarity method ("word2vec", "tfidf", or "cosine").
            similarity_threshold (float): Threshold for similarity detection (only for text pairs).
            max_features (int): Maximum features for TF-IDF vectorization.

        Returns:
            dict: Contains:
                - 'similar_text_pairs': List of high-similarity text pairs.
                - 'word2vec_similarity': Pairwise similarity scores (if applicable).
                - 'tfidf_matrix': DataFrame of TF-IDF vector representations.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        results = {}

        # Step 1: Find highly similar text pairs
        results["similar_text_pairs"] = self.find_text_pairs(column, similarity_threshold)

        # Step 2: Compute word2vec similarity (if selected)
        if similarity_method == "word2vec":
            results["word2vec_similarity"] = self.compute_text_similarity(column, method="word2vec")

        # Step 3: Compute TF-IDF vectorization
        results["tfidf_matrix"] = self.text_vectorization_analysis(column, method="tfidf", max_features=max_features)

        return results







