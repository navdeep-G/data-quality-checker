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

