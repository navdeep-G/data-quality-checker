import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter
from langdetect import detect
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from difflib import SequenceMatcher
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import phonenumbers


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
            "duplicates": self.check_duplicates(),
            "outliers": self.check_outliers(),
        }
        return report

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
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric.")

        if metric not in ['mean', 'median', 'std']:
            raise ValueError("Invalid metric. Choose from 'mean', 'median', 'std'.")

        data = self.data[column].dropna().values
        bootstrap_estimates = []

        # Perform bootstrap sampling
        np.random.seed(42)  # For reproducibility
        for _ in range(n_iterations):
            sample = np.random.choice(data, size=len(data), replace=True)
            if metric == 'mean':
                bootstrap_estimates.append(np.mean(sample))
            elif metric == 'median':
                bootstrap_estimates.append(np.median(sample))
            elif metric == 'std':
                bootstrap_estimates.append(np.std(sample))

        # Calculate confidence interval
        lower_bound = np.percentile(bootstrap_estimates, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(bootstrap_estimates, (1 + confidence_level) / 2 * 100)
        bootstrap_estimate = np.mean(bootstrap_estimates) if metric == 'mean' else np.median(bootstrap_estimates)

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
        if test_type not in ['t-test', 'chi-squared', 'anova']:
            raise ValueError("Invalid test_type. Choose from 't-test', 'chi-squared', or 'anova'.")

        if column1 not in self.data.columns:
            raise ValueError(f"Column '{column1}' does not exist in the dataset.")

        if test_type == 't-test':
            if column2 is None or column2 not in self.data.columns:
                raise ValueError(f"Column '{column2}' must be specified and exist in the dataset for a t-test.")

            # Perform independent t-test
            from scipy.stats import ttest_ind
            group1 = self.data[column1].dropna()
            group2 = self.data[column2].dropna()
            test_statistic, p_value = ttest_ind(group1, group2, equal_var=False)

        elif test_type == 'chi-squared':
            if column2 is None or column2 not in self.data.columns:
                raise ValueError(
                    f"Column '{column2}' must be specified and exist in the dataset for a chi-squared test.")

            # Perform chi-squared test
            from scipy.stats import chi2_contingency
            contingency_table = pd.crosstab(self.data[column1], self.data[column2])
            test_statistic, p_value, _, _ = chi2_contingency(contingency_table)

        elif test_type == 'anova':
            if group_column is None or group_column not in self.data.columns:
                raise ValueError(f"Group column '{group_column}' must be specified and exist in the dataset for ANOVA.")

            # Perform one-way ANOVA
            from scipy.stats import f_oneway
            groups = [group[column1].dropna().values for _, group in self.data.groupby(group_column)]
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

        # Check for numeric column
        if pd.api.types.is_numeric_dtype(self.data[column]):
            from scipy.stats import kstest
            # Normalize data range to [0, 1]
            col_data = self.data[column].dropna()
            min_val, max_val = col_data.min(), col_data.max()
            scaled_data = (col_data - min_val) / (max_val - min_val)

            # Perform KS test against uniform distribution
            ks_stat, p_value = kstest(scaled_data, 'uniform')
            return {
                "is_uniform": p_value > p_value_threshold,
                "p_value": p_value
            }

        # Check for categorical column
        elif pd.api.types.is_object_dtype(self.data[column]):
            from scipy.stats import chisquare
            # Calculate observed frequencies
            col_data = self.data[column].dropna()
            observed_freq = col_data.value_counts()
            expected_freq = [len(col_data) / len(observed_freq)] * len(observed_freq)

            # Perform chi-squared test
            chi_stat, p_value = chisquare(observed_freq, expected_freq)
            return {
                "is_uniform": p_value > p_value_threshold,
                "p_value": p_value
            }
        else:
            raise ValueError(f"Column '{column}' is not numeric or categorical.")

    def detect_empty_columns(self):
        """
        Identify columns in the dataset that are completely empty or contain only null values.

        Returns:
            list: A list of column names that are completely empty.

        Raises:
            ValueError: If the dataset is empty.
        """
        if self.data.empty:
            raise ValueError("The dataset is empty.")

        # Check for columns where all values are null
        empty_columns = [col for col in self.data.columns if self.data[col].isnull().all()]
        return empty_columns

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

    def check_duplicates(self):
        """
        Identify duplicate rows in the dataset.

        Returns:
            pd.DataFrame: A DataFrame containing duplicate rows.

        Raises:
            ValueError: If the dataset is empty.
        """
        if self.data.empty:
            raise ValueError("Dataset is empty.")
        duplicates = self.data[self.data.duplicated()]
        return duplicates

    def check_outliers(self, threshold=3):
        """
        Detect outliers in numeric columns using the Z-score method.

        Args:
            threshold (float): The Z-score threshold for identifying outliers.

        Returns:
            pd.Series: A series containing the number of outliers per numeric column.

        Raises:
            ValueError: If no numeric columns are available in the dataset.
        """
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        if numeric_data.empty:
            raise ValueError("No numeric columns available for outlier detection.")
        outliers = ((numeric_data - numeric_data.mean()).abs() > threshold * numeric_data.std()).sum()
        return outliers[outliers > 0]

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


### 3. TimeSeriesAnalyzer Class (3 methods)
import pandas as pd
import statsmodels.tsa.seasonal as sm

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

