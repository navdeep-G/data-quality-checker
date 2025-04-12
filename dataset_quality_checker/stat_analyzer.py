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

        test_statistic = None
        p_value = None

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
