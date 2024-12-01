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


### 1. DataQualityChecker Class (20 methods)
class DataQualityChecker:
    def __init__(self, data):
        self.data = data

    def generate_report(self):
        """
        Generate a summary report of data quality issues including missing values,
        duplicates, and outliers.
        """
        report = {
            "missing_values": self.check_missing_values(),
            "duplicates": self.check_duplicates(),
            "outliers": self.check_outliers(),
        }
        return report

    def check_cross_column_dependency(self, column1, column2, rule):
        """
        Check for violations of cross-column dependency rules.
        """
        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise ValueError("One or both specified columns do not exist in the dataset.")
        violations = self.data[~self.data.apply(lambda row: rule(row[column1], row[column2]), axis=1)]
        return violations

    def check_multicollinearity(self, threshold=10):
        """
        Check for multicollinearity using Variance Inflation Factor (VIF).
        """
        numeric_data = self.data.select_dtypes(include=["float64", "int64"]).dropna()
        vif_data = pd.DataFrame()
        vif_data["feature"] = numeric_data.columns
        vif_data["VIF"] = [
            variance_inflation_factor(numeric_data.values, i) for i in range(numeric_data.shape[1])
        ]
        return vif_data[vif_data["VIF"] > threshold]

    def target_feature_relationship(self, target_column, feature_columns):
        """
        Plot the relationship between the target column and numeric features.
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

    def check_data_coverage(self, reference_data, columns):
        """
        Check if the dataset sufficiently covers all unique values in specified columns of a reference dataset.

        Args:
            reference_data (pd.DataFrame): Reference dataset for comparison.
            columns (list): Columns to evaluate for coverage.

        Returns:
            dict: Missing values for each column.
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
        Detect potential data leaks by checking for high correlation between features and the target.

        Args:
            target_column (str): Name of the target column.
            feature_columns (list): List of feature column names to evaluate.

        Returns:
            dict: Feature columns with correlation above 0.8.
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

    def check_missing_values(self):
        return self.data.isnull().mean() * 100

    def check_duplicates(self):
        return self.data[self.data.duplicated()]

    def check_outliers(self, threshold=3):
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        return ((numeric_data - numeric_data.mean()).abs() > threshold * numeric_data.std()).sum()

    def check_imbalance(self, column):
        return self.data[column].value_counts(normalize=True) * 100

    def check_data_type_consistency(self):
        inconsistent = {}
        for col in self.data.columns:
            types = self.data[col].map(type).nunique()
            if types > 1:
                inconsistent[col] = types
        return inconsistent

    def check_correlation(self, threshold=0.9):
        corr_matrix = self.data.corr()
        return [
            (x, y) for x in corr_matrix.columns for y in corr_matrix.columns
            if x != y and abs(corr_matrix.loc[x, y]) > threshold
        ]

    def check_unique_values(self):
        return [col for col in self.data.columns if self.data[col].nunique() == 1]

    def validate_schema(self, schema_file):
        with open(schema_file, "r") as file:
            schema = json.load(file)
        missing_columns = [
            col for col in schema["columns"] if col not in self.data.columns
        ]
        return missing_columns

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

    def check_column_naming_convention(self, regex_pattern=r"^[a-z_]+$"):
        pattern = re.compile(regex_pattern)
        return [col for col in self.data.columns if not pattern.match(col)]

    def detect_anomalies(self, column, contamination=0.05):
        iso = IsolationForest(contamination=contamination)
        self.data["anomaly"] = iso.fit_predict(self.data[[column]])
        return self.data["anomaly"] == -1

    def check_sampling_bias(self, column, baseline_distribution):
        actual_distribution = self.data[column].value_counts(normalize=True)
        return {
            cat: actual_distribution.get(cat, 0) - baseline_distribution.get(cat, 0)
            for cat in baseline_distribution
        }


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
class TimeSeriesAnalyzer:
    def __init__(self, data):
        self.data = data

    def check_time_series_gaps(self, timestamp_column):
        self.data[timestamp_column] = pd.to_datetime(self.data[timestamp_column])
        gaps = self.data[timestamp_column].diff().dt.total_seconds()
        unordered = (gaps < 0).sum()
        return {"gaps": gaps.isnull().sum(), "unordered": unordered}

    def time_series_decomposition(self, column, frequency):
        series = self.data[column]
        decomp = sm.tsa.seasonal_decompose(series.dropna(), period=frequency)
        decomp.plot()
        plt.show()
        return decomp

    def check_rare_events(self, column, z_threshold=3):
        z_scores = (self.data[column] - self.data[column].mean()) / self.data[column].std()
        return self.data[z_scores.abs() > z_threshold]


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

