import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import statsmodels.api as sm


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

    from difflib import SequenceMatcher

    def check_text_similarity(self, column, similarity_threshold=0.8):
        """
        Identify pairs of text entries in a column with high similarity.

        Args:
            column (str): Name of the column to check.
            similarity_threshold (float): Threshold for similarity (0 to 1).

        Returns:
            list of tuples: Pairs of similar text entries.
        """
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

    def check_cross_column_dependency(self, column1, column2, rule):
        """
        Check for violations of a cross-column dependency rule.

        Args:
            column1 (str): First column involved in the rule.
            column2 (str): Second column involved in the rule.
            rule (callable): A function that takes two arguments and returns True if the rule is satisfied.

        Returns:
            pd.DataFrame: Rows where the rule is violated.
        """
        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise ValueError("One or both columns do not exist in the dataset.")

        violations = self.data[~self.data.apply(lambda row: rule(row[column1], row[column2]), axis=1)]
        return violations

    from sklearn.ensemble import IsolationForest

    def detect_anomalies(self, column, contamination=0.05):
        """
        Detect anomalies in a numeric column using Isolation Forest.

        Args:
            column (str): Name of the numeric column to check.
            contamination (float): Proportion of anomalies in the data.

        Returns:
            pd.Series: Boolean series indicating anomalies.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' is not numeric.")

        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.data['anomaly'] = isolation_forest.fit_predict(self.data[[column]])
        return self.data['anomaly'] == -1

    def check_conditional_probability(self, column1, column2, expected_probabilities):
        """
        Check if conditional probabilities deviate from expected values.

        Args:
            column1 (str): The column representing conditions.
            column2 (str): The column representing outcomes.
            expected_probabilities (dict): Mapping of conditions to expected probabilities of outcomes.

        Returns:
            dict: Actual conditional probabilities compared to expected probabilities.
        """
        actual_probabilities = self.data.groupby(column1)[column2].value_counts(normalize=True).to_dict()
        deviations = {
            condition: {
                outcome: actual_probabilities.get((condition, outcome), 0) - expected_prob
                for outcome, expected_prob in outcomes.items()
            }
            for condition, outcomes in expected_probabilities.items()
        }
        return deviations

    def plot_cdf(self, column):
        """
        Plot the CDF of a numeric column.
        Args:
            column (str): Name of the numeric column.
        Returns:
            None
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        data = self.data[column].dropna()
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        plt.plot(sorted_data, cdf)
        plt.title(f"CDF of {column}")
        plt.xlabel(column)
        plt.ylabel("CDF")
        plt.grid(True)
        plt.show()

    def check_multicollinearity(self, threshold=10):
        """Check for multicollinearity using Variance Inflation Factor (VIF)."""
        numeric_data = self.data.select_dtypes(include=['float64', 'int64']).dropna()
        vif_data = pd.DataFrame()
        vif_data["feature"] = numeric_data.columns
        vif_data["VIF"] = [variance_inflation_factor(numeric_data.values, i) for i in range(numeric_data.shape[1])]
        return vif_data[vif_data["VIF"] > threshold]

    def check_column_naming_convention(self, regex_pattern=r"^[a-z_]+$"):
        """Check if column names adhere to a specific naming convention."""
        pattern = re.compile(regex_pattern)
        inconsistent_columns = [col for col in self.data.columns if not pattern.match(col)]
        return inconsistent_columns

    def check_rare_events(self, column, z_threshold=3):
        """Identify rare events in a time-series or numeric column based on deviations from the mean."""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found.")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' is not numeric.")

        z_scores = (self.data[column] - self.data[column].mean()) / self.data[column].std()
        return self.data[z_scores.abs() > z_threshold]

    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords

    def check_common_words(self, column, top_n=10):
        """
        Identify the most common words in a text column.

        Args:
            column (str): Name of the text column.
            top_n (int): Number of top common words to return.

        Returns:
            pd.DataFrame: Top N most common words and their counts.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise TypeError(f"Column '{column}' is not of string type.")

        vectorizer = CountVectorizer(stop_words=stopwords.words("english"))
        word_counts = vectorizer.fit_transform(self.data[column].dropna())
        word_freq = pd.DataFrame(
            word_counts.toarray(), columns=vectorizer.get_feature_names_out()
        ).sum().sort_values(ascending=False).head(top_n)
        return word_freq

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

    def check_outlier_impact(self, column, method='mean'):
        """
        Assess the impact of outliers on aggregate statistics.

        Args:
            column (str): Column to analyze.
            method (str): Aggregation method ('mean' or 'median').

        Returns:
            float: Difference in the aggregate value with and without outliers.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if method not in ['mean', 'median']:
            raise ValueError(f"Method '{method}' is not supported.")
        z_scores = (self.data[column] - self.data[column].mean()) / self.data[column].std()
        non_outliers = self.data[z_scores.abs() <= 3]
        original_stat = getattr(self.data[column], method)()
        adjusted_stat = getattr(non_outliers[column], method)()
        return original_stat - adjusted_stat

    def check_sampling_bias(self, column, baseline_distribution):
        """
        Compare the distribution of a column with a baseline distribution.

        Args:
            column (str): Column to compare.
            baseline_distribution (dict): Expected distribution as a dictionary.

        Returns:
            dict: Deviation of actual distribution from the baseline.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        actual_distribution = self.data[column].value_counts(normalize=True).to_dict()
        deviations = {
            category: actual_distribution.get(category, 0) - baseline_distribution.get(category, 0)
            for category in baseline_distribution
        }
        return deviations

    def check_feature_engineering_quality(self, features):
        """
        Assess if engineered features add value by checking their correlation with the target.

        Args:
            features (list): List of engineered feature column names.

        Returns:
            dict: Features with their correlation values against the target.
        """
        target_column = "target"  # Replace with the actual target column
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' does not exist.")
        correlations = {}
        for feature in features:
            if feature not in self.data.columns:
                raise ValueError(f"Feature column '{feature}' does not exist.")
            correlations[feature] = self.data[feature].corr(self.data[target_column])
        return correlations

    def category_feature_interaction(self, categorical_column, numeric_column):
        """
        Analyze the interaction between a categorical column and a numerical column.
        Args:
            categorical_column (str): Name of the categorical column.
            numeric_column (str): Name of the numeric column.
        Returns:
            pd.DataFrame: Statistics of the numeric column grouped by the categorical column.
        """
        if categorical_column not in self.data.columns or numeric_column not in self.data.columns:
            raise ValueError("One or both specified columns do not exist in the dataset.")

        interaction_stats = self.data.groupby(categorical_column)[numeric_column].describe()
        return interaction_stats

    def low_variance_features(self, threshold=0.01):
        """
        Identify features with low variance.
        Args:
            threshold (float): Variance threshold below which features are flagged.
        Returns:
            list: Features with variance below the threshold.
        """
        variances = self.data.var()
        low_variance = variances[variances < threshold].index.tolist()
        return low_variance

    def check_skewness(self):
        """
        Calculate skewness for numeric features.
        Returns:
            pd.Series: Skewness values for numeric columns.
        """
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        skewness = numeric_data.skew()
        return skewness

    import seaborn as sns
    import matplotlib.pyplot as plt

    def target_feature_relationship(self, target_column, feature_columns):
        """
        Plot the relationship between the target column and numeric features.
        Args:
            target_column (str): The target variable.
            feature_columns (list): List of feature columns to analyze.
        Returns:
            None
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

    def cross_column_value_patterns(self, column1, column2):
        """
        Check for specific value patterns between two columns.
        Args:
            column1 (str): First column.
            column2 (str): Second column.
        Returns:
            pd.DataFrame: Frequency table of value pairs.
        """
        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise ValueError("One or both specified columns do not exist in the dataset.")

        patterns = self.data.groupby([column1, column2]).size().unstack(fill_value=0)
        return patterns

    def plot_correlation_heatmap(self):
        """
        Plot a heatmap for numeric feature correlations.
        Returns:
            None
        """
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_data.corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    def value_distribution_summary(self):
        """
        Summarize the value distribution for each column.
        Returns:
            dict: Summary statistics for numeric and value counts for categorical columns.
        """
        summary = {}
        for column in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                summary[column] = self.data[column].describe()
            elif pd.api.types.is_object_dtype(self.data[column]):
                summary[column] = self.data[column].value_counts().to_dict()
        return summary

    def time_series_decomposition(self, column, frequency):
        """
        Decompose a time-series column into trend, seasonality, and residuals.
        Args:
            column (str): Name of the time-series column.
            frequency (int): Seasonal frequency.
        Returns:
            sm.tsa.seasonal_decompose: Decomposed components.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        ts_data = self.data[column].dropna()
        decomposition = sm.tsa.seasonal_decompose(ts_data, period=frequency)
        decomposition.plot()
        plt.show()
        return decomposition

    def analyze_text_length(self, column, min_length=5, max_length=500):
        """
        Analyze the length of text data in a column.
        Args:
            column (str): Text column to analyze.
            min_length (int): Minimum acceptable length.
            max_length (int): Maximum acceptable length.
        Returns:
            pd.DataFrame: Rows with text outside the acceptable range.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found.")
        lengths = self.data[column].str.len()
        return self.data[(lengths < min_length) | (lengths > max_length)]


if __name__ == "__main__":
    df = pd.read_csv("../data/sample_data.csv")
    checker = DatasetQualityChecker(df)
    report = checker.generate_report()
    print("Dataset Quality Report:\n", report)
