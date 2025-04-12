# Dataset Quality Checker

A Python package for evaluating and maintaining dataset quality by identifying common data issues such as missing values, duplicates, outliers, and class imbalances. It also supports advanced assessments like schema validation, semantic consistency checks, hypothesis testing, time-series analysis, and data drift detection—making it a comprehensive tool for ensuring data reliability.

---

## 🔧 Features
### 🔍 Basic Checks
- **Missing Values**: Detect columns with missing data and calculate missing percentages.
- **Duplicate Rows**: Identify and return duplicate rows.
- **Outliers**: Detect numeric outliers using Z-score, IQR, or Isolation Forest.
- **Class Imbalance**: Analyze class distribution in categorical columns.
- **Empty Columns**: Identify columns with no non-null values.
- **Column Duplicates**: Detect duplicate values within specific columns.
- **Null Rows**: Identify rows where all values are null.
- **Null Proportions by Group**: Check for significant differences in null proportions across groups.
- **Data Completeness**: Ensure all required columns are present in the dataset.
- **Date Format Validation**: Identify invalid date formats in columns.
- **Encoding Consistency**: Verify text column encoding.
- **Redundant Columns**: Identify correlated or identical columns.
- **Granularity Inconsistency**: Detect inconsistent date granularity.
- **Sparse Columns**: Detect empty or sparse columns based on threshold.
### 🚀 Advanced Checks
- **Schema Validation**: Validate against a predefined schema with type, format, and range constraints.
- **Data Type Consistency**: Identify columns with inconsistent or mixed data types.
- **Correlation Analysis**: Flag highly correlated numeric features.
- **Unique Values**: Detect columns with only a single unique value.
- **Rare Categories**: Identify infrequent categories based on a frequency threshold.
- **Multicollinearity**: Detect using Variance Inflation Factor (VIF).
- **Unexpected Values**: Flag invalid entries in categorical columns.
- **Naming Convention Check**: Verify adherence to column naming patterns.
- **Cross-Column Rules**: Validate inter-column dependencies and logical rules.
- **Conditional Probability Deviations**: Identify deviations in expected conditional distributions.
- **Sampling Bias**: Compare column distribution with expected distribution.
- **Numeric Precision**: Validate that numeric values do not exceed specified precision.
- **String Length Outliers**: Identify text entries that are too short or too long.
- **Email Validity**: Validate format of email addresses.
- **Phone Number Validity**: Validate format and correctness of phone numbers.
- **Inconsistent Casing**: Detect inconsistent casing in text columns.
- **Partition Completeness**: Validate presence of data across partitions.
- **Foreign Key Validity**: Check validity of foreign keys against a reference.
- **Aggregation Validity**: Compare aggregates to raw data for validation.
- **Cross-Partition Type Consistency**: Ensure consistent data types across partitions.
### 📊 Statistical Testing
- **Hypothesis Testing**: Perform t-tests, chi-squared tests, or ANOVA.
- **Confidence Intervals**: Compute for numeric columns.
- **Outlier Impact**: Assess how outliers affect key statistics.
- **Uniformity Test**: Check if data follows a uniform distribution.
- **Distribution Comparison**: Compare distributions to identify data drift.
### 📈 Time-Series Analysis
- **Time Gaps**: Identify missing or unordered timestamps.
- **Trend Detection**: Spot anomalies or trends in numeric time-series data.
- **Seasonality**: Detect recurring seasonal patterns.
- **Change Point Detection**: Identify structural breaks or shifts.
- **Stationarity Test**: Use ADF test to check for stationarity.
- **Forecast Metrics**: Evaluate time-series model accuracy.
- **Temporal Consistency**: Validate timestamp order and overlapping intervals.
### 📝 Text and NLP Analysis
- **Text Length Check**: Detect excessively long text entries.
- **Text Similarity**: Find highly similar text pairs.
- **Sentiment Analysis**: Assign sentiment scores.
- **NER**: Extract entities like names, dates, organizations.
- **Keyword Extraction**: Use RAKE for identifying key terms.
- **Spelling Correction**: Auto-correct spelling in text columns.
- **Topic Modeling**: Extract dominant topics via LDA.
- **Tokenization**: Break text into words or sentences.
- **Language Detection**: Detect text language.
- **Stopword Analysis**: Count common stopwords.
- **N-gram Analysis**: Identify frequent n-gram patterns.
### 📊 Data Drift and Anomaly Detection
- **Data Drift**: Compare distributions to a baseline dataset.
- **Anomalies**: Use Isolation Forest to identify numeric anomalies.
- **Rare Events**: Detect unusual numeric events using statistical thresholds.
- **Drift by Row**: Compare row-level values to a reference.
- **Drift Detection via Subsets**: Check drift in specific subsets or keys.
### 🌐 Semantic & Cross-Column Validation
- **Semantic Checks**: Validate data against expected semantic mappings.
- **Dependency Validation**: Enforce and validate inter-column logic.
- **Conditional Probabilities**: Detect inconsistencies in dependent probabilities.
- **Consistency Across Columns**: Validate values across columns using custom rules.
- **Column Relationship Validation**: Check logical relationships between multiple columns.
- **Duplicate Detection by Subset**: Detect duplicates in subsets of columns.
- **Data Integrity After Joins**: Verify correctness after data merges.
- **Overlapping Categories**: Detect overlapping values in categorical fields.
