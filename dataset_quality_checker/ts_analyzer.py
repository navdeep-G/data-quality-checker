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
