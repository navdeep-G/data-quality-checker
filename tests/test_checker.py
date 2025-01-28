import pandas as pd
import pytest
from dataset_quality_checker import DataQualityChecker


# Fixture to set up the DatasetQualityChecker instance
@pytest.fixture
def dataset_checker():
    """
    Fixture to create a DatasetQualityChecker instance with sample data.
    """
    data = pd.DataFrame({
        "A": [1, 2, None, 4, 5],
        "B": [1, 1, 2, 2, 2],
        "C": [1, 200, 300, 400, 500],
        "D": ["Low", "Low", "Medium", "High", "High"]
    })
    return DataQualityChecker(data)


# Test for missing values
def test_missing_values(dataset_checker):
    """
    Test to ensure the correct percentage of missing values is detected.
    """
    missing_values = dataset_checker.check_missing_values()
    assert missing_values["A"] == 20.0, "Incorrect percentage of missing values for column 'A'."


# Test for duplicates
def test_duplicates(dataset_checker):
    """
    Test to ensure no duplicate rows are detected in the dataset.
    """
    duplicates = dataset_checker.detect_duplicates()
    assert len(duplicates['duplicate_columns']) == 0, "Duplicate rows found in the dataset."


# Test for outliers
def test_outliers(dataset_checker):
    """
    Test to ensure no outliers are detected in the dataset.
    """
    outliers = dataset_checker.check_outliers()
    assert (outliers['Outlier_Count'] == 0).all(), "Some columns have outliers!"


# Test for class imbalance
def test_class_imbalance(dataset_checker):
    """
    Test to ensure the class distribution is correctly analyzed for column 'D'.
    """
    imbalance = dataset_checker.check_imbalance("D")
    assert pytest.approx(imbalance[2], rel=1e-2) == 20.0, "Incorrect class imbalance percentage for category 'Medium'."
