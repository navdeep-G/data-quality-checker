import pandas as pd
import pytest
from src.checker import DataQualityChecker

# Fixture to set up the DatasetQualityChecker instance
@pytest.fixture
def dataset_checker():
    data = pd.DataFrame({
        "A": [1, 2, None, 4, 5],
        "B": [1, 1, 2, 2, 2],
        "C": [1, 200, 300, 400, 500],
        "D": ["Low", "Low", "Medium", "High", "High"]
    })
    return DataQualityChecker(data)

# Test for missing values
def test_missing_values(dataset_checker):
    missing_values = dataset_checker.check_missing_values()
    assert missing_values["A"] == 20.0


# Test for duplicates
def test_duplicates(dataset_checker):
    duplicates = dataset_checker.check_duplicates()
    assert duplicates.shape[0] == 0


# Test for outliers
def test_outliers(dataset_checker):
    outliers = dataset_checker.check_outliers()
    assert (outliers['Outlier_Count'] == 0).all(), "Some columns have outliers!"


# Test for class imbalance
def test_class_imbalance(dataset_checker):
    imbalance = dataset_checker.check_imbalance("D")
    assert pytest.approx(imbalance[2], rel=1e-2) == 20.0
