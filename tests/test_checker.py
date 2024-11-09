import pandas as pd
import unittest
from src.checker import DatasetQualityChecker

class TestDatasetQualityChecker(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "A": [1, 2, None, 4, 5],
            "B": [1, 1, 2, 2, 2],
            "C": [1, 200, 300, 400, 500]
        })
        self.checker = DatasetQualityChecker(self.data)

    def test_missing_values(self):
        missing_values = self.checker.check_missing_values()
        self.assertEqual(missing_values["A"], 20.0)

    def test_duplicates(self):
        duplicates = self.checker.check_duplicates()
        self.assertEqual(duplicates, 0)

    def test_outliers(self):
        outliers = self.checker.check_outliers()
        self.assertEqual(outliers["C"], 3)

    def test_class_imbalance(self):
        imbalance = self.checker.check_imbalance("B")
        self.assertAlmostEqual(imbalance[2], 60.0)

if __name__ == "__main__":
    unittest.main()
