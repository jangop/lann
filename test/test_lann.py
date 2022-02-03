"""
Test file handling.
"""

import unittest

import sklearn.utils.estimator_checks

from lann import LocallyAdaptiveNeighborsClassifier


class SklearnTest(unittest.TestCase):
    def test_classifier(self):
        parameters = {
            "max_neighbors": [None, 10],
            "random_state": [None, 0],
            "regularization": [0, 0.5],
            "auxiliary_path": [None, "/tmp/test_lann"],
            "record_fingerprints": [True, False],
        }

        for parameter, values in parameters.items():
            for value in values:
                with self.subTest(parameter=value):
                    sklearn.utils.estimator_checks.check_estimator(
                        LocallyAdaptiveNeighborsClassifier(**{parameter: value})
                    )


if __name__ == "__main__":
    unittest.main()
