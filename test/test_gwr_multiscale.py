import sys
import os
import unittest
import numpy as np
import pandas as pd
import geopandas as gp
from pygwmodel import GWRMultiscale, ParallelType, BandwidthWeight, CRSDistance

TEST_DATA = os.environ.get("PYGW_TEST_DATA", sys.argv[1] if len(sys.argv) > 1 else "test/londonhp100.csv")
ENABLE_OPENMP = (lambda s: False if s is None else (s.lower() in ['true', '1', 't', 'y', 'yes', 'on']))(os.getenv("ENABLE_OPENMP"))


class TestGWRMultiscale(unittest.TestCase):

    def setUp(self):
        londonhp_csv = pd.read_csv(TEST_DATA)
        self.londonhp = gp.GeoDataFrame(londonhp_csv, geometry=gp.points_from_xy(londonhp_csv.x, londonhp_csv.y))
        self.depen = 'PURCHASE'
        self.indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
        self.parallel_case = {
            ParallelType.SerialOnly: dict()
        }
        if ENABLE_OPENMP:
            self.parallel_case[ParallelType.OpenMP] = {'threads': 4}

    def test_fit_specified_bandwidths(self):
        """Test fit with manually specified bandwidths (fast and deterministic)."""
        n_var = len(self.indep) + 1  # including intercept
        weights = [BandwidthWeight(36.0, True) for _ in range(n_var)]

        for p, pargs in self.parallel_case.items():
            with self.subTest(parallel=p):
                algorithm = GWRMultiscale(
                    self.londonhp, self.depen, self.indep,
                    weights,
                    bandwidth_initilize=None,  # defaults to Null (auto)
                    bandwidth_selection_approach=None,  # defaults to CV
                    has_hat_matrix=True
                )
                algorithm.enable_parallel(p, **pargs).fit()

                # Check result structure
                self.assertIsNotNone(algorithm.result_layer)
                expected_cols = ['Intercept', 'Intercept_SE', 'Intercept_TV']
                for v in self.indep:
                    expected_cols += [v, f'{v}_SE', f'{v}_TV']
                expected_cols.append('fitted')
                for col in expected_cols:
                    self.assertIn(col, algorithm.result_layer.columns)

                # Check betas shape: (n_samples, n_vars)
                self.assertEqual(algorithm.betas.shape, (len(self.londonhp), n_var))
                self.assertEqual(algorithm.betasSE.shape, (len(self.londonhp), n_var))
                self.assertEqual(algorithm.betasTV.shape, (len(self.londonhp), n_var))

                # Check diagnostic
                diag = algorithm.diagnostic
                self.assertIsNotNone(diag)
                for key in ['RSS', 'AICc', 'ENP', 'EDF', 'RSquare', 'RSquareAdjust']:
                    self.assertIn(key, diag)
                self.assertGreater(diag['RSquare'], 0)
                self.assertLess(diag['RSquare'], 1)

    def test_fit_result_layer(self):
        """Test that result_layer contains finite values."""
        n_var = len(self.indep) + 1
        weights = [BandwidthWeight(50.0, True) for _ in range(n_var)]

        algorithm = GWRMultiscale(
            self.londonhp, self.depen, self.indep,
            weights,
            bandwidth_initilize=None,
            bandwidth_selection_approach=None,
            has_hat_matrix=True
        )
        algorithm.max_iteration = 10
        algorithm.fit()

        result = algorithm.result_layer
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(self.londonhp))
        # Check no NaN in fitted values
        self.assertFalse(np.any(np.isnan(result['fitted'].values)))

    def test_specified_bandwidths_skip_selection(self):
        """Test with Specified bandwidth init type skips auto-selection."""
        n_var = len(self.indep) + 1
        weights = [BandwidthWeight(36.0, True) for _ in range(n_var)]

        algorithm = GWRMultiscale(
            self.londonhp, self.depen, self.indep,
            weights,
            bandwidth_initilize=[GWRMultiscale.BandwidthInitilizeType.Specified] * n_var,
            bandwidth_selection_approach=[GWRMultiscale.BandwidthSelectionCriterionType.CV] * n_var,
            has_hat_matrix=True
        )
        algorithm.max_iteration = 10
        algorithm.fit()

        # Bandwidths should remain as specified (not auto-optimized)
        for w in algorithm.weights:
            self.assertEqual(w.bandwidth, 36.0)

        # Verify result layer is produced
        self.assertIsNotNone(algorithm.result_layer)
        diag = algorithm.diagnostic
        self.assertIsNotNone(diag)

    def test_no_hat_matrix(self):
        """Test fitting without hat matrix (faster, less memory)."""
        n_var = len(self.indep) + 1
        weights = [BandwidthWeight(36.0, True) for _ in range(n_var)]

        algorithm = GWRMultiscale(
            self.londonhp, self.depen, self.indep,
            weights,
            bandwidth_initilize=[GWRMultiscale.BandwidthInitilizeType.Specified] * n_var,
            bandwidth_selection_approach=[GWRMultiscale.BandwidthSelectionCriterionType.CV] * n_var,
            has_hat_matrix=False
        )
        algorithm.max_iteration = 10
        algorithm.fit()

        self.assertIsNotNone(algorithm.result_layer)
        diag = algorithm.diagnostic
        self.assertIsNotNone(diag)

    def test_fixed_bandwidth(self):
        """Test with fixed (non-adaptive) bandwidths."""
        n_var = len(self.indep) + 1
        weights = [BandwidthWeight(5000.0, False) for _ in range(n_var)]

        algorithm = GWRMultiscale(
            self.londonhp, self.depen, self.indep,
            weights,
            bandwidth_initilize=[GWRMultiscale.BandwidthInitilizeType.Specified] * n_var,
            bandwidth_selection_approach=[GWRMultiscale.BandwidthSelectionCriterionType.CV] * n_var,
            has_hat_matrix=True
        )
        algorithm.max_iteration = 10
        algorithm.fit()

        self.assertIsNotNone(algorithm.result_layer)
        diag = algorithm.diagnostic
        self.assertIsNotNone(diag)
        self.assertGreater(diag['RSquare'], 0)
        self.assertLess(diag['RSquare'], 1)

    def test_backfitting_criterion(self):
        """Test with CVR backfitting criterion type."""
        n_var = len(self.indep) + 1
        weights = [BandwidthWeight(36.0, True) for _ in range(n_var)]

        algorithm = GWRMultiscale(
            self.londonhp, self.depen, self.indep,
            weights,
            bandwidth_initilize=[GWRMultiscale.BandwidthInitilizeType.Specified] * n_var,
            has_hat_matrix=True
        )
        algorithm.criterion_type = GWRMultiscale.BackFittingCriterionType.CVR
        algorithm.max_iteration = 10
        algorithm.fit()

        self.assertIsNotNone(algorithm.result_layer)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)
