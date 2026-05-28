import os
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import geopandas as gp
from pygwmodel import GTWR, ParallelType, BandwidthWeight
from pygwmodel.spatial_weight import CRSSTDistance

TEST_DATA = os.environ.get("PYGW_TEST_DATA")
if TEST_DATA is None and len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
    TEST_DATA = sys.argv[1]
if TEST_DATA is None:
    TEST_DATA = str(Path(__file__).with_name("londonhp100.csv"))
ENABLE_OPENMP = (lambda s: False if s is None else (s.lower() in ['true', '1', 't', 'y', 'yes', 'on']))(os.getenv("ENABLE_OPENMP"))


class TestGTWR(unittest.TestCase):

    def setUp(self):
        londonhp_csv = pd.read_csv(TEST_DATA)
        self.londonhp = gp.GeoDataFrame(
            londonhp_csv,
            geometry=gp.points_from_xy(londonhp_csv.x, londonhp_csv.y))
        self.londonhp['TIME'] = np.arange(len(self.londonhp),
                                           dtype=np.float64)
        self.depen = 'PURCHASE'
        self.indep = ["FLOORSZ", "UNEMPLOY", "PROF"]
        self.distance = CRSSTDistance(lambda_=0.5)
        self.parallel_case = {
            ParallelType.SerialOnly: dict()
        }
        if ENABLE_OPENMP:
            self.parallel_case[ParallelType.OpenMP] = {'threads': 4}

    def test_minimal(self):
        for p, pargs in self.parallel_case.items():
            with self.subTest(parallel=p):
                algorithm = GTWR(
                    self.londonhp, self.depen, self.indep, times='TIME',
                    weight=BandwidthWeight(36.0, True),
                    distance=self.distance
                ).enable_parallel(p, **pargs).fit()

                self.assertIsNotNone(algorithm.result_layer)
                indep_var_names = ['Intercept'] + self.indep
                for name in indep_var_names:
                    self.assertIn(name, algorithm.result_layer.columns)
                self.assertIn('Intercept_SE', algorithm.result_layer.columns)
                self.assertIn('fitted', algorithm.result_layer.columns)

                diag = algorithm.diagnostic
                self.assertIsNotNone(diag)
                self.assertIn('AIC', diag)
                self.assertIn('AICc', diag)
                self.assertIn('RSquare', diag)
                self.assertGreater(diag['RSquare'], 0)
                self.assertLess(diag['RSquare'], 1)

    def test_autoselect_bandwidth(self):
        for p, pargs in self.parallel_case.items():
            with self.subTest(parallel=p):
                algorithm = GTWR(
                    self.londonhp, self.depen, self.indep, times='TIME',
                    weight=BandwidthWeight(36.0, True),
                    distance=self.distance
                ).enable_parallel(p, **pargs).fit(
                    optimize_bandwidth=GTWR.BandwidthSelectionCriterionType.CV
                )
                self.assertIsNotNone(algorithm.weight.bandwidth)
                self.assertGreater(algorithm.weight.bandwidth, 0)

    def test_predict(self):
        for p, pargs in self.parallel_case.items():
            with self.subTest(parallel=p):
                algorithm = GTWR(
                    self.londonhp, self.depen, self.indep, times='TIME',
                    weight=BandwidthWeight(36.0, True),
                    distance=self.distance
                ).enable_parallel(p, **pargs).fit()
                prediction = algorithm.predict(self.londonhp)
                self.assertIn("y_hat", prediction.columns)
                self.assertIn("residual", prediction.columns)

    def test_fixed_bandwidth(self):
        for p, pargs in self.parallel_case.items():
            with self.subTest(parallel=p):
                algorithm = GTWR(
                    self.londonhp, self.depen, self.indep, times='TIME',
                    weight=BandwidthWeight(100.0, True),
                    distance=self.distance
                ).enable_parallel(p, **pargs).fit()

                self.assertIsNotNone(algorithm.result_layer)
                diag = algorithm.diagnostic
                self.assertIsNotNone(diag)
                self.assertGreater(diag['RSquare'], 0)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)
