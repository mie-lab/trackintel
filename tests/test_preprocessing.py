import pytest
import sys

import trackintel as ti
from trackintel.preprocessing import positionfixes
from trackintel.preprocessing import staypoints


class TestPreprocessing:
    def test_extract_staypoints_sliding_min(self):
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        assert len(spts) == len(pfs), "With small thresholds, staypoint extraction should yield each positionfix"
        
    def test_extract_staypoints_sliding_max(self):
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=sys.maxsize, 
                                                       time_threshold=sys.maxsize)
        assert len(spts) == 0, "With large thresholds, staypoint extraction should not yield positionfixes"

    def test_cluster_staypoints_dbscan_min(self):
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        clusters = spts.as_staypoints.extract_places(method='dbscan', epsilon=1e-18, num_samples=0)
        assert len(clusters) == len(spts), "With small hyperparameters, clustering should not reduce the number"

    def test_cluster_staypoints_dbscan_max(self):
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        clusters = spts.as_staypoints.extract_places(method='dbscan', epsilon=1e18, num_samples=1000)
        assert len(clusters) == 0, "With large hyperparameters, everything is an outlier"
