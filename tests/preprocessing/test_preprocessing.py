import pytest
import sys

import trackintel as ti
from trackintel.preprocessing import positionfixes
from trackintel.preprocessing import staypoints


class TestPreprocessing():
    def test_extract_staypoints_sliding_min(self):
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        assert len(spts) == len(pfs), "With small thresholds, staypoint extraction should yield each positionfix"
        
    def test_extract_staypoints_sliding_max(self):
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=sys.maxsize, 
                                                       time_threshold=sys.maxsize)
        assert len(spts) == 0, "With large thresholds, staypoint extraction should not yield positionfixes"
        
    def test_extract_triplegs_staypoint(self):
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        tpls1 = pfs.as_positionfixes.extract_triplegs()
        tpls2 = pfs.as_positionfixes.extract_triplegs(spts)
        assert len(tpls1) > 0, "There should be more than zero triplegs"
        assert len(tpls2) > 0, "There should be more than zero triplegs"
        assert len(tpls1) == len(tpls2), "If we extract the staypoints in the same way, it should lead to " + \
            "the same number of triplegs"

    def test_cluster_staypoints_dbscan_min(self):
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        _, clusters = spts.as_staypoints.extract_locations(method='dbscan', epsilon=1e-18, num_samples=0)
        assert len(clusters) == len(spts), "With small hyperparameters, clustering should not reduce the number"

    def test_cluster_staypoints_dbscan_max(self):
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        _, clusters = spts.as_staypoints.extract_locations(method='dbscan', epsilon=1e18, num_samples=1000)
        assert len(clusters) == 0, "With large hyperparameters, everything is an outlier"
        
    def test_cluster_staypoints_dbscan_user_dataset(self):
        spts = ti.read_staypoints_csv('tests/data/geolife/geolife_staypoints.csv')
        # take the first row and duplicate once
        spts = spts.head(1)
        spts = spts.append(spts, ignore_index=True)
        # assign a different user_id to the second row
        spts.iloc[1, 5] = 1
        # duplicate for a certain number 
        spts = spts.append([spts]*5,ignore_index=True)
        _, locs_ds = spts.as_staypoints.extract_locations(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='dataset')
        _, locs_us = spts.as_staypoints.extract_locations(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='user')
        loc_ds_num = locs_ds['location_id'].unique().shape[0]
        loc_us_num = locs_us['location_id'].unique().shape[0]
        assert  loc_ds_num == 1, "Considering all staypointsat once, there will be only 1 location"
        assert  loc_us_num == 2, "Considering user separately, there will be 2 location"
