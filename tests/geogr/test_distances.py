import pytest
import sys
import os
import filecmp

import pandas as pd
import trackintel as ti
import numpy as np
from trackintel.geogr import distances
from trackintel.geogr.distances import haversine_dist
from trackintel.geogr.distances import meters_to_decimal_degrees, calculate_distance_matrix
from sklearn.metrics import pairwise_distances
from math import radians
import time

class TestCalculate_distance_matrix:
    def test_compare_haversine_to_scikit_xy(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        our_d_matrix = calculate_distance_matrix(x=spts, y=spts, dist_metric='haversine')

        x = spts.geometry.x.values
        y = spts.geometry.y.values

        x_rad = np.asarray([radians(_) for _ in x])
        y_rad = np.asarray([radians(_) for _ in y])
        yx = np.concatenate((y_rad.reshape(-1, 1), x_rad.reshape(-1, 1)), axis=1)

        their_d_matrix = pairwise_distances(yx, metric='haversine') * 6371000
        assert np.allclose(np.abs(our_d_matrix - their_d_matrix), 0, atol=0.001) # atol = 1mm

    def test_trajectory_distance(self):
        tpls = ti.read_triplegs_csv(os.path.join('tests', 'data', 'geolife', 'geolife_triplegs.csv'))
        t_start = time.time()
        D_single = calculate_distance_matrix(x=tpls.iloc[0:4], dist_metric='dtw', n_jobs=1)
        t_single = time.time() - t_start
        t_start = time.time()
        D_multi = calculate_distance_matrix(x=tpls.iloc[0:4], dist_metric='dtw', n_jobs=4)
        t_multi = time.time() - t_start

        print(t_single, t_multi)
        assert np.isclose(np.sum(np.abs(D_single - D_multi)), 0)



class TestMetersToDecimalDegrees:
    def test_meters_to_decimal_degrees(self):
        input_result_dict = {1.0: {0: 111320, 23: 102470, 45: 78710, 67: 43496},
                             0.1: {0: 11132, 23: 10247, 45: 7871, 67: 4349.6},
                             0.01: {0: 1113.2, 23: 1024.7, 45: 787.1, 67: 434.96},
                             0.001: {0: 111.32, 23: 102.47, 45: 78.71, 67: 43.496}}

        for degree, lat_output in input_result_dict.items():
            for lat, meters in lat_output.items():

                decimal_degree_output = meters_to_decimal_degrees(meters, lat)
                assert np.isclose(decimal_degree_output, degree, atol=0.1)


