import os
from math import radians

import numpy as np
from sklearn.metrics import pairwise_distances

import trackintel as ti
from trackintel.geogr.distances import meters_to_decimal_degrees, calculate_distance_matrix


class TestCalculate_distance_matrix:

    def test_shape_for_different_array_length(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))

        x = spts.iloc[0:5]
        y = spts.iloc[5:15]

        d_euc1 = calculate_distance_matrix(X=x, Y=y, dist_metric='euclidean')
        d_euc2 = calculate_distance_matrix(X=y, Y=x, dist_metric='euclidean')
        d_hav1 = calculate_distance_matrix(X=x, Y=y, dist_metric='haversine')
        d_hav2 = calculate_distance_matrix(X=y, Y=x, dist_metric='haversine')

        assert d_euc1.shape == d_hav1.shape == (5, 10)
        assert d_euc2.shape == d_hav2.shape == (10, 5)
        assert np.isclose(0, np.sum(np.abs(d_euc1 - d_euc2.T)))
        assert np.isclose(0, np.sum(np.abs(d_hav1 - d_hav2.T)))

    def test_keyword_combinations(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))

        x = spts.iloc[0:5]
        y = spts.iloc[5:15]

        _ = calculate_distance_matrix(X=x, Y=y, dist_metric='euclidean', n_jobs=-1)
        _ = calculate_distance_matrix(X=y, Y=x, dist_metric='haversine', n_jobs=-1)
        d_mink1 = calculate_distance_matrix(X=x, Y=x, dist_metric='minkowski', p=1)
        d_mink2 = calculate_distance_matrix(X=x, Y=x, dist_metric='minkowski', p=2)
        d_euc = calculate_distance_matrix(X=x, Y=x, dist_metric='euclidean')

        assert not np.array_equal(d_mink1,d_mink2)
        assert np.array_equal(d_euc, d_mink2)


    def test_compare_haversine_to_scikit_xy(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        our_d_matrix = calculate_distance_matrix(X=spts, Y=spts, dist_metric='haversine')

        x = spts.geometry.x.values
        y = spts.geometry.y.values

        x_rad = np.asarray([radians(_) for _ in x])
        y_rad = np.asarray([radians(_) for _ in y])
        yx = np.concatenate((y_rad.reshape(-1, 1), x_rad.reshape(-1, 1)), axis=1)

        their_d_matrix = pairwise_distances(yx, metric='haversine') * 6371000
        assert np.allclose(np.abs(our_d_matrix - their_d_matrix), 0, atol=0.001) # atol = 1mm

    def test_trajectory_distance(self):
        tpls = ti.read_triplegs_csv(os.path.join('tests', 'data', 'geolife', 'geolife_triplegs.csv'))
        D_single = calculate_distance_matrix(X=tpls.iloc[0:4], dist_metric='dtw', n_jobs=1)
        D_multi = calculate_distance_matrix(X=tpls.iloc[0:4], dist_metric='dtw', n_jobs=4)

        assert np.isclose(np.sum(np.abs(D_single - D_multi)), 0)

    def test_trajectory_distance_via_accessor_x(self):
        tpls = ti.read_triplegs_csv(os.path.join('tests', 'data', 'geolife', 'geolife_triplegs.csv'))


        D_single = tpls.iloc[0:4].as_triplegs.similarity(dist_metric='dtw', n_jobs=1)
        D_multi = tpls.iloc[0:4].as_triplegs.similarity(dist_metric='dtw', n_jobs=4)

        assert np.isclose(np.sum(np.abs(D_single - D_multi)), 0)

    def test_trajectory_distance_via_accessor_xy(self):
        tpls = ti.read_triplegs_csv(os.path.join('tests', 'data', 'geolife', 'geolife_triplegs.csv'))

        x = tpls.iloc[0:2]
        y = tpls.iloc[4:8]

        D_single = x.as_triplegs.similarity(Y=y, dist_metric='dtw', n_jobs=1)
        D_multi = x.as_triplegs.similarity(Y=y, dist_metric='dtw', n_jobs=4)

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


