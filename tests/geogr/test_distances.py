import os
from math import radians

import geopandas as gpd
import numpy as np
import pytest
from shapely import wkt
from shapely.geometry import LineString
from sklearn.metrics import pairwise_distances

import trackintel as ti
from trackintel.geogr.distances import meters_to_decimal_degrees, calculate_distance_matrix, \
    calculate_haversine_length, _calculate_haversine_length_single


@pytest.fixture
def ls_short():
    return LineString([(13.476808430, 48.573711823), (13.506804, 48.939008), (13.4664690, 48.5706414)])


@pytest.fixture
def ls_long():
    return LineString([(13.476808430, 48.573711823), (11.5675446, 48.1485459), (8.5067847, 47.4084269)])


@pytest.fixture
def gdf_ls(ls_short, ls_long):
    a_list = [(0, ls_short), (1, ls_long)]
    gdf = gpd.GeoDataFrame(a_list, columns=['id', 'geometry']).set_geometry('geometry')
    gdf = gdf.set_crs('wgs84')
    return gdf


@pytest.fixture
def single_linestring():
    # measured length in qgis: ~1024 m

    return wkt.loads("""LineString(13.47671401745228259 
    48.57364142178052901, 13.47510901146785933
    48.5734004715611789, 13.47343656825720082
    48.57335585102421049, 13.47172366271079369
    48.57318629262447018, 13.4697275208142031
    48.57325768570418489, 13.4680415901582915
    48.57348971251707326, 13.46604544826169914
    48.57348971251707326, 13.46473716607271243
    48.57319521676494389, 13.46319959731452798
    48.57253482611510975, 13.46319959731452798
    48.57253482611510975)""")


class TestCalculate_distance_matrix:

    def test_shape_for_different_array_length(self):
        spts_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        spts = ti.read_staypoints_csv(spts_file, tz='utc', index_col='id')

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
        spts_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        spts = ti.read_staypoints_csv(spts_file, tz='utc', index_col='id')

        x = spts.iloc[0:5]
        y = spts.iloc[5:15]

        _ = calculate_distance_matrix(X=x, Y=y, dist_metric='euclidean', n_jobs=-1)
        _ = calculate_distance_matrix(X=y, Y=x, dist_metric='haversine', n_jobs=-1)
        d_mink1 = calculate_distance_matrix(X=x, Y=x, dist_metric='minkowski', p=1)
        d_mink2 = calculate_distance_matrix(X=x, Y=x, dist_metric='minkowski', p=2)
        d_euc = calculate_distance_matrix(X=x, Y=x, dist_metric='euclidean')

        assert not np.array_equal(d_mink1, d_mink2)
        assert np.array_equal(d_euc, d_mink2)

    def test_compare_haversine_to_scikit_xy(self):
        spts_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        spts = ti.read_staypoints_csv(spts_file, tz='utc', index_col='id')
        our_d_matrix = calculate_distance_matrix(X=spts, Y=spts, dist_metric='haversine')

        x = spts.geometry.x.values
        y = spts.geometry.y.values

        x_rad = np.asarray([radians(_) for _ in x])
        y_rad = np.asarray([radians(_) for _ in y])
        yx = np.concatenate((y_rad.reshape(-1, 1), x_rad.reshape(-1, 1)), axis=1)

        their_d_matrix = pairwise_distances(yx, metric='haversine') * 6371000
        assert np.allclose(np.abs(our_d_matrix - their_d_matrix), 0, atol=0.001)  # atol = 1mm

    def test_trajectory_distance(self):
        tpls_file = os.path.join('tests', 'data', 'geolife', 'geolife_triplegs.csv')
        tpls = ti.read_triplegs_csv(tpls_file, tz='utc', index_col='id')
        D_single = calculate_distance_matrix(X=tpls.iloc[0:4], dist_metric='dtw', n_jobs=1)
        D_multi = calculate_distance_matrix(X=tpls.iloc[0:4], dist_metric='dtw', n_jobs=4)

        assert np.isclose(np.sum(np.abs(D_single - D_multi)), 0)

    def test_trajectory_distance_via_accessor_x(self):
        tpls_file = os.path.join('tests', 'data', 'geolife', 'geolife_triplegs.csv')
        tpls = ti.read_triplegs_csv(tpls_file, tz='utc', index_col='id')

        D_single = tpls.iloc[0:4].as_triplegs.similarity(dist_metric='dtw', n_jobs=1)
        D_multi = tpls.iloc[0:4].as_triplegs.similarity(dist_metric='dtw', n_jobs=4)

        assert np.isclose(np.sum(np.abs(D_single - D_multi)), 0)

    def test_trajectory_distance_via_accessor_xy(self):
        tpls_file = os.path.join('tests', 'data', 'geolife', 'geolife_triplegs.csv')
        tpls = ti.read_triplegs_csv(tpls_file, tz='utc', index_col='id')

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


class Testcalc_haversine_length_of_linestring:
    def test_length(self, gdf_ls):
        """check if `calculate_haversine_length` runs without errors"""
        length = calculate_haversine_length(gdf_ls)

        assert length[0] < length[1]


class Test_calculate_haversine_length_single:
    def Test_length(self, single_linestring):
        """check if the length of a longer linestring is calculated correctly up to some meters"""
        length = _calculate_haversine_length_single(single_linestring)

        assert 1020 < length < 1030
