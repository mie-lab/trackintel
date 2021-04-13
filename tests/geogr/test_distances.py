import os
from math import radians

import geopandas as gpd
import numpy as np
import pytest
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
from sklearn.metrics import pairwise_distances
from geopandas.testing import assert_geodataframe_equal

import trackintel as ti
from trackintel.geogr.distances import (
    check_gdf_crs,
    meters_to_decimal_degrees,
    calculate_distance_matrix,
    calculate_haversine_length,
    _calculate_haversine_length_single,
)


@pytest.fixture
def gdf_lineStrings():
    """Construct a gdf that contains two LineStrings."""
    ls_short = LineString([(13.476808430, 48.573711823), (13.506804, 48.939008), (13.4664690, 48.5706414)])
    ls_long = LineString([(13.476808430, 48.573711823), (11.5675446, 48.1485459), (8.5067847, 47.4084269)])

    a_list = [(0, ls_short), (1, ls_long)]
    gdf = gpd.GeoDataFrame(a_list, columns=["id", "geometry"]).set_geometry("geometry")
    gdf = gdf.set_crs("wgs84")
    return gdf


@pytest.fixture
def single_linestring():
    """Construct LineString that has ~1024 m in QGIS."""
    return wkt.loads(
        """LineString(13.47671401745228259 
    48.57364142178052901, 13.47510901146785933
    48.5734004715611789, 13.47343656825720082
    48.57335585102421049, 13.47172366271079369
    48.57318629262447018, 13.4697275208142031
    48.57325768570418489, 13.4680415901582915
    48.57348971251707326, 13.46604544826169914
    48.57348971251707326, 13.46473716607271243
    48.57319521676494389, 13.46319959731452798
    48.57253482611510975, 13.46319959731452798
    48.57253482611510975)"""
    )


@pytest.fixture
def geolife_tpls():
    """Read geolife triplegs for testing."""
    tpls_file = os.path.join("tests", "data", "geolife", "geolife_triplegs.csv")
    tpls = ti.read_triplegs_csv(tpls_file, tz="utc", index_col="id")
    return tpls


class TestCalculate_distance_matrix:
    """Tests for the calculate_distance_matrix() function."""

    def test_shape_for_different_array_length(self):
        stps_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        stps = ti.read_staypoints_csv(stps_file, tz="utc", index_col="id")

        x = stps.iloc[0:5]
        y = stps.iloc[5:15]

        d_euc1 = calculate_distance_matrix(X=x, Y=y, dist_metric="euclidean")
        d_euc2 = calculate_distance_matrix(X=y, Y=x, dist_metric="euclidean")
        d_hav1 = calculate_distance_matrix(X=x, Y=y, dist_metric="haversine")
        d_hav2 = calculate_distance_matrix(X=y, Y=x, dist_metric="haversine")

        assert d_euc1.shape == d_hav1.shape == (5, 10)
        assert d_euc2.shape == d_hav2.shape == (10, 5)
        assert np.isclose(0, np.sum(np.abs(d_euc1 - d_euc2.T)))
        assert np.isclose(0, np.sum(np.abs(d_hav1 - d_hav2.T)))

    def test_keyword_combinations(self):
        stps_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        stps = ti.read_staypoints_csv(stps_file, tz="utc", index_col="id")

        x = stps.iloc[0:5]
        y = stps.iloc[5:15]

        _ = calculate_distance_matrix(X=x, Y=y, dist_metric="euclidean", n_jobs=-1)
        _ = calculate_distance_matrix(X=y, Y=x, dist_metric="haversine", n_jobs=-1)
        d_mink1 = calculate_distance_matrix(X=x, Y=x, dist_metric="minkowski", p=1)
        d_mink2 = calculate_distance_matrix(X=x, Y=x, dist_metric="minkowski", p=2)
        d_euc = calculate_distance_matrix(X=x, Y=x, dist_metric="euclidean")

        assert not np.array_equal(d_mink1, d_mink2)
        assert np.array_equal(d_euc, d_mink2)

    def test_compare_haversine_to_scikit_xy(self):
        stps_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        stps = ti.read_staypoints_csv(stps_file, tz="utc", index_col="id")
        our_d_matrix = calculate_distance_matrix(X=stps, Y=stps, dist_metric="haversine")

        x = stps.geometry.x.values
        y = stps.geometry.y.values

        x_rad = np.asarray([radians(_) for _ in x])
        y_rad = np.asarray([radians(_) for _ in y])
        yx = np.concatenate((y_rad.reshape(-1, 1), x_rad.reshape(-1, 1)), axis=1)

        their_d_matrix = pairwise_distances(yx, metric="haversine") * 6371000
        assert np.allclose(np.abs(our_d_matrix - their_d_matrix), 0, atol=0.001)  # atol = 1mm

    def test_trajectory_distance_dtw(self, geolife_tpls):
        """Calculate Linestring length using dtw, single and multi core."""
        tpls = geolife_tpls

        D_all = calculate_distance_matrix(X=tpls.iloc[0:4], dist_metric="dtw", n_jobs=-1)
        D_zero = calculate_distance_matrix(X=tpls.iloc[0:4], dist_metric="dtw", n_jobs=0)

        D_single = calculate_distance_matrix(X=tpls.iloc[0:4], dist_metric="dtw", n_jobs=1)
        D_multi = calculate_distance_matrix(X=tpls.iloc[0:4], dist_metric="dtw", n_jobs=4)

        assert np.isclose(np.sum(np.abs(D_single - D_multi)), 0)
        assert np.isclose(np.sum(np.abs(D_all - D_multi)), 0)
        assert np.isclose(np.sum(np.abs(D_zero - D_multi)), 0)

    def test_trajectory_distance_frechet(self, geolife_tpls):
        """Calculate Linestring length using frechet, single and multi core."""
        tpls = geolife_tpls

        D_single = calculate_distance_matrix(X=tpls.iloc[0:4], dist_metric="frechet", n_jobs=1)
        D_multi = calculate_distance_matrix(X=tpls.iloc[0:4], dist_metric="frechet", n_jobs=4)

        assert np.isclose(np.sum(np.abs(D_single - D_multi)), 0)

    def test_trajectory_distance_via_accessor_x(self, geolife_tpls):
        """Calculate Linestring length using dtw via accessor."""
        tpls = geolife_tpls

        D_single = tpls.iloc[0:4].as_triplegs.calculate_distance_matrix(dist_metric="dtw", n_jobs=1)
        D_multi = tpls.iloc[0:4].as_triplegs.calculate_distance_matrix(dist_metric="dtw", n_jobs=4)

        assert np.isclose(np.sum(np.abs(D_single - D_multi)), 0)

    def test_trajectory_distance_via_accessor_xy(self, geolife_tpls):
        """Calculate Linestring length using dtw via accessor."""
        tpls = geolife_tpls

        x = tpls.iloc[0:2]
        y = tpls.iloc[4:8]

        D_single = x.as_triplegs.calculate_distance_matrix(Y=y, dist_metric="dtw", n_jobs=1)
        D_multi = x.as_triplegs.calculate_distance_matrix(Y=y, dist_metric="dtw", n_jobs=4)

        assert np.isclose(np.sum(np.abs(D_single - D_multi)), 0)

    def test_trajectory_distance_error(self, geolife_tpls):
        """Test if the an error is raised when passing unknown 'dist_metric'."""
        tpls = geolife_tpls

        with pytest.raises(AttributeError):
            tpls.iloc[0:4].as_triplegs.calculate_distance_matrix(dist_metric=12345, n_jobs=1)
        with pytest.raises(AttributeError):
            tpls.iloc[0:4].as_triplegs.calculate_distance_matrix(dist_metric="random", n_jobs=1)

    def test_distance_error(self, single_linestring):
        """Test if the an error is raised when wrong geometry is passed."""
        # construct a gdf with two MultiLineStrings
        multi = MultiLineString([single_linestring, single_linestring])
        a_list = [(0, multi), (1, multi)]
        gdf = gpd.GeoDataFrame(a_list, columns=["id", "geometry"]).set_geometry("geometry")
        gdf = gdf.set_crs("wgs84")

        with pytest.raises(AttributeError):
            calculate_distance_matrix(X=gdf, dist_metric="dtw", n_jobs=1)


class TestCheck_gdf_crs:
    """Tests for check_gdf_crs() method."""

    def test_transformation(self):
        """Check if data gets transformed."""
        file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(file, sep=";", crs="EPSG:4326", index_col=None)
        pfs_2056 = pfs.to_crs("EPSG:2056")
        _, pfs_4326 = check_gdf_crs(pfs_2056, transform=True)
        assert_geodataframe_equal(pfs, pfs_4326, check_less_precise=True)

    def test_crs_warning(self):
        """Check if warning is raised for data without crs."""
        file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(file, sep=";", crs=None, index_col=None)
        with pytest.warns(UserWarning):
            check_gdf_crs(pfs)


class TestMetersToDecimalDegrees:
    """Tests for the meters_to_decimal_degrees() function."""

    def test_meters_to_decimal_degrees(self):
        input_result_dict = {
            1.0: {0: 111320, 23: 102470, 45: 78710, 67: 43496},
            0.1: {0: 11132, 23: 10247, 45: 7871, 67: 4349.6},
            0.01: {0: 1113.2, 23: 1024.7, 45: 787.1, 67: 434.96},
            0.001: {0: 111.32, 23: 102.47, 45: 78.71, 67: 43.496},
        }

        for degree, lat_output in input_result_dict.items():
            for lat, meters in lat_output.items():
                decimal_degree_output = meters_to_decimal_degrees(meters, lat)
                assert np.isclose(decimal_degree_output, degree, atol=0.1)


class Testcalc_haversine_length:
    """Tests for the calculate_haversine_length() function."""

    def test_length(self, gdf_lineStrings):
        """Check if `calculate_haversine_length` runs without errors."""
        length = calculate_haversine_length(gdf_lineStrings)

        assert length[0] < length[1]


class Test_calculate_haversine_length_single:
    """Tests for the _calculate_haversine_length_single() function."""

    def Test_length(self, single_linestring):
        """Check if the length of a longer linestring is calculated correctly up to some meters."""
        length = _calculate_haversine_length_single(single_linestring)

        assert 1020 < length < 1030
