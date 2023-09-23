import os
from math import radians

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import haversine_distances

import trackintel as ti
from trackintel.geogr.distances import (
    calculate_distance_matrix,
    calculate_haversine_length,
    check_gdf_planar,
    get_speed_positionfixes,
    point_haversine_dist,
    meters_to_decimal_degrees,
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


@pytest.fixture
def load_positionfixes():
    """Load test positionfixes"""
    path = os.path.join("tests", "data", "positionfixes.csv")
    pfs = ti.read_positionfixes_csv(path, sep=";", index_col="id", crs="EPSG:4326")
    # the correct speeds were computed manually in Python
    correct_speeds = np.array([8.82100607, 8.82100607, 0.36585538, 1.93127652, 19.60643425, 2.07086017]) / 3.6
    return pfs, correct_speeds


@pytest.fixture
def example_triplegs():
    """Generate input data for trip generation from geolife positionfixes"""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, sp = pfs.as_positionfixes.generate_staypoints(
        method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
    )
    sp = sp.as_staypoints.create_activity_flag(time_threshold=15)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
    return pfs, tpls


class TestHaversineDist:
    def test_haversine_dist(self):
        """
        input_latlng saves different combinations of haversine-distances in meters and the longitude & latitudes from
        two different points in WGS84

        References
        ------
        https://community.esri.com/groups/coordinate-reference-systems/blog/2017/10/05/haversine-formula
        """

        # {haversine-distance in meters[longitude_P1, latitudes_P1, longitude_P2, latitudes_P2]}
        input_latlng = {
            18749: [8.5, 47.3, 8.7, 47.2],  # Source: see Information to function
            5897658.289: [-0.116773, 51.510357, -77.009003, 38.889931],
            3780627: [0.0, 4.0, 0.0, 38],
            # Source for next lines: self-computation with formula from link above
            2306879.363: [-7.345, -7.345, 7.345, 7.345],
            13222121.519: [-0.118746, 73.998, -120.947783, -21.4783],
            785767.221: [50, 0, 45, 5],
        }

        for haversine, latlng in input_latlng.items():
            haversine_output = point_haversine_dist(latlng[0], latlng[1], latlng[2], latlng[3])
            assert np.isclose(haversine_output, haversine, atol=0.1)

    def test_haversine_vectorized(self):
        sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        sp = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id")
        x = sp.geometry.x.values
        y = sp.geometry.y.values

        n = len(x)
        # our distance
        ix_1, ix_2 = np.triu_indices(n, k=1)

        x1 = x[ix_1]
        y1 = y[ix_1]
        x2 = x[ix_2]
        y2 = y[ix_2]

        d_ours = point_haversine_dist(x1, y1, x2, y2)

        # their distance
        x_rad = np.asarray([radians(_) for _ in x])
        y_rad = np.asarray([radians(_) for _ in y])
        yx = np.concatenate((y_rad.reshape(-1, 1), x_rad.reshape(-1, 1)), axis=1)

        D_theirs = haversine_distances(yx, yx) * 6371000
        d_theirs = D_theirs[ix_1, ix_2]
        assert np.sum(np.abs(d_ours - d_theirs)) < 0.01  # 1cm for 58 should be good enough

    def test_example_from_sklean(self):
        bsas = [-34.83333, -58.5166646]
        paris = [49.0083899664, 2.53844117956]
        bsas_in_radians = [radians(_) for _ in bsas]
        paris_in_radians = [radians(_) for _ in paris]
        d_theirs = haversine_distances([bsas_in_radians, paris_in_radians]) * 6371000

        d_ours = point_haversine_dist(bsas[1], bsas[0], paris[1], paris[0])

        assert np.abs(d_theirs[1][0] - d_ours) < 0.01


class TestCalculate_distance_matrix:
    """Tests for the calculate_distance_matrix() function."""

    def test_shape_for_different_array_length(self):
        sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        sp = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id")

        x = sp.iloc[0:5]
        y = sp.iloc[5:15]

        d_euc1 = calculate_distance_matrix(X=x, Y=y, dist_metric="euclidean")
        d_euc2 = calculate_distance_matrix(X=y, Y=x, dist_metric="euclidean")
        d_hav1 = calculate_distance_matrix(X=x, Y=y, dist_metric="haversine")
        d_hav2 = calculate_distance_matrix(X=y, Y=x, dist_metric="haversine")

        assert d_euc1.shape == d_hav1.shape == (5, 10)
        assert d_euc2.shape == d_hav2.shape == (10, 5)
        assert np.isclose(0, np.sum(np.abs(d_euc1 - d_euc2.T)))
        assert np.isclose(0, np.sum(np.abs(d_hav1 - d_hav2.T)))

    def test_keyword_combinations(self):
        sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        sp = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id")

        x = sp.iloc[0:5]
        y = sp.iloc[5:15]

        _ = calculate_distance_matrix(X=x, Y=y, dist_metric="euclidean", n_jobs=-1)
        _ = calculate_distance_matrix(X=y, Y=x, dist_metric="haversine", n_jobs=-1)
        d_mink1 = calculate_distance_matrix(X=x, Y=x, dist_metric="minkowski", p=1)
        d_mink2 = calculate_distance_matrix(X=x, Y=x, dist_metric="minkowski", p=2)
        d_euc = calculate_distance_matrix(X=x, Y=x, dist_metric="euclidean")

        assert not np.array_equal(d_mink1, d_mink2)
        assert np.array_equal(d_euc, d_mink2)

    def test_compare_haversine_to_scikit_xy(self):
        """Test the results using our haversine function and scikit function."""
        sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        sp = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id")
        our_d_matrix = calculate_distance_matrix(X=sp, Y=sp, dist_metric="haversine")

        x = sp.geometry.x.values
        y = sp.geometry.y.values

        x_rad = np.asarray([radians(_) for _ in x])
        y_rad = np.asarray([radians(_) for _ in y])
        yx = np.concatenate((y_rad.reshape(-1, 1), x_rad.reshape(-1, 1)), axis=1)

        their_d_matrix = pairwise_distances(yx, metric="haversine") * 6371000
        # atol = 10mm
        assert np.allclose(our_d_matrix, their_d_matrix, atol=0.01)

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


class TestCheck_gdf_planar:
    """Tests for check_gdf_planar() method."""

    def test_transformation(self):
        """Check if data gets transformed."""
        file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(file, sep=";", crs="EPSG:4326", index_col=None)
        pfs_2056 = pfs.to_crs("EPSG:2056")
        bool, pfs_4326 = check_gdf_planar(pfs_2056, transform=True)
        assert not bool
        assert_geodataframe_equal(pfs, pfs_4326, check_less_precise=True)

    def test_crs_warning(self):
        """Check if warning is raised for data without crs."""
        file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(file, sep=";", crs=None, index_col=None)
        with pytest.warns(UserWarning):
            assert check_gdf_planar(pfs) is False

    def test_if_planer(self):
        """Check if planer crs is successfully checked."""
        p1 = Point(8.5067847, 47.4)
        t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")

        list_dict = [
            {"user_id": 0, "started_at": t1, "finished_at": t1, "geom": p1},
        ]
        # a geographic crs different than wgs1984
        sp = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4610")
        assert check_gdf_planar(sp) is False

        # wgs1984
        sp = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
        assert check_gdf_planar(sp) is False

        # wgs1984 to swiss planer
        sp = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
        sp = sp.to_crs("EPSG:2056")
        assert check_gdf_planar(sp) is True

    def test_none_crs_transform(self):
        """Check if crs gets set to WGS84."""
        file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(file, sep=";", crs=None, index_col=None)
        bool, pfs_4326 = check_gdf_planar(pfs, transform=True)
        assert not bool
        pfs.crs = "EPSG:4326"
        assert_geodataframe_equal(pfs, pfs_4326)


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
        ls1, ls2 = gdf_lineStrings.geometry
        ls1, ls2 = np.array(ls1.coords), np.array(ls2.coords)
        assert length[0] == np.sum(point_haversine_dist(ls1[:-1, 0], ls1[:-1, 1], ls1[1:, 0], ls1[1:, 1]))
        assert length[1] == np.sum(point_haversine_dist(ls2[:-1, 0], ls2[:-1, 1], ls2[1:, 0], ls2[1:, 1]))


class TestSpeedPositionfixes:
    def test_positionfixes_stable(self, load_positionfixes):
        """Test whether the positionfixes stay the same apart from the new speed column"""
        pfs, _ = load_positionfixes
        speed_pfs = get_speed_positionfixes(pfs)
        assert_geodataframe_equal(pfs, speed_pfs.drop(columns=["speed"]))

    def test_accessor(self, load_positionfixes):
        """Test whether the accessor yields the same output as the function"""
        pfs, _ = load_positionfixes
        speed_pfs_acc = pfs.as_positionfixes.get_speed()
        speed_pfs_normal = get_speed_positionfixes(pfs)
        assert_geodataframe_equal(speed_pfs_acc, speed_pfs_normal)

    def test_speed_correct(self, load_positionfixes):
        """Test whether the correct speed values are computed"""
        pfs, correct_speeds = load_positionfixes
        # assert first two are the same
        speed_pfs = get_speed_positionfixes(pfs)
        assert speed_pfs.loc[speed_pfs.index[0], "speed"] == speed_pfs.loc[speed_pfs.index[1], "speed"]
        assert np.all(np.isclose(speed_pfs["speed"].values, correct_speeds, rtol=1e-06))

    def test_one_speed(self, load_positionfixes):
        """Test for each individual speed whether is is correct"""
        pfs, correct_speeds = load_positionfixes
        # compute speeds
        speed_pfs = get_speed_positionfixes(pfs)
        computed_speeds = speed_pfs["speed"].values
        # test for each row whether the speed is correct
        for ind in range(1, len(correct_speeds)):
            ind_prev = ind - 1
            time_diff = (pfs.loc[ind, "tracked_at"] - pfs.loc[ind_prev, "tracked_at"]).total_seconds()
            point1 = pfs.loc[ind_prev, "geom"]
            point2 = pfs.loc[ind, "geom"]
            dist = point_haversine_dist(point1.x, point1.y, point2.x, point2.y)[0]
            assert np.isclose(dist / time_diff, computed_speeds[ind], rtol=1e-06)
            assert np.isclose(dist / time_diff, correct_speeds[ind], rtol=1e-06)

    def test_geometry_name(self, load_positionfixes):
        """Test if the geometry name can be set freely."""
        pfs, _ = load_positionfixes
        pfs.rename(columns={"geom": "freely_chosen_geometry_name"}, inplace=True)
        pfs.set_geometry("freely_chosen_geometry_name", inplace=True)
        get_speed_positionfixes(pfs)

    def test_planar_geometry(self):
        """Test function for geometry that is planar."""
        start_time = pd.Timestamp("2022-05-26 23:59:59")
        second = pd.Timedelta("1s")
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 1.0)  # distance of sqrt(2)
        p3 = Point(4.0, 5.0)  # distance of 5
        d = [
            {"tracked_at": start_time, "g": p1},
            {"tracked_at": start_time + second, "g": p2},
            {"tracked_at": start_time + 3 * second, "g": p3},
        ]
        pfs = gpd.GeoDataFrame(d, geometry="g", crs="EPSG:2056")
        pfs = get_speed_positionfixes(pfs)
        correct_speed = np.array((np.sqrt(2), np.sqrt(2), 5 / 2))
        assert np.all(np.isclose(pfs["speed"].to_numpy(), correct_speed, rtol=1e-6))


class TestPfsMeanSpeedTriplegs:
    def test_triplegs_stable(self, example_triplegs):
        """Test whether the triplegs stay the same apart from the new speed column"""
        pfs, tpls = example_triplegs
        tpls_speed = ti.geogr.distances.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")
        assert_geodataframe_equal(tpls, tpls_speed.drop(columns=["speed"]))

    def test_tripleg_id_assertion(self, load_positionfixes):
        """Test whether an error is triggered if wrong posistionfixes are used as input"""
        pfs, tpls = load_positionfixes
        with pytest.raises(Exception) as e_info:
            _ = ti.geogr.distances.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")
            assert e_info == "Positionfixes must include column tripleg_id"

    def test_pfs_exist_assertion(self, load_positionfixes):
        """Test whether an error is triggered if wrong posistionfixes are used as input"""
        _, tpls = load_positionfixes
        with pytest.raises(Exception) as e_info:
            _ = ti.geogr.distances.get_speed_triplegs(tpls, None, method="pfs_mean_speed")
            assert e_info == "Method pfs_mean_speed requires positionfixes as input"

    def test_one_speed_correct(self, example_triplegs):
        """Test whether speed computation is correct with one example"""
        pfs, tpls = example_triplegs
        tpls_speed = ti.geogr.distances.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")
        # compute speed for one tripleg manually
        test_tpl = tpls.index[0]
        test_pfs = pfs[pfs["tripleg_id"] == test_tpl]
        pfs_speed = get_speed_positionfixes(test_pfs)
        test_tpl_speed = np.mean(pfs_speed["speed"].values[1:])
        # compare to the one computed in the function
        computed_tpls_speed = tpls_speed.loc[test_tpl]["speed"]
        assert test_tpl_speed == computed_tpls_speed

    def test_accessor(self, example_triplegs):
        """Test whether the accessor yields the same output as the function"""
        pfs, tpls = example_triplegs
        tpls_speed_acc = tpls.as_triplegs.get_speed(pfs, method="pfs_mean_speed")
        tpls_speed_normal = ti.geogr.distances.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")
        assert_geodataframe_equal(tpls_speed_acc, tpls_speed_normal)

    def test_pfs_input_assertion(self, example_triplegs):
        """Test whether an AttributeError is raised if no positionfixes are provided as input"""
        error_msg = 'Method "pfs_mean_speed" requires positionfixes as input.'
        _, tpls = example_triplegs
        with pytest.raises(AttributeError, match=error_msg):
            ti.geogr.distances.get_speed_triplegs(tpls, None, method="pfs_mean_speed")

    def test_pfs_tripleg_id_assertion(self, example_triplegs):
        """Test whether an AttributeError is raised if positionfixes do not provide column "tripleg_id"."""
        error_msg = 'Positionfixes must include column "tripleg_id".'
        pfs, tpls = example_triplegs
        pfs.drop(columns=["tripleg_id"], inplace=True)
        with pytest.raises(AttributeError, match=error_msg):
            ti.geogr.distances.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")


class TestSimpleSpeedTriplegs:
    def test_triplegs_stable(self, example_triplegs):
        """Test whether the triplegs stay the same apart from the new speed column"""
        _, tpls = example_triplegs
        tpls_speed = ti.geogr.distances.get_speed_triplegs(tpls)
        assert_geodataframe_equal(tpls, tpls_speed.drop(columns=["speed"]))

    def test_one_speed_correct(self, example_triplegs):
        """Test with one example whether the computed speeds are correct"""
        _, tpls = example_triplegs
        tpls_speed = ti.geogr.distances.get_speed_triplegs(tpls)
        test_tpl = tpls.index[0]
        gt_distance = 2025.650764
        test_tpl_speed = (
            gt_distance / (tpls.loc[test_tpl]["finished_at"] - tpls.loc[test_tpl]["started_at"]).total_seconds()
        )
        # compare to the one computed in the function
        computed_tpls_speed = tpls_speed.loc[test_tpl]["speed"]
        assert np.isclose(test_tpl_speed, computed_tpls_speed, rtol=1e-04)

    def test_accessor(self, example_triplegs):
        """Test whether the accessor yields the same output as the function"""
        _, tpls = example_triplegs
        tpls_speed_acc = tpls.as_triplegs.get_speed()
        tpls_speed_normal = ti.geogr.distances.get_speed_triplegs(tpls)
        assert_geodataframe_equal(tpls_speed_acc, tpls_speed_normal)

    def test_method_error(self):
        """Test whether an error is triggered if wrong posistionfixes are used as input"""
        with pytest.raises(Exception) as e_info:
            _ = ti.geogr.distances.get_speed_triplegs(None, None, method="wrong_method")
            assert e_info == "Method wrong_method not known for speed computation."
