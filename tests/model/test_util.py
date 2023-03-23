import os
from functools import WRAPPER_ASSIGNMENTS

from geopandas import GeoDataFrame
import numpy as np
from pandas import Timestamp, Timedelta
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point

import trackintel as ti
from trackintel.io.postgis import read_trips_postgis
from trackintel.model.util import _copy_docstring, get_speed_positionfixes


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


@pytest.fixture
def load_positionfixes():
    """Load test positionfixes"""
    path = os.path.join("tests", "data", "positionfixes.csv")
    pfs = ti.read_positionfixes_csv(path, sep=";", index_col="id", crs="EPSG:4326")
    # the correct speeds were computed manually in Python
    correct_speeds = np.array([8.82100607, 8.82100607, 0.36585538, 1.93127652, 19.60643425, 2.07086017]) / 3.6
    return pfs, correct_speeds


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
            dist = ti.geogr.point_distances.haversine_dist(point1.x, point1.y, point2.x, point2.y)[0]
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
        start_time = Timestamp("2022-05-26 23:59:59")
        second = Timedelta("1s")
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 1.0)  # distance of sqrt(2)
        p3 = Point(4.0, 5.0)  # distance of 5
        d = [
            {"tracked_at": start_time, "g": p1},
            {"tracked_at": start_time + second, "g": p2},
            {"tracked_at": start_time + 3 * second, "g": p3},
        ]
        pfs = GeoDataFrame(d, geometry="g", crs="EPSG:2056")
        pfs = get_speed_positionfixes(pfs)
        correct_speed = np.array((np.sqrt(2), np.sqrt(2), 5 / 2))
        assert np.all(np.isclose(pfs["speed"].to_numpy(), correct_speed, rtol=1e-6))


class TestPfsMeanSpeedTriplegs:
    def test_triplegs_stable(self, example_triplegs):
        """Test whether the triplegs stay the same apart from the new speed column"""
        pfs, tpls = example_triplegs
        tpls_speed = ti.model.util.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")
        assert_geodataframe_equal(tpls, tpls_speed.drop(columns=["speed"]))

    def test_tripleg_id_assertion(self, load_positionfixes):
        """Test whether an error is triggered if wrong posistionfixes are used as input"""
        pfs, tpls = load_positionfixes
        with pytest.raises(Exception) as e_info:
            _ = ti.model.util.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")
            assert e_info == "Positionfixes must include column tripleg_id"

    def test_pfs_exist_assertion(self, load_positionfixes):
        """Test whether an error is triggered if wrong posistionfixes are used as input"""
        _, tpls = load_positionfixes
        with pytest.raises(Exception) as e_info:
            _ = ti.model.util.get_speed_triplegs(tpls, None, method="pfs_mean_speed")
            assert e_info == "Method pfs_mean_speed requires positionfixes as input"

    def test_one_speed_correct(self, example_triplegs):
        """Test whether speed computation is correct with one example"""
        pfs, tpls = example_triplegs
        tpls_speed = ti.model.util.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")
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
        tpls_speed_normal = ti.model.util.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")
        assert_geodataframe_equal(tpls_speed_acc, tpls_speed_normal)

    def test_pfs_input_assertion(self, example_triplegs):
        """Test whether an AttributeError is raised if no positionfixes are provided as input"""
        error_msg = 'Method "pfs_mean_speed" requires positionfixes as input.'
        _, tpls = example_triplegs
        with pytest.raises(AttributeError, match=error_msg):
            ti.model.util.get_speed_triplegs(tpls, None, method="pfs_mean_speed")

    def test_pfs_tripleg_id_assertion(self, example_triplegs):
        """Test whether an AttributeError is raised if positionfixes do not provide column "tripleg_id"."""
        error_msg = 'Positionfixes must include column "tripleg_id".'
        pfs, tpls = example_triplegs
        pfs.drop(columns=["tripleg_id"], inplace=True)
        with pytest.raises(AttributeError, match=error_msg):
            ti.model.util.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")


class TestSimpleSpeedTriplegs:
    def test_triplegs_stable(self, example_triplegs):
        """Test whether the triplegs stay the same apart from the new speed column"""
        _, tpls = example_triplegs
        tpls_speed = ti.model.util.get_speed_triplegs(tpls)
        assert_geodataframe_equal(tpls, tpls_speed.drop(columns=["speed"]))

    def test_one_speed_correct(self, example_triplegs):
        """Test with one example whether the computed speeds are correct"""
        _, tpls = example_triplegs
        tpls_speed = ti.model.util.get_speed_triplegs(tpls)
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
        tpls_speed_normal = ti.model.util.get_speed_triplegs(tpls)
        assert_geodataframe_equal(tpls_speed_acc, tpls_speed_normal)

    def test_method_error(self):
        """Test whether an error is triggered if wrong posistionfixes are used as input"""
        with pytest.raises(Exception) as e_info:
            _ = ti.model.util.get_speed_triplegs(None, None, method="wrong_method")
            assert e_info == "Method wrong_method not known for speed computation."


class Test_copy_docstring:
    def test_default(self):
        @_copy_docstring(read_trips_postgis)
        def bar(b: int) -> int:
            """Old docstring."""
            pass

        old_docs = """Old docstring."""
        print(type(old_docs))

        for wa in WRAPPER_ASSIGNMENTS:
            attr_foo = getattr(read_trips_postgis, wa)
            attr_bar = getattr(bar, wa)
            if wa == "__doc__":
                assert attr_foo == attr_bar
                assert attr_bar != old_docs
            else:
                assert attr_foo != attr_bar
