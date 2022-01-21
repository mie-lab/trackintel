import os
from functools import WRAPPER_ASSIGNMENTS
import pytest
import trackintel as ti
import numpy as np
from geopandas.testing import assert_geodataframe_equal

from trackintel.model.util import _copy_docstring
from trackintel.io.postgis import read_trips_postgis


@pytest.fixture
def example_triplegs():
    """Generate input data for trip generation from geolife positionfixes"""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, sp = pfs.as_positionfixes.generate_staypoints(
        method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
    )
    sp = sp.as_staypoints.create_activity_flag(time_threshold=15)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
    # the correct speeds were computed manually in Python
    correct_mean_pfs = [
        10.130884454034259,
        10.678073112625503,
        5.926792247569468,
        6.050364744922858,
        4.491716804294943,
        10.674771564277739,
        8.45313299158666,
        6.043023605853386,
        7.410984604978216,
        3.8711394932699066,
        12.97103423249977,
        14.43808554236438,
        2.9742346175310743,
        13.420204804955388,
        8.53979455145249,
        9.829183886902651,
        14.079778494712114,
        8.353638261463793,
        10.48314263336621,
        4.454370862584902,
        11.526663056213618,
        10.65636700582968,
    ]
    # convert to ms
    correct_mean_pfs = np.array(correct_mean_pfs) / 3.6

    correct_simple_speed = [
        2.829121178320859,
        3.915482482569027,
        1.3784047725622866,
        1.6636467482223185,
        1.3309113622426305,
        2.965214323410483,
        1.5825486291732953,
        1.7511353695291132,
        2.0586068347161715,
        1.1931794724765687,
        2.958540587467701,
        3.6256270589453927,
        0.8261762826475207,
        3.258519963963093,
        1.9820872090585189,
        2.1899718143436426,
        2.10801868184735,
        1.173059235444356,
        1.9918980847659702,
        1.1889846356657174,
        2.8269066914147554,
        1.991111700250081,
    ]
    return pfs, tpls, correct_mean_pfs, correct_simple_speed


@pytest.fixture
def load_positionfixes():
    """Load test positionfixes"""
    pfs = ti.io.file.read_positionfixes_csv(os.path.join("tests", "data", "positionfixes.csv"), sep=";", index_col="id")
    # the correct speeds were computed manually in Python
    correct_speeds = np.array([8.82100607, 8.82100607, 0.36585538, 1.93127652, 19.60643425, 2.07086017]) / 3.6
    return pfs, correct_speeds


class TestSpeedPositionfixes:
    def test_positionfixes_stable(self, load_positionfixes):
        """Test whether the positionfixes stay the same apart from the new speed column"""
        pfs, _ = load_positionfixes
        speed_pfs = ti.model.util.get_speed_positionfixes(pfs)
        assert_geodataframe_equal(pfs, speed_pfs.drop(columns=["speed"]))

    def test_accessor(self, load_positionfixes):
        """Test whether the positionfixes stay the same apart from the new speed column"""
        pfs, _ = load_positionfixes
        speed_pfs = pfs.as_positionfixes.get_speed()
        assert_geodataframe_equal(pfs, speed_pfs.drop(columns=["speed"]))

    def test_speed_correct(self, load_positionfixes):
        """Test whether the correct speed values are computed"""
        pfs, correct_speeds = load_positionfixes
        # assert first two are the same
        speed_pfs = ti.model.util.get_speed_positionfixes(pfs)
        assert speed_pfs.loc[speed_pfs.index[0], "speed"] == speed_pfs.loc[speed_pfs.index[1], "speed"]
        assert np.all(np.isclose(speed_pfs["speed"].values, correct_speeds, rtol=1e-06))

    def test_one_speed(self, load_positionfixes):
        """Test for each individual speed whether is is correct"""
        pfs, correct_speeds = load_positionfixes
        # compute speeds
        speed_pfs = ti.model.util.get_speed_positionfixes(pfs)
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


class TestPfsMeanSpeedTriplegs:
    def test_triplegs_stable(self, example_triplegs):
        """Test whether the triplegs stay the same apart from the new speed column"""
        pfs, tpls, _, _ = example_triplegs
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

    def test_tripleg_speed_correct(self, example_triplegs):
        """Test whether the computed mean speed values correspond to the one yielded from linestrings"""
        pfs, tpls, ground_truth_speed, _ = example_triplegs
        tpls_speed = ti.model.util.get_speed_triplegs(tpls, pfs, method="pfs_mean_speed")
        assert np.all(np.isclose(tpls_speed["speed"].values, ground_truth_speed, rtol=1e-06))

    def test_accessor(self, example_triplegs):
        """Test whether the computed mean speed values correspond to the one yielded from linestrings"""
        pfs, tpls, ground_truth_speed, _ = example_triplegs
        tpls_speed = tpls.as_triplegs.get_speed(pfs, method="pfs_mean_speed")
        assert np.all(np.isclose(tpls_speed["speed"].values, ground_truth_speed, rtol=1e-06))


class TestSimpleSpeedTriplegs:
    def test_triplegs_stable(self, example_triplegs):
        """Test whether the triplegs stay the same apart from the new speed column"""
        _, tpls, _, _ = example_triplegs
        tpls_speed = ti.model.util.get_speed_triplegs(tpls)
        assert_geodataframe_equal(tpls, tpls_speed.drop(columns=["speed"]))

    def test_tripleg_speed_correct(self, example_triplegs):
        """Test whether the computed mean speed values correspond to the one yielded from linestrings"""
        _, tpls, _, ground_truth_speed = example_triplegs
        tpls_speed = ti.model.util.get_speed_triplegs(tpls)
        assert np.all(np.isclose(tpls_speed["speed"].values, ground_truth_speed, rtol=1e-06))

    def test_accessor(self, example_triplegs):
        """Test whether the computed mean speed values correspond to the one yielded from linestrings"""
        _, tpls, _, ground_truth_speed = example_triplegs
        tpls_speed = tpls.as_triplegs.get_speed()
        assert np.all(np.isclose(tpls_speed["speed"].values, ground_truth_speed, rtol=1e-06))

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
