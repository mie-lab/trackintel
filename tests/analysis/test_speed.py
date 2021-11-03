import os
import pytest
import trackintel as ti
import numpy as np
from geopandas.testing import assert_geodataframe_equal


@pytest.fixture
def example_triplegs():
    """Generate input data for trip generation from geolife positionfixes"""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, sp = pfs.as_positionfixes.generate_staypoints(
        method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
    )
    sp = sp.as_staypoints.create_activity_flag(time_threshold=15)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
    return pfs, sp, tpls


@pytest.fixture
def load_positionfixes():
    """Load test positionfixes"""
    pfs = ti.io.file.read_positionfixes_csv(os.path.join("tests", "data", "positionfixes.csv"), sep=";", index_col="id")
    correct_speeds = np.array([8.82100607, 8.82100607, 0.36585538, 1.93127652, 19.60643425, 2.07086017])
    return pfs, correct_speeds


class TestSpeedPositionfixes:
    def test_positionfixes_stable(self, load_positionfixes):
        """Test whether the positionfixes stay the same apart from the new speed column"""
        pfs, _ = load_positionfixes
        speed_pfs = ti.analysis.speed.speed_positionfixes(pfs)
        assert_geodataframe_equal(pfs, speed_pfs.drop(columns=["speed"]))

    def test_speed_correct(self, load_positionfixes):
        """Test whether the correct speed values are computed"""
        pfs, correct_speeds = load_positionfixes
        # assert first two are the same
        speed_pfs = ti.analysis.speed.speed_positionfixes(pfs)
        assert speed_pfs.loc[speed_pfs.index[0], "speed"] == speed_pfs.loc[speed_pfs.index[1], "speed"]
        assert np.all(np.isclose(speed_pfs["speed"].values, correct_speeds))

    def test_one_speed(self, load_positionfixes):
        """Test for each individual speed whether is is correct"""
        pfs, correct_speeds = load_positionfixes
        # compute speeds
        speed_pfs = ti.analysis.speed.speed_positionfixes(pfs)
        computed_speeds = speed_pfs["speed"].values
        # test for each row whether the speed is correct
        for ind in range(1, len(correct_speeds)):
            ind_prev = ind - 1
            time_diff = (pfs.loc[ind, "tracked_at"] - pfs.loc[ind_prev, "tracked_at"]).total_seconds()
            point1 = pfs.loc[ind_prev, "geom"]
            point2 = pfs.loc[ind, "geom"]
            dist = ti.geogr.point_distances.haversine_dist(point1.x, point1.y, point2.x, point2.y)[0]
            assert np.isclose(3.6 * dist / time_diff, computed_speeds[ind])
            assert np.isclose(3.6 * dist / time_diff, correct_speeds[ind])


class TestSpeedTriplegs:
    def test_triplegs_stable(self, example_triplegs):
        """Test whether the triplegs stay the same apart from the new speed column"""
        pfs, _, tpls = example_triplegs
        tpls_speed = ti.analysis.speed.mean_speed_triplegs(pfs, tpls)
        assert_geodataframe_equal(tpls, tpls_speed.drop(columns=["mean_speed"]))

    def test_tripleg_id_assertion(self, load_positionfixes):
        """Test whether an error is triggered if wrong posistionfixes are used as input"""
        pfs, _ = load_positionfixes
        with pytest.raises(Exception) as e_info:
            _ = ti.analysis.speed.mean_speed_triplegs(pfs, None)
            assert e_info == "Positionfixes must include column tripleg_id"

    def test_tripleg_speed_correct(self, example_triplegs):
        """Test whether the computed mean speed values correspond to the one yielded from linestrings"""
        pfs, _, tpls = example_triplegs
        tpls_speed = ti.analysis.speed.mean_speed_triplegs(pfs, tpls)
        ground_truth_speed = [
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
        assert np.all(np.isclose(tpls_speed["mean_speed"].values, ground_truth_speed))
