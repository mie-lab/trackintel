import os
import pandas as pd
import pytest

import trackintel as ti


class TestGenerate_tours:
    """Tests for generate_tours() method."""

    def test_generate_trips(self):
        trips = ti.io.file.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")
        trips_, tours = ti.preprocessing.trips.generate_tours(trips, max_dist=100)

        tours_loaded = ti.io.file.read_tours_csv(
            os.path.join("tests", "data", "geolife_long", "tours.csv"), index_col="id"
        )
        # check only first five columns because journey and location ID are empty in this case
        pd.testing.assert_frame_equal(tours_loaded.iloc[:, :5], tours.iloc[:, :5])

    def test_tours_location(self):
        """Test generate tours method with the locations given as an argument"""
        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
        )
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)
        stps, tpls, trips = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=15)
        # generate locations
        stps, locs = stps.as_staypoints.generate_locations(method="dbscan", epsilon=100, num_samples=1)

        tours_loaded = ti.io.file.read_tours_csv(
            os.path.join("tests", "data", "geolife_long", "tours.csv"), index_col="id"
        )
        tours_loaded["journey"] = tours_loaded["journey"].astype(object)  # need to cast because of NaNs

        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, stps_w_locs=stps)

        pd.testing.assert_frame_equal(tours_loaded, tours)

    def test_accessor(self):
        """Test if the accessor leads to the same results as the explicit function."""
        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
        )
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)
        stps, tpls, trips = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=15)

        # generate tours using the explicit function import
        trips_expl, tours_expl = ti.preprocessing.trips.generate_tours(trips)

        # generate tours using the accessor
        trips_acc, tours_acc = trips.as_trips.generate_tours()

        pd.testing.assert_frame_equal(trips_expl, trips_acc)
        pd.testing.assert_frame_equal(tours_expl, tours_acc)

    def test_tours_with_gaps(self):
        """Test if the argument max_gap_size leads to correct behaviour"""
        trips = ti.io.file.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")
        # make a gap
        trips = trips.drop(10)
        _, tours_no_gaps_allowed = trips.as_trips.generate_tours(max_dist=30, max_gap_size=0)
        assert tours_no_gaps_allowed.empty  # no tours can be found with this max_dist if no gaps are allowed
        _, tours_one_gap_allowed = trips.as_trips.generate_tours(max_dist=30, max_gap_size=1)
        assert len(tours_one_gap_allowed) == 1

    def test_trips_on_tour(self):
        trips = ti.io.file.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, max_dist=100)

        # check datatype
        assert trips_out["tour_id"].dtype == "Int64"
        # check correct
        trips_on_tour = trips_out[~pd.isna(trips_out["tour_id"])]
        for tour_id, df in trips_on_tour.groupby("tour_id"):
            # start and end staypoints must correspond
            assert tours.loc[tour_id, "origin_staypoint_id"] == df.iloc[0]["origin_staypoint_id"]
            assert tours.loc[tour_id, "destination_staypoint_id"] == df.iloc[-1]["destination_staypoint_id"]
            # start and end times must correspont
            assert tours.loc[tour_id, "started_at"] == df.iloc[0]["started_at"]
            assert tours.loc[tour_id, "finished_at"] == df.iloc[-1]["finished_at"]
