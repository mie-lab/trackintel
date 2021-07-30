import os
import pandas as pd
import pytest
from shapely.geometry import MultiPoint, Point
import datetime
import geopandas as gpd

import trackintel as ti


@pytest.fixture
def example_trip_data():
    """Trips and locations for tour generation."""
    # define points for each staypoint
    stp_geom_mapping = {
        1: Point(8.5067847, 47.4),
        2: Point(8.5067847, 47.40001),
        3: Point(8.5067847, 47.6),
        4: Point(8.5067847, 47.7),
        5: Point(8.5067847, 47.399),
        6: Point(8.5067847, 47.60001),
    }

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")
    t4 = pd.Timestamp("1971-01-02 08:00:00", tz="utc")
    t5 = pd.Timestamp("1971-01-02 09:00:00", tz="utc")
    t6 = pd.Timestamp("1971-01-02 10:00:00", tz="utc")
    t7 = pd.Timestamp("1971-01-02 11:00:00", tz="utc")
    t8 = pd.Timestamp("1971-01-02 12:00:00", tz="utc")
    one_hour = datetime.timedelta(hours=1)

    trips_list_dict = [
        # loop
        {
            "id": 1,
            "user_id": 0,
            "started_at": t1,
            "finished_at": t2,
            "origin_staypoint_id": 1,
            "destination_staypoint_id": 2,
        },
        # this should not belong to tour because time-wise not connected
        {
            "id": 5,
            "user_id": 0,
            "started_at": t2,
            "finished_at": t2,
            "origin_staypoint_id": 2,
            "destination_staypoint_id": 3,
        },
        # long tour
        {
            "id": 2,
            "user_id": 0,
            "started_at": t3,
            "finished_at": t4,
            "origin_staypoint_id": 3,
            "destination_staypoint_id": 4,
        },
        {
            "id": 6,
            "user_id": 0,
            "started_at": t4,
            "finished_at": t5,
            "origin_staypoint_id": 4,
            "destination_staypoint_id": 5,
        },
        {
            "id": 15,
            "user_id": 0,
            "started_at": t5,
            "finished_at": t6,
            "origin_staypoint_id": 5,
            "destination_staypoint_id": 6,
        },
        # new user - gap in tour
        {
            "id": 7,
            "user_id": 1,
            "started_at": t3,
            "finished_at": t4,
            "origin_staypoint_id": 3,
            "destination_staypoint_id": 5,
        },
        {
            "id": 80,
            "user_id": 1,
            "started_at": t4,
            "finished_at": t5,
            "origin_staypoint_id": 4,
            "destination_staypoint_id": 3,
        },
        # no tour
        {
            "id": 3,
            "user_id": 1,
            "started_at": t5,
            "finished_at": t6,
            "origin_staypoint_id": 1,
            "destination_staypoint_id": 4,
        },
    ]
    # fill geom based on staypoints
    for row_dict in trips_list_dict:
        row_dict["geom"] = MultiPoint(
            (stp_geom_mapping[row_dict["origin_staypoint_id"]], stp_geom_mapping[row_dict["destination_staypoint_id"]])
        )

    # assign location IDs (only for testing, not a trackintel stps standard!)
    stps_locs = pd.DataFrame({"location_id": [1, 1, 2, 3, 1, 2]}, index=[1, 2, 3, 4, 5, 6])  # todo: name index?

    trips = gpd.GeoDataFrame(data=trips_list_dict, geometry="geom", crs="EPSG:4326")
    trips = trips.set_index("id")
    assert trips.as_trips
    return trips, stps_locs


class TestGenerate_tours:
    """Tests for generate_tours() method."""

    def test_generate_tours(self, example_trip_data):
        trips, stps_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips)
        # check that nothing else than the new column has changed in trips df
        assert all(trips_out.iloc[:, :6] == trips)
        # check that the two tours were found
        assert trips_out.loc[1, "tour_id"] == 0
        assert (
            trips_out.loc[2, "tour_id"] == 1 and trips_out.loc[6, "tour_id"] == 1 and trips_out.loc[15, "tour_id"] == 1
        )
        # all others have tour id nan
        user_1_df = trips_out[trips_out["user_id"] == 1]
        assert all(pd.isna(user_1_df["tour_id"]))

    def test_tours_with_gap(self, example_trip_data):
        trips, stps_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, max_nr_gaps=1)
        # new tour was found for user 1
        assert len(tours) == 3
        assert trips_out.loc[7, "tour_id"] == 2
        assert trips_out.loc[80, "tour_id"] == 2

    def test_tour_times(self, example_trip_data):
        trips, stps_locs = example_trip_data
        max_time = datetime.timedelta(days=1)
        trips_out, tours = ti.preprocessing.trips.generate_tours(
            trips, max_nr_gaps=1, max_time=datetime.timedelta(days=1)
        )
        # check that all times are below the max time
        for i, row in tours.iterrows():
            time_diff = row["finished_at"] - row["started_at"]
            assert time_diff > datetime.timedelta(0) and time_diff < max_time
        # check that all times are taken correctly from the trips table
        for tour_id, tour_df in trips_out.groupby("tour_id"):
            gt_start = tour_df.iloc[0]["started_at"]
            gt_end = tour_df.iloc[-1]["finished_at"]
            assert gt_start == tours.loc[tour_id, "started_at"]
            assert gt_end == tours.loc[tour_id, "finished_at"]

    def test_tour_geom(self, example_trip_data):
        trips, stps_locs = example_trip_data
        trips.rename(columns={"geom": "other_geom_name"}, inplace=True)
        trips = trips.set_geometry("other_geom_name")
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips)
        # check that nothing else than the new column has changed in trips df
        assert all(trips_out.iloc[:, :6] == trips)

    def test_tour_max_time(self, example_trip_data):
        trips, stps_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, max_time=datetime.timedelta(hours=2))
        assert len(tours) == 0
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, max_time=datetime.timedelta(hours=3))
        assert len(tours) == 1

    def test_tours_locations(self, example_trip_data):
        trips, stps_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, stps_w_locs=stps_locs, max_nr_gaps=1)
        assert all(tours["location_id"] == pd.Series([1, 2, 2]))
        for tour_id, tour_df in trips_out.groupby("tour_id"):
            gt_start = tour_df.iloc[0]["origin_staypoint_id"]
            gt_loc = stps_locs.loc[gt_start, "location_id"]
            gt_end = tour_df.iloc[-1]["destination_staypoint_id"]
            assert gt_start == tours.loc[tour_id, "origin_staypoint_id"]
            assert gt_end == tours.loc[tour_id, "destination_staypoint_id"]
            assert gt_loc == tours.loc[tour_id, "location_id"]

    def test_generate_tours_geolife(self):
        """Test tour generation also on the geolife dataset"""
        trips = ti.io.file.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")
        trips_, tours = ti.preprocessing.trips.generate_tours(trips, max_dist=100)

        tours_loaded = ti.io.file.read_tours_csv(
            os.path.join("tests", "data", "geolife_long", "tours.csv"), index_col="id"
        )
        # check only first five columns because location ID is empty in this case
        pd.testing.assert_frame_equal(tours_loaded.iloc[:, :5], tours.iloc[:, :5])

    def test_accessor(self, example_trip_data):
        """Test if the accessor leads to the same results as the explicit function."""
        # generate tours
        trips, stps_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, stps_w_locs=stps_locs, max_nr_gaps=1)

        # generate tours using the explicit function import
        trips_expl, tours_expl = ti.preprocessing.trips.generate_tours(trips)

        # generate tours using the accessor
        trips_acc, tours_acc = trips.as_trips.generate_tours()

        pd.testing.assert_frame_equal(trips_expl, trips_acc)
        pd.testing.assert_frame_equal(tours_expl, tours_acc)

    def test_print_progress_flag(self, example_trip_data, capsys):
        """Test if the print_progress bar controls the printing behavior."""
        trips, stps_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, print_progress=True)
        captured_print = capsys.readouterr()
        assert captured_print.err != ""

        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, print_progress=False)
        captured_noprint = capsys.readouterr()
        assert captured_noprint.err == ""
