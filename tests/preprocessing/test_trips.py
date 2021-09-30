import os
import pandas as pd
import pytest
from shapely.geometry import MultiPoint, Point
import datetime
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

import trackintel as ti


@pytest.fixture
def example_trip_data():
    """Trips and locations for tour generation."""
    # define points for each staypoint
    sp_geom_mapping = {
        1: Point(8.5067847, 47.4),
        2: Point(8.5067847, 47.40001),
        3: Point(8.5067847, 47.6),
        4: Point(8.5067847, 47.7),
        5: Point(8.5067847, 47.399),
        6: Point(8.5067847, 47.60001),
        7: Point(9.5067847, 47.20001),
    }

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")
    t4 = pd.Timestamp("1971-01-02 08:00:00", tz="utc")
    t5 = pd.Timestamp("1971-01-02 09:00:00", tz="utc")
    t6 = pd.Timestamp("1971-01-02 10:00:00", tz="utc")

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
            "destination_staypoint_id": 7,
        },
    ]
    # fill geom based on staypoints
    for row_dict in trips_list_dict:
        row_dict["geom"] = MultiPoint(
            (sp_geom_mapping[row_dict["origin_staypoint_id"]], sp_geom_mapping[row_dict["destination_staypoint_id"]])
        )

    # assign location IDs (only for testing, not a trackintel staypoints standard!)
    sp_locs = pd.DataFrame({"location_id": [1, 1, 2, 3, 1, 2, 4]}, index=[1, 2, 3, 4, 5, 6, 7])

    trips = gpd.GeoDataFrame(data=trips_list_dict, geometry="geom", crs="EPSG:4326")
    trips = trips.set_index("id")
    # assert valid trips
    trips.as_trips
    return trips, sp_locs


@pytest.fixture
def example_nested_tour(example_trip_data):
    """Helper function to create a nested trip"""
    trips, _ = example_trip_data

    # construct trips that lie between trips 6 and 15 and form a tour on their own
    # define start and end points of these trips
    first_trip_subtour = MultiPoint((trips.loc[15, "geom"][0], Point(9.5067847, 47.20001)))
    second_trip_subtour = MultiPoint((Point(9.5067847, 47.20001), trips.loc[15, "geom"][0]))
    # define time of start and time at the intermediate point
    start_time_subtour = pd.Timestamp("1971-01-02 08:45:00", tz="utc")
    middle_time = pd.Timestamp("1971-01-02 08:55:00", tz="utc")
    trips.loc[6, "finished_at"] = start_time_subtour
    # trip 6 starts at 8:45 at sp 5, then at 8:45 the user goes to 7 and back to 5
    trips.loc[100] = [0, start_time_subtour, middle_time, 5, 7, first_trip_subtour]
    trips.loc[200] = [0, middle_time, trips.loc[15, "started_at"], 7, 5, second_trip_subtour]
    trips.sort_values(by=["user_id", "started_at", "origin_staypoint_id", "destination_staypoint_id"], inplace=True)
    return trips


class TestGenerate_tours:
    """Tests for generate_tours() method."""

    def test_generate_tours(self, example_trip_data):
        """Test general functionality of generate tours function"""
        trips, sp_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips)
        # check that nothing else than the new column has changed in trips df
        assert all(trips_out.iloc[:, :6] == trips)
        # check that the two tours were found
        assert trips_out.loc[1, "tour_id"] == [0]
        assert (
            trips_out.loc[2, "tour_id"] == [1]
            and trips_out.loc[6, "tour_id"] == [1]
            and trips_out.loc[15, "tour_id"] == [1]
        )
        # all others have tour id nan
        user_1_df = trips_out[trips_out["user_id"] == 1]
        assert all(pd.isna(user_1_df["tour_id"]))

    def test_tours_with_gap(self, example_trip_data):
        """Test functionality of max_nr_gaps parameter in tour generation"""
        trips, sp_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, max_nr_gaps=1)
        # new tour was found for user 1
        assert len(tours) == 3
        assert trips_out.loc[7, "tour_id"] == [2]
        assert trips_out.loc[80, "tour_id"] == [2]

    def test_tour_times(self, example_trip_data):
        """Check whether the start and end times of generated tours are correct"""
        trips, sp_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, max_nr_gaps=1, max_time="1d")
        # check that all times are below the max time
        for i, row in tours.iterrows():
            time_diff = row["finished_at"] - row["started_at"]
            assert time_diff > pd.to_timedelta("0m") and time_diff < pd.to_timedelta("1d")

        # group trips by tour and check that start and end of each tour are correct
        grouped_trips = ti.preprocessing.trips.get_trips_grouped(trips, tours)
        for tour_id, tour_df in grouped_trips:
            gt_start = tour_df.iloc[0]["started_at"]
            gt_end = tour_df.iloc[-1]["finished_at"]
            assert gt_start == tours.loc[tour_id, "started_at"]
            assert gt_end == tours.loc[tour_id, "finished_at"]

    def test_tour_geom(self, example_trip_data):
        """Test whether tour generation is invariant to the name of the geometry column"""
        trips, sp_locs = example_trip_data
        trips.rename(columns={"geom": "other_geom_name"}, inplace=True)
        trips = trips.set_geometry("other_geom_name")
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips)
        # check that nothing else than the new column has changed in trips df
        assert all(trips_out.iloc[:, :6] == trips)

    def test_tour_max_time(self, example_trip_data):
        """Test functionality of max time argument in tour generation"""
        trips, sp_locs = example_trip_data
        with pytest.warns(UserWarning, match="No tours can be generated, return empty tours"):
            _, tours = ti.preprocessing.trips.generate_tours(trips, max_time="2h")  # only 2 hours allowed
            assert len(tours) == 0
        _, tours = ti.preprocessing.trips.generate_tours(trips, max_time="3h")  # increase to 3 hours
        assert len(tours) == 1

    def test_tours_locations(self, example_trip_data):
        """Test whether tour generation with locations as input yields correct results as well"""
        trips, sp_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, staypoints=sp_locs, max_nr_gaps=1)
        assert all(tours["location_id"] == pd.Series([1, 2, 2]))

        # group trips by tour and check that the locations of start and end of each tour are correct
        grouped_trips = ti.preprocessing.trips.get_trips_grouped(trips, tours)
        for tour_id, tour_df in grouped_trips:
            gt_start = tour_df.iloc[0]["origin_staypoint_id"]
            gt_loc = sp_locs.loc[gt_start, "location_id"]
            gt_end = tour_df.iloc[-1]["destination_staypoint_id"]
            assert gt_start == tours.loc[tour_id, "origin_staypoint_id"]
            assert gt_end == tours.loc[tour_id, "destination_staypoint_id"]
            assert gt_loc == tours.loc[tour_id, "location_id"]

    def test_tours_crs(self, example_trip_data):
        """Test if the tours generation works with projected coordinate system"""
        trips, _ = example_trip_data
        trips_crs = trips.copy()
        # normal baseline
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips)
        # set other crs
        trips_crs.set_crs("WGS84", inplace=True)
        trips_crs.to_crs("EPSG:2056", inplace=True)
        trips_out_crs, tours_crs = ti.preprocessing.trips.generate_tours(trips_crs)
        trips_out_crs.to_crs("WGS84", inplace=True)
        # assert equal
        assert_geodataframe_equal(trips_out, trips_out_crs, check_less_precise=True)
        pd.testing.assert_frame_equal(tours, tours_crs)

    def test_generate_tours_geolife(self):
        """Test tour generation also on the geolife dataset"""
        path_trips = os.path.join("tests", "data", "geolife_long", "trips.csv")
        trips = ti.io.file.read_trips_csv(path_trips, index_col="id", geom_col="geom", crs="EPSG:4326")
        _, tours = ti.preprocessing.trips.generate_tours(trips, max_dist=100)

        path_tours = os.path.join("tests", "data", "geolife_long", "tours.csv")
        tours_loaded = ti.io.file.read_tours_csv(path_tours, index_col="id")
        # check only first five columns because location ID is empty in this case
        pd.testing.assert_frame_equal(tours_loaded.iloc[:, :5], tours.iloc[:, :5])

    def test_accessor(self, example_trip_data):
        """Test if the accessor leads to the same results as the explicit function."""
        # generate tours
        trips, _ = example_trip_data

        # generate tours using the explicit function import
        trips_expl, tours_expl = ti.preprocessing.trips.generate_tours(trips)

        # generate tours using the accessor
        trips_acc, tours_acc = trips.as_trips.generate_tours()

        pd.testing.assert_frame_equal(trips_expl, trips_acc)
        pd.testing.assert_frame_equal(tours_expl, tours_acc)

    def test_print_progress_flag(self, example_trip_data, capsys):
        """Test if the print_progress bar controls the printing behavior."""
        trips, sp_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, print_progress=True)
        captured_print = capsys.readouterr()
        assert captured_print.err != ""

        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, print_progress=False)
        captured_noprint = capsys.readouterr()
        assert captured_noprint.err == ""

    def test_index_stability(self, example_trip_data):
        """Test if the index of the trips remains stable"""
        trips, sp_locs = example_trip_data
        trips_out, tours = ti.preprocessing.trips.generate_tours(trips, print_progress=True)

        assert trips.index.equals(trips_out.index)

    def test_nested_tour(self, example_nested_tour):
        """
        Test whether we get two tours (a long one and a nested short one), for example for home-work-lunch-work-home
        """
        trips_out, tours = ti.preprocessing.trips.generate_tours(example_nested_tour)
        assert len(tours) == 3
        # the nested tour of length 2 should be found
        assert tours.loc[1, "trips"] == [100, 200]
        # the big tour should be found
        assert tours.loc[2, "trips"] == [2, 6, 100, 200, 15]
        # in the trips table, trip 100 and 200 is assigned the tour id of its smaller tour
        assert trips_out.loc[100, "tour_id"] == [1, 2]
        assert trips_out.loc[200, "tour_id"] == [1, 2]

    def test_tours_warn_existing_column(self, example_trip_data):
        trips, _ = example_trip_data
        trips["tour_id"] = 1
        with pytest.warns(UserWarning, match="Deleted existing column 'tour_id' from trips."):
            _ = ti.preprocessing.trips.generate_tours(trips, print_progress=True)

    def test_tours_time_gap_error(self, example_trip_data):
        trips, _ = example_trip_data
        # check that an int as max time raises a TypeError
        with pytest.raises(Exception) as e_info:
            _ = ti.preprocessing.trips.generate_tours(trips, max_time=24)
            assert e_info == "Parameter max_time must be either of type String or pd.Timedelta!"


class TestTourHelpers:
    """Test auxiliary function for trip grouping"""

    def test_get_trips_grouped(self, example_nested_tour):
        trips, tours = ti.preprocessing.trips.generate_tours(example_nested_tour)
        grouped_trips = ti.preprocessing.trips.get_trips_grouped(trips, tours)
        for tour_id, trips_on_tour in grouped_trips:
            # check that all trips belong to the tour
            for i, id in enumerate(trips_on_tour["trip_id"]):
                assert id in list(tours.loc[tour_id, "trips"])
