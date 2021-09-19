import os
import pytest
from shapely.geometry import Point, MultiPoint

import trackintel as ti


@pytest.fixture
def testdata_trips():
    """Read trips test data from file."""
    path = os.path.join("tests", "data", "geolife_long", "trips.csv")
    test_trips = ti.read_trips_csv(path, index_col="id", geom_col="geom")
    return test_trips


class TestTrips:
    """Tests for the TripsAccessor."""

    def test_accessor(self, testdata_trips):
        """Test if the as_trips accessor checks the required column for trips."""
        trips = testdata_trips.copy()
        assert trips.as_trips

        # user_id
        with pytest.raises(AttributeError):
            trips.drop(["user_id"], axis=1).as_trips

    def test_accessor_geometry_type(self, testdata_trips):
        """Test if the as_trips accessor requires MultiPoint geometry."""
        trips_wrong_geom = testdata_trips.copy()

        # check geometry type
        with pytest.raises(AttributeError, match="The geometry must be a MultiPoint"):
            trips_wrong_geom["geom"] = Point([(13.476808430, 48.573711823)])
            trips_wrong_geom.as_trips

    def test_accessor_geometry_name(self, testdata_trips):
        """Test that it also works with a different geometry column name"""
        trips_other_geom_name = testdata_trips.copy()

        trips_other_geom_name["other_geom"] = Point(13.476808430, 48.573711823)
        trips_other_geom_name.set_geometry("other_geom", inplace=True)
        trips_other_geom_name.drop(columns=["geom"], inplace=True)

        # check that it works with other geometry name
        with pytest.raises(AttributeError, match="The geometry must be a MultiPoint"):
            trips_other_geom_name.as_trips
