import os
import pytest
from shapely.geometry import Point

import trackintel as ti


@pytest.fixture
def testdata_trips():
    """Read location test data from files."""
    trips = ti.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")
    return trips


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
