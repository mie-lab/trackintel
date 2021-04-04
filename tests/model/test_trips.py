import os
import pytest

import trackintel as ti


class TestTrips:
    """Tests for the TripsAccessor."""

    def test_accessor(self):
        """Test if the as_trips accessor checks the required column for trips."""
        trips = ti.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")
        assert trips.as_trips

        # user_id
        with pytest.raises(AttributeError):
            trips.drop(["user_id"], axis=1).as_trips
