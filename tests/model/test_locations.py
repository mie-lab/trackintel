import os
import pytest

from shapely.geometry import LineString

import trackintel as ti


@pytest.fixture
def testdata_locs():
    """Read location test data from files."""
    sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
    sp = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id")
    sp, locs = sp.as_staypoints.generate_locations(
        method="dbscan", epsilon=10, num_samples=0, distance_metric="haversine", agg_level="dataset"
    )
    return locs


class TestLocations:
    """Tests for the LocationsAccessor."""

    def test_accessor_column(self, testdata_locs):
        """Test if the as_locations accessor checks the required column for locations."""
        locs = testdata_locs.copy()

        with pytest.raises(AttributeError, match="To process a DataFrame as a collection of locations"):
            locs.drop(["user_id"], axis=1).as_locations

    def test_accessor_geometry_type(self, testdata_locs):
        """Test if the as_locations accessor requires Point geometry."""
        locs = testdata_locs.copy()
        with pytest.raises(AttributeError, match="The center geometry must be a Point"):
            locs["center"] = LineString(
                [(13.476808430, 48.573711823), (13.506804, 48.939008), (13.4664690, 48.5706414)]
            )
            locs.as_locations
