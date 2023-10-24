import pytest
import os

import geopandas as gpd

import trackintel as ti
from trackintel import Tours


@pytest.fixture
def test_tours():
    """Read test data from file"""
    tours_file = os.path.join("tests", "data", "tours.csv")
    tours = ti.read_tours_csv(tours_file, sep=";", index_col="id")
    # we want test the accessor of GeoDataFrames and not Tours
    return gpd.GeoDataFrame(tours)


class TestTours:
    """Tests for the Tours class."""

    def test_missing_column(self, test_tours):
        """Test if as_tours accessor check required columns"""
        with pytest.raises(AttributeError, match="To process a DataFrame as a collection of tours"):
            test_tours.drop(columns="user_id").as_tours

    def test_accessor_recursive(self, test_tours):
        tours = test_tours.as_tours
        assert type(tours) is Tours
        assert id(tours) == id(tours.as_tours)
