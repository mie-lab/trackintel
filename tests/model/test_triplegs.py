import os

import pytest
import geopandas as gpd
from shapely.geometry import Point

import trackintel as ti
from trackintel import Triplegs


@pytest.fixture
def testdata_tpls():
    """Read triplegs test data from files."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
    pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    _, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
    # we want test the accessor of GeoDataFrames and not Triplegs
    return gpd.GeoDataFrame(tpls)


class TestTriplegs:
    """Tests for the Triplegs class."""

    def test_accessor_column(self, testdata_tpls):
        """Test if the as_triplegs accessor checks the required column for triplegs."""
        tpls = testdata_tpls.copy()

        tpls.as_triplegs

        # check user_id
        with pytest.raises(AttributeError):
            tpls.drop(["user_id"], axis=1).as_triplegs

    def test_accessor_geometry(self, testdata_tpls):
        """Test if the as_triplegs accessor requires geometry column."""
        tpls = testdata_tpls.copy()

        # check geometry
        with pytest.raises(AttributeError):
            tpls.drop(["geom"], axis=1).as_triplegs

    def test_accessor_geometry_type(self, testdata_tpls):
        """Test if the as_triplegs accessor requires LineString geometry."""
        tpls = testdata_tpls.copy()

        # check geometry type
        with pytest.raises(AttributeError, match="The geometry must be a LineString"):
            tpls["geom"] = Point([(13.476808430, 48.573711823)])
            tpls.as_triplegs

    def test_accessor_recursive(self, testdata_tpls):
        tpls = testdata_tpls.as_triplegs
        assert type(tpls) is Triplegs
        assert id(tpls) == id(tpls.as_triplegs)
