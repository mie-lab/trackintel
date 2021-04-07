import os
import pytest
import numpy as np

from shapely.geometry import LineString

import trackintel as ti


@pytest.fixture
def testdata_geolife():
    """Read geolife test data from files."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
    return pfs


class TestPositionfixes:
    """Tests for the PositionfixesAccessor."""

    def test_accessor_column(self, testdata_geolife):
        """Test if the as_positionfixes accessor checks the required column for positionfixes."""
        pfs = testdata_geolife.copy()
        assert pfs.as_positionfixes

        # check user_id
        with pytest.raises(AttributeError, match="To process a DataFrame as a collection of positionfixes"):
            pfs.drop(["user_id"], axis=1).as_positionfixes

    def test_accessor_geometry(self, testdata_geolife):
        """Test if the as_positionfixes accessor requires geometry column."""
        pfs = testdata_geolife.copy()

        # check geometry
        with pytest.raises(AttributeError, match="No geometry data set yet"):
            pfs.drop(["geom"], axis=1).as_positionfixes

    def test_accessor_geometry_type(self, testdata_geolife):
        """Test if the as_positionfixes accessor requires Point geometry."""
        pfs = testdata_geolife.copy()

        # check geometry type
        with pytest.raises(AttributeError, match="The geometry must be a Point"):
            pfs["geom"] = LineString([(13.476808430, 48.573711823), (13.506804, 48.939008), (13.4664690, 48.5706414)])
            pfs.as_positionfixes

    def test_center(self, testdata_geolife):
        """Check if pfs has center method and returns (lat, lon) pairs as geometry."""
        pfs = testdata_geolife
        assert len(pfs.as_positionfixes.center) == 2

    def test_similarity_matrix(self, testdata_geolife):
        """Check the similarity_matrix function called through accessor runs as expected."""
        pfs = testdata_geolife

        accessor_result = pfs.as_positionfixes.calculate_distance_matrix(dist_metric="haversine", n_jobs=1)
        function_result = ti.geogr.distances.calculate_distance_matrix(pfs, dist_metric="haversine", n_jobs=1)
        assert np.allclose(accessor_result, function_result)
