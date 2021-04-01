import os
import pytest

import trackintel as ti


@pytest.fixture
def testdata_stps():
    """Read stps test data from files."""
    stps_file = os.path.join('tests', 'data', 'staypoints.csv')
    stps = ti.read_staypoints_csv(stps_file, sep=';', index_col='id')
    return stps

class TestStaypoints:
    """Tests for the StaypointsAccessor."""

    def test_accessor(self, testdata_stps):
        """Test if the as_staypoints accessor checks the required column for staypoints."""
        stps = testdata_stps
        assert stps.as_staypoints
        
        # geometery
        with pytest.raises(AttributeError):
            stps.drop(['geom'], axis=1).as_staypoints
            
        # user_id
        with pytest.raises(AttributeError):
            stps.drop(['user_id'], axis=1).as_staypoints
            
        # check geometry type
        with pytest.raises(AttributeError):
            pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
            pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5 * 60)
            _, tpls = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
            tpls.as_staypoints

    def test_staypoints_center(self, testdata_stps):
        """Check if stps has center method and returns (lat, lon) pairs as geometry."""
        stps = testdata_stps
        assert len(stps.as_staypoints.center) == 2
