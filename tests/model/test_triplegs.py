import os
import pytest

import trackintel as ti

@pytest.fixture
def testdata_geolife():
    """Read geolife test data from files."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
    return pfs


class TestTriplegs:
    """Tests for the TriplegsAccessor."""

    def test_accessor(self, testdata_geolife):
        """Test if the as_triplegs accessor checks the required column for triplegs."""
        pfs = testdata_geolife
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5 * 60)
        _, tpls = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
        
        assert tpls.as_triplegs
            
        # check user_id
        with pytest.raises(AttributeError):
            tpls.drop(['user_id'], axis=1).as_triplegs
            
        # check geometry type
        with pytest.raises(AttributeError):
            stps.as_triplegs