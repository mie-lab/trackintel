import os
import pytest
import numpy as np

import trackintel as ti


@pytest.fixture
def testdata_geolife():
    """Read geolife test data from files."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
    return pfs

class TestPositionfixes:
    """Tests for the PositionfixesAccessor."""
    
    def test_accessor(self, testdata_geolife):
        """Test if the as_positionfixes accessor checks the required column for positionfixes."""
        pfs = testdata_geolife
        assert pfs.as_positionfixes
        
        # check geometry
        with pytest.raises(AttributeError):
            pfs.drop(['geom'], axis=1).as_positionfixes
            
        # check user_id
        with pytest.raises(AttributeError):
            pfs.drop(['user_id'], axis=1).as_positionfixes
            
        # check geometry type
        with pytest.raises(AttributeError):
            pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5 * 60)
            _, tpls = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
            tpls.as_positionfixes

    def test_center(self, testdata_geolife):
        """Check if pfs has center method and returns (lat, lon) pairs as geometry."""
        pfs = testdata_geolife
        assert len(pfs.as_positionfixes.center) == 2
        
    def test_similarity_matrix(self, testdata_geolife):
        """Check the similarity_matrix function called through accessor runs as expected."""
        pfs = testdata_geolife
        
        accessor_result = pfs.as_positionfixes.calculate_distance_matrix(dist_metric='haversine', n_jobs=1)
        function_result = ti.geogr.distances.calculate_distance_matrix(pfs, dist_metric='haversine', n_jobs=1)
        assert np.allclose(accessor_result, function_result)
