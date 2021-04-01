
import os
import pytest

import trackintel as ti


class TestLocations:
    """Tests for the LocationsAccessor."""
    
    def test_accessor(self):
        """Test if the as_locations accessor checks the required column for locations."""
        stps_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        stps = ti.read_staypoints_csv(stps_file, tz='utc', index_col='id')
        # 
        stps, locs = stps.as_staypoints.generate_locations(method='dbscan', 
                                                           epsilon=10, 
                                                           num_samples=0, 
                                                           distance_matrix_metric='haversine',
                                                           agg_level='dataset')
        assert locs.as_locations
        
        # user_id
        with pytest.raises(AttributeError):
            locs.drop(['user_id'], axis=1).as_locations
        
        # check geometry type
        with pytest.raises(AttributeError):
            pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
            pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5 * 60)
            _, tpls = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
            tpls.as_locations