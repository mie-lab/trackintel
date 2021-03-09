import os

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

import trackintel as ti
from trackintel.geogr.distances import calculate_distance_matrix


class TestGenerate_locations():
    def test_generate_locations_dbscan_hav_euc(self):
        stps_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        stps = ti.read_staypoints_csv(stps_file, tz='utc', index_col='id')
        
        # haversine calculation 
        _, loc_har = stps.as_staypoints.generate_locations(method='dbscan', epsilon=100, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='dataset')
        # WGS_1984
        stps.crs = 'epsg:4326'
        # WGS_1984_UTM_Zone_49N 
        stps = stps.to_crs("epsg:32649")
        
        # euclidean calculation 
        _, loc_eu = stps.as_staypoints.generate_locations(method='dbscan', epsilon=100, 
                                                         num_samples=0, distance_matrix_metric='euclidean',
                                                         agg_level='dataset')
        
        assert len(loc_har) == len(loc_eu) , "The #location should be the same for haversine" + \
            "and euclidean distances"
            
    def test_generate_locations_dbscan_haversine(self):
        stps_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        stps = ti.read_staypoints_csv(stps_file, tz='utc', index_col='id')
        
        # haversine calculation using sklearn.metrics.pairwise_distances
        stps, locs = stps.as_staypoints.generate_locations(method='dbscan', epsilon=10, 
                                                    num_samples=0, distance_matrix_metric='haversine',
                                                    agg_level='dataset')
        
        # calculate pairwise haversine matrix and fed to dbscan
        sp_distance_matrix = calculate_distance_matrix(stps, dist_metric="haversine")
        db = DBSCAN(eps=10, min_samples=0, metric="precomputed")
        labels = db.fit_predict(sp_distance_matrix)
        
        assert len(set(locs.index)) == len(set(labels)) , "The number of locations should be the same"
    
    def test_generate_locations_dbscan_loc(self):
        stps_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        stps = ti.read_staypoints_csv(stps_file, tz='utc', index_col='id')
        stps, locs = stps.as_staypoints.generate_locations(method='dbscan', epsilon=10,
                                                           num_samples=0, distance_matrix_metric='haversine',
                                                           agg_level='dataset')

        # create locations as grouped staypoints, another way to create locations
        other_locs = pd.DataFrame(columns=['user_id', 'id','center'])
        grouped_df = stps.groupby(['user_id', 'location_id'])
        for combined_id, group in grouped_df:
            user_id, location_id = combined_id
            group.set_geometry(stps.geometry.name, inplace=True)

            if int(location_id) != -1:
                temp_loc = {}
                temp_loc['user_id'] = user_id
                temp_loc['id'] = location_id
                
                # point geometry of place
                temp_loc['center'] = Point(group.geometry.x.mean(), group.geometry.y.mean())
                other_locs = other_locs.append(temp_loc, ignore_index=True)

        other_locs = gpd.GeoDataFrame(other_locs, geometry='center', crs=stps.crs)
        other_locs.set_index('id', inplace=True)
        
        assert all(other_locs['center'] == locs['center']), "The location geometry should be the same"
        assert all(other_locs.index == locs.index), "The location id should be the same"
    
    def test_generate_locations_dbscan_user_dataset(self):
        stps_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        stps = ti.read_staypoints_csv(stps_file, tz='utc', index_col='id')
        # take the first row and duplicate once
        stps = stps.head(1)
        stps = stps.append(stps, ignore_index=True)
        # assign a different user_id to the second row
        stps.iloc[1, 4] = 1
        
        # duplicate for a certain number 
        stps = stps.append([stps]*5,ignore_index=True)
        _, locs_ds = stps.as_staypoints.generate_locations(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='dataset')
        _, locs_us = stps.as_staypoints.generate_locations(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='user')
        loc_ds_num = len(locs_ds.index.unique())
        loc_us_num = len(locs_us.index.unique())
        assert loc_ds_num == 1, "Considering all staypoints at once, there should be only one location"
        assert loc_us_num == 2, "Considering user staypoints separately, there should be two locations"
    
    def test_generate_locations_dbscan_min(self):
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        _, stps = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        _, locs_user = stps.as_staypoints.generate_locations(method='dbscan', epsilon=1e-18, 
                                                            num_samples=0, agg_level='user')
        _, locs_data = stps.as_staypoints.generate_locations(method='dbscan', epsilon=1e-18, 
                                                            num_samples=0, agg_level='dataset')
        assert len(locs_user) == len(stps), "With small hyperparameters, clustering should not reduce the number"
        assert len(locs_data) == len(stps), "With small hyperparameters, clustering should not reduce the number"

    def test_generate_locations_dbscan_max(self):
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        _, stps = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        _, locs_user = stps.as_staypoints.generate_locations(method='dbscan', epsilon=1e18,
                                                             num_samples=1000, agg_level='user')
        _, locs_data = stps.as_staypoints.generate_locations(method='dbscan', epsilon=1e18,
                                                             num_samples=1000, agg_level='dataset')
        assert len(locs_user) == 0, "With large hyperparameters, every user location is an outlier"
        assert len(locs_data) == 0, "With large hyperparameters, every dataset location is an outlier"
    
    def test_generate_locations_missing_link(self):
        """Test nan is assigned for missing link between stps and locs."""
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        _, stps = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        stps, _ = stps.as_staypoints.generate_locations(method='dbscan', epsilon=1e18,
                                                        num_samples=1000, agg_level='user')
    
        assert pd.isna(stps['location_id']).any()
        
    def test_generate_locations_dtype_consistent(self):
        """Test the dtypes for the generated columns."""
        stps_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        stps = ti.read_staypoints_csv(stps_file, tz='utc', index_col='id')
        # 
        stps, locs = stps.as_staypoints.generate_locations(method='dbscan', 
                                                           epsilon=10, 
                                                           num_samples=0, 
                                                           distance_matrix_metric='haversine',
                                                           agg_level='dataset')
        assert stps['user_id'].dtype == locs['user_id'].dtype
        assert stps['location_id'].dtype == 'Int64'
        assert locs.index.dtype == 'int64'
        # change the user_id to string
        stps['user_id'] = stps['user_id'].apply(lambda x: str(x))
        stps, locs = stps.as_staypoints.generate_locations(method='dbscan', 
                                                           epsilon=10, 
                                                           num_samples=0, 
                                                           distance_matrix_metric='haversine',
                                                           agg_level='dataset')
        assert stps['user_id'].dtype == locs['user_id'].dtype
        assert stps['location_id'].dtype == 'Int64'
        assert locs.index.dtype == 'int64'
        
    def test_generate_locations_index_start(self):
        """Test the generated index start from 0 for different methods."""
        stps_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        stps = ti.read_staypoints_csv(stps_file, tz='utc', index_col='id')
        
        _, locs_ds = stps.as_staypoints.generate_locations(method='dbscan',
                                                           epsilon=10,
                                                           num_samples=0,
                                                           distance_matrix_metric='haversine',
                                                           agg_level='dataset')
        _, locs_us = stps.as_staypoints.generate_locations(method='dbscan',
                                                           epsilon=10,
                                                           num_samples=0,
                                                           distance_matrix_metric='haversine',
                                                           agg_level='user')
        
        assert (locs_ds.index == np.arange(len(locs_ds))).any()
        assert (locs_us.index == np.arange(len(locs_us))).any()
        
class TestCreate_activity_flag():
    pass