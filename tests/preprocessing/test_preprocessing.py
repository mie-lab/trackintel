import pytest
import sys, os

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from math import radians

import trackintel as ti
from trackintel.preprocessing import positionfixes
from trackintel.preprocessing import staypoints
from trackintel.geogr.distances import calculate_distance_matrix

class TestPreprocessing():
    def test_extract_staypoints_sliding_min(self):
        pfs = ti.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'), sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        assert len(spts) == len(pfs), "With small thresholds, staypoint extraction should yield each positionfix"
        
    def test_extract_staypoints_sliding_max(self):
        pfs = ti.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'), sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=sys.maxsize, 
                                                       time_threshold=sys.maxsize)
        assert len(spts) == 0, "With large thresholds, staypoint extraction should not yield positionfixes"
        
    def test_extract_triplegs_staypoint(self):
        pfs = ti.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'), sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        tpls1 = pfs.as_positionfixes.extract_triplegs()
        tpls2 = pfs.as_positionfixes.extract_triplegs(spts)
        assert len(tpls1) > 0, "There should be more than zero triplegs"
        assert len(tpls2) > 0, "There should be more than zero triplegs"
        assert len(tpls1) == len(tpls2), "If we extract the staypoints in the same way, it should lead to " + \
            "the same number of triplegs"

    def test_cluster_staypoints_dbscan_min(self):
        pfs = ti.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'), sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        _, locs_user = spts.as_staypoints.extract_locations(method='dbscan', epsilon=1e-18, 
                                                            num_samples=0, agg_level='user')
        _, locs_data = spts.as_staypoints.extract_locations(method='dbscan', epsilon=1e-18, 
                                                            num_samples=0, agg_level='dataset')
        assert len(locs_user) == len(spts), "With small hyperparameters, clustering should not reduce the number"
        assert len(locs_data) == len(spts), "With small hyperparameters, clustering should not reduce the number"

    def test_cluster_staypoints_dbscan_max(self):
        pfs = ti.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'), sep=';')
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        _, locs_user = spts.as_staypoints.extract_locations(method='dbscan', epsilon=1e18, 
                                                            num_samples=1000, agg_level='user')
        _, locs_data = spts.as_staypoints.extract_locations(method='dbscan', epsilon=1e18, 
                                                            num_samples=1000, agg_level='dataset')
        assert len(locs_user) == 0, "With large hyperparameters, every user location is an outlier"
        assert len(locs_data) == 0, "With large hyperparameters, every dataset location is an outlier"
        
        
    def test_cluster_staypoints_dbscan_user_dataset(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        # take the first row and duplicate once
        spts = spts.head(1)
        spts = spts.append(spts, ignore_index=True)
        # assign a different user_id to the second row
        spts.iloc[1, 5] = 1
        # duplicate for a certain number 
        spts = spts.append([spts]*5,ignore_index=True)
        _, locs_ds = spts.as_staypoints.extract_locations(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='dataset')
        _, locs_us = spts.as_staypoints.extract_locations(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='user')
        loc_ds_num = locs_ds['location_id'].unique().shape[0]
        loc_us_num = locs_us['location_id'].unique().shape[0]
        assert loc_ds_num == 1, "Considering all staypoints at once, there should be only one location"
        assert loc_us_num == 2, "Considering user staypoints separately, there should be two locations"
        
    def test_cluster_staypoints_dbscan_loc(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        spts, locs = spts.as_staypoints.extract_locations(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='dataset')

        # create locations as grouped staypoints, another way to create locations
        other_locs = pd.DataFrame(columns=['user_id', 'location_id','center'])
        grouped_df = spts.groupby(['user_id', 'location_id'])
        for combined_id, group in grouped_df:
            user_id, location_id = combined_id
            group.set_geometry(spts.geometry.name, inplace=True)

            if int(location_id) != -1:
                temp_loc = {}
                temp_loc['user_id'] = user_id
                temp_loc['location_id'] = location_id
                
                # point geometry of place
                temp_loc['center'] = Point(group.geometry.x.mean(), group.geometry.y.mean())
                other_locs = other_locs.append(temp_loc, ignore_index=True)

        other_locs = gpd.GeoDataFrame(other_locs, geometry='center', crs=spts.crs)
        
        assert all(other_locs['center'] == locs['center']), "The location geometry should be the same"
        assert all(other_locs['location_id'] == locs['location_id']), "The location id should be the same"
        
    def test_cluster_staypoints_functions(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        _, locs_1 = spts.as_staypoints.extract_locations(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='dataset')
        _, locs_2 = spts.as_staypoints.cluster_staypoints(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='dataset')
        
        assert all(locs_1['center'] == locs_2['center']), "The location geometry should be the same"
        assert all(locs_1['location_id'] == locs_2['location_id']), "The location id should be the same"
    
    def test_cluster_staypoints_dbscan_haversine(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        
        # haversine calculation using sklearn.metrics.pairwise_distances, epsilon converted to radius
        spts, locs = spts.as_staypoints.extract_locations(method='dbscan', epsilon=10, 
                                                    num_samples=0, distance_matrix_metric='haversine',
                                                    agg_level='dataset')
        
        # calculate pairwise haversine matrix and fed to dbscan
        sp_distance_matrix = calculate_distance_matrix(spts, dist_metric="haversine")
        db = DBSCAN(eps=10, min_samples=0, metric="precomputed")
        labels = db.fit_predict(sp_distance_matrix)
        
        assert len(set(locs['location_id'])) == len(set(labels)) , "The #location should be the same"
    
    def test_cluster_staypoints_dbscan_hav_euc(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        
        # haversine calculation 
        _, loc_har = spts.as_staypoints.extract_locations(method='dbscan', epsilon=100, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='dataset')
        # WGS_1984
        spts.crs = 'epsg:4326'
        # WGS_1984_UTM_Zone_49N 
        spts = spts.to_crs("epsg:32649")
        
        # euclidean calculation 
        _, loc_eu = spts.as_staypoints.extract_locations(method='dbscan', epsilon=100, 
                                                         num_samples=0, distance_matrix_metric='euclidean',
                                                         agg_level='dataset')
        
        assert len(loc_har) == len(loc_eu) , "The #location should be the same for haversine" + \
            "and euclidean distances"
            
    