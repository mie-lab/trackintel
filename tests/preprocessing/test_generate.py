import pytest
import sys, os

import geopandas as gpd
import pandas as pd
import numpy as np
import datetime

from shapely.geometry import Point, LineString
from sklearn.cluster import DBSCAN

import trackintel as ti
from trackintel.geogr.distances import calculate_distance_matrix

class TestGenerate():
    def test_generate_staypoints_sliding_min(self):
        pfs = ti.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'), sep=';')
        spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        assert len(spts) == len(pfs), "With small thresholds, staypoint extraction should yield each positionfix"
        
    def test_generate_staypoints_sliding_max(self):
        pfs = ti.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'), sep=';')
        spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=sys.maxsize, 
                                                       time_threshold=sys.maxsize)
        assert len(spts) == 0, "With large thresholds, staypoint extraction should not yield positionfixes"
        

            
    def test_generate_triplegs_global(self):
        # generate triplegs from raw-data
        pfs = ti.io.dataset_reader.read_geolife(os.path.join('tests', 'data', 'geolife'))
        spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=25, time_threshold=5 * 60)
        tpls = pfs.as_positionfixes.generate_triplegs(spts)

        # load pregenerated test-triplegs
        tpls_test = ti.read_triplegs_csv(os.path.join('tests', 'data', 'geolife', 'geolife_triplegs_short.csv'))

        assert len(tpls) > 0
        assert len(tpls) == len(tpls)

        distance_sum = 0
        for i in range(len(tpls)):
            distance = tpls.geom.iloc[i].distance(tpls_test.geom.iloc[i])
            distance_sum = distance_sum + distance
        np.testing.assert_almost_equal(distance_sum, 0.0)
        
    def test_generate_staypoint_triplegs(self):
        pfs = ti.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'), sep=';')
        spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        tpls1 = pfs.as_positionfixes.generate_triplegs()
        tpls2 = pfs.as_positionfixes.generate_triplegs(spts)
        assert len(tpls1) > 0, "There should be more than zero triplegs"
        assert len(tpls2) > 0, "There should be more than zero triplegs"
        assert len(tpls1) == len(tpls2), "If we extract the staypoints in the same way, it should lead to " + \
            "the same number of triplegs"
            
    def test_generate_staypoints_triplegs_overlap(self):
        """
        Triplegs and staypoints should not overlap when generated using the default extract triplegs method.
        This test extracts triplegs and staypoints from positionfixes and stores them in a single dataframe.
        The dataframe is sorted by date, then we check if the staypoint/tripleg from the row before was finished when
        the next one started.
        """
        pfs = ti.io.dataset_reader.read_geolife(os.path.join('tests', 'data', 'geolife_long'))
        spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=25, time_threshold=5 * 60)
        tpls = pfs.as_positionfixes.generate_triplegs(spts)

        spts_tpls = spts[['started_at', 'finished_at', 'user_id']].append(
            tpls[['started_at', 'finished_at', 'user_id']])
        spts_tpls.sort_values(by=['user_id', 'started_at'], inplace=True)
        for user_id_this in spts['user_id'].unique():
            spts_tpls_this = spts_tpls[spts_tpls['user_id'] == user_id_this]
            diff = spts_tpls_this['started_at'] - spts_tpls_this['finished_at'].shift(1)
            # transform to numpy array and drop first values (always nan due to shift operation)
            diff = diff.values[1:]

            # all values have to greater or equal to zero. Otherwise there is an overlap
            assert all(diff >= np.timedelta64(datetime.timedelta()))
            
    def test_generate_locations_dbscan_min(self):
        pfs = ti.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'), sep=';')
        spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        _, locs_user = spts.as_staypoints.generate_locations(method='dbscan', epsilon=1e-18, 
                                                            num_samples=0, agg_level='user')
        _, locs_data = spts.as_staypoints.generate_locations(method='dbscan', epsilon=1e-18, 
                                                            num_samples=0, agg_level='dataset')
        assert len(locs_user) == len(spts), "With small hyperparameters, clustering should not reduce the number"
        assert len(locs_data) == len(spts), "With small hyperparameters, clustering should not reduce the number"

    def test_generate_locations_dbscan_max(self):
        pfs = ti.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'), sep=';')
        spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        _, locs_user = spts.as_staypoints.generate_locations(method='dbscan', epsilon=1e18, 
                                                            num_samples=1000, agg_level='user')
        _, locs_data = spts.as_staypoints.generate_locations(method='dbscan', epsilon=1e18, 
                                                            num_samples=1000, agg_level='dataset')
        assert len(locs_user) == 0, "With large hyperparameters, every user location is an outlier"
        assert len(locs_data) == 0, "With large hyperparameters, every dataset location is an outlier"
        
        
    def test_generate_locations_dbscan_user_dataset(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        # take the first row and duplicate once
        spts = spts.head(1)
        spts = spts.append(spts, ignore_index=True)
        # assign a different user_id to the second row
        spts.iloc[1, 5] = 1
        # duplicate for a certain number 
        spts = spts.append([spts]*5,ignore_index=True)
        _, locs_ds = spts.as_staypoints.generate_locations(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='dataset')
        _, locs_us = spts.as_staypoints.generate_locations(method='dbscan', epsilon=10, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='user')
        loc_ds_num = locs_ds['location_id'].unique().shape[0]
        loc_us_num = locs_us['location_id'].unique().shape[0]
        assert loc_ds_num == 1, "Considering all staypoints at once, there should be only one location"
        assert loc_us_num == 2, "Considering user staypoints separately, there should be two locations"
        
    def test_generate_locations_dbscan_loc(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        spts, locs = spts.as_staypoints.generate_locations(method='dbscan', epsilon=10, 
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
    
    def test_generate_locations_dbscan_haversine(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        
        # haversine calculation using sklearn.metrics.pairwise_distances, epsilon converted to radius
        spts, locs = spts.as_staypoints.generate_locations(method='dbscan', epsilon=10, 
                                                    num_samples=0, distance_matrix_metric='haversine',
                                                    agg_level='dataset')
        
        # calculate pairwise haversine matrix and fed to dbscan
        sp_distance_matrix = calculate_distance_matrix(spts, dist_metric="haversine")
        db = DBSCAN(eps=10, min_samples=0, metric="precomputed")
        labels = db.fit_predict(sp_distance_matrix)
        
        assert len(set(locs['location_id'])) == len(set(labels)) , "The #location should be the same"
    
    def test_generate_locations_dbscan_hav_euc(self):
        spts = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))
        
        # haversine calculation 
        _, loc_har = spts.as_staypoints.generate_locations(method='dbscan', epsilon=100, 
                                                          num_samples=0, distance_matrix_metric='haversine',
                                                          agg_level='dataset')
        # WGS_1984
        spts.crs = 'epsg:4326'
        # WGS_1984_UTM_Zone_49N 
        spts = spts.to_crs("epsg:32649")
        
        # euclidean calculation 
        _, loc_eu = spts.as_staypoints.generate_locations(method='dbscan', epsilon=100, 
                                                         num_samples=0, distance_matrix_metric='euclidean',
                                                         agg_level='dataset')
        
        assert len(loc_har) == len(loc_eu) , "The #location should be the same for haversine" + \
            "and euclidean distances"
            
    def test_generate_trips(self):
        """
        Test if we can generate the example trips based on example data
        """
        gap_threshold = 15
        # load pregenerated trips
        trips_loaded = pd.read_csv(os.path.join('tests', 'data', 'geolife_long', 'trips.csv'), index_col='id')
        trips_loaded['started_at'] = pd.to_datetime(trips_loaded['started_at'])
        trips_loaded['finished_at'] = pd.to_datetime(trips_loaded['finished_at'])

        # create trips from geolife (based on positionfixes)
        pfs = ti.io.dataset_reader.read_geolife(os.path.join('tests', 'data', 'geolife_long'))
        spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=25,
                                                       time_threshold=5 * 60)
        spts = spts.as_staypoints.create_activity_flag()
        tpls = pfs.as_positionfixes.generate_triplegs(spts)

        # temporary fix ID bug (issue  #56) so that we work with valid staypoint/tripleg files
        spts = spts.set_index('id')
        tpls = tpls.set_index('id')

        # generate trips and a joint staypoint/triplegs dataframe
        spts, tpls, trips = ti.preprocessing.triplegs.generate_trips(spts, tpls, gap_threshold=gap_threshold, id_offset=0)
        # test if generated trips are equal
        pd.testing.assert_frame_equal(trips_loaded, trips)
        
    def test_generate_trips_id_management(self):
        """
        Test if we can generate the example trips based on example data
        """
        gap_threshold = 15

        spts_tpls_loaded = pd.read_csv(os.path.join('tests', 'data', 'geolife_long', 'tpls_spts.csv'), index_col='id')
        spts_tpls_loaded['started_at'] = pd.to_datetime(spts_tpls_loaded['started_at'])
        spts_tpls_loaded['started_at_next'] = pd.to_datetime(spts_tpls_loaded['started_at_next'])
        spts_tpls_loaded['finished_at'] = pd.to_datetime(spts_tpls_loaded['finished_at'])

        # create trips from geolife (based on positionfixes)
        pfs = ti.io.dataset_reader.read_geolife(os.path.join('tests', 'data', 'geolife_long'))
        spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=25, time_threshold=5 * 60)
        spts = spts.as_staypoints.create_activity_flag()
        tpls = pfs.as_positionfixes.generate_triplegs(spts)

        # temporary fix ID bug (issue  #56) so that we work with valid staypoint/tripleg files
        spts = spts.set_index('id')
        tpls = tpls.set_index('id')

        # generate trips and a joint staypoint/triplegs dataframe
        spts, tpls, trips = ti.preprocessing.triplegs.generate_trips(spts, tpls, gap_threshold=gap_threshold, id_offset=0)
        spts_tpls = create_debug_spts_tpls_data(spts, tpls, gap_threshold=gap_threshold)

        # test if generated staypoints/triplegs are equal (especially important for trip ids)
        pd.testing.assert_frame_equal(spts_tpls_loaded, spts_tpls, check_dtype=False)
        
    def test_generate_trips_gap_detection(self):
        """
        Test different gap cases:
        - activity - tripleg - activity [gap] activity - tripleg - activity
        - activity - tripleg -  [gap]  - tripleg - activity
        - activity - tripleg -  [gap]  activity - tripleg - activity
        - activity - tripleg -  [gap]  activity - tripleg - activity
        - activity - tripleg - activity [gap] - tripleg - tripleg - tripleg - activity
        - tripleg - [gap] - tripleg - tripleg - [gap] - tripleg
        Returns
        -------

        """
        gap_threshold = 15

        # load data and add dummy geometry
        spts_in = pd.read_csv(os.path.join('.', 'tests', 'data', 'trips', 'staypoints_gaps.csv'),
                              sep=';', index_col='id', parse_dates=[0, 1],
                              infer_datetime_format=True, dayfirst=True)
        spts_in['geom'] = Point(1, 1)
        spts_in = gpd.GeoDataFrame(spts_in, geometry='geom')
        assert spts_in.as_staypoints

        tpls_in = pd.read_csv(os.path.join('.', 'tests', 'data', 'trips', 'triplegs_gaps.csv'),
                              sep=';', index_col='id', parse_dates=[0, 1],
                              infer_datetime_format=True, dayfirst=True)
        tpls_in['geom'] = LineString([[1, 1], [2, 2]])
        tpls_in = gpd.GeoDataFrame(tpls_in, geometry='geom')
        assert tpls_in.as_triplegs

        # load ground truth data
        trips_loaded = pd.read_csv(os.path.join('.', 'tests', 'data', 'trips', 'trips_gaps.csv'), index_col='id')
        trips_loaded['started_at'] = pd.to_datetime(trips_loaded['started_at'])
        trips_loaded['finished_at'] = pd.to_datetime(trips_loaded['finished_at'])

        spts_tpls_loaded = pd.read_csv(os.path.join('.', 'tests', 'data', 'trips', 'stps_tpls_gaps.csv')
                                       , index_col='id')
        spts_tpls_loaded['started_at'] = pd.to_datetime(spts_tpls_loaded['started_at'])
        spts_tpls_loaded['started_at_next'] = pd.to_datetime(spts_tpls_loaded['started_at_next'])
        spts_tpls_loaded['finished_at'] = pd.to_datetime(spts_tpls_loaded['finished_at'])

        # generate trips and a joint staypoint/triplegs dataframe
        spts_proc, tpls_proc, trips = ti.preprocessing.triplegs.generate_trips(spts_in, tpls_in,
                                                                               gap_threshold=15, 
                                                                               id_offset=0)
        spts_tpls = create_debug_spts_tpls_data(spts_proc, tpls_proc, gap_threshold=gap_threshold)

        # test if generated trips are equal
        pd.testing.assert_frame_equal(trips_loaded, trips)

        # test if generated staypoints/triplegs are equal (especially important for trip ids)
        pd.testing.assert_frame_equal(spts_tpls_loaded, spts_tpls, check_dtype=False)



# helper function for "test_generate_trips_*"
def create_debug_spts_tpls_data(spts, tpls, gap_threshold):
    spts = spts.copy()
    tpls = tpls.copy()

    # create table with relevant information from triplegs and staypoints.
    tpls['type'] = 'tripleg'
    spts['type'] = 'staypoint'
    spts_tpls = spts[['started_at', 'finished_at', 'user_id', 'type', 'activity', 'trip_id',
                      'prev_trip_id', 'next_trip_id']].append(
        tpls[['started_at', 'finished_at', 'user_id', 'type', 'trip_id']])

    # transform nan to bool
    spts_tpls['activity'] = spts_tpls['activity'] == True
    spts_tpls.sort_values(by=['user_id', 'started_at'], inplace=True)
    spts_tpls['started_at_next'] = spts_tpls['started_at'].shift(-1)
    spts_tpls['activity_next'] = spts_tpls['activity'].shift(-1)

    spts_tpls['gap'] = (spts_tpls['started_at_next'] - spts_tpls['finished_at']).dt.seconds / 60 > gap_threshold

    return spts_tpls