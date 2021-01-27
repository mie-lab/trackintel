import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

import trackintel as ti

class TestSmoothen_triplegs():
    def test_smoothen_triplegs(self):
        tpls = ti.read_triplegs_csv(os.path.join('tests','data','triplegs_with_too_many_points_test.csv'), sep=';')
        tpls_smoothed = ti.preprocessing.triplegs.smoothen_triplegs(tpls, tolerance=0.0001)
        line1 = tpls.iloc[0].geom
        line1_smoothed = tpls_smoothed.iloc[0].geom
        line2 = tpls.iloc[1].geom
        line2_smoothed = tpls_smoothed.iloc[1].geom

        assert line1.length == line1_smoothed.length
        assert line2.length == line2_smoothed.length
        assert len(line1.coords) == 10
        assert len(line2.coords) == 7
        assert len(line1_smoothed.coords) == 4
        assert len(line2_smoothed.coords) == 3

class TestGenerate_trips():
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
        spts_tpls = _create_debug_spts_tpls_data(spts_proc, tpls_proc, gap_threshold=gap_threshold)

        # test if generated trips are equal
        pd.testing.assert_frame_equal(trips_loaded, trips)

        # test if generated staypoints/triplegs are equal (especially important for trip ids)
        pd.testing.assert_frame_equal(spts_tpls_loaded, spts_tpls, check_dtype=False)
        
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
        spts, tpls, _ = ti.preprocessing.triplegs.generate_trips(spts, tpls, gap_threshold=gap_threshold, id_offset=0)
        spts_tpls = _create_debug_spts_tpls_data(spts, tpls, gap_threshold=gap_threshold)

        # test if generated staypoints/triplegs are equal (especially important for trip ids)
        pd.testing.assert_frame_equal(spts_tpls_loaded, spts_tpls, check_dtype=False)
        
    

# helper function for "test_generate_trips_*"
def _create_debug_spts_tpls_data(spts, tpls, gap_threshold):
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