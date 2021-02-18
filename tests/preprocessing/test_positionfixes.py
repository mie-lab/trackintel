import datetime
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from math import radians

import trackintel as ti
from trackintel.geogr.distances import haversine_dist

class TestGenerate_staypoints():
    def test_generate_staypoints_sliding_min(self):
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        pfs, spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        assert len(spts) == len(pfs), "With small thresholds, staypoint extraction should yield each positionfix"
        
    def test_generate_staypoints_sliding_max(self):
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        _, spts = pfs.as_positionfixes.generate_staypoints(method='sliding', 
                                                           dist_threshold=sys.maxsize, 
                                                           time_threshold=sys.maxsize)
        assert len(spts) == 0, "With large thresholds, staypoint extraction should not yield positionfixes"
        
    def test_generate_staypoints_dtype_consistent(self):
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        pfs, spts = pfs.as_positionfixes.generate_staypoints(method='sliding', 
                                                             dist_threshold=25, 
                                                             time_threshold=5 * 60)
        assert pfs['user_id'].dtype == spts['user_id'].dtype
        # assert pfs['staypoint_id'].dtype == spts['id'].dtype
        
    def test_generate_staypoints_groupby_sliding(self):
        """Test the 'sliding' result obtained using user_id for loop (previous) with groupby.apply (current)."""
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs_ori = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        
        # stps detection using groupby
        pfs_groupby, spts_groupby = pfs_ori.as_positionfixes.generate_staypoints(method='sliding', 
                                                                                 dist_threshold=25, 
                                                                                 time_threshold=300)
        # stps detection using for loop
        pfs_for, spts_for = _generate_staypoints_original(pfs_ori, 
                                                          method='sliding', 
                                                          dist_threshold=25, 
                                                          time_threshold=300)
        
        pd.testing.assert_frame_equal(spts_groupby, spts_for, check_dtype=False)
        pd.testing.assert_frame_equal(pfs_groupby, pfs_for, check_dtype=False)
        
    def test_generate_staypoints_groupby_dbscan(self):
        """Test the 'dbscan' result obtained using user_id for loop (previous) with groupby.apply (current)."""
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs_ori = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        
        # stps detection using groupby
        pfs_groupby, spts_groupby = pfs_ori.as_positionfixes.generate_staypoints(method='dbscan')
        # stps detection using for loop
        pfs_for, spts_for = _generate_staypoints_original(pfs_ori, method='dbscan')
        
        pd.testing.assert_frame_equal(spts_groupby, spts_for, check_dtype=False)
        pd.testing.assert_frame_equal(pfs_groupby, pfs_for, check_dtype=False)
        

class TestGenerate_triplegs():
    def test_generate_triplegs_global(self):
        # generate triplegs from raw-data
        pfs = ti.io.dataset_reader.read_geolife(os.path.join('tests', 'data', 'geolife'))
        pfs, spts = pfs.as_positionfixes.generate_staypoints(method='sliding', 
                                                             dist_threshold=25, 
                                                             time_threshold=5 * 60)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(spts)

        # load pregenerated test-triplegs
        tpls_file = os.path.join('tests', 'data', 'geolife', 'geolife_triplegs_short.csv')
        tpls_test = ti.read_triplegs_csv(tpls_file, tz='utc')

        assert len(tpls) > 0
        assert len(tpls) == len(tpls)

        distance_sum = 0
        for i in range(len(tpls)):
            distance = tpls.geom.iloc[i].distance(tpls_test.geom.iloc[i])
            distance_sum = distance_sum + distance
        np.testing.assert_almost_equal(distance_sum, 0.0)
    
    def test_generate_triplegs_dtype_consistent(self):
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        pfs, spts = pfs.as_positionfixes.generate_staypoints(method='sliding', 
                                                             dist_threshold=25, 
                                                             time_threshold=5 * 60)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(spts)
        assert pfs['user_id'].dtype == tpls['user_id'].dtype
        # assert pfs['tripleg_id'].dtype == tpls['id'].dtype
        
    def test_generate_staypoint_triplegs(self):
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        pfs, spts = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        _, tpls1 = pfs.as_positionfixes.generate_triplegs()
        _, tpls2 = pfs.as_positionfixes.generate_triplegs(spts)
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
        pfs_file = os.path.join('tests', 'data', 'geolife_long')
        pfs = ti.io.dataset_reader.read_geolife(pfs_file)
        pfs, spts = pfs.as_positionfixes.generate_staypoints(method='sliding', 
                                                             dist_threshold=25, 
                                                             time_threshold=5 * 60)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(spts)

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

            

def _generate_staypoints_original(positionfixes, method='sliding',
                                  dist_threshold=50, time_threshold= 300, epsilon=100,
                                  dist_func=haversine_dist, num_samples=1):
    """
    Original function using 'user_id' for-loop to generate staypoints based on pfs.
    
    Used for comparing with the current groupby.apply() method.
    """
    # copy the original pfs for adding 'staypoint_id' column
    ret_pfs = positionfixes.copy()
    ret_pfs.sort_values(by='user_id', inplace=True)
    
    elevation_flag = 'elevation' in ret_pfs.columns # if there is elevation data

    name_geocol = ret_pfs.geometry.name
    ret_spts = pd.DataFrame(columns=['id', 'user_id', 'started_at', 'finished_at', 'geom'])

    if method == 'sliding':
        # Algorithm from Li et al. (2008). For details, please refer to the paper.
        staypoint_id_counter = 0
        ret_pfs['staypoint_id'] = np.NaN  # this marks all that are not part of a SP

        for user_id_this in ret_pfs['user_id'].unique():

            positionfixes_user_this = ret_pfs.loc[ret_pfs['user_id'] == user_id_this]  # this is no copy

            positionfixes_user_this = positionfixes_user_this.sort_values('tracked_at')
            pfs = positionfixes_user_this.to_dict('records')
            idx = positionfixes_user_this.index.to_list()
            num_pfs = len(pfs)

            posfix_staypoint_matching = {}

            i = 0
            j = 0  # is zero because it gets incremented in the beginning
            while i < num_pfs:
                if j == num_pfs:
                    # We're at the end, this can happen if in the last "bin", 
                    # the dist_threshold is never crossed anymore.
                    break
                else:
                    j = i + 1
                while j < num_pfs:
                    # TODO: Can we make distance function independent of projection?
                    dist = dist_func(pfs[i][name_geocol].x, pfs[i][name_geocol].y,
                                     pfs[j][name_geocol].x, pfs[j][name_geocol].y)

                    if dist > dist_threshold:
                        delta_t = pfs[j]['tracked_at'] - pfs[i]['tracked_at']
                        if delta_t.total_seconds() > time_threshold:
                            staypoint = {}
                            staypoint['user_id'] = pfs[i]['user_id']
                            staypoint[name_geocol] = Point(np.mean([pfs[k][name_geocol].x for k in range(i, j)]),
                                                           np.mean([pfs[k][name_geocol].y for k in range(i, j)]))
                            if elevation_flag:
                                staypoint['elevation'] = np.mean([pfs[k]['elevation'] for k in range(i, j)])
                            staypoint['started_at'] = pfs[i]['tracked_at']
                            staypoint['finished_at'] = pfs[j - 1]['tracked_at']
                            staypoint['id'] = staypoint_id_counter

                            # store matching 
                            posfix_staypoint_matching[staypoint_id_counter] = [idx[k] for k in range(i, j)]
                            staypoint_id_counter += 1

                            # add staypoint
                            ret_spts = ret_spts.append(staypoint, ignore_index=True)

                            # TODO Discussion: Is this last point really a staypoint? As we don't know if the
                            #      person "moves on" afterwards...
                            if j == num_pfs - 1:
                                staypoint = {}
                                staypoint['user_id'] = pfs[j]['user_id']
                                staypoint[name_geocol] = Point(pfs[j][name_geocol].x, pfs[j][name_geocol].y)
                                if elevation_flag:
                                    staypoint['elevation'] = pfs[j]['elevation']
                                staypoint['started_at'] = pfs[j]['tracked_at']
                                staypoint['finished_at'] = pfs[j]['tracked_at']
                                staypoint['id'] = staypoint_id_counter

                                # store matching
                                posfix_staypoint_matching[staypoint_id_counter] = [idx[j]] 
                                staypoint_id_counter += 1
                                ret_spts = ret_spts.append(staypoint, ignore_index=True)
                        i = j
                        break
                    j = j + 1

            # add matching to original positionfixes (for every user)

            for staypoints_id, posfix_idlist in posfix_staypoint_matching.items():
                # note that we use .loc because above we have saved the id 
                # of the positionfixes not thier absolut position
                ret_pfs.loc[posfix_idlist, 'staypoint_id'] = staypoints_id
        

    elif method == 'dbscan':

        db = DBSCAN(eps=epsilon / 6371000, min_samples=num_samples, algorithm='ball_tree', metric='haversine')

        for user_id_this in ret_pfs['user_id'].unique():

            user_positionfixes = ret_pfs[ret_pfs['user_id'] == user_id_this]  # this is not a copy!

            # TODO: enable transformations to temporary (metric) system
            transform_crs = None
            if transform_crs is not None:
                pass

            # get staypoint matching
            coordinates = np.array([[radians(g.y), radians(g.x)] for g in user_positionfixes[name_geocol]])
            labels = db.fit_predict(coordinates)

            # add positionfixes - staypoint matching to original positionfixes
            ret_pfs.loc[user_positionfixes.index, 'staypoint_id'] = labels

        # create staypoints as the center of the grouped positionfixes
        grouped_df = ret_pfs.groupby(['user_id', 'staypoint_id'])
        for combined_id, group in grouped_df:
            user_id, staypoint_id = combined_id

            if int(staypoint_id) != -1:
                staypoint = {}
                staypoint['user_id'] = user_id
                staypoint['id'] = staypoint_id

                # point geometry of staypoint
                staypoint[name_geocol] = Point(group[name_geocol].x.mean(),
                                           group[name_geocol].y.mean())

                ret_spts = ret_spts.append(staypoint, ignore_index=True)

    ret_spts = gpd.GeoDataFrame(ret_spts, geometry=name_geocol,crs=ret_pfs.crs)
    ret_pfs = gpd.GeoDataFrame(ret_pfs, geometry=name_geocol,crs=ret_pfs.crs)
    
    ## ensure dtype consistency 
    ret_spts['id'] = ret_spts['id'].astype('int64')
    ret_spts.set_index('id', inplace=True)
    ret_pfs['staypoint_id'] = ret_pfs['staypoint_id'].astype('float')

    ret_spts['user_id'] = ret_spts['user_id'].astype(ret_pfs['user_id'].dtype)

    return ret_pfs, ret_spts