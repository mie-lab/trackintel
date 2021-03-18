import datetime
import os
import sys
from math import radians

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

import trackintel as ti
from trackintel.geogr.distances import haversine_dist


@pytest.fixture
def pfs_geolife():
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join('tests', 'data', 'geolife'))
    return pfs


@pytest.fixture
def pfs_geolife_long():
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join('tests', 'data', 'geolife_long'))
    return pfs


@pytest.fixture
def geolife_pfs_stps_short(pfs_geolife):
    pfs, stps = pfs_geolife.as_positionfixes.generate_staypoints(method='sliding',
                                                                 dist_threshold=25,
                                                                 time_threshold=5 * 60)
    return pfs, stps


@pytest.fixture
def geolife_pfs_stps_long(pfs_geolife_long):
    pfs, stps = pfs_geolife_long.as_positionfixes.generate_staypoints(method='sliding',
                                                                      dist_threshold=25,
                                                                      time_threshold=5 * 60)
    return pfs, stps


@pytest.fixture
def testdata_tpls_geolife():
    tpls_file = os.path.join('tests', 'data', 'geolife', 'geolife_triplegs_short.csv')
    tpls = ti.read_triplegs_csv(tpls_file, tz='utc', index_col='id')
    tpls.geometry = tpls.geometry.set_crs(epsg=4326)
    return tpls


@pytest.fixture
def testdata_pfs_ids_geolife():
    pfs_file = os.path.join('tests', 'data', 'geolife', 'geolife_positionfixes_with_ids.csv')
    pfs = ti.read_positionfixes_csv(pfs_file, index_col='id')
    pfs['staypoint_id'] = pfs['staypoint_id'].astype('Int64')
    pfs['tripleg_id'] = pfs['tripleg_id'].astype('Int64')
    pfs.geometry = pfs.geometry.set_crs(epsg=4326)
    return pfs


class TestGenerate_staypoints():
    def test_generate_staypoints_sliding_min(self):
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method='sliding', dist_threshold=0, time_threshold=0)
        assert len(stps) == len(pfs), "With small thresholds, staypoint extraction should yield each positionfix"

    def test_generate_staypoints_sliding_max(self):
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        _, stps = pfs.as_positionfixes.generate_staypoints(method='sliding',
                                                           dist_threshold=sys.maxsize,
                                                           time_threshold=sys.maxsize)
        assert len(stps) == 0, "With large thresholds, staypoint extraction should not yield positionfixes"

    def test_generate_staypoints_missing_link(self, geolife_pfs_stps_long):
        """Test nan is assigned for missing link between pfs and stps."""
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        pfs, _ = pfs.as_positionfixes.generate_staypoints(method='sliding',
                                                          dist_threshold=sys.maxsize,
                                                          time_threshold=sys.maxsize)

        assert pd.isna(pfs['staypoint_id']).any()

    def test_generate_staypoints_dtype_consistent(self):
        """Test the dtypes for the generated columns."""
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method='sliding',
                                                             dist_threshold=25,
                                                             time_threshold=5 * 60)
        assert pfs['user_id'].dtype == stps['user_id'].dtype
        assert pfs['staypoint_id'].dtype == "Int64"
        assert stps.index.dtype == "int64"

    def test_generate_staypoints_index_start(self):
        """Test the generated index start from 0 for different methods."""
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs_ori = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')

        _, stps_sliding = pfs_ori.as_positionfixes.generate_staypoints(method='sliding',
                                                                       dist_threshold=25,
                                                                       time_threshold=300)
        _, stps_dbscan = pfs_ori.as_positionfixes.generate_staypoints(method='dbscan')

        assert (stps_sliding.index == np.arange(len(stps_sliding))).any()
        assert (stps_dbscan.index == np.arange(len(stps_dbscan))).any()

    def test_generate_staypoints_groupby_sliding(self):
        """Test the 'sliding' result obtained using user_id for loop (previous) with groupby.apply (current)."""
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs_ori = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')

        # stps detection using groupby
        pfs_groupby, stps_groupby = pfs_ori.as_positionfixes.generate_staypoints(method='sliding',
                                                                                 dist_threshold=25,
                                                                                 time_threshold=300)
        # stps detection using for loop
        pfs_for, stps_for = _generate_staypoints_original(pfs_ori,
                                                          method='sliding',
                                                          dist_threshold=25,
                                                          time_threshold=300)

        pd.testing.assert_frame_equal(stps_groupby, stps_for, check_dtype=False)
        pd.testing.assert_frame_equal(pfs_groupby, pfs_for, check_dtype=False)

    def test_generate_staypoints_groupby_dbscan(self):
        """Test the 'dbscan' result obtained using user_id for loop (previous) with groupby.apply (current)."""
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs_ori = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')

        # stps detection using groupby
        pfs_groupby, stps_groupby = pfs_ori.as_positionfixes.generate_staypoints(method='dbscan')
        # stps detection using for loop
        pfs_for, stps_for = _generate_staypoints_original(pfs_ori, method='dbscan')

        pd.testing.assert_frame_equal(stps_groupby, stps_for, check_dtype=False)
        pd.testing.assert_frame_equal(pfs_groupby, pfs_for, check_dtype=False)


class TestGenerate_triplegs():
    def test_generate_triplegs_case1(self, geolife_pfs_stps_short, testdata_tpls_geolife):
        """Test tripleg generation using the 'between_staypoints' method for case 1.
        case 1: triplegs from positionfixes with column 'staypoint_id' and staypoints."""

        pfs, stps = geolife_pfs_stps_short
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps, method='between_staypoints')
        
        assert_geodataframe_equal(tpls, testdata_tpls_geolife, check_less_precise=True)

    def test_generate_triplegs_case2(self, geolife_pfs_stps_short, testdata_tpls_geolife):
        """Test tripleg generation using the 'between_staypoints' method for case 2.
        case 2: triplegs from positionfixes (without column 'staypoint_id') and staypoints."""

        pfs, stps = geolife_pfs_stps_short
        pfs.drop('staypoint_id', axis=1, inplace=True)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps, method='between_staypoints')
        
        assert_geodataframe_equal(tpls, testdata_tpls_geolife, check_less_precise=True)

    def test_generate_triplegs_case3(self, geolife_pfs_stps_short, testdata_tpls_geolife):
        """Test tripleg generation using the 'between_staypoints' method for case 3.
        case 3: triplegs from positionfixes with column 'staypoint_id' but without staypoints."""

        pfs, _ = geolife_pfs_stps_short
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(method='between_staypoints')
        
        assert_geodataframe_equal(tpls, testdata_tpls_geolife, check_less_precise=True)

    def test_tripleg_ids_in_positionfixes_case1(self, geolife_pfs_stps_short, testdata_pfs_ids_geolife):
        """
        checks if positionfixes are not altered and if staypoint/tripleg ids where set correctly during tripleg
        generation using the 'between_staypoints' method with positionfixes with 'staypoint_id' and staypoints as
        input (case 2).
        """
        pfs, stps = geolife_pfs_stps_short
        pfs, _ = pfs.as_positionfixes.generate_triplegs(stps, method='between_staypoints')
        
        assert_geodataframe_equal(pfs, testdata_pfs_ids_geolife, check_less_precise=True, check_like=True)

    def test_tripleg_ids_in_positionfixes_case2(self, geolife_pfs_stps_short, testdata_pfs_ids_geolife):
        """
        checks if positionfixes are not altered and if staypoint/tripleg ids where set correctly during tripleg
        generation using the 'between_staypoints' method with positionfixes without 'staypoint_id' and staypoints as
        input (case 2).
        """
        pfs, stps = geolife_pfs_stps_short
        pfs.drop('staypoint_id', axis=1, inplace=True)
        pfs, _ = pfs.as_positionfixes.generate_triplegs(stps, method='between_staypoints')

        testdata_pfs_ids_geolife.drop('staypoint_id', axis=1, inplace=True)
        assert_geodataframe_equal(pfs, testdata_pfs_ids_geolife, check_less_precise=True, check_like=True)

    def test_tripleg_ids_in_positionfixes_case3(self, geolife_pfs_stps_short, testdata_pfs_ids_geolife):
        """
        checks if positionfixes are not altered and if staypoint/tripleg ids where set correctly during tripleg
        generation using the 'between_staypoints' method with only positionfixes as input (case 3).
        """
        pfs, _ = geolife_pfs_stps_short
        pfs, _ = pfs.as_positionfixes.generate_triplegs(method='between_staypoints')
        
        assert_geodataframe_equal(pfs, testdata_pfs_ids_geolife, check_less_precise=True, check_like=True)

    def test_tripleg_generation_case2_empty_staypoints(self, geolife_pfs_stps_short, testdata_pfs_ids_geolife):
        """
        checks if it is safe to have users that have positionfixes but no staypoints
        """
        pfs, stps = geolife_pfs_stps_short
        pfs.drop('staypoint_id', axis=1, inplace=True)
        pfs.loc[0, 'user_id'] = 5000
        pfs, _ = pfs.as_positionfixes.generate_triplegs(stps, method='between_staypoints')

        # only test that it can run without error
        assert True

    def test_tripleg_generation_stability(self, geolife_pfs_stps_long):
        """
        checks if the results are same if different variants of the tripleg_generation method 'between_staypoints'
        are used
        """

        pfs, stps = geolife_pfs_stps_long

        pfs_case1, tpls_case1 = pfs.as_positionfixes.generate_triplegs(stps, method='between_staypoints')
        pfs_case2, tpls_case2 = pfs.drop('staypoint_id',
                                         axis=1).as_positionfixes.generate_triplegs(stps, method='between_staypoints')
        pfs_case3, tpls_case3 = pfs.as_positionfixes.generate_triplegs(method='between_staypoints')
        
        assert_geodataframe_equal(pfs_case1.drop('staypoint_id', axis=1), pfs_case2, check_less_precise=True, check_like=True)
        assert_geodataframe_equal(pfs_case1, pfs_case3, check_less_precise=True, check_like=True)
        
        assert_geodataframe_equal(tpls_case1, tpls_case2, check_less_precise=True, check_like=True)
        assert_geodataframe_equal(tpls_case1, tpls_case3, check_less_precise=True, check_like=True)

    def test_generate_triplegs_dtype_consistent(self, pfs_geolife):
        """Test the dtypes for the generated columns."""
        pfs, stps = pfs_geolife.as_positionfixes.generate_staypoints(method='sliding',
                                                                     dist_threshold=25,
                                                                     time_threshold=5 * 60)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)
        assert pfs['user_id'].dtype == tpls['user_id'].dtype
        assert pfs['tripleg_id'].dtype == "Int64"
        assert tpls.index.dtype == "int64"

    def test_generate_triplegs_missing_link(self, geolife_pfs_stps_long):
        """Test nan is assigned for missing link between pfs and tpls."""
        pfs, stps = geolife_pfs_stps_long

        pfs, _ = pfs.as_positionfixes.generate_triplegs(stps, method='between_staypoints')

        assert pd.isna(pfs['tripleg_id']).any()

    def test_generate_triplegs_index_start(self, geolife_pfs_stps_long):
        """Test the generated index start from 0 for different methods."""
        pfs, stps = geolife_pfs_stps_long

        _, tpls_case1 = pfs.as_positionfixes.generate_triplegs(stps, method='between_staypoints')
        _, tpls_case2 = pfs.drop('staypoint_id',
                                 axis=1).as_positionfixes.generate_triplegs(stps, method='between_staypoints')
        _, tpls_case3 = pfs.as_positionfixes.generate_triplegs(method='between_staypoints')

        assert (tpls_case1.index == np.arange(len(tpls_case1))).any()
        assert (tpls_case2.index == np.arange(len(tpls_case2))).any()
        assert (tpls_case3.index == np.arange(len(tpls_case3))).any()

    def test_generate_staypoints_triplegs_overlap(self, pfs_geolife_long):
        """
        Triplegs and staypoints should not overlap when generated using the default extract triplegs method.
        This test extracts triplegs and staypoints from positionfixes and stores them in a single dataframe.
        The dataframe is sorted by date, then we check if the staypoint/tripleg from the row before was finished when
        the next one started.
        """
        pfs = pfs_geolife_long
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method='sliding',
                                                             dist_threshold=25,
                                                             time_threshold=5 * 60)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        stps_tpls = stps[['started_at', 'finished_at', 'user_id']].append(
            tpls[['started_at', 'finished_at', 'user_id']])
        stps_tpls.sort_values(by=['user_id', 'started_at'], inplace=True)
        for user_id_this in stps['user_id'].unique():
            stps_tpls_this = stps_tpls[stps_tpls['user_id'] == user_id_this]
            diff = stps_tpls_this['started_at'] - stps_tpls_this['finished_at'].shift(1)
            # transform to numpy array and drop first values (always nan due to shift operation)
            diff = diff.values[1:]

            # all values have to greater or equal to zero. Otherwise there is an overlap
            assert all(diff >= np.timedelta64(datetime.timedelta()))


def _generate_staypoints_original(positionfixes, method='sliding',
                                  dist_threshold=50, time_threshold=300, epsilon=100,
                                  dist_func=haversine_dist, num_samples=1):
    """
    Original function using 'user_id' for-loop to generate staypoints based on pfs.
    
    Used for comparing with the current groupby.apply() method.
    """
    # copy the original pfs for adding 'staypoint_id' column
    ret_pfs = positionfixes.copy()
    ret_pfs.sort_values(by='user_id', inplace=True)

    elevation_flag = 'elevation' in ret_pfs.columns  # if there is elevation data

    name_geocol = ret_pfs.geometry.name
    ret_stps = pd.DataFrame(columns=['id', 'user_id', 'started_at', 'finished_at', 'geom'])

    if method == 'sliding':
        # Algorithm from Li et al. (2008). For details, please refer to the paper.
        staypoint_id_counter = 0
        ret_pfs['staypoint_id'] = np.nan  # this marks all that are not part of a SP

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
                            ret_stps = ret_stps.append(staypoint, ignore_index=True)

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
                                ret_stps = ret_stps.append(staypoint, ignore_index=True)
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

                ret_stps = ret_stps.append(staypoint, ignore_index=True)

    ret_stps = gpd.GeoDataFrame(ret_stps, geometry=name_geocol, crs=ret_pfs.crs)
    ret_pfs = gpd.GeoDataFrame(ret_pfs, geometry=name_geocol, crs=ret_pfs.crs)

    ## ensure dtype consistency 
    ret_stps['id'] = ret_stps['id'].astype('int64')
    ret_stps.set_index('id', inplace=True)
    ret_pfs['staypoint_id'] = ret_pfs['staypoint_id'].astype('float')

    ret_stps['user_id'] = ret_stps['user_id'].astype(ret_pfs['user_id'].dtype)

    return ret_pfs, ret_stps
