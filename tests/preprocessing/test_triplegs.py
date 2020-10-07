import os

import pandas as pd

from trackintel.io.dataset_reader import read_geolife
from trackintel.preprocessing.triplegs import generate_trips


class TestGenerate_trips():
    def test_1(self):
        pfs = read_geolife(os.path.join('tests', 'data', 'geolife_long'))
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=25, time_threshold=5 * 60)
        spts = spts.as_staypoints.create_activity_flag()
        tpls = pfs.as_positionfixes.extract_triplegs(spts)

        trips = generate_trips(tpls, spts, gap_threshold=15, id_offset=0)
        tpls['type'] = 'tripleg'
        spts['type'] = 'staypoint'

        # create table with relevant information from triplegs and staypoints.
        spts_tpls = spts[['started_at', 'finished_at', 'user_id', 'id', 'type', 'activity']].append(
            tpls[['started_at', 'finished_at', 'user_id', 'id', 'type']])

        # transform nan to bool
        spts_tpls['activity'] = (spts_tpls['activity'] == True)

        spts_tpls.sort_values(by='started_at', inplace=True)
        debug_spts = pd.DataFrame(spts)
        x = 2
        #
        # #check if valid staypoints
        #
        # trip_id = 0
        #
        # # get all user_ids
        # user_ids = np.concatenate([tpls.user_id.unique().astype('int'), spts.user_id.unique().astype('int')])
        # user_ids = np.unique(user_ids)
        #
        # for user_id_this in user_ids:
        #     pass
        #
        #
        # combined_df = tpls.loc[tpls['user_id']==user_id_this, ['id', 'user_id', 'started_at', 'finished_at']]
        # combined_df['type'] = 'tripleg'
        # combined_df = combined_df.append(spts.loc[spts['user_id']==user_id_this, ['id', 'user_id', 'started_at', 'finished_at', 'activity']])
        #
        # combined_df['type'] = combined_df['type'].fillna(value='staypoint')
        # combined_df.sort_values(by='started_at', inplace=True)
        #
        #
        #
        #
