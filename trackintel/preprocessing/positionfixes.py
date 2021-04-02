import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN

import datetime
from math import radians

from trackintel.geogr.distances import haversine_dist




def generate_triplegs(pfs_input, stps_input, method="between_staypoints", gap_threshold=15):
    """Generate triplegs from positionfixes.

    Parameters
    ----------
    pfs_input : GeoDataFrame (as trackintel positionfixes)
        The pfs have to follow the standard definition for positionfixes DataFrames. 
        If 'staypoint_id' column is not found, stps_input needs to be given.

    stps_input : GeoDataFrame (as trackintel staypoints), optional
        The stps (corresponding to the positionfixes). If this is not passed, the
        positionfixes need 'staypoint_id' associated with them.

    method: {'between_staypoints'}
        Method to create triplegs. 'between_staypoints' method defines a tripleg as all pfs 
        between two stps. This method requires either a column 'staypoint_id' on 
        the pfs or passing stps as an input.
            
    gap_threshold: float, default 15 (minutes)
        Maximum allowed temporal gap size in minutes. If tracking data is missing for more than 
        `gap_threshold` minutes, then a new tripleg will be generated.

    Returns
    -------
    pfs: GeoDataFrame (as trackintel positionfixes)
        The original positionfixes with a new column ``[`tripleg_id`]``.
        
    tpls: GeoDataFrame (as trackintel triplegs)
        The generated triplegs.

    Notes
    -----
    Methods 'between_staypoints' requires either a column 'staypoint_id' on the 
    positionfixes or passing some staypoints that correspond to the positionfixes! 
    This means you usually should call ``generate_staypoints()`` first.
    
    The first pfs after a stp is regarded as the first pfs of the generated tpl.

    Examples
    --------
    >>> pfs.as_positionfixes.generate_triplegs('between_staypoints', gap_threshold=15)
    """
    # copy the original pfs for adding 'staypoint_id' column
    pfs = pfs_input.copy()

    if method == "between_staypoints":

        # get case:
        # Case 1: pfs have a column 'staypoint_id'
        # Case 2: pfs do not have a column 'staypoint_id' but stps_input are provided

        if "staypoint_id" not in pfs.columns:
            case = 2
        else:
            case = 1

        # Preprocessing for case 2:
        # - step 1: Assign staypoint ids to positionfixes by matching timestamps (per user)
        # - step 2: Find first positionfix after a staypoint
        # (relevant if the pfs of stps are not provided, and we can only infer the pfs after stps through time)
        if case == 2:
            # initialize the index list of pfs where a tpl will begin
            insert_index_ls = []
            pfs["staypoint_id"] = pd.NA
            for user_id_this in pfs["user_id"].unique():
                spts_user = stps_input[stps_input["user_id"] == user_id_this]
                pfs_user = pfs[pfs["user_id"] == user_id_this]

                # step 1
                # All positionfixes with timestamp between staypoints are assigned the value 0
                # Intersect all positionfixes of a user with all staypoints of the same user
                intervals = pd.IntervalIndex.from_arrays(spts_user["started_at"], spts_user["finished_at"], closed="both")
                is_in_interval = pfs_user["tracked_at"].apply(lambda x: intervals.contains(x).any()).astype("bool")
                pfs.loc[is_in_interval[is_in_interval].index, "staypoint_id"] = 0

                # step 2
                # Identify first positionfix after a staypoint
                # find index of closest positionfix with equal or greater timestamp.
                tracked_at_sorted = pfs_user["tracked_at"].sort_values()
                insert_position_user = tracked_at_sorted.searchsorted(spts_user["finished_at"])
                insert_index_user = tracked_at_sorted.iloc[insert_position_user].index

                # store the insert insert_position_user in an array
                insert_index_ls.extend(list(insert_index_user))
            #
            cond_staypoints_case2 = pd.Series(False, index=pfs.index)
            cond_staypoints_case2.loc[insert_index_ls] = True

        # initialize tripleg_id with pd.NA and fill all pfs that belong to staypoints with -1
        # pd.NA will be replaced later with tripleg ids
        pfs["tripleg_id"] = pd.NA
        pfs.loc[~pd.isna(pfs["staypoint_id"]), "tripleg_id"] = -1

        # we need to ensure pfs is properly ordered
        pfs.sort_values(by=["user_id", "tracked_at"], inplace=True)
        # get all conditions that trigger a new tripleg.
        # condition 1: a positionfix belongs to a new tripleg if the user changes. For this we need to sort pfs.
        # The first positionfix of the new user is the start of a new tripleg (if it is no staypoint)
        cond_new_user = ((pfs["user_id"] - pfs["user_id"].shift(1)) != 0) & pd.isna(pfs["staypoint_id"])

        # condition 2: Temporal gaps
        # if there is a gap that is longer than gap_threshold minutes, we start a new tripleg
        cond_gap = pfs["tracked_at"] - pfs["tracked_at"].shift(1) > datetime.timedelta(minutes=gap_threshold)

        # condition 3: stps
        # By our definition the pf after a stp is the first pf of a tpl.
        # this works only for numeric staypoint ids, TODO: can we change?
        _stp_id = (pfs["staypoint_id"] + 1).fillna(0)  
        cond_stp = (_stp_id - _stp_id.shift(1)) != 0

        # special check for case 2: pfs that belong to stp might not present in the data.
        # We need to select these pfs using time.
        if case == 2:
            cond_stp = cond_stp | cond_staypoints_case2

        # combine conditions
        cond_all = cond_new_user | cond_gap | cond_stp
        # make sure not to create triplegs within staypoints:
        cond_all = cond_all & pd.isna(pfs["staypoint_id"])

        cond_all.sort_index(inplace=True)
        # get the start position of tpls
        tpls_starts = np.where(cond_all)[0]

        # a valid linestring needs 2 points
        tpls_lengths = np.diff(tpls_starts)
        cond_to_remove = np.take(tpls_starts, np.where(tpls_lengths < 2)[0])
        cond_all.iloc[cond_to_remove] = False
        pfs.loc[pfs.index.isin(cond_to_remove), "tripleg_id"] = -1

        # assign an incrementing id to all positionfixes that start a tripleg
        # create triplegs
        pfs.loc[cond_all, "tripleg_id"] = np.arange(cond_all.sum())

        # fill the pd.NAs with the previously observed tripleg_id
        # pfs not belonging to tripleg are also propagated (with -1)
        pfs["tripleg_id"] = pfs["tripleg_id"].fillna(method="ffill")
        # assign back pd.NA to -1
        pfs.loc[pfs["tripleg_id"] == -1, "tripleg_id"] = pd.NA

        posfix_grouper = pfs.groupby("tripleg_id")

        tpls = posfix_grouper.agg(
            {"user_id": ["mean"], "tracked_at": [min, max], "geom": list}
        )  # could add a "number of pfs": can be any column "count"

        # prepare dataframe: Rename columns; read/set geometry/crs;
        # Order of column has to correspond to the order of the groupby statement
        tpls.columns = ["user_id", "started_at", "finished_at", "geom"]
        tpls["geom"] = tpls["geom"].apply(LineString)
        tpls = tpls.set_geometry("geom")
        tpls.crs = pfs.crs

        # check the correctness of the generated tpls
        assert tpls.as_triplegs

        if case == 2:
            pfs.drop(columns="staypoint_id", inplace=True)

        # dtype consistency
        pfs["tripleg_id"] = pfs["tripleg_id"].astype("Int64")
        tpls.index = tpls.index.astype("int64")
        tpls.index.name = "id"

        # user_id of tpls should be the same as pfs
        tpls["user_id"] = tpls["user_id"].astype(pfs["user_id"].dtype)

        return pfs, tpls

    else:
        raise AttributeError(f"Method unknown. We only support 'between_staypoints'. You passed {method}")

