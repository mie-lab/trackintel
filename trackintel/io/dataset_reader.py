# -*- coding: utf-8 -*-

import glob
import os
from collections import defaultdict
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from trackintel.preprocessing.util import calc_temp_overlap

FEET2METER = 0.3048
CRS_WGS84 = "epsg:4326"


def read_geolife(geolife_path, print_progress=False):
    """
    Read raw geolife data and return trackintel positionfixes.

    This functions parses all geolife data available in the directory ``geolife_path``

    Parameters
    ----------
    geolife_path: str
        Path to the directory with the geolife data

    print_progress: Bool, default False
        Show per-user progress if set to True.

    Returns
    -------
    gdf: GeoDataFrame (as trackintel positionfixes)
        Contains all loaded geolife positionfixes

    labels: dict
        Dictionary with the available mode labels.
        Keys are user ids of users that have a "labels.txt" in their folder.

    Notes
    -----
    The geopandas dataframe has the following columns and datatype: 'elevation': float64 (in meters); 'tracked_at': datetime64[ns];
    'user_id': int64; 'geom': shapely geometry; 'accuracy': None;

    For some users, travel mode labels are provided as .txt file. These labels are read and returned as label dictionary.
    The label dictionary contains the user ids as keys and DataFrames with the available labels as values.
    Labels can be added to each user at the tripleg level, see
    :func:`trackintel.io.dataset_reader.geolife_add_modes_to_triplegs` for more details.

    The folder structure within the geolife directory needs to be identical with the folder structure
    available from the official download. The means that the top level folder (provided with 'geolife_path')
    contains the folders for the different users:

    | geolife_path
    | ├── 000
    | │   ├── Trajectory
    | │   │   ├── 20081023025304.plt
    | │   │   ├── 20081024020959.plt
    | │   │   ├── 20081026134407.plt
    | │   │   └── ...
    | ├── 001
    | │   ├── Trajectory
    | │   │   └── ...
    | │   ...
    | ├── 010
    | │   ├── labels.txt
    | │   ├── Trajectory
    | │   │   └── ...
    | └── ...

    the geolife dataset as it can be downloaded from:

    https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

    References
    ----------
    [1] Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. Mining interesting locations and travel sequences from
    GPS trajectories. In Proceedings of International conference on World Wild Web (WWW 2009),
    Madrid Spain. ACM Press: 791-800.

    [2] Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, Wei-Ying Ma. Understanding Mobility Based on GPS Data.
    In Proceedings of ACM conference on Ubiquitous Computing (UbiComp 2008), Seoul, Korea. ACM Press: 312-321.

    [3] Yu Zheng, Xing Xie, Wei-Ying Ma, GeoLife: A Collaborative Social Networking Service among User, location
    and trajectory. Invited paper, in IEEE Data Engineering Bulletin. 33, 2, 2010, pp. 32-40.

    Example
    ----------
    >>> from trackintel.io.dataset_reader import read_geolife
    >>> pfs, mode_labels = read_geolife(os.path.join('downloads', 'Geolife Trajectories 1.3'))
    """
    # u are strings in the format "052", "002".
    uids = [u for u in os.listdir(geolife_path) if os.path.isdir(os.path.join(geolife_path, u))]

    if len(uids) == 0:
        raise FileNotFoundError("No user folders found at path {}".format(geolife_path))

    for user_id in uids:
        try:
            int(user_id)
        except ValueError as err:
            errmsg = (
                "Invalid user_id '{}' found in geolife path '{}'. The geolife path can only contain folders"
                " named with integers that represent the user id.".format(user_id, os.path.join(geolife_path, user_id))
            )
            raise ValueError(errmsg) from err

    labels = _get_labels(geolife_path, uids)
    # get the dfs in form of an generator and concatinate them
    gdf = pd.concat(_get_df(geolife_path, uids, print_progress), axis=0, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geom", crs=CRS_WGS84)
    gdf["accuracy"] = np.nan
    gdf.index.name = "id"
    return gdf, labels


def _get_labels(geolife_path, uids):
    """Generate dictionary with the available mode labels.

    Parameters
    ----------
    geolife_path : str
        Path to the directory with the geolife data.
    uids : iterable
        User folders in the geolife data directory.

    Returns
    -------
    dict
        dict containing the mode labels with the uids in the keys.

    Notes
    -----
    No further checks are done on user ids, they must be convertable to ints.
    """
    labels_rename = {"Start Time": "started_at", "End Time": "finished_at", "Transportation Mode": "mode"}
    label_dict = {}  # output dict for the labels

    # get paths to all "labels.txt" files.
    possible_label_paths = ((os.path.join(geolife_path, user_id, "labels.txt"), user_id) for user_id in uids)
    label_paths = ((path, user_id) for path, user_id in possible_label_paths if os.path.exists(path))

    # insert all labels into the output dict
    for path, user_id in label_paths:
        labels = pd.read_csv(path, delimiter="\t")
        labels.rename(columns=labels_rename, inplace=True)
        labels["started_at"] = pd.to_datetime(labels["started_at"], format="%Y/%m/%d %H:%M:%S", utc=True)
        labels["finished_at"] = pd.to_datetime(labels["finished_at"], format="%Y/%m/%d %H:%M:%S", utc=True)
        label_dict[int(user_id)] = labels
    return label_dict


def _get_df(geolife_path, uids, print_progress):
    """Create a generator that yields single trajectory dataframes.

    Parameters
    ----------
    geolife_path : str
        Path to the directory with the geolife data.
    uids : iterable
        User folders in the geolife data directory.
    print_progress : bool
        Show per-user progress if set to True.

    Yields
    -------
    pd.DataFrame
        A single DataFrame from a single trajectory file.

    Notes
    -----
    No further checks are done on user ids, they must be convertable to ints.
    """
    disable = not print_progress
    names = ["latitude", "longitude", "zeros", "elevation", "date days", "date", "time"]
    usecols = ["latitude", "longitude", "elevation", "date", "time"]

    for user_id in tqdm(uids, disable=disable):
        pattern = os.path.join(geolife_path, user_id, "Trajectory", "*.plt")
        for traj_file in glob.glob(pattern):
            data = pd.read_csv(traj_file, skiprows=6, header=None, names=names, usecols=usecols)
            data["tracked_at"] = pd.to_datetime(data["date"] + " " + data["time"], format="%Y-%m-%d %H:%M:%S", utc=True)
            data["geom"] = gpd.points_from_xy(data["longitude"], data["latitude"])
            data["user_id"] = int(user_id)
            data["elevation"] = data["elevation"] * FEET2METER
            data.drop(columns=["date", "time", "longitude", "latitude"], inplace=True)
            yield data


def geolife_add_modes_to_triplegs(
    triplegs, labels, ratio_threshold=0.5, max_triplegs=20, max_duration_tripleg=7 * 24 * 60 * 60
):
    """
    Add available mode labels to geolife data.

    The Geolife dataset provides a set of tripleg labels that are defined by a duration but are not matched to the
    Geolife tracking data. This function matches the labels to triplegs based on their temporal overlap.

    Parameters
    ----------
    triplegs : GeoDataFrame (as trackintel triplegs)
        Geolife triplegs.

    labels : dictionary
        Geolife labels as provided by the trackintel `read_geolife` function.

    ratio_threshold : float, default 0.5
        How much a label needs to overlap a tripleg to assign a the to this tripleg.

    max_triplegs : int, default 20
        Number of neighbors that are considered in the search for matching triplegs.

    max_duration_tripleg : float, default 7 * 24 * 60 * 60 (seconds)
        Used for a primary filter. All triplegs that are further away in time than 'max_duration_tripleg' from a
        label won't be considered for matching.

    Returns
    -------
    tpls : GeoDataFrame (as trackintel triplegs)
        triplegs with mode labels.

    Notes
    ------
    In the case that several labels overlap with the same tripleg the label with the highest overlap (relative to the
    tripleg) is chosen

    Example
    ----------
    >>> from trackintel.io.dataset_reader import read_geolife, geolife_add_modes_to_triplegs
    >>> pfs, mode_labels = read_geolife(os.path.join('downloads', 'Geolife Trajectories 1.3'))
    >>> pfs, sp = pfs.as_positionfixes.generate_staypoints()
    >>> pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
    >>> tpls = geolife_add_modes_to_triplegs(tpls, mode_labels)
    """
    tpls = triplegs.copy()
    # temp time fields for nn query
    tpls["started_at_s"] = (tpls["started_at"] - pd.Timestamp("1970-01-01", tz="utc")) // pd.Timedelta("1s")
    tpls["finished_at_s"] = (tpls["finished_at"] - pd.Timestamp("1970-01-01", tz="utc")) // pd.Timedelta("1s")
    # tpls_id_mode_list is used to collect tripleg-mode matches. It will be filled with dictionaries with the
    # following keys: [id', 'label_id', 'mode']
    tpls_id_mode_list = list()

    for user_this in labels.keys():
        tpls_this = tpls[tpls["user_id"] == user_this]
        labels_this = labels[user_this]

        labels_this["started_at_s"] = (
            labels_this["started_at"] - pd.Timestamp("1970-01-01", tz="utc")
        ) // pd.Timedelta("1s")
        labels_this["finished_at_s"] = (
            labels_this["finished_at"] - pd.Timestamp("1970-01-01", tz="utc")
        ) // pd.Timedelta("1s")

        # fit search tree on timestamps
        if tpls_this.shape[0] < max_triplegs:
            max_triplegs = tpls_this.shape[0]
        nn = NearestNeighbors(n_neighbors=max_triplegs, metric="chebyshev")
        nn.fit(tpls_this[["started_at_s", "finished_at_s"]])

        # find closest neighbours for timestamps in labels
        distances, candidates = nn.kneighbors(labels_this[["started_at_s", "finished_at_s"]])

        # filter anything above max_duration_tripleg (max distance start or end)
        pre_filter = distances > max_duration_tripleg

        candidates = pd.DataFrame(candidates, dtype="Int64")
        candidates[pre_filter] = np.nan
        candidates.dropna(how="all", inplace=True)

        # collect the tripleg - mode matches in the
        tpls_id_mode_list.extend(_calc_overlap_for_candidates(candidates, tpls_this, labels_this, ratio_threshold))

    if len(tpls_id_mode_list) == 0:
        tpls["mode"] = np.nan
    else:
        tpls_id_mode = pd.DataFrame(tpls_id_mode_list)

        # chose label with highest overlap
        tpls_id_mode = tpls_id_mode.sort_values(by=["id", "ratio"])
        # keep last (df sorted ascending)
        tpls_id_mode = tpls_id_mode.drop_duplicates(subset="id", keep="last").set_index("id")

        tpls = tpls.join(tpls_id_mode)
        tpls = tpls.astype({"label_id": "Int64"})

    tpls.drop(["started_at_s", "finished_at_s"], axis=1, inplace=True)

    try:
        tpls.drop(["ratio"], axis=1, inplace=True)
    except KeyError:
        pass

    return tpls


def _calc_overlap_for_candidates(candidates, tpls_this, labels_this, ratio_threshold):
    """
    Iterate all candidate triplegs and labels for a single user.

    Parameters
    ----------
    candidates : DataFrame
        A dataframe that has the following properties:
        index = Reference to position in the label_this dataframe
        columns: nb of neighbors, sorted by temporal distance
        values: Reference to position in the tpls_this dataframe

    tpls_this : GeoDataFrame (as trackintel triplegs)
        triplegs of a single user

    labels_this : DataFrame
        labels of a single user

    ratio_threshold : float, optional
        How much a label needs to overlap a tripleg to assign a the to this tripleg.

    Returns
    -------
    tpls_id_mode_list : list
        tpls_id_mode_list is used to collect tripleg-mode matches. It will be filled with dictionaries with the
        following keys: [id', 'label_id', 'mode']

    Notes
    -----
    Candidates is a matrix with one row per label and where each column corresponds to a potential tripleg match. All
    potential tripleg matches that are overlapped (in time) by more than ratio_threshold by a label are
    assigned this label.
    """
    tpls_id_mode_list = []

    # iterate all rows
    for label_pos, row in candidates.iterrows():
        potential_label = labels_this.iloc[label_pos, :]
        # for every row, iterate all columns. Unused column index would indicate the nth column.
        for _, tpls_pos in row.iteritems():

            # skip if tripleg was prefiltered and set to nan
            if pd.isna(tpls_pos):
                continue

            potential_tripleg = tpls_this.iloc[tpls_pos, :]
            ratio_this = calc_temp_overlap(
                potential_tripleg["started_at"],
                potential_tripleg["finished_at"],
                potential_label["started_at"],
                potential_label["finished_at"],
            )

            if ratio_this >= ratio_threshold:
                # assign label to tripleg (by storing matching in dictionary)
                tpls_id_mode_list.append(
                    {
                        "id": potential_tripleg.name,
                        "label_id": potential_label.name,
                        "mode": potential_label["mode"],
                        "ratio": ratio_this,
                    }
                )

    return tpls_id_mode_list
