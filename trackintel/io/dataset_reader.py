# -*- coding: utf-8 -*-

import glob
import ntpath
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

FEET2METER = 0.3048

CRS_WGS84 = "epsg:4326"

from trackintel.preprocessing.util import calc_temp_overlap


def read_geolife(geolife_path, print_progress=False):
    """
    Read raw geolife data and return trackintel positionfixes.

    This functions parses all geolife data available in the directory ``geolife_path``

    Parameters
    ----------
    geolife_path: str
        path to the directory with the geolife data

    print_progress: Bool, default False
        Show per-user progress if set to True.

    Returns
    -------
    gdf: GeoDataFrame (as trackintel positionfixes)
        Contains all loaded geolife positionfixes

    labels: dict
        Dictionary with the available mode labels.

    Notes
    -----
    The geopandas dataframe has the following columns and datatype: 'latitude': float64, Latitude WGS84;
    'longitude': float64, Longitude WGS84; 'elevation': float64, in meters; 'tracked_at': datetime64[ns];
    'user_id': int64; 'geom': geopandas/shapely geometry; 'accuracy': None;

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
    geolife_path = os.path.join(geolife_path, "*")
    user_folder = sorted(glob.glob(geolife_path))

    df_list_users = []
    label_dict = dict()

    if len(user_folder) == 0:
        raise NameError("No folders found with working directory {} and path {}".format(os.getcwd(), geolife_path))

    for user_folder_this in tqdm(user_folder, disable=not print_progress):

        # skip files
        if not os.path.isdir(user_folder_this):
            continue

        # check if labels are available
        try:
            labels = pd.read_csv(os.path.join(user_folder_this, "labels.txt"), delimiter="\t")
            rename_dict = {"Start Time": "started_at", "End Time": "finished_at", "Transportation Mode": "mode"}
            labels.rename(rename_dict, axis=1, inplace=True)
            labels["started_at"] = pd.to_datetime(labels["started_at"], format="%Y/%m/%d %H:%M:%S", utc=True)
            labels["finished_at"] = pd.to_datetime(labels["finished_at"], format="%Y/%m/%d %H:%M:%S", utc=True)
        except OSError:
            labels = pd.DataFrame(columns=["started_at", "finished_at", "mode"])

        # extract user id from path
        _, tail = ntpath.split(user_folder_this)
        try:
            user_id = int(tail)
        except ValueError as err:
            errmsg = (
                "Invalid user_id '{}' found in geolife path '{}'. The geolife path can only contain folders"
                " named with integers that represent the user id.".format(tail, user_folder_this)
            )
            raise ValueError(errmsg) from err

        input_files = sorted(glob.glob(os.path.join(user_folder_this, "Trajectory", "*.plt")))
        df_list_days = []

        # read every day of every user and concatenate input files
        for input_file_this in input_files:
            data_this = pd.read_csv(
                input_file_this,
                skiprows=6,
                header=None,
                names=["latitude", "longitude", "zeros", "elevation", "date days", "date", "time"],
            )

            data_this["tracked_at"] = pd.to_datetime(
                data_this["date"] + " " + data_this["time"], format="%Y-%m-%d %H:%M:%S", utc=True
            )

            data_this.drop(["zeros", "date days", "date", "time"], axis=1, inplace=True)
            data_this["user_id"] = user_id
            data_this["elevation"] = data_this["elevation"] * FEET2METER

            data_this["geom"] = list(zip(data_this["longitude"], data_this["latitude"]))
            data_this["geom"] = data_this["geom"].apply(Point)

            df_list_days.append(data_this)

        # concat all days of a user into a single dataframe
        df_user_this = pd.concat(df_list_days, axis=0, ignore_index=True)
        label_dict[user_id] = labels

        df_list_users.append(df_user_this)

    df = pd.concat(df_list_users, axis=0, ignore_index=True)
    gdf = gpd.GeoDataFrame(df, geometry="geom", crs=CRS_WGS84)
    gdf["accuracy"] = np.nan

    gdf.index.name = "id"

    return gdf, label_dict


def geolife_add_modes_to_triplegs(
    tpls_in, labels, ratio_threshold=0.5, max_triplegs=20, max_duration_tripleg=7 * 24 * 60 * 60
):
    """
    Add available mode labels to geolife data.

    The Geolife dataset provides a set of tripleg labels that are defined by a duration but are not matched to the
    Geolife tracking data. This function matches the labels to triplegs based on their temporal overlap.

    Parameters
    ----------
    tpls_in : GeoDataFrame (as trackintel triplegs)
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

    Example
    ----------
    >>> from trackintel.io.dataset_reader import read_geolife, geolife_add_modes_to_triplegs
    >>> pfs, mode_labels = read_geolife(os.path.join('downloads', 'Geolife Trajectories 1.3'))
    >>> pfs, spts = pfs.as_positionfixes.generate_staypoints()
    >>> pfs, tpls = pfs.as_positionfixes.generate_triplegs(spts)
    >>> tpls = geolife_add_modes_to_triplegs(tpls, mode_labels)
    """
    tpls = tpls_in.copy()
    # temp time fields for nn query
    tpls["started_at_s"] = (tpls["started_at"] - pd.Timestamp("1970-01-01", tz="utc")) // pd.Timedelta("1s")
    tpls["finished_at_s"] = (tpls["finished_at"] - pd.Timestamp("1970-01-01", tz="utc")) // pd.Timedelta("1s")
    all_users = tpls["user_id"].unique()
    # tpls_id_mode_list is used to collect tripleg-mode matches. It will be filled with dictionaries with the
    # following keys: [id', 'label_id', 'mode']
    tpls_id_mode_list = list()

    for user_this in all_users:
        tpls_this = tpls[tpls["user_id"] == user_this]
        labels_this = labels[user_this]
        if labels_this.empty:
            continue

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
        tpls_id_mode = pd.DataFrame(tpls_id_mode_list).set_index("id")
        tpls = tpls.join(tpls_id_mode)
        tpls = tpls.astype({"label_id": "Int64"})

    try:
        tpls.drop(["started_at_s", "finished_at_s"], axis=1, inplace=True)
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
                    {"id": potential_tripleg.name, "label_id": potential_label.name, "mode": potential_label["mode"]}
                )

    return tpls_id_mode_list
