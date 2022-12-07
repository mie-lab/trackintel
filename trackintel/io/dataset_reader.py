# -*- coding: utf-8 -*-

import glob
import os
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from trackintel.preprocessing.util import calc_temp_overlap

FEET2METER = 0.3048
CRS_WGS84 = 4326
CRS_CH1903 = 21781
MZMV_encoding = "latin1"


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


def read_mzmv(mzmv_path):
    """Read the data from Swiss "Mikrozensus Mobilität und Verkehr"

    Parameters
    ----------
    mzmv_path : str
        Path to unzipped data folder of MZMB (everything else should be left zipped).

    Returns
    -------
    trips : GeoDataFrame (as trackintel trips)
    sp : GeoDataFrame (as trackintel staypoints)
    tpls : GeoDataFrame (as trackintel triplegs)

    Notes
    -----
    !important! As geometry (column `geom`) for triplegs we set verification points (VP_XY), the quality of these
    points is low
    but they are available for all entries (fit into trackintel model). In the worst case we have only start and end
    points of the tripleg. Prefer whenever possible the column `geometry` this geometry is created by a routing tool
    but it is not available for all entries.

    To fit the trackintel model, we rename `HHNR` to `user_id`, rename [`f51100time`, `f51400time`] to
    [`started_at`, `finished_at`].
    Since the MZMV data contains only time information and no date information, the data is set to 1970-01-01.
    """
    shp = os.path.join(mzmv_path, "5_Routen(Geometriefiles)\\CH_routen.zip")
    db_csv = os.path.join(mzmv_path, "4_DB_csv\\CH_CSV.zip")

    zf = ZipFile(db_csv)
    with zf.open("wege.csv") as f:
        wege = pd.read_csv(f, encoding="latin")
    wege.index.name = "trip_id"
    # make copy to merge trip_id to triplegs
    rename_columns = {"HHNR": "user_id", "f51100time": "started_at", "f51400time": "finished_at"}
    wege.rename(columns=rename_columns, inplace=True)
    # string to times + add date
    wege["started_at"] = _mzmv_to_datetime(wege["started_at"])
    wege["finished_at"] = _mzmv_to_datetime(wege["finished_at"])
    trip_id_merge = wege[["user_id", "WEGNR"]].reset_index()

    with zf.open("etappen.csv") as f:
        etappen = pd.read_csv(f, encoding=MZMV_encoding)
    etappen.rename(columns=rename_columns, inplace=True)
    etappen["started_at"] = _mzmv_to_datetime(etappen["started_at"])
    etappen["finished_at"] = _mzmv_to_datetime(etappen["finished_at"])
    etappen = pd.merge(etappen, trip_id_merge, on=["user_id", "WEGNR"])
    sp = _mzmv_generate_sp(etappen, zf)

    # Read Geometry: #
    # possible to pass zip folder as argument as folder contains only one file
    routen = gpd.read_file(shp)[["HHNR", "ETNR", "geometry"]]  # takes long
    routen.rename(columns=rename_columns, inplace=True)
    etappen = pd.merge(etappen, routen, on=["user_id", "ETNR"], how="left")

    vp = _mzmv_verification_points(zf, "verifikationspunkte.csv", etappen)
    etappen = pd.merge(etappen, vp, on=["user_id", "ETNR"], how="left")

    etappen = gpd.GeoDataFrame(etappen, geometry="VP_XY", crs=CRS_WGS84)
    etappen.index.name = "tripleg_id"
    return wege, sp, etappen


def _mzmv_verification_points(zf, filepath, etappen):
    """Extracts verifications points as LineStrings.

    Start and endpoint of tripleg is added from etappen to gain valid LineStrings.

    Parameters
    ----------
    zf : zipfile.ZipFile
        ZipFile with which we can open filepath
    filepath : str
        path to verification points file within zf.
    etappen : DataFrame
        etappen with renamed columns.

    Returns
    -------
    pd.DataFrame
        Filled with aggregated verification points per tripleg.
        Columns `VP_XY`, `VP_XY_CH1903` contain geometries.
    """
    # MZMV stores 6 points + border in long format
    # this method aggregates up to 6 points (w/o border) into geometry
    num_points = 6

    with zf.open(filepath) as f:
        vp = pd.read_csv(f, encoding=MZMV_encoding)
    vp.rename(columns={"HHNR": "user_id"}, inplace=True)  # to be inline with etappen

    # insert nan to later drop point w/o geometry
    geom_cols = ["{}X", "{}Y", "{}X_CH1903", "{}Y_CH1903"]
    # e.g. point 2 has geometry R2_X, R2_Y, R2_X_CH1903, R2_Y_CH1903
    na_997 = [c.format(f"R{i}_") for i in range(1, num_points + 1) for c in geom_cols]
    for col in na_997:
        vp.loc[vp[col] == -997, col] = np.nan

    # we only keep information to join back + geometry
    cols = ["user_id", "ETNR"] + geom_cols
    group = [[c.format(f"R{i}_") for c in cols] for i in range(1, num_points + 1)]
    rename_target = [c.format("") for c in cols]
    rename_maps = [{g_col: r_col for (g_col, r_col) in zip(g, rename_target)} for g in group]

    # split by verification point number, rename for concat, drop verification point w/o geometry
    gcol = ["X", "Y", "X_CH1903", "Y_CH1903"]
    vps = [vp[g].rename(columns=rm).dropna(subset=gcol, how="all") for (g, rm) in zip(group, rename_maps)]
    vp = pd.concat(vps)  # inner order: R1, R2, ..., R6

    # gather start and end from etappen and put them into the same format
    sp = etappen[["user_id", "ETNR", "S_X", "S_Y", "S_X_CH1903", "S_Y_CH1903"]].copy()
    ep = etappen[["user_id", "ETNR", "Z_X", "Z_Y", "Z_X_CH1903", "Z_Y_CH1903"]].copy()
    sp.rename(columns={"S_X": "X", "S_Y": "Y", "S_X_CH1903": "X_CH1903", "S_Y_CH1903": "Y_CH1903"}, inplace=True)
    ep.rename(columns={"Z_X": "X", "Z_Y": "Y", "Z_X_CH1903": "X_CH1903", "Z_Y_CH1903": "Y_CH1903"}, inplace=True)
    vp = pd.concat((sp, vp, ep))  # right order is important (S, R1, ..., R6, E)

    vp["VP_XY"] = gpd.points_from_xy(x=vp["X"], y=vp["Y"], crs=CRS_WGS84)
    vp["VP_XY_CH1903"] = gpd.points_from_xy(x=vp["X_CH1903"], y=vp["Y_CH1903"], crs=CRS_CH1903)
    vp.drop(columns=["X", "Y", "X_CH1903", "Y_CH1903"], inplace=True)

    # aggregate points in the same etappe into a linestring
    aggfuncs = {"VP_XY": lambda xy: LineString(xy.to_list()), "VP_XY_CH1903": lambda xy: LineString(xy.to_list())}
    # groupby keeps innner order!
    vp = vp.groupby(["user_id", "ETNR"], as_index=False).agg(aggfuncs)
    return vp


def _mzmv_to_datetime(col):
    """Convert time from mzmv to pd.Timestamp on fix date 1970-01-01.

    Parameters
    ----------
    col : pd.Series
        Times stored as strings.

    Returns
    -------
    pd.Series
        Times stored as pd.Timestamp[ns]

    """
    postfix = [" 1970-01-01"] * len(col)  # no broadcasting possible
    midnight = col == "24:00:00"
    col = col.str.cat(postfix)
    # 24:00:00 is no valid time
    # to keep everything on same day loose 1 sec
    col[midnight] = "23:59:59 1970-01-01"
    return pd.to_datetime(col, format="%H:%M:%S %Y-%m-%d")


def _mzmv_generate_sp(etappen, zf):
    """Generate staypoints only from etappen.

    Parameters
    ----------
    etappen : pd.DataFrame
        DataFrame with renamed columns. Sorted by "user_id" and "WEGNR" ascending.
        Must contain column "trip_id".
    zf : zipfile.ZipFile
        zipfile with which we can open "haushalte.csv" and "zielpersonen.csv"
        to add column purpose to staypoints.

    Returns
    -------
    staypoints : GeoDataFrame (as trackintel staypoints)
        staypoints with mandatory columns, all the columns for staypoints from mzmv,
        and additionally "prev_trip_id", "trip_id", "next_trip_id", "is_activity",
        "purpose", "purpose_tpls"

    Notes
    -----
    Encoding for values of "purpose_tpls" can be looked up in the documentation of MZMV.
    """
    assert "trip_id" in etappen.columns  # small regression test
    etappen.sort_values(by=["user_id", "ETNR"], inplace=True)
    first_tpls = etappen["ETNR"] == 1  # first tripleg of user (ETNR is unique per user)
    last_tpls = first_tpls.shift(-1, fill_value=True)
    # create staypoints from starts
    # if previous staypoint is different user/trip -> staypoint is activity
    etappen["S_is_activity"] = (etappen[["user_id", "WEGNR"]] != etappen[["user_id", "WEGNR"]].shift(1)).any(axis=1)
    # quick and dirty copy trip ids and delete most of in next step
    etappen["S_prev_trip_id"] = etappen["trip_id"].shift(1)
    etappen["S_next_trip_id"] = etappen["trip_id"]
    etappen["S_trip_id"] = etappen["trip_id"]  # to not overwrite it
    # staypoints that aren't activity are in a trip (don't have prev or next)
    etappen.loc[~etappen["S_is_activity"], "S_prev_trip_id"] = np.nan
    etappen.loc[~etappen["S_is_activity"], "S_next_trip_id"] = np.nan
    # activity is outside of trips -> no trip id
    etappen.loc[etappen["S_is_activity"], "S_trip_id"] = np.nan

    etappen["S_finished_at"] = etappen["started_at"]  # time you leave staypoint
    # end of next etappe within same weg is *always* identical to start in previous one.
    # end of weg within same user is *always* identical to start in previous one.
    etappen["S_started_at"] = etappen["finished_at"].shift(1, fill_value=pd.NaT)
    # first weg -> we only know finish time for staypoints
    etappen.loc[first_tpls, "S_started_at"] = pd.NaT
    # add purpose of triplegs to staypoints
    etappen["S_purpose_tpls"] = etappen["f52900"].shift(1)
    etappen.loc[first_tpls, "S_purpose_tpls"] = None
    # **all** the columns that are associated with the staypoints
    col = [
        "X",
        "Y",
        "QAL",
        "BFS",
        "PLZ",
        "Ort",
        "Str",
        "hnr",
        "LND",
        "X_CH1903",
        "Y_CH1903",
        "SPRACHE",
        "REGION",
        "KANTON",
        "NUTS3",
        "AGGLO2000",
        "AGGLO_GROESSE2000",
        "STRUKTUR_2000",
        "STRUKTUR_AGG_2000",
        "struktur_bfs9_2000",
        "AGGLO2012",
        "AGGLO_GROESSE2012",
        "staedt_char_2012",
        "stat_stadt_2012",
        "DEGURBA",
        "is_activity",
        "started_at",
        "finished_at",
        "prev_trip_id",
        "next_trip_id",
        "trip_id",
        "purpose_tpls"
    ]
    # W_X_CH1903, X coordinate of home, CH1903 as integers are better to join 
    s_col = ["user_id", "WEGNR", "ETNR", "W_X_CH1903", "W_Y_CH1903"] + ["S_" + c for c in col]
    sp = etappen[s_col].copy()
    sp.rename(columns={"S_" + c: c for c in col}, inplace=True)

    # what now is missing is the last staypoint of the last trip
    etappen["Z_is_activity"] = True  # we filter later
    etappen["Z_prev_trip_id"] = etappen["trip_id"]
    etappen["Z_next_trip_id"] = np.nan  # is always last trip
    etappen["Z_trip_id"] = np.nan  # outside of trips
    etappen["Z_started_at"] = etappen["finished_at"]
    etappen["Z_finished_at"] = pd.NaT
    etappen["Z_purpose_tpls"] = etappen["f52900"]
    z_col = ["user_id", "WEGNR", "ETNR", "W_X_CH1903", "W_Y_CH1903"] + ["Z_" + c for c in col]
    sp_last = etappen.loc[last_tpls, z_col]
    sp_last.rename(columns={"Z_" + c: c for c in col}, inplace=True)

    sp = pd.concat((sp, sp_last))
    # now we add column purpose to show if work (home we got from etappen W_..)
    with zf.open("zielpersonen.csv") as f:
        usecols = ["HHNR", "A_X_CH1903", "A_Y_CH1903", "AU_X_CH1903", "AU_Y_CH1903"]
        zielpersonen = pd.read_csv(f, encoding=MZMV_encoding, usecols=usecols).rename(columns={"HHNR": "user_id"})
    sp = pd.merge(sp, zielpersonen, how="left", on=["user_id"])
    # A_<...> for work, AU_<...> for education
    work = ((sp["A_X_CH1903"] == sp["X_CH1903"]) & (sp["A_Y_CH1903"] == sp["Y_CH1903"])) | (
        (sp["AU_X_CH1903"] == sp["X_CH1903"]) & (sp["AU_Y_CH1903"] == sp["Y_CH1903"])
    )
    home = (sp["W_X_CH1903"] == sp["X_CH1903"]) & (sp["W_Y_CH1903"] == sp["Y_CH1903"])
    sp.loc[work, "purpose"] = "work"
    sp.loc[home, "purpose"] = "home"  # potentially overwrite work

    sp.reset_index(drop=True, inplace=True)  # drop etappen index

    sp["XY"] = gpd.points_from_xy(sp["X"], sp["Y"], crs=CRS_WGS84)
    sp["XY_CH1904"] = gpd.points_from_xy(sp["X_CH1903"], sp["Y_CH1903"], crs=CRS_CH1903)
    sp = gpd.GeoDataFrame(sp, geometry="XY", crs=CRS_WGS84)
    sp.index.name = "staypoint_id"

    # clean up
    sp_drop = ["X", "Y", "X_CH1903", "Y_CH1903"]
    sp_drop += ["W_X_CH1903", "W_Y_CH1903", "A_X_CH1903", "A_Y_CH1903", "AU_X_CH1903", "AU_Y_CH1903"]
    sp.drop(columns=sp_drop, inplace=True)
    added_cols = [b + c for b in ("S_", "Z_") for c in col[-7:]]
    etappen.drop(columns=added_cols, inplace=True)
    return sp
