# -*- coding: utf-8 -*-

import glob
import ntpath
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

FEET2METER = 0.3048

CRS_WGS84 = 'epsg:4326'


def read_geolife(geolife_path):
    """ Read raw geolife data and return trackintel positionfixes

    This functions parses all geolife data available in the directory ``geolife_path``

    Parameters
    ----------
    geolife_path: str
        path to the directory with the geolife data

    Returns
    -------
    gdf: GeoPandas DataFrame
        Contains all loaded geolife positionfixes
    labels: dict
        Dictionary with the available (optional) mode labels.

    Notes
    ------
    The geopandas dataframe has the following columns and datatype: 'lat': float64, Latitude WGS84; 'lon': float64, Latitude WGS84; 'elevation': float64, in meters;
    'tracked_at': datetime64[ns]; 'user_id': int64; 'geom': geopandas/shapely geometry; 'accuracy': None;

    The label dictionary contains the user ids as keys and DataFrames with the available labels as values.

    The folder structure within the geolife directory needs to be identical with the folder structure
    available from the official download. The means that the top level folder (provided with 'geolife_path')
    contains the folders for the different users:

    | geolife_path
    | ├── 000
    | │   ├── Trajectory
    | │   │   ├── 20081023025304.plt
    | │   │   ├── 20081024020959.plt
    | │   │   └── 20081026134407.plt
    | ├── 001
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
    >>> geolife_pfs, labels = read_geolife(os.path.join('downloads', 'Geolife Trajectories 1.3'))
    """

    geolife_path = os.path.join(geolife_path, '*')
    user_folder = sorted(glob.glob(geolife_path))

    df_list_users = []
    label_dict = dict()

    if len(user_folder) == 0:
        raise NameError('No folders found with working directory {} and path {}'.format(os.getcwd(), geolife_path))

    for user_folder_this in user_folder:

        # skip files
        if not os.path.isdir(user_folder_this):
            continue

        # check if labels are available
        try:
            labels = pd.read_csv(os.path.join(user_folder_this, 'labels.txt'), delimiter="\t")
            rename_dict = {"Start Time": "t_start", "End Time": "t_end", "Transportation Mode": "mode"}
            labels.rename(rename_dict, axis=1, inplace=True)
            labels['t_start'] = pd.to_datetime(labels['t_start'], format="%Y/%m/%d %H:%M:%S", utc=True)
            labels['t_end'] = pd.to_datetime(labels['t_end'], format="%Y/%m/%d %H:%M:%S", utc=True)
        except OSError:
            labels = pd.DataFrame(columns=["t_start", "t_end", "mode"])

        # extract user id from path
        _, tail = ntpath.split(user_folder_this)
        user_id = int(tail)
        print("start importing geolife user_id: ", user_id)

        input_files = sorted(glob.glob(os.path.join(
            user_folder_this, "Trajectory", "*.plt")))
        df_list_days = []

        # read every day of every user and concatenate input files
        for input_file_this in input_files:
            data_this = pd.read_csv(input_file_this, skiprows=6, header=None,
                                    names=['lat', 'lon', 'zeros', 'elevation',
                                           'date days', 'date', 'time'])

            data_this['tracked_at'] = pd.to_datetime(data_this['date']
                                                     + ' ' + data_this['time'], format="%Y-%m-%d %H:%M:%S", utc=True)

            data_this.drop(['zeros', 'date days', 'date', 'time'], axis=1,
                           inplace=True)
            data_this['user_id'] = user_id
            data_this['elevation'] = data_this['elevation'] * FEET2METER

            data_this['geom'] = list(zip(data_this.lon, data_this.lat))
            data_this['geom'] = data_this['geom'].apply(Point)

            df_list_days.append(data_this)

        # concat all days of a user into a single dataframe
        df_user_this = pd.concat(df_list_days, axis=0, ignore_index=True)
        label_dict[user_id] = labels
        print("finished user_id: ", user_id)

        df_list_users.append(df_user_this)

    df = pd.concat(df_list_users, axis=0, ignore_index=True)
    gdf = gpd.GeoDataFrame(df, geometry="geom", crs=CRS_WGS84)
    gdf["accuracy"] = np.nan

    gdf.index.name = 'id'

    return gdf, label_dict
