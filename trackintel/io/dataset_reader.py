# -*- coding: utf-8 -*-

import glob
import ntpath
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

FEET2METER = 0.3048

CRS_WGS84 = {'init': 'epsg:4326'}


def read_geolife(geolife_path):
    """ Read raw geolife data and return geopandas dataframe

    The geolife dataset as it can be downloaded from
    https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
    contains a folder for every tracked user with one file with tracking data for every day. We iterate every
    folder and concatenate all files of 1 user into a single geopandas dataframe that is compatible with
    trackintel.
    Relevant publications:
    [1] Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. Mining interesting locations and travel sequences from
    GPS trajectories. In Proceedings of International conference on World Wild Web (WWW 2009),
    Madrid Spain. ACM Press: 791-800.
    [2] Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, Wei-Ying Ma. Understanding Mobility Based on GPS Data.
    In Proceedings of ACM conference on Ubiquitous Computing (UbiComp 2008), Seoul, Korea. ACM Press: 312-321.
    [3] Yu Zheng, Xing Xie, Wei-Ying Ma, GeoLife: A Collaborative Social Networking Service among User, location
    and trajectory. Invited paper, in IEEE Data Engineering Bulletin. 33, 2, 2010, pp. 32-40.

    Parameters
    ----------
    geolife_path: str
    Path to the top level directory of the downloaded geolife data. E.g., the folder that contains the folders for the
    different users:
    -geolife_path
    -- 000
    --- Trajectory
    ---- 20081023025304.plt
    ---- 20081024020959.plt
    ---- 20081026134407.plt
    ---- ...
    -- 001
    --- Trajectory
    ---- ...
    -- ...

    Returns
    -------
    gdf: geopandas dataframe
    A geopandas dataframe with the following columns:
    'lat': float64, Latitude WGS84; 'lon': float64, Latitude WGS84; 'elevation': float64, in meters;
    'tracked_at': datetime64[ns]; 'user_id': int64; 'geom': geopandas/shapely geometry; 'accuracy': None
    """


    geolife_path = os.path.join(geolife_path, '*')
    user_folder = sorted(glob.glob(geolife_path))

    df_list_users = []

    if len(user_folder) == 0:
        raise NameError('No folders found with working directory {} and path {}'.format(os.getcwd(), geolife_path))

    for user_folder_this in user_folder:

        # skip files
        if not os.path.isdir(user_folder_this):
            continue

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
                                                     + ' ' + data_this['time'], format="%Y-%m-%d %H:%M:%S")

            data_this.drop(['zeros', 'date days', 'date', 'time'], axis=1,
                           inplace=True)
            data_this['user_id'] = user_id
            data_this['elevation'] = data_this['elevation'] * FEET2METER

            data_this['geom'] = list(zip(data_this.lon, data_this.lat))
            data_this['geom'] = data_this['geom'].apply(Point)

            df_list_days.append(data_this)

        # concat all days of a user into a single dataframe
        df_user_this = pd.concat(df_list_days, axis=0, ignore_index=True)
        print("finished user_id: ", user_id)

        df_list_users.append(df_user_this)

    df = pd.concat(df_list_users, axis=0, ignore_index=True)
    gdf = gpd.GeoDataFrame(df, geometry="geom", crs=CRS_WGS84)
    gdf["accuracy"] = np.nan

    gdf.index.name = 'id'

    return gdf
