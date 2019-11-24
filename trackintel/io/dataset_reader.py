# -*- coding: utf-8 -*-

import os
import time
import json
import ntpath
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
import psycopg2
import trackintel as ti
from shapely.geometry import Point
import matplotlib.pyplot as plt

FEET2METER = 0.3048

CRS_WGS84 = {'init' :'epsg:4326'}



def read_geolife(geolife_path):
   

    # In the geolife data, every user has a folder with a file with tracking data
    # for every day. We iterate every folder concatenate all files of 1 user into
    # a single pandas dataframe and send it to the postgres database.
    geolife_path = os.path.join(geolife_path,'*')
    user_folder = glob.glob(geolife_path)


    df_list_users = []

    for user_folder_this in user_folder:
        # extract user id from path
        _, tail = ntpath.split(user_folder_this)
        user_id = int(tail)
        print("start importing geolife user_id: ", user_id)
        
        input_files = glob.glob(os.path.join(
                    user_folder_this, "Trajectory", "*.plt"))
        df_list_days = []

        # read every day of every user and concatenate input files
        for input_file_this in input_files:
            
            data_this = pd.read_csv(input_file_this, skiprows=6, header=None,
                                    names=['lat', 'lon', 'zeros', 'elevation', 
                                           'date days', 'date', 'time'])

            data_this['tracked_at'] = pd.to_datetime(data_this['date']
                                                     + ' ' + data_this['time'])

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
    df["accuracy"] = None
    gdf = gpd.GeoDataFrame(df, geometry="geom", crs=CRS_WGS84)
    gdf["accuracy"] = None

    return gdf