import dateutil.parser

import pandas as pd
import geopandas as gpd
import shapely

from shapely.geometry import Point


def read_positionfixes_csv(*args, **kwargs):
    """Wraps the pandas read_csv function, extracts longitude and latitude and 
    builds a geopandas GeoDataFrame.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.
    """
    df = pd.read_csv(*args, **kwargs)
    df['geom'] = list(zip(df.longitude, df.latitude))
    df['geom'] = df['geom'].apply(Point)
    df['tracked_at'] = df['tracked_at'].apply(dateutil.parser.parse)
    return gpd.GeoDataFrame(df, geometry='geom')
