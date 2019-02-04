import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point


def read_positionfixes_csv(*args, **kwargs):
    """
    Wraps the pandas read_csv function, extracts longitude and latitude and 
    builds a geopandas GeoDataFrame.
    """
    df = pd.read_csv(*args, **kwargs)
    df['geom'] = list(zip(df.longitude, df.latitude))
    df['geom'] = df['geom'].apply(Point)
    return gpd.GeoDataFrame(df, geometry='geom')
