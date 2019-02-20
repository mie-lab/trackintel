import dateutil.parser

import pandas as pd
import geopandas as gpd
import shapely

from shapely.geometry import Point


def read_positionfixes_csv(*args, **kwargs):
    """Wraps the pandas read_csv function, extracts longitude and latitude and 
    builds a geopandas GeoDataFrame. This also validates that the ingested data
    conforms to the trackintel understanding of positionfixes. 
    See :doc:`/modules/model`.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.
    """
    df = pd.read_csv(*args, **kwargs)
    df['geometry'] = list(zip(df.longitude, df.latitude))
    df['geometry'] = df['geometry'].apply(Point)
    df['tracked_at'] = df['tracked_at'].apply(dateutil.parser.parse)
    df = df.drop(['longitude', 'latitude'], axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    assert gdf.as_positionfixes
    return gdf
