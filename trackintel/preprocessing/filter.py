import geopandas as gpd



def spatial_filter(source, areas, method="within", re_project=False):
    """Filter staypoints, locations or triplegs with a geo extent. 

    Parameters
    ----------
    source : GeoDataFrame
        The source feature to perform the spatial filtering 

    areas : GeoDataFrame
        The areas used to perform the spatial filtering. Note, you can have multiple Polygons 
        and it will return all the features intersect with ANY of those geometries.

    method : str, {'within', 'intersects', 'crosses'}, default 'within'
        - 'within'    : return instances in 'source' where no points of these instances lies in the \
            exterior of the 'areas' and at least one point of the interior of these instances lies \
            in the interior of 'areas'.
        - 'intersects': return instances in 'source' where the boundary or interior of these instances \
            intersect in any way with those of the 'areas'
        - 'crosses'   : return instances in 'source' where the interior of these instances intersects \
            the interior of the 'areas' but does not contain it, and the dimension of the intersection \
            is less than the dimension of the one of the 'areas'.
            
    re_project : bool, default False
        If this is set to True, the 'source' will be projected to the coordinate reference system of 'areas' 
    
    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame containing the features after the spatial filtering.
        
    Examples
    --------
    >>> spts.as_staypoints.spatial_filter(areas, method="within", re_project=False)
    """
    
    gdf = source.copy()
    
    if re_project:
        init_crs = gdf.crs
        gdf = gdf.to_crs(areas.crs)
    
    # build spatial index for pre filtering
    source_sindex = gdf.sindex
    possible_matches_index = []
    for other in areas.itertuples():
        bounds = other.geometry.bounds
        c = list(source_sindex.intersection(bounds))
        possible_matches_index += c

    # Get unique candidates
    unique_candidate_matches = list(set(possible_matches_index))
    possible_matches = gdf.iloc[unique_candidate_matches]
    
    # get final result
    if method == 'within':
        ret_gdf = possible_matches.loc[possible_matches.within(areas.unary_union)]
    elif method == 'intersects':
        ret_gdf = possible_matches.loc[possible_matches.intersects(areas.unary_union)]
    elif method == 'crosses':
        ret_gdf = possible_matches.loc[possible_matches.crosses(areas.unary_union)]
    else:
        raise AttributeError("method unknown. We only support ['within', 'intersects', 'crosses']. "
                                f"You passed {method}")
        
    if re_project:
        return ret_gdf.to_crs(init_crs)
    else:
        return ret_gdf