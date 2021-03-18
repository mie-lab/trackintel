import numpy as np

from trackintel.geogr.distances import check_wgs_for_distance_calculation, calculate_haversine_length


def predict_transport_mode(triplegs, method='simple-coarse', **kwargs):
    """
    Predict the transport mode of triplegs.
    
    Predict/impute the transport mode that was likely chosen to cover the given
    tripleg, e.g., car, bicycle, or walk.

    Parameters
    ----------
    method: {'simple-coarse'}
        The following methods are available for transport mode inference/prediction:
        
        - 'simple-coarse' : Uses simple heuristics to predict coarse transport classes. 
    
    Returns
    -------
    triplegs : GeoDataFrame (as trackintel triplegs)
        The triplegs with added column mode, containing the predicted transport modes.
    
    Notes
    -----
    ``simple-coarse`` method includes ``{'slow_mobility', 'motorized_mobility', 'fast_mobility'}``. 
    In the default classification, ``slow_mobility`` (<15 km/h) includes transport modes such as 
    walking or cycling, ``motorized_mobility`` (<100 km/h) modes such as car or train, and 
    ``fast_mobility`` (>100 km/h) modes such as high-speed rail or airplanes. 
    These categories are default values and can be overwritten using the keyword argument categories.
        
    """
    if method == 'simple-coarse':
        # implemented as keyword argument if later other methods that don't use categories are added
        categories = kwargs.pop('categories', {15 / 3.6: 'slow_mobility', 100 / 3.6: 'motorized_mobility',
                                               np.inf: 'fast_mobility'})

        return predict_transport_mode_simple_coarse(triplegs, categories)
    else:
        raise NameError(f'Method {method} not known for predicting tripleg transport modes.')


def predict_transport_mode_simple_coarse(triplegs_in, categories):
    """
    Predict a transport mode out of three coarse classes. 
    
    Implements a simple speed based heuristic (over the whole tripleg). 
    As such, it is very fast, but also very simple and coarse.

    Parameters
    ----------
    triplegs : trackintel triplegs GeoDataFrame
        The triplegs for the transport mode prediction.
        
    categories : dict, optional
        The categories for the speed classification {upper_boundary:'category_name'}. 
        The unit for the upper boundary is m/s.
        The default is {15/3.6: 'slow_mobility', 100/3.6: 'motorized_mobility', np.inf: 'fast_mobility'}.

    Raises
    ------
    ValueError
        In case the boundaries of the categories are not in ascending order.

    Returns
    -------
    triplegs : trackintel triplegs GeoDataFrame
        the triplegs with added column mode, containing the predicted transport modes.

    For additional documentation, see
    :func:`trackintel.analysis.transport_mode_identification.predict_transport_mode`.

    """
    if not (check_categories(categories)):
        raise ValueError('the categories must be in increasing order')

    triplegs = triplegs_in.copy()
    wgs = check_wgs_for_distance_calculation(triplegs.crs)
    #
    if wgs:
        triplegs['distance'] = calculate_haversine_length(triplegs)
    else:
        triplegs['distance'] = triplegs.length

    def identify_mode(tripleg, categories):
        """
        Identify the mode based on the (overall) tripleg speed.

        Parameters
        ----------
        tripleg : trackintel triplegs GeoDataFrame
            the tripleg to analyse
        wgs : bool
            whether the tripleg is in WGS84 or not.
        categories : dict
            the upper boundaries (as keys) and the names of the categories as values.

        Returns
        -------
        str
            the identified mode.
        """

        duration = (tripleg['finished_at'] - tripleg['started_at']).total_seconds()
        speed = tripleg['distance'] / duration  # The unit of the speed is m/s

        for bound in categories:
            if speed < bound:
                return categories[bound]

    triplegs['mode'] = triplegs.apply(identify_mode, args=((categories,)), axis=1)
    return triplegs


def check_categories(cat):
    """
    Check if the keys of a dictionary are in ascending order.

    Parameters
    ----------
    cat : disct
        the dictionary to be checked.

    Returns
    -------
    correct : bool
        True if dict keys are in ascending order False otherwise.

    """
    correct = True
    bounds = list(cat.keys())
    for i in range(len(bounds) - 1):
        if bounds[i] >= bounds[i + 1]:
            correct = False
    return correct
