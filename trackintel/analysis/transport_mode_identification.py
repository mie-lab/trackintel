# -*- coding: utf-8 -*-

import trackintel as ti
from trackintel.geogr.distances import haversine_dist


def predict_transport_mode(triplegs, method='simple-coarse'):
    """
    Predicts/imputes the transport mode that was likely chosen to cover the 
    given tripleg, e.g., car, bicycle, or walk.

    Parameters
    ----------
    method: str, {'simple-coarse'}, default 'simple-coarse'
        The following methods are available for transport mode inference/prediction:
        'simple-coarse' : Uses simple heuristics to predict coarse transport classes.
            These include ``{'slow_mobility', 'motorized_mobility', 'fast_mobility'}``.
            In this classification, ``slow_mobility`` includes transport modes such as
            walking or cycling, ``motorized_mobility`` modes such as car or train, and
            ``fast_mobility`` modes such as high-speed rail or airplanes.
    """
    if method == 'simple-coarse':
        return predict_transport_mode_simple_coarse(triplegs)
    else:
        raise NameError(f'Method {method} not known for predicting tripleg transport modes.')


def predict_transport_mode_simple_coarse(triplegs):
    """
    Predicts a transport mode out of three coarse classes. Implements a simple speed-
    based heuristic (over the whole tripleg) that uses the cutoffs 15 km/h and 100 km/h. 
    As such, it is very fast, but also very simple and coarse.

    For additional documentation, see 
    :func:`trackintel.analysis.transport_mode_identification.predict_transport_mode`.
    """
    triplegs = triplegs.copy()

    def identify_mode(tripleg):
        """Identifies the mode based on the (overall) tripleg speed."""
        # Computes distance over whole tripleg geometry (using the Haversine distance).
        distance = sum([haversine_dist(pt1[0], pt1[1], pt2[0], pt2[1]) for pt1, pt2 \
            in zip(tripleg.geom.coords[:-1], tripleg.geom.coords[1:])])
        duration = (tripleg['finished_at'] - tripleg['started_at']).total_seconds()
        speed = distance / duration

        if speed < 15 / 3.6:
            return 'slow_mobility'
        elif speed < 100 / 3.6:
            return 'motorized_mobility'
        else:
            return 'fast_mobility'

    triplegs['mode'] = triplegs.apply(identify_mode, axis=1)
    return triplegs
