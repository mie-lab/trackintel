import pytest
import sys
import os
import filecmp

import pandas as pd
import trackintel as ti
import numpy as np
from trackintel.geogr import distances
from trackintel.geogr.distances import haversine_dist


class Test_haversine_dist:
    def test_haversine_dist(self):

        # input_latlng saves different combinations of haversine-distances in meters and the longitude & latitudes from
        # two different points in WGS84
        # {haversine-distance in meters[longitude_P1, latitudes_P1, longitude_P2, latitudes_P2]}
        input_latlng = {18749: [8.5, 47.3, 8.7, 47.2],  #Source: see Information to function
                        5897658.289: [-0.116773, 51.510357, -77.009003, 38.889931], #Source: https://community.esri.com/groups/coordinate-reference-systems/blog/2017/10/05/haversine-formula
                        3780627: [0.0, 4.0, 0.0, 38],  # Source for next lines: self-computation with formula from link above
                        2306879.363: [-7.345, -7.345, 7.345, 7.345],
                        13222121.519: [-0.118746, 73.998, -120.947783, -21.4783],
                        785767.221: [50, 0, 45, 5]}

        for haversine, latlng in input_latlng.items():
            haversine_output = haversine_dist(latlng[0], latlng[1], latlng[2], latlng[3])
            assert np.isclose(haversine_output, haversine, atol=0.1)

