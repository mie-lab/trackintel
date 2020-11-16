import pytest
import sys
import os
import filecmp

import pandas as pd
import trackintel as ti
import numpy as np
from trackintel.geogr import distances
from trackintel.geogr.distances import haversine_dist
from trackintel.geogr.distances import meters_to_decimal_degrees


class TestHaversineDist:
    def test_haversine_dist(self):
        # input_latlng saves different combinations of haversine-distances in meters and the longitude & latitudes from
        # two different points in WGS84
        # {haversine-distance in meters[longitude_P1, latitudes_P1, longitude_P2, latitudes_P2]}
        input_latlng = {18749: [8.5, 47.3, 8.7, 47.2],  # Source: see Information to function
                        5897658.289: [-0.116773, 51.510357, -77.009003, 38.889931],
                        # Source: https://community.esri.com/groups/coordinate-reference-systems/blog/2017/10/05/haversine-formula
                        3780627: [0.0, 4.0, 0.0, 38],
                        # Source for next lines: self-computation with formula from link above
                        2306879.363: [-7.345, -7.345, 7.345, 7.345],
                        13222121.519: [-0.118746, 73.998, -120.947783, -21.4783],
                        785767.221: [50, 0, 45, 5]}

        for haversine, latlng in input_latlng.items():
            haversine_output = haversine_dist(latlng[0], latlng[1], latlng[2], latlng[3])
            assert np.isclose(haversine_output, haversine, atol=0.1)



class TestMetersToDecimalDegrees:
    def test_meters_to_decimal_degrees(self):
        input_result_dict = {1.0: {0: 111320, 23: 102470, 45: 78710, 67: 43496},
                             0.1: {0: 11132, 23: 10247, 45: 7871, 67: 4349.6},
                             0.01: {0: 1113.2, 23: 1024.7, 45: 787.1, 67: 434.96},
                             0.001: {0: 111.32, 23: 102.47, 45: 78.71, 67: 43.496}}

        for degree, lat_output in input_result_dict.items():
            for lat, meters in lat_output.items():

                decimal_degree_output = meters_to_decimal_degrees(meters, lat)
                assert np.isclose(decimal_degree_output, degree, atol=0.1)


