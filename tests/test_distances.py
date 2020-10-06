import pytest
import numpy as np
from trackintel.geogr.distances import  meters_to_decimal_degrees

class Test_meters_to_decimal_degrees:
    def test_sample_values(self):
        input_resupt_dict = {1.0: {0: 111320, 23: 102470, 45: 78710, 67: 43496},
                                    0.1: {0: 11132, 23: 10247, 45: 7871, 67: 4349.6},
                                    0.01: {0: 1113.2, 23: 1024.7, 45: 787.1, 67: 434.96},
                                    0.001: {0: 111.32, 23: 102.47, 45: 78.71, 67: 43.496}}

        for degree, lat_output in input_resupt_dict.items():
            for lat, meters in lat_output.items():

                decimal_degree_output = meters_to_decimal_degrees(meters, lat)
                assert np.isclose(decimal_degree_output, degree, atol=0.1)
