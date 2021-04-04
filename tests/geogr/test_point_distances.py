import os
from math import radians

import numpy as np
from sklearn.metrics.pairwise import haversine_distances

import trackintel as ti
from trackintel.geogr.distances import haversine_dist


class TestHaversineDist:
    def test_haversine_dist(self):
        """
        input_latlng saves different combinations of haversine-distances in meters and the longitude & latitudes from
        two different points in WGS84

        References
        ------
        https://community.esri.com/groups/coordinate-reference-systems/blog/2017/10/05/haversine-formula
        """

        # {haversine-distance in meters[longitude_P1, latitudes_P1, longitude_P2, latitudes_P2]}
        input_latlng = {
            18749: [8.5, 47.3, 8.7, 47.2],  # Source: see Information to function
            5897658.289: [-0.116773, 51.510357, -77.009003, 38.889931],
            3780627: [0.0, 4.0, 0.0, 38],
            # Source for next lines: self-computation with formula from link above
            2306879.363: [-7.345, -7.345, 7.345, 7.345],
            13222121.519: [-0.118746, 73.998, -120.947783, -21.4783],
            785767.221: [50, 0, 45, 5],
        }

        for haversine, latlng in input_latlng.items():
            haversine_output = haversine_dist(latlng[0], latlng[1], latlng[2], latlng[3])
            assert np.isclose(haversine_output, haversine, atol=0.1)

    def test_haversine_vectorized(self):
        stps_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        stps = ti.read_staypoints_csv(stps_file, tz="utc", index_col="id")
        x = stps.geometry.x.values
        y = stps.geometry.y.values

        n = len(x)
        # our distance
        ix_1, ix_2 = np.triu_indices(n, k=1)

        x1 = x[ix_1]
        y1 = y[ix_1]
        x2 = x[ix_2]
        y2 = y[ix_2]

        d_ours = haversine_dist(x1, y1, x2, y2)

        # their distance
        x_rad = np.asarray([radians(_) for _ in x])
        y_rad = np.asarray([radians(_) for _ in y])
        yx = np.concatenate((y_rad.reshape(-1, 1), x_rad.reshape(-1, 1)), axis=1)

        D_theirs = haversine_distances(yx, yx) * 6371000
        d_theirs = D_theirs[ix_1, ix_2]
        assert np.sum(np.abs(d_ours - d_theirs)) < 0.01  #  1cm for 58 should be good enough

    def test_example_from_sklean(self):

        bsas = [-34.83333, -58.5166646]
        paris = [49.0083899664, 2.53844117956]
        bsas_in_radians = [radians(_) for _ in bsas]
        paris_in_radians = [radians(_) for _ in paris]
        d_theirs = haversine_distances([bsas_in_radians, paris_in_radians]) * 6371000

        d_ours = haversine_dist(bsas[1], bsas[0], paris[1], paris[0])

        assert np.abs(d_theirs[1][0] - d_ours) < 0.01
