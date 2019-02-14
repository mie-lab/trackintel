# This is not needed if the trackintel library is installed. ==================
import sys
sys.path.append("..")
sys.path.append("../trackintel")
# =============================================================================

import logging

import matplotlib.pyplot as plt
import trackintel as ti

from trackintel.geogr.distances import meters_to_decimal_degrees


logging.basicConfig(filename='log/preprocessing.log', level=logging.INFO, filemode='w')

positionfixes = ti.read_positionfixes_csv('data/gpsies_trajectory.csv', sep=';')
staypoints = ti.extract_staypoints(positionfixes, method='sliding', dist_threshold=100, time_threshold=5*60)

ti.plot_staypoints(staypoints, out_filename='out/gpsies_trajectory_staypoints.png',
                   radius=meters_to_decimal_degrees(100, 47.5), positionfixes=positionfixes, plot_osm=True)
