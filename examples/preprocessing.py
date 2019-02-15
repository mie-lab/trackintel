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

# GPSies trajectory.
positionfixes = ti.read_positionfixes_csv('data/gpsies_trajectory.csv', sep=';')
staypoints = ti.extract_staypoints(positionfixes, method='sliding', dist_threshold=100, time_threshold=5*60)

ti.plot_staypoints(staypoints, out_filename='out/gpsies_trajectory_staypoints.png',
                   radius=meters_to_decimal_degrees(100, 47.5), positionfixes=positionfixes, plot_osm=True)

# Geolife trajectory.
positionfixes = ti.read_positionfixes_csv('data/geolife_trajectory.csv', sep=';')
staypoints = ti.extract_staypoints(positionfixes, method='sliding', dist_threshold=100, time_threshold=10*60)
ti.plot_staypoints(staypoints, out_filename='out/geolife_trajectory_staypoints.png',
                   radius=meters_to_decimal_degrees(100, 47.5), positionfixes=positionfixes, plot_osm=True)

# Google trajectory.
positionfixes = ti.read_positionfixes_csv('data/google_trajectory.csv', sep=';')
staypoints = ti.extract_staypoints(positionfixes, method='sliding', dist_threshold=75, time_threshold=10*60)
ti.plot_staypoints(staypoints, out_filename='out/google_trajectory_staypoints.png',
                   radius=meters_to_decimal_degrees(75, 47.5), positionfixes=positionfixes, plot_osm=True)

# Posmo trajectory.
positionfixes = ti.read_positionfixes_csv('data/posmo_trajectory.csv', sep=';')
staypoints = ti.extract_staypoints(positionfixes, method='sliding', dist_threshold=50, time_threshold=1*60)
ti.plot_staypoints(staypoints, out_filename='out/posmo_trajectory_staypoints.png',
                   radius=meters_to_decimal_degrees(50, 47.5), positionfixes=positionfixes, plot_osm=False)
