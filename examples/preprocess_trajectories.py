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
pfs = ti.read_positionfixes_csv('data/geolife_trajectory.csv', sep=';')
# pfs.as_positionfixes.plot(out_filename='out/gpsies_trajectory_positionfixes.png', plot_osm=True)

spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=100, time_threshold=5*60)
# spts.as_staypoints.plot(out_filename='out/gpsies_trajectory_staypoints.png',
#                         radius=meters_to_decimal_degrees(100, 47.5), positionfixes=pfs, plot_osm=True)

plcs = spts.as_staypoints.extract_places(method='dbscan', epsilon=meters_to_decimal_degrees(120, 47.5), 
                                         num_samples=3)
# plcs.as_places.plot(out_filename='out/gpsies_trajectory_places.png', radius=meters_to_decimal_degrees(120, 47.5), 
#                     positionfixes=pfs, staypoints=spts, staypoints_radius=meters_to_decimal_degrees(100, 47.5), 
#                     plot_osm=True)

tpls = pfs.as_positionfixes.extract_triplegs()
tpls.as_triplegs.plot(staypoints=spts, staypoints_radius=meters_to_decimal_degrees(100, 47.5))
sys.exit(0)

# Geolife trajectory.
pfs = ti.read_positionfixes_csv('data/geolife_trajectory.csv', sep=';')
pfs.as_positionfixes.plot(out_filename='out/geolife_trajectory_positionfixes.png', plot_osm=False)

spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=100, time_threshold=10*60)
spts.as_staypoints.plot(out_filename='out/geolife_trajectory_staypoints.png',
                        radius=meters_to_decimal_degrees(100, 47.5), positionfixes=pfs, plot_osm=True)

# Google trajectory.
positionfixes = ti.read_positionfixes_csv('data/google_trajectory.csv', sep=';')
spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=75, time_threshold=10*60)
spts.as_staypoints.plot(out_filename='out/google_trajectory_staypoints.png',
                        radius=meters_to_decimal_degrees(75, 47.5), positionfixes=pfs, plot_osm=True)

# Posmo trajectory.
positionfixes = ti.read_positionfixes_csv('data/posmo_trajectory.csv', sep=';')
spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=50, time_threshold=1*60)
spts.as_staypoints.plot(out_filename='out/posmo_trajectory_staypoints.png',
                        radius=meters_to_decimal_degrees(50, 47.5), positionfixes=pfs, plot_osm=False)
