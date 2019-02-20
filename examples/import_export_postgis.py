# This is not needed if the trackintel library is installed. ==================
import sys
sys.path.append("..")
sys.path.append("../trackintel")
# =============================================================================

import logging

import matplotlib.pyplot as plt
import trackintel as ti


logging.basicConfig(filename='log/import_export_postgis.log', level=logging.INFO, filemode='w')

conn_string = 'postgresql://test:1234@localhost:5432/trackintel-tests'

# Geolife trajectory to PostGIS.
pfs = ti.read_positionfixes_csv('data/geolife_trajectory.csv', sep=';')
pfs.as_positionfixes.to_postgis(conn_string, 'positionfixes')

# Geolife trajectory from PostGIS.
pfs = ti.read_positionfixes_postgis(conn_string, 'positionfixes')
pfs.as_positionfixes.plot()
