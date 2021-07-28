# This is not needed if the trackintel library is installed. ==================
import sys

sys.path.append("..")
sys.path.append("../trackintel")
# =============================================================================

import logging

import matplotlib.pyplot as plt
import trackintel as ti


logging.basicConfig(filename="examples/log/import_export_postgis.log", level=logging.INFO, filemode="w")

conn_string = "postgresql://test:1234@localhost:5432/trackintel-tests"

# Geolife trajectory to PostGIS.
pfs = ti.read_positionfixes_csv("examples/data/geolife_trajectory.csv", sep=";")
# pfs.as_positionfixes.to_postgis('positionfixes', conn_string)

# Geolife trajectory from PostGIS.
# pfs = ti.io.read_positionfixes_postgis('positionfixes', conn_string)
pfs.as_positionfixes.plot()
