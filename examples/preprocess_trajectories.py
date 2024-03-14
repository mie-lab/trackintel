# This is not needed if the trackintel library is installed. ==================
import sys

sys.path.append("..")
sys.path.append("../trackintel")
# =============================================================================

import logging

import trackintel as ti

logging.basicConfig(filename="examples/log/preprocessing.log", level=logging.INFO, filemode="w")

# GPS trajectory.
pfs = ti.read_positionfixes_csv("examples/data/geolife_trajectory.csv", sep=";", crs="EPSG:4326", index_col=None)
ti.plot(filename="examples/out/gps_trajectory_positionfixes.png", positionfixes=pfs, plot_osm=True)

pfs, sp = pfs.generate_staypoints(method="sliding", dist_threshold=100, time_threshold=5)
ti.plot(
    filename="examples/out/gps_trajectory_staypoints.png",
    staypoints=sp,
    radius_sp=100,
    positionfixes=pfs,
    plot_osm=True,
)

_, locs = sp.generate_locations(method="dbscan", epsilon=100, num_samples=3)
ti.plot(
    filename="examples/out/gps_trajectory_locations.png",
    locations=locs,
    radius_locs=120,
    positionfixes=pfs,
    staypoints=sp,
    radius_sp=100,
    plot_osm=True,
)

_, tpls = pfs.generate_triplegs(staypoints=sp)
ti.plot(
    filename="examples/out/gpsies_trajectory_triplegs.png", triplegs=tpls, staypoints=sp, radius_sp=100, plot_osm=True
)

# Geolife trajectory.
pfs = ti.read_positionfixes_csv("examples/data/geolife_trajectory.csv", sep=";", crs="EPSG:4326", index_col=None)
ti.plot(filename="examples/out/geolife_trajectory_positionfixes.png", positionfixes=pfs)

_, sp = pfs.generate_staypoints(method="sliding", dist_threshold=100, time_threshold=10)
ti.plot(
    filename="examples/out/geolife_trajectory_staypoints.png",
    staypoints=sp,
    radius_sp=100,
    positionfixes=pfs,
    plot_osm=True,
)

# Google trajectory.
pfs = ti.read_positionfixes_csv("examples/data/google_trajectory.csv", sep=";", crs="EPSG:4326", index_col=None)
_, sp = pfs.generate_staypoints(method="sliding", dist_threshold=75, time_threshold=10)
ti.plot(
    filename="examples/out/google_trajectory_staypoints.png",
    staypoints=sp,
    radius_sp=75,
    positionfixes=pfs,
    plot_osm=True,
)

# Posmo trajectory.
pfs = ti.read_positionfixes_csv(
    "examples/data/posmo_trajectory.csv", sep=";", crs="EPSG:4326", index_col=None, tz="UTC"
)
_, sp = pfs.generate_staypoints(method="sliding", dist_threshold=50, time_threshold=1)
ti.plot(
    filename="examples/out/posmo_trajectory_staypoints.png",
    staypoints=sp,
    radius_sp=50,
    positionfixes=pfs,
    plot_osm=False,
)
