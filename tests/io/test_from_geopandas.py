import os

import geopandas as gpd
import pandas as pd

import trackintel as ti


class TestFromGeopandas:
    def test_positionfixes_from_gpd(self):
        gdf = gpd.read_file(os.path.join('tests', 'data', 'positionfixes.geojson'))
        pfs_from_gpd = ti.io.from_geopandas.positionfixes_from_gpd(gdf, user_id='User', geom='geometry', tz='utc')
        pfs_from_csv = ti.read_positionfixes_csv(os.path.join('tests', 'data', 'positionfixes.csv'), sep=';')

        pd.testing.assert_frame_equal(pfs_from_gpd, pfs_from_csv, check_exact=False)

    def test_triplegs_from_gpd(self):
        gdf = gpd.read_file(os.path.join('tests', 'data', 'triplegs.geojson'))
        tpls_from_gpd = ti.io.from_geopandas.triplegs_from_gpd(gdf, user_id='User', geom='geometry', tz='utc')
        tpls_from_csv = ti.read_triplegs_csv(os.path.join('tests', 'data', 'triplegs.csv'), sep=';')

        pd.testing.assert_frame_equal(tpls_from_gpd, tpls_from_csv, check_exact=False)

    def test_staypoints_from_gpd(self):
        gdf = gpd.read_file(os.path.join('tests', 'data', 'staypoints.geojson'))
        stps_from_gpd = ti.io.from_geopandas.staypoints_from_gpd(gdf, 'start_time', 'end_time', geom='geometry',
                                                                 tz='utc')
        stps_from_csv = ti.read_staypoints_csv(os.path.join('tests', 'data', 'staypoints.csv'), sep=';')

        pd.testing.assert_frame_equal(stps_from_gpd, stps_from_csv, check_exact=False)

    def test_locations_from_gpd(self):
        # TODO: Problem with multiple geometry columns and geojson format
        gdf = gpd.read_file(os.path.join('tests', 'data', 'locations.geojson'))
        plcs_from_gpd = ti.io.from_geopandas.locations_from_gpd(gdf, user_id='User', center='geometry')
        plcs_from_csv = ti.read_locations_csv(os.path.join('tests', 'data', 'locations.csv'), sep=';')

        # drop the second geometry column manually because not storable in GeoJSON (from Geopandas)
        plcs_from_csv = plcs_from_csv.drop(columns='extent')
        pd.testing.assert_frame_equal(plcs_from_csv, plcs_from_gpd, check_exact=False)

    def test_trips_from_gpd(self):
        df = pd.read_csv(os.path.join('tests', 'data', 'trips.csv'), sep=';')
        trips_from_gpd = ti.io.from_geopandas.trips_from_gpd(df, tz='utc')
        trips_from_csv = ti.read_trips_csv(os.path.join('tests', 'data', 'trips.csv'), sep=';')

        pd.testing.assert_frame_equal(trips_from_gpd, trips_from_csv, check_exact=False)

    def test_tours_from_gpd(self):
        # TODO: implement tests for reading tours from Geopandas
        pass
