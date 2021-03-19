import os

import geopandas as gpd
import pandas as pd

import trackintel as ti


class TestFromGeopandas:
    def test_read_positionfixes_gpd(self):
        gdf = gpd.read_file(os.path.join('tests', 'data', 'positionfixes.geojson'))
        gdf.set_index('id', inplace=True)
        pfs_from_gpd = ti.io.from_geopandas.read_positionfixes_gpd(gdf, user_id='User', geom='geometry', tz='utc')
        
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs_from_csv = ti.read_positionfixes_csv(pfs_file, sep=';', tz='utc', index_col='id')

        pd.testing.assert_frame_equal(pfs_from_gpd, pfs_from_csv, check_exact=False)

    def test_triplegs_from_gpd(self):
        gdf = gpd.read_file(os.path.join('tests', 'data', 'triplegs.geojson'))
        gdf.set_index('id', inplace=True)
        tpls_from_gpd = ti.io.from_geopandas.triplegs_from_gpd(gdf, user_id='User', geom='geometry', tz='utc')
        
        tpls_file = os.path.join('tests', 'data', 'triplegs.csv')
        tpls_from_csv = ti.read_triplegs_csv(tpls_file, sep=';', tz='utc', index_col='id')

        pd.testing.assert_frame_equal(tpls_from_gpd, tpls_from_csv, check_exact=False)

    def test_staypoints_from_gpd(self):
        gdf = gpd.read_file(os.path.join('tests', 'data', 'staypoints.geojson'))
        gdf.set_index('id', inplace=True)
        stps_from_gpd = ti.io.from_geopandas.staypoints_from_gpd(gdf, 'start_time', 'end_time', geom='geometry',
                                                                 tz='utc')
        
        stps_file = os.path.join('tests', 'data', 'staypoints.csv')
        stps_from_csv = ti.read_staypoints_csv(stps_file, sep=';', tz='utc', index_col='id')

        pd.testing.assert_frame_equal(stps_from_gpd, stps_from_csv, check_exact=False)

    def test_locations_from_gpd(self):
        # TODO: Problem with multiple geometry columns and geojson format
        gdf = gpd.read_file(os.path.join('tests', 'data', 'locations.geojson'))
        gdf.set_index('id', inplace=True)
        locs_from_gpd = ti.io.from_geopandas.locations_from_gpd(gdf, user_id='User', center='geometry')
        
        locs_file = os.path.join('tests', 'data', 'locations.csv')
        locs_from_csv = ti.read_locations_csv(locs_file, sep=';', index_col='id')

        # drop the second geometry column manually because not storable in GeoJSON (from Geopandas)
        locs_from_csv = locs_from_csv.drop(columns='extent')
        pd.testing.assert_frame_equal(locs_from_csv, locs_from_gpd, check_exact=False)

    def test_trips_from_gpd(self):
        df = pd.read_csv(os.path.join('tests', 'data', 'trips.csv'), sep=';')
        df.set_index('id', inplace=True)
        trips_from_gpd = ti.io.from_geopandas.trips_from_gpd(df, tz='utc')
        
        trips_file = os.path.join('tests', 'data', 'trips.csv')
        trips_from_csv = ti.read_trips_csv(trips_file, sep=';', tz='utc', index_col='id')

        pd.testing.assert_frame_equal(trips_from_gpd, trips_from_csv, check_exact=False)

    def test_tours_from_gpd(self):
        # TODO: implement tests for reading tours from Geopandas
        pass
