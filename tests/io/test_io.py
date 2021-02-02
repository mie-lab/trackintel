import filecmp
import os

import trackintel as ti
import geopandas as gpd
import pandas as pd



class TestIO:
    def test_positionfixes_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'positionfixes.csv')
        tmp_file = os.path.join('tests', 'data', 'positionfixes_test.csv')
        pfs = ti.read_positionfixes_csv(orig_file, sep=';')
        pfs['tracked_at'] = pfs['tracked_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        pfs.as_positionfixes.to_csv(tmp_file, sep=';',
                                    columns=['user_id', 'tracked_at', 'latitude', 'longitude', 'elevation', 'accuracy'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
        
    def test_positionfixes_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass
    
    def test_positionfixes_from_gpd(self):
        
        file = os.path.join('tests','data','positionfixes.geojson')
        csv_file = os.path.join('tests', 'data', 'positionfixes.csv')
        gdf = gpd.read_file(file)
        pfs = ti.io.from_geopandas.positionfixes_from_gpd(gdf, user_id='User', geom='geometry')
        pfs_csv = ti.read_positionfixes_csv(csv_file, sep=';')
        pfs_csv['tracked_at'] = pfs_csv['tracked_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
        assert pfs.equals(pfs_csv)

    def test_triplegs_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'triplegs.csv')
        tmp_file = os.path.join('tests', 'data', 'triplegs_test.csv')
        tpls = ti.read_triplegs_csv(orig_file, sep=';')
        tpls['started_at'] = tpls['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        tpls['finished_at'] = tpls['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        tpls.as_triplegs.to_csv(tmp_file, sep=';',
                                columns=['user_id', 'started_at', 'finished_at', 'geom'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)


    def test_triplegs_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass
    
    def test_triplegs_from_gpd(self):
        file=os.path.join('tests','data','triplegs.geojson')
        csv_file = os.path.join('tests','data','triplegs.csv')
        gdf = gpd.read_file(file)
        tpls = ti.io.from_geopandas.triplegs_from_gpd(gdf,user_id='User',geom='geometry')
        tpls_csv = ti.read_triplegs_csv(csv_file,sep=';')
        tpls_csv['started_at'] = tpls_csv['started_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
        tpls_csv['finished_at'] = tpls_csv['finished_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
        
        assert tpls.equals(tpls_csv)

    def test_staypoints_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'staypoints.csv')
        tmp_file = os.path.join('tests', 'data', 'staypoints_test.csv')
        stps = ti.read_staypoints_csv(orig_file, sep=';')
        stps['started_at'] = stps['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        stps['finished_at'] = stps['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        stps.as_staypoints.to_csv(tmp_file, sep=';',
                                  columns=['user_id', 'started_at', 'finished_at', 'elevation', 'geom'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
        
    def test_staypoints_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass
    
    def test_staypoints_from_gpd(self):
        file=os.path.join('tests','data','staypoints.geojson')
        csv_file = os.path.join('tests','data','staypoints.csv')
        gdf = gpd.read_file(file)
        stps = ti.io.from_geopandas.staypoints_from_gpd(gdf,'start_time','end_time', geom='geometry')
        stps_csv = ti.read_staypoints_csv(csv_file,sep=';')
        stps_csv['started_at'] = stps_csv['started_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
        stps_csv['finished_at'] = stps_csv['finished_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
        
        assert stps.equals(stps_csv)

    def test_locations_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'locations.csv')
        tmp_file = os.path.join('tests', 'data', 'locations_test.csv')
        plcs = ti.read_locations_csv(orig_file, sep=';')
        plcs.as_locations.to_csv(tmp_file, sep=';',
                              columns=['user_id', 'elevation', 'center', 'extent'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
        
    def test_locations_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass

    def test_locations_from_gpd(self): #TODO: Problem with multiple geometry columns and geojson format
        file=os.path.join('tests','data','locations.geojson')
        csv_file = os.path.join('tests','data','locations.csv')
        gdf = gpd.read_file(file)
        plcs = ti.io.from_geopandas.locations_from_gpd(gdf,user_id='User',center='geometry')
        plcs_csv = ti.read_locations_csv(csv_file,sep=';')
        plcs_csv = plcs_csv.drop(columns='extent') #drop the second geometry column manually because not storable in GeoJSON (from Geopandas)
        assert plcs.equals(plcs_csv)

    def test_trips_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'trips.csv')
        tmp_file = os.path.join('tests', 'data', 'trips_test.csv')
        tpls = ti.read_trips_csv(orig_file, sep=';')
        tpls['started_at'] = tpls['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        tpls['finished_at'] = tpls['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        tpls.as_trips.to_csv(tmp_file, sep=';',
                             columns=['user_id', 'started_at', 'finished_at', 'origin_staypoint_id',
                                      'destination_staypoint_id'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
        
    def test_trips_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass

    def test_trips_from_gpd(self): 
        csv_file = os.path.join('tests','data','trips.csv')
        df = pd.read_csv(csv_file, sep=';')
        trps = ti.io.from_geopandas.trips_from_gpd(df)
        trps_csv = ti.read_trips_csv(csv_file,sep=';')
        trps_csv['started_at'] = trps_csv['started_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
        trps_csv['finished_at'] = trps_csv['finished_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
        trps_csv['started_at'] = trps_csv['started_at'].apply(lambda d: d.replace('+00:00', 'Z'))
        trps_csv['finished_at'] = trps_csv['finished_at'].apply(lambda d: d.replace('+00:00', 'Z'))
        
        assert trps.equals(trps_csv)
