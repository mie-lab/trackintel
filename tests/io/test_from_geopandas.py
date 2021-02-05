import os

import trackintel as ti
import geopandas as gpd
import pandas as pd

class TestFromGeopandas:
        def test_positionfixes_from_gpd(self):
        
            file = os.path.join('tests','data','positionfixes.geojson')
            csv_file = os.path.join('tests', 'data', 'positionfixes.csv')
            gdf = gpd.read_file(file)
            pfs = ti.io.from_geopandas.positionfixes_from_gpd(gdf, user_id='User', geom='geometry')
            pfs_csv = ti.read_positionfixes_csv(csv_file, sep=';')
            pfs_csv['tracked_at'] = pfs_csv['tracked_at'].apply(lambda d: d.isoformat().replace(' ', 'T')) #datetime format not yet implemented and chekced in model
            assert pfs.equals(pfs_csv)
            
        def test_triplegs_from_gpd(self):
            file=os.path.join('tests','data','triplegs.geojson')
            csv_file = os.path.join('tests','data','triplegs.csv')
            gdf = gpd.read_file(file)
            tpls = ti.io.from_geopandas.triplegs_from_gpd(gdf,user_id='User',geom='geometry')
            tpls_csv = ti.read_triplegs_csv(csv_file,sep=';')
            tpls_csv['started_at'] = tpls_csv['started_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))  #datetime format not yet implemented and chekced in model
            tpls_csv['finished_at'] = tpls_csv['finished_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
            
            assert tpls.equals(tpls_csv)
            
        def test_staypoints_from_gpd(self):
            file=os.path.join('tests','data','staypoints.geojson')
            csv_file = os.path.join('tests','data','staypoints.csv')
            gdf = gpd.read_file(file)
            stps = ti.io.from_geopandas.staypoints_from_gpd(gdf,'start_time','end_time', geom='geometry')
            stps_csv = ti.read_staypoints_csv(csv_file,sep=';')
            stps_csv['started_at'] = stps_csv['started_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))  #datetime format not yet implemented and chekced in model
            stps_csv['finished_at'] = stps_csv['finished_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
            
            assert stps.equals(stps_csv)
            
        def test_locations_from_gpd(self): #TODO: Problem with multiple geometry columns and geojson format
            file=os.path.join('tests','data','locations.geojson')
            csv_file = os.path.join('tests','data','locations.csv')
            gdf = gpd.read_file(file)
            plcs = ti.io.from_geopandas.locations_from_gpd(gdf,user_id='User',center='geometry')
            plcs_csv = ti.read_locations_csv(csv_file,sep=';')
            plcs_csv = plcs_csv.drop(columns='extent') #drop the second geometry column manually because not storable in GeoJSON (from Geopandas)
            assert plcs.equals(plcs_csv)
            
        def test_trips_from_gpd(self): 
            csv_file = os.path.join('tests','data','trips.csv')
            df = pd.read_csv(csv_file, sep=';')
            trps = ti.io.from_geopandas.trips_from_gpd(df)
            trps_csv = ti.read_trips_csv(csv_file,sep=';')
            trps_csv['started_at'] = trps_csv['started_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))  #datetime format not yet implemented and chekced in model
            trps_csv['finished_at'] = trps_csv['finished_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
            trps_csv['started_at'] = trps_csv['started_at'].apply(lambda d: d.replace('+00:00', 'Z'))
            trps_csv['finished_at'] = trps_csv['finished_at'].apply(lambda d: d.replace('+00:00', 'Z'))
            
            assert trps.equals(trps_csv)