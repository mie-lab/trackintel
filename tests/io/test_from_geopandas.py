import os

import trackintel as ti
import geopandas as gpd
import pandas as pd

class TestFromGeopandas:
        def test_positionfixes_from_gpd(self):
        
            gdf = gpd.read_file(os.path.join('tests','data','positionfixes.geojson'))
            pfs_from_gpd = ti.io.from_geopandas.positionfixes_from_gpd(gdf, user_id='User', geom='geometry')
            pfs_from_csv = ti.read_positionfixes_csv(os.path.join('tests', 'data', 'positionfixes.csv'), sep=';')
            
            # TODO: datetime format not yet implemented and checked in model
            pfs_from_csv['tracked_at'] = pfs_from_csv['tracked_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
            pfs_from_csv['tracked_at'] = pfs_from_csv['tracked_at'].apply(lambda d: d.replace('+00:00', ''))
            
            assert pfs_from_gpd.equals(pfs_from_csv)
            
        def test_triplegs_from_gpd(self):
            gdf = gpd.read_file(os.path.join('tests','data','triplegs.geojson'))
            tpls_from_gpd = ti.io.from_geopandas.triplegs_from_gpd(gdf,user_id='User',geom='geometry')
            tpls_from_csv = ti.read_triplegs_csv(os.path.join('tests','data','triplegs.csv'),sep=';')
            
            # TODO: datetime format not yet implemented and checked in model
            tpls_from_csv['started_at'] = tpls_from_csv['started_at'].apply(lambda d: d.isoformat().replace(' ', 'T')) 
            tpls_from_csv['finished_at'] = tpls_from_csv['finished_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
            tpls_from_csv['started_at'] = tpls_from_csv['started_at'].apply(lambda d: d.replace('+00:00', ''))
            tpls_from_csv['finished_at'] = tpls_from_csv['finished_at'].apply(lambda d: d.replace('+00:00', ''))
            
            assert tpls_from_gpd.equals(tpls_from_csv)
            
        def test_staypoints_from_gpd(self):
            gdf = gpd.read_file(os.path.join('tests','data','staypoints.geojson'))
            stps_from_gpd = ti.io.from_geopandas.staypoints_from_gpd(gdf,'start_time','end_time', geom='geometry')
            stps_from_csv = ti.read_staypoints_csv(os.path.join('tests','data','staypoints.csv'),sep=';')
            
            # TODO: datetime format not yet implemented and checked in model
            stps_from_csv['started_at'] = stps_from_csv['started_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))  
            stps_from_csv['finished_at'] = stps_from_csv['finished_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
            stps_from_csv['started_at'] = stps_from_csv['started_at'].apply(lambda d: d.replace('+00:00', ''))
            stps_from_csv['finished_at'] = stps_from_csv['finished_at'].apply(lambda d: d.replace('+00:00', ''))
            
            assert stps_from_gpd.equals(stps_from_csv)
            
        def test_locations_from_gpd(self): 
            #TODO: Problem with multiple geometry columns and geojson format
            gdf = gpd.read_file(os.path.join('tests','data','locations.geojson'))
            plcs_from_gpd = ti.io.from_geopandas.locations_from_gpd(gdf,user_id='User',center='geometry')
            plcs_from_csv = ti.read_locations_csv(os.path.join('tests','data','locations.csv'),sep=';')
            
            # drop the second geometry column manually because not storable in GeoJSON (from Geopandas)
            plcs_from_csv = plcs_from_csv.drop(columns='extent') 
            assert plcs_from_gpd.equals(plcs_from_csv)
            
        def test_trips_from_gpd(self): 
            df = pd.read_csv(os.path.join('tests','data','trips.csv'), sep=';')
            trips_from_gpd = ti.io.from_geopandas.trips_from_gpd(df)
            trips_from_csv = ti.read_trips_csv(os.path.join('tests','data','trips.csv'),sep=';')
            
            # TODO: datetime format not yet implemented and checked in model
            trips_from_csv['started_at'] = trips_from_csv['started_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))  
            trips_from_csv['finished_at'] = trips_from_csv['finished_at'].apply(lambda d: d.isoformat().replace(' ', 'T'))
            trips_from_csv['started_at'] = trips_from_csv['started_at'].apply(lambda d: d.replace('+00:00', 'Z'))
            trips_from_csv['finished_at'] = trips_from_csv['finished_at'].apply(lambda d: d.replace('+00:00', 'Z'))
            
            assert trips_from_gpd.equals(trips_from_csv)
            
        def test_tours_from_gpd(self):
            # TODO: implement tests for reading tours from Geopandas
            pass