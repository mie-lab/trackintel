import filecmp
import os
import trackintel as ti
import geopandas as gpd


class TestIO:
    def test_positionfixes_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'positionfixes.csv')
        mod_file = os.path.join('tests','data', 'positionfixes_mod_columns.csv')
        tmp_file = os.path.join('tests', 'data', 'positionfixes_test.csv')
        pfs = ti.read_positionfixes_csv(orig_file, sep=';')
        mod_pfs = ti.read_positionfixes_csv(mod_file, columns={'lat':'latitude', 'lon':'longitude', 'time':'tracked_at'},sep=';')
        assert mod_pfs.equals(pfs)
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
        pfs = ti.io.geopandas.read_positionfixes_gpd(gdf, user_id='User',geom='geometry')
        pfs_csv = ti.read_positionfixes_csv(csv_file, sep=';')
        pfs_csv['tracked_at'] = pfs_csv['tracked_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        assert pfs.equals(pfs_csv)
        

    def test_triplegs_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'triplegs.csv')
        mod_file = os.path.join('tests','data','triplegs_mod_columns.csv')
        tmp_file = os.path.join('tests', 'data', 'triplegs_test.csv')
        tpls = ti.read_triplegs_csv(orig_file, sep=';')
        mod_tpls = ti.read_triplegs_csv(mod_file, columns={'start_time':'started_at','end_time':'finished_at', 'tripleg':'geom'},sep=';')
        assert mod_tpls.equals(tpls)
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
        tpls = ti.io.geopandas.read_triplegs_gpd(gdf,user_id='User')
        tpls_csv = ti.read_triplegs_csv(csv_file,sep=';')
        tpls_csv['started_at'] = tpls_csv['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        tpls_csv['finished_at'] = tpls_csv['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        
        assert tpls.equals(tpls_csv)
        

    def test_staypoints_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'staypoints.csv')
        mod_file = os.path.join('tests', 'data', 'staypoints_mod_columns.csv')
        tmp_file = os.path.join('tests', 'data', 'staypoints_test.csv')
        stps = ti.read_staypoints_csv(orig_file, sep=';')
        mod_stps = ti.read_staypoints_csv(mod_file, columns={'User':'user_id'},sep=';')
        assert mod_stps.equals(stps)
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
        stps = ti.io.geopandas.read_triplegs_gpd(gdf,'start_time','end_time')
        stps_csv = ti.read_triplegs_csv(csv_file,sep=';')
        stps_csv['started_at'] = stps_csv['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        stps_csv['finished_at'] = stps_csv['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        
        assert stps.equals(stps_csv)

    def test_locations_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'locations.csv')
        mod_file = os.path.join('tests','data','locations_mod_columns.csv')
        tmp_file = os.path.join('tests', 'data', 'locations_test.csv')
        mod_plcs = ti.read_locations_csv(mod_file, columns={'geom':'center'},sep=';')
        plcs = ti.read_locations_csv(orig_file, sep=';')
        assert mod_plcs.equals(plcs)
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
        plcs = ti.io.geopandas.read_locations_gpd(gdf,user_id='User')
        plcs_csv = ti.read_locations_csv(csv_file,sep=';')
        plcs_csv['started_at'] = plcs_csv['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        plcs_csv['finished_at'] = plcs_csv['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        
        assert plcs.equals(plcs_csv)

    def test_trips_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'trips.csv')
        mod_file = os.path.join('tests', 'data', 'trips_mod_columns.csv')
        tmp_file = os.path.join('tests', 'data', 'trips_test.csv')
        tpls = ti.read_trips_csv(orig_file, sep=';')
        mod_tpls = ti.read_trips_csv(mod_file, columns= {'orig_stp':'origin_staypoint_id','dest_stp':'destination_staypoint_id'},sep=';')
        assert mod_tpls.equals(tpls)
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


    def test_trips_from_gpd(self): #TODO: empty geometry in trips.csv
        file=os.path.join('tests','data','trips.geojson')
        csv_file = os.path.join('tests','data','trips.csv')
        gdf = gpd.read_file(file)
        trps = ti.io.geopandas.read_trips_gpd(gdf,user_id='User')
        trps_csv = ti.read_triplegs_csv(csv_file,sep=';')
        trps_csv['started_at'] = trps_csv['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        trps_csv['finished_at'] = trps_csv['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        
        assert trps.equals(trps_csv)
