import filecmp
import os
import pytest

import trackintel as ti


class TestFile:
    def test_positionfixes_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'positionfixes.csv')
        mod_file = os.path.join('tests','data', 'positionfixes_mod_columns.csv')
        tmp_file = os.path.join('tests', 'data', 'positionfixes_test.csv')

        pfs = ti.read_positionfixes_csv(orig_file, sep=';', index_col="id")
        
        column_mapping = {'lat':'latitude', 'lon':'longitude', 'time':'tracked_at'}
        mod_pfs = ti.read_positionfixes_csv(mod_file, sep=';', index_col="id", columns=column_mapping)
        assert mod_pfs.equals(pfs)
        pfs['tracked_at'] = pfs['tracked_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        
        columns = ['user_id', 'tracked_at', 'latitude', 'longitude', 'elevation', 'accuracy']
        pfs.as_positionfixes.to_csv(tmp_file, sep=';', columns=columns)
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
        
    def test_positionfixes_csv_index_warning(self):
        """Test if a warning is raised when not parsing the index_col arguement."""
        file = os.path.join('tests', 'data', 'positionfixes.csv')
        with pytest.warns(UserWarning):
            ti.read_positionfixes_csv(file, sep=';')
        
    def test_positionfixes_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass
    


    def test_triplegs_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'triplegs.csv')
        mod_file = os.path.join('tests','data','triplegs_mod_columns.csv')
        tmp_file = os.path.join('tests', 'data', 'triplegs_test.csv')
        tpls = ti.read_triplegs_csv(orig_file, sep=';', tz='utc')
        
        column_mapping = {'start_time': 'started_at', 'end_time': 'finished_at', 'tripleg': 'geom'}
        mod_tpls = ti.read_triplegs_csv(mod_file, sep=';', columns=column_mapping)
        
        assert mod_tpls.equals(tpls)
        tpls['started_at'] = tpls['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        tpls['finished_at'] = tpls['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        
        columns = ['user_id', 'started_at', 'finished_at', 'geom']
        tpls.as_triplegs.to_csv(tmp_file, sep=';', columns=columns)
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)


    def test_triplegs_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass
    
    

    def test_staypoints_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'staypoints.csv')
        mod_file = os.path.join('tests', 'data', 'staypoints_mod_columns.csv')
        tmp_file = os.path.join('tests', 'data', 'staypoints_test.csv')
        stps = ti.read_staypoints_csv(orig_file, sep=';', tz='utc', index_col="id")
        mod_stps = ti.read_staypoints_csv(mod_file, columns={'User': 'user_id'}, sep=';', index_col="id")
        assert mod_stps.equals(stps)
        stps['started_at'] = stps['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        stps['finished_at'] = stps['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        
        columns = ['user_id', 'started_at', 'finished_at', 'elevation', 'geom']
        stps.as_staypoints.to_csv(tmp_file, sep=';', columns=columns)
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
    
    def test_staypoints_csv_index_warning(self):
        """Test if a warning is raised when not parsing the index_col arguement."""
        file = os.path.join('tests', 'data', 'staypoints.csv')
        with pytest.warns(UserWarning):
            ti.read_staypoints_csv(file, sep=';')
            
    def test_staypoints_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass
    

    def test_locations_from_to_csv(self):
        orig_file = os.path.join('tests', 'data', 'locations.csv')
        mod_file = os.path.join('tests','data','locations_mod_columns.csv')
        tmp_file = os.path.join('tests', 'data', 'locations_test.csv')
        mod_plcs = ti.read_locations_csv(mod_file, columns={'geom':'center'},sep=';')
        locs = ti.read_locations_csv(orig_file, sep=';')
        assert mod_plcs.equals(locs)
        locs.as_locations.to_csv(tmp_file, sep=';',
                              columns=['user_id', 'elevation', 'center', 'extent'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
        
        
    def test_locations_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass

    

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
    
    
    def test_tours_from_to_csv(self):
        # TODO Implement some tests for reading and writing tours.
        pass
    
