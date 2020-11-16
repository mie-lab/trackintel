import filecmp
import os

import trackintel as ti


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

