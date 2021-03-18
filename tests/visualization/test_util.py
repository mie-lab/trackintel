import os
import pytest
from geopandas.testing import assert_geodataframe_equal

import trackintel as ti


class TestTransform_gdf_to_wgs84:
    def test_transformation(self):
        """Check if data gets transformed."""

        file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(file, sep=';', crs='EPSG:4326', index_col=None)
        pfs_2056 = pfs.to_crs("EPSG:2056")
        pfs_4326 = ti.visualization.util.transform_gdf_to_wgs84(pfs_2056)
        assert_geodataframe_equal(pfs, pfs_4326, check_less_precise=True)

    def test_crs_warning(self):
        """Check if warning is raised for data without crs."""

        file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(file, sep=';', crs=None, index_col=None)
        with pytest.warns(UserWarning):
            ti.visualization.util.transform_gdf_to_wgs84(pfs)
