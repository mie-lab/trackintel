import os
import pytest
from geopandas.testing import assert_geodataframe_equal
import numpy as np

import trackintel as ti
from trackintel.visualization.util import a4_figsize


class TestTransform_gdf_to_wgs84:
    """Tests for transform_gdf_to_wgs84() method."""

    def test_transformation(self):
        """Check if data gets transformed."""
        file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(file, sep=";", crs="EPSG:4326", index_col=None)
        pfs_2056 = pfs.to_crs("EPSG:2056")
        pfs_4326 = ti.visualization.util.transform_gdf_to_wgs84(pfs_2056)
        assert_geodataframe_equal(pfs, pfs_4326, check_less_precise=True)

    def test_crs_warning(self):
        """Check if warning is raised for data without crs."""
        file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(file, sep=";", crs=None, index_col=None)
        with pytest.warns(UserWarning):
            ti.visualization.util.transform_gdf_to_wgs84(pfs)


class TestA4_figsize:
    """Tests for a4_figsize() method."""

    def test_parameter(self, caplog):
        """Test different parameter configurations."""
        fig_width, fig_height = a4_figsize(columns=1)
        assert np.allclose([3.30708661, 2.04389193], [fig_width, fig_height])

        fig_width, fig_height = a4_figsize(columns=1.5)
        assert np.allclose([5.07874015, 3.13883403], [fig_width, fig_height])

        fig_width, fig_height = a4_figsize(columns=2)
        assert np.allclose([6.85039370, 4.23377614], [fig_width, fig_height])

        with pytest.raises(ValueError):
            a4_figsize(columns=3)

        a4_figsize(fig_height_mm=250)
        assert "fig_height too large" in caplog.text
