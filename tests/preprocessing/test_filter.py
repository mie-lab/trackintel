import os
import pytest
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

import trackintel as ti


@pytest.fixture
def locs_from_geolife():
    """Create locations from geolife staypoints."""
    # read staypoints
    sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
    sp = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id")

    # cluster staypoints to locations
    _, locs = sp.as_staypoints.generate_locations(
        method="dbscan", epsilon=10, num_samples=0, distance_metric="haversine", agg_level="dataset"
    )

    # the projection needs to be defined: WGS84
    locs.crs = "epsg:4326"
    return locs


class TestSpatial_filter:
    """Tests for the spatial_filter function."""

    def test_filter_staypoints(self):
        """Test if spatial_filter works for staypoints."""
        # read staypoints and area file
        sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        sp = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id")
        extent = gpd.read_file(os.path.join("tests", "data", "area", "tsinghua.geojson"))

        # the projection needs to be defined: WGS84
        sp.crs = "epsg:4326"
        within_sp = sp.as_staypoints.spatial_filter(areas=extent, method="within", re_project=True)
        intersects_sp = sp.as_staypoints.spatial_filter(areas=extent, method="intersects", re_project=True)
        crosses_sp = sp.as_staypoints.spatial_filter(areas=extent, method="crosses", re_project=True)

        # the result obtained from ArcGIS
        gis_within_num = 13

        assert len(within_sp) == gis_within_num, (
            "The spatial filtered sp number should be the same as" + "the one from the result with ArcGIS"
        )
        assert len(crosses_sp) == 0, "There will be no point crossing area"

        # For staypoints the result of within and intersects should be the same
        assert_geodataframe_equal(within_sp, intersects_sp, check_less_precise=True)

    def test_filter_triplegs(self):
        """Test if spatial_filter works for triplegs."""
        # read triplegs and area file
        tpls_file = os.path.join("tests", "data", "geolife", "geolife_triplegs.csv")
        tpls = ti.read_triplegs_csv(tpls_file, tz="utc", index_col="id")
        extent = gpd.read_file(os.path.join("tests", "data", "area", "tsinghua.geojson"))

        # the projection needs to be defined: WGS84
        tpls.crs = "epsg:4326"
        within_tl = tpls.as_triplegs.spatial_filter(areas=extent, method="within", re_project=True)
        intersects_tl = tpls.as_triplegs.spatial_filter(areas=extent, method="intersects", re_project=True)
        crosses_tl = tpls.as_triplegs.spatial_filter(areas=extent, method="crosses", re_project=True)

        # the result obtained from ArcGIS
        gis_within_num = 9
        gis_intersects_num = 20

        assert len(within_tl) == gis_within_num, (
            "The within tripleg number should be the same as" + "the one from the result with ArcGIS"
        )
        assert len(intersects_tl) == gis_intersects_num, (
            "The intersects tripleg number should be " + "the same as the one from the result with ArcGIS"
        )
        assert len(crosses_tl) == len(intersects_tl) - len(within_tl), (
            "The crosses tripleg number"
            + "should equal the number of intersect triplegs minus the number of within triplegs"
        )

    def test_filter_locations(self, locs_from_geolife):
        """Test if spatial_filter works for locations."""
        locs = locs_from_geolife
        extent = gpd.read_file(os.path.join("tests", "data", "area", "tsinghua.geojson"))

        # filter locations with the area
        within_loc = locs.as_locations.spatial_filter(areas=extent, method="within", re_project=True)
        intersects_loc = locs.as_locations.spatial_filter(areas=extent, method="intersects", re_project=True)
        crosses_loc = locs.as_locations.spatial_filter(areas=extent, method="crosses", re_project=True)

        # the result obtained from ArcGIS
        gis_within_num = 12

        assert len(within_loc) == gis_within_num, (
            "The spatial filtered location number should be the same as" + "the one from the result with ArcGIS"
        )
        assert len(crosses_loc) == 0, "There will be no point crossing area"

        # For location the result of within and intersects should be the same
        assert_geodataframe_equal(within_loc, intersects_loc, check_less_precise=True)

    def test_re_project(self, locs_from_geolife):
        """Test if passing the re_project parameter will reproject the input gdf."""
        locs = locs_from_geolife
        extent = gpd.read_file(os.path.join("tests", "data", "area", "tsinghua.geojson"))

        # filter locations with the area, reproject
        within_loc_reProj = locs.as_locations.spatial_filter(areas=extent, method="within", re_project=True)

        # manual reproject
        init_crs = locs.crs
        locs = locs.to_crs(extent.crs)
        within_loc = locs.as_locations.spatial_filter(areas=extent, method="within", re_project=False)
        within_loc = within_loc.to_crs(init_crs)

        assert_geodataframe_equal(within_loc_reProj, within_loc, check_less_precise=True)

    def test_method_error(self, locs_from_geolife):
        """Test if the an error is raised when passing unknown 'method' to spatial_filter()."""
        locs = locs_from_geolife
        extent = gpd.read_file(os.path.join("tests", "data", "area", "tsinghua.geojson"))
        with pytest.raises(AttributeError):
            locs.as_locations.spatial_filter(areas=extent, method=12345)
