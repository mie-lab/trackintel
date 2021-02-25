import os

import geopandas as gpd

import trackintel as ti


class TestSpatial_filter():
    
    def test_filter_staypoints(self):
        # read staypoints and area file
        spts_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        spts = ti.read_staypoints_csv(spts_file, tz='utc', index_col='id')
        extent = gpd.read_file(os.path.join('tests', 'data', 'area', 'tsinghua.geojson'))
        
        # the projection needs to be defined: WGS84
        spts.crs = 'epsg:4326'
        within_spts = spts.as_staypoints.spatial_filter(areas=extent, method="within", re_project=True)
        intersects_spts = spts.as_staypoints.spatial_filter(areas=extent, method="intersects", re_project=True)
        crosses_spts = spts.as_staypoints.spatial_filter(areas=extent, method="crosses", re_project=True)
        
        # the result obtained from ArcGIS
        gis_within_num = 13
        
        assert len(within_spts) == gis_within_num, "The spatial filtered sp number should be the same as" + \
            "the one from the result with ArcGIS"
        assert all(within_spts.geometry == intersects_spts.geometry), "For sp the result of within and" + \
            "intersects should be the same"
        assert len(crosses_spts) == 0, "There will be no point crossing area"
        
    def test_filter_triplegs(self):
        # read triplegs and area file
        tpls_file = os.path.join('tests', 'data', 'geolife', 'geolife_triplegs.csv')
        tpls = ti.read_triplegs_csv(tpls_file, tz='utc', index_col='id')
        extent = gpd.read_file(os.path.join('tests', 'data', 'area', 'tsinghua.geojson'))
        
        # the projection needs to be defined: WGS84
        tpls.crs = 'epsg:4326'
        within_tl = tpls.as_triplegs.spatial_filter(areas=extent, method="within", re_project=True)
        intersects_tl = tpls.as_triplegs.spatial_filter(areas=extent, method="intersects", re_project=True)
        crosses_tl = tpls.as_triplegs.spatial_filter(areas=extent, method="crosses", re_project=True)
        
        # the result obtained from ArcGIS
        gis_within_num = 9
        gis_intersects_num = 20
        
        assert len(within_tl) == gis_within_num, "The within tripleg number should be the same as" + \
            "the one from the result with ArcGIS"
        assert len(intersects_tl) == gis_intersects_num, "The intersects tripleg number should be " + \
            "the same as the one from the result with ArcGIS"
        assert len(crosses_tl) == len(intersects_tl) - len(within_tl), "The crosses tripleg number" + \
            "should equal the number of intersect triplegs minus the number of within triplegs"
    
    def test_filter_locations(self):
        # read staypoints and area file
        spts_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        spts = ti.read_staypoints_csv(spts_file, tz='utc', index_col='id')
        extent = gpd.read_file(os.path.join('tests', 'data', 'area', 'tsinghua.geojson'))
        
        # cluster staypoints to locations
        _, locs = spts.as_staypoints.generate_locations(method='dbscan', epsilon=10, 
                                                       num_samples=0, distance_matrix_metric='haversine',
                                                       agg_level='dataset')
        
        # the projection needs to be defined: WGS84
        locs.crs = 'epsg:4326'
        
        # filter locations with the area
        within_loc = locs.as_locations.spatial_filter(areas=extent, method="within", re_project=True)
        intersects_loc = locs.as_locations.spatial_filter(areas=extent, method="intersects", re_project=True)
        crosses_loc = locs.as_locations.spatial_filter(areas=extent, method="crosses", re_project=True)
        
        # the result obtained from ArcGIS
        gis_within_num = 12
        
        assert len(within_loc) == gis_within_num, "The spatial filtered location number should be the same as" + \
            "the one from the result with ArcGIS"
        assert all(within_loc.geometry == intersects_loc.geometry), "For location the result of within and" + \
            "intersects should be the same"
        assert len(crosses_loc) == 0, "There will be no point crossing area"