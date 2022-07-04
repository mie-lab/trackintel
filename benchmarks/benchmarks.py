import os
import pandas as pd

import sys
sys.path.append("/Users/nishant/Documents/GitHub/trackintel")
os.chdir("/Users/nishant/Documents/GitHub/trackintel")

import trackintel as ti

class TimeSuite_Generate_Staypoints:
    """ Run time tests """
    
    def time_gen_sp_geolife_long(self):
        """Generate sp only"""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long_10_MB"))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        return pfs, sp

    def time_gen_sp_tpls_geolife_long(self):
        """Generate sp and tpls """
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long_10_MB"))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        return pfs, tpls
        
    


class MemSuite_Generate_staypoints:
    """Memory tests"""

    def mem_gen_sp_geolife_long(self):
        """Generate sp only"""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long_10_MB"))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        return pfs, sp

    def mem_gen_sp_tpls_geolife_long(self):
        """Generate sp and tpls """
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long_10_MB"))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        return pfs, tpls
