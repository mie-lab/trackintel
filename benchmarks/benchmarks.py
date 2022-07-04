import os
import pandas as pd

import sys
sys.path.append("/Users/nishant/Documents/GitHub/trackintel")
os.chdir("/Users/nishant/Documents/GitHub/trackintel")

import trackintel as ti

class TimeSuite_Generate_Staypoints:
    """Run time tests for generate_staypoints() method."""

    def time_sp_tpls_geolife_long(self):
        """Generate sp and tpls sequences of the original pfs for subsequent testing."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

        tpls["type"] = "tripleg"
        sp["type"] = "staypoint"
        sp_tpls = pd.concat((sp, tpls), ignore_index=True).sort_values(by="started_at")
        return sp_tpls

    def time_all_geolife_long(self):
        """Generate sp, tpls and trips of the original pfs for subsequent testing."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        sp = sp.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        sp, tpls, trips = ti.preprocessing.triplegs.generate_trips(sp, tpls, gap_threshold=15)

        return sp, tpls, trips


class MemSuite_Generate_staypoints:
    """Memory tests for generate_staypoints() method."""

    def mem_sp_tpls_geolife_long(self):
        """Generate sp and tpls sequences of the original pfs for subsequent testing."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

        tpls["type"] = "tripleg"
        sp["type"] = "staypoint"
        sp_tpls = pd.concat((sp, tpls), ignore_index=True).sort_values(by="started_at")
        return sp_tpls

    def mem_all_geolife_long(self):
        """Generate sp, tpls and trips of the original pfs for subsequent testing."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        sp = sp.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        sp, tpls, trips = ti.preprocessing.triplegs.generate_trips(sp, tpls, gap_threshold=15)

        return sp, tpls, trips