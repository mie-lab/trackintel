import os
import trackintel as ti
from pathlib import Path

datasetlist = ["geolife_long", "geolife_long_10_MB"]
bm_dataset = datasetlist[0]


trackintel_root = Path(__file__).parents[1]


class BM_Read_PFS:
    """Benchmarks for read positionfixes"""

    def common_func(self):
        os.chdir(trackintel_root)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", bm_dataset))
        return pfs

    def time_read_pfs(self):
        self.common_func()

    def mem_read_pfs(self):
        return self.common_func()

    def peakmem_read_pfs(self):
        self.common_func()


class BM_Generate_SP:
    """Benchmarks for generate staypoints"""

    def setup(self):
        os.chdir(trackintel_root)
        self.pfs, self._ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", bm_dataset))

    def common_func(self):
        """Generate sp"""
        pfs, sp = self.pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        return sp

    def time_gen_sp_geolife_long(self):
        self.common_func()

    def mem_gen_sp_geolife_long(self):
        return self.common_func()

    def peakmem_gen_sp_geolife_long(self):
        self.common_func()


class BM_Generate_TPLS:
    """Benchmarks for generate triplegs"""

    def setup(self):
        os.chdir(trackintel_root)
        self.pfs, self._ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", bm_dataset))
        self.pfs, self.sp = self.pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=25, time_threshold=5
        )

    def common_func(self):
        """Generate TPLS"""
        _, tpls = self.pfs.as_positionfixes.generate_triplegs(self.sp, method="between_staypoints")
        return tpls

    def time_gen_tpls_geolife_long(self):
        self.common_func()

    def mem_gen_tpls_geolife_long(self):
        return self.common_func()

    def peakmem_gen_tpls_geolife_long(self):
        self.common_func()


class BM_Generate_TRIPS:
    """Benchmarks for generate trips"""

    def setup(self):
        os.chdir(trackintel_root)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", bm_dataset))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        pfs, self.tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        self.sp = sp.as_staypoints.create_activity_flag(time_threshold=15)

    def common_func(self):
        """Generate Trips"""
        _, _, trips = ti.preprocessing.generate_trips(self.sp, self.tpls)
        return trips

    def time_gen_trips_geolife_long(self):
        self.common_func()

    def mem_gen_trips_geolife_long(self):
        return self.common_func()

    def peakmem_gen_trips_geolife_long(self):
        self.common_func()


#
class BM_Generate_TOURS:
    """Benchmarks for generate tours"""

    def setup(self):
        os.chdir(trackintel_root)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", bm_dataset))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        sp = sp.as_staypoints.create_activity_flag(time_threshold=15)
        _, _, self.trips = ti.preprocessing.generate_trips(sp, tpls)

    def common_func(self):
        """Generate Tours"""
        _, tours = ti.preprocessing.generate_tours(self.trips, max_dist=100)
        return tours

    def time_gen_tours_geolife_long(self):
        self.common_func()

    def mem_gen_tours_geolife_long(self):
        return self.common_func()

    def peakmem_gen_tours_geolife_long(self):
        self.common_func()
