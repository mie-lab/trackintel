import os
import pytest

import trackintel as ti


class TestPositionfixes:
    def test_as_positionfixes_accessor(self):
        pfs_file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col="id")
        assert pfs.as_positionfixes

        pfs = pfs.drop(["geom"], axis=1)
        with pytest.raises(AttributeError):
            pfs.as_positionfixes

    def test_positionfixes_center(self):
        pfs_file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col="id")
        assert len(pfs.as_positionfixes.center) == 2
