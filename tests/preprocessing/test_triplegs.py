import os

import trackintel as ti

class TestTriplegs():
    def test_smoothen_triplegs(self):
        tpls = ti.read_triplegs_csv(os.path.join('tests','data','triplegs_with_too_many_points_test.csv'), sep=';')
        tpls_smoothed = ti.preprocessing.triplegs.smoothen_triplegs(tpls, tolerance=0.0001)
        line1 = tpls.iloc[0].geom
        line1_smoothed = tpls_smoothed.iloc[0].geom
        line2 = tpls.iloc[1].geom
        line2_smoothed = tpls_smoothed.iloc[1].geom

        assert line1.length == line1_smoothed.length
        assert line2.length == line2_smoothed.length
        assert len(line1.coords) == 10
        assert len(line2.coords) == 7
        assert len(line1_smoothed.coords) == 4
        assert len(line2_smoothed.coords) == 3
