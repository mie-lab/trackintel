# -*- coding: utf-8 -*-

import trackintel
import os
from trackintel.similarity.dtw import e_dtw as dtw

class TestDtw:
    def test_dtw(self):
        
        s = ';'
        t1 = trackintel.io.file.read_positionfixes_csv('/Users/svenruf/code/trackintel/tests/data/sim1.csv',sep=s)
        t2 = trackintel.io.file.read_positionfixes_csv('/Users/svenruf/code/trackintel/tests/data/sim2.csv',sep=s)
        #t2 is t1 with one additional point
        
        
        assert dtw(t1,t2)
        assert dtw(t1,t1) == dtw(t2,t2)
        assert dtw(t1,t2) == 0
        assert False
        
        