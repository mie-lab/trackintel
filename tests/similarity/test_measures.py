# -*- coding: utf-8 -*-

import trackintel as ti
import os
from trackintel.similarity.measures import *

class TestMeasures:
    def test_dtw(self):
        
        s = ';'
        t1 = ti.io.file.read_positionfixes_csv(os.path.join('tests','data','sim1.csv'),sep=s)
        t2 = ti.io.file.read_positionfixes_csv(os.path.join('tests','data','sim2.csv'),sep=s)
       
        
        
        assert e_dtw(t1,t2) is float
        assert e_dtw(t1,t1) == e_dtw(t2,t2) == 0
#        assert dtw(t1,t2) == 0
#        assert False
        
    def test_edr(self):
        tp1 = ti.io.file.read_positionfixes_csv(os.path.join('tests','data','test_edr_no.csv'))
        tp2 = ti.io.file.read_positionfixes_csv(os.path.join('tests','data','test_edr_outlier.csv'))
        eps = ti.geogr.distances.meters_to_decimal_degrees(20, tp1.geom.y.mean())
        assert e_edr(tp1,tp2,eps) == 1/len(tp1) #only one point of the two trajectories is really different
        
        
        
        