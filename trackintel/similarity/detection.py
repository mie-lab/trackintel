# -*- coding: utf-8 -*-
from trackintel.similarity import measures
import numpy as np
from scipy.sparse import dok_matrix
from console_progressbar import ProgressBar


def min_dist_first_points(tp1,tp2):
    p10 = tp1.iloc[0]['geom']
    p11 = tp1.iloc[1]['geom']
    p20 = tp2.iloc[0]['geom']
    p21 = tp2.iloc[1]['geom']
    a = p10.distance(p20)
    b = p10.distance(p21)
    c = p11.distance(p20)
    d = p11.distance(p21)
    return min(a,b,c,d)

def similarity_detection(data, method, trsh=1000):   #ToDo: Default Treshold calculation
    """Method detects similar trajectories in a data set.
    
    INPUT:
        method:         type of similarity measure, available: DynamicTimeWarping (dtw), Edit Distance on real Sequence (edr)
        data:           a positionfixes GDF with tripleg_ids
        trsh:           for 2 trajectories with a bigger distance than the treshold the similarity will be set to zero
        
    RETURN: 
        sim:            sparse similarity matrix (inverted distances of trajectories)
                        Be aware that also non existing tripleg_ids have similarity zero in the sim matrix!
        """
    try:
        assert data.as_positionfixes
        assert 'tripleg_id' in data.columns
    except: 
        raise Exception('Input data format must be positionfixes with added tripleg_id')
        
    calc_dist = None
    
    if method == 'dtw':
        calc_dist = getattr(measures, 'e_dtw')
    elif method == 'edr':
        calc_dist = getattr(measures, 'e_edr')
        
        
        it = data['tripleg_id'].unique()  
        it = it[it != -1] #array with all tripleg_ids to iterate over
        sim = dok_matrix((int(max(it)+1),int(max(it)+1)))
        bar = ProgressBar(total=len(it))
        for i in range(len(it)):
            tp1 = data.loc[data['tripleg_id']==it[i]].sort_values('tracked_at')  #slice dataframe to extract a tripleg
            for j in range(int(i)+1,len(it)):
                tp2 = data.loc[data['tripleg_id']==it[j]].sort_values('tracked_at')
                if min_dist_first_points(tp1,tp2)>trsh:
                    s=0
                else:
                    d = calc_dist(tp1,tp2)
                    if d==0:
                        s=np.inf #avoid division by zero
                    else:       #in all other cases the inverted trajectory distance is stored
                        s=1/d
                         
                sim[i,j] = s
                sim[j,i] = s
            bar.print_progress_bar(i)
        sim.setdiag(np.inf)
        bar.finish()
        
    return sim


