# -*- coding: utf-8 -*-
from trackintel.similarity import measures
import numpy as np
from scipy.sparse import dok_matrix
from console_progressbar import ProgressBar


def e_dist_tuples(c1,c2):
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def similarity_matrix(data, method, field='tripleg_id', trsh=None, eps=None, dist=False, **kwargs):   
    """Method calculates similarity (/distance) matrix of trajectories in a data set.
    
    INPUT:
        data:           a positionfixes GDF with tripleg_ids, recommended: Use tripleg_ids in range(0,n) to avoid oversize of similarity/distanc matrices.
        method:         type of similarity measure, available: DynamicTimeWarping (dtw), Edit Distance on real Sequence (edr). Start End Distance (sed) See doc for details.
        field:          Which field of the GDF should be used to distinguish the trajectories
        trsh:           value to pre-check the distance. If the distance of the two trajectory centers (mean of all points) is greater than trsh, the calculation
                        will not be performed and the similarity will be set to zero.
        eps:            Epsilon parameter for EDR. Points closer to eachother than eps are considered as equal.
        dist:           False: Output is a similarity (sparse)matrix
                        True: Output is a distance matrix. Not recommended for large data sets! Treshold will be ignored!
                               
    RETURN: 
        sim             sparse similarity matrix (inverted distances of trajectories) 
        dist=False      Be aware that also non existing tripleg_ids have similarity zero in the sim matrix!
        
        sim
        dist=True       Distance matrix
                        Be aware that also non existing tripleg_ids have distance inf in the distance matrix!
                        
        Pay attention on the projection of your data. Parameters as trsh, eps may have to be converted!
        To convert a value in meters to decimal degrees, the method geogr.distances.meters_to_decimal_degrees can be used.
        """
    try:
        assert data.as_positionfixes
        assert field in data.columns
    except: 
        raise Exception('Input data format must be positionfixes with added tripleg_id')
        
    calc_dist = None
    
    if method == 'dtw':
        calc_dist = getattr(measures, 'e_dtw')
        if trsh==None:
            trsh=1000 #ToDo: Implement default treshold for dtw
        
    elif method == 'edr':
        try:
            assert float(eps)
        except: 
            raise Exception('EDR requires epsilon parameter')
        calc_dist = getattr(measures, 'e_edr')
        
    elif method == 'ses':
        try:
            time_trsh = kwargs.get('time_trsh')
        except:
            raise Exception('for start end similarity, also a time treshold has to be defined. (time_trsh=someNumber)')
        
        calc_dist = getattr(measures, 'start_end_dist')
    else:
        raise NotImplementedError
        
      
    it = data[field].unique()  
    it = it[it != -1]
    bar = ProgressBar(total=len(it))#array with all tripleg_ids to iterate over
    
    if dist:
        sim = np.ones((int(max(it)+1),int(max(it)+1)))*np.inf
    else:
        sim = dok_matrix((int(max(it)+1),int(max(it)+1)))
    if not method == 'ses':
        for i in range(len(it)):
            tp1 = data.loc[data[field]==it[i]].sort_values('tracked_at')  #slice dataframe to extract a tripleg
            for j in range(int(i)+1,len(it)):
                tp2 = data.loc[data[field]==it[j]].sort_values('tracked_at')
                
                
                
                if dist:
                    d = calc_dist(tp1,tp2,eps=eps)
                    sim[int(it[i]),int(it[j])] = d
                    sim[int(it[j]),int(it[i])] = d
                    sim[int(it[i]),int(it[i])] = 0
                else:
                    if e_dist_tuples(tp1.as_positionfixes.center, tp2.as_positionfixes.center)>trsh:   #pre check the trajectories, if the distance of the centers exceeds the threshold
                        s=0
                    else:
                        d = calc_dist(tp1,tp2,eps=eps)
                        
                        if d==0:
                            s=np.inf #avoid division by zero
                        else:       #in all other cases the inverted trajectory distance is stored
                            s=1/d
                             
                    sim[it[i],it[j]] = s
                    sim[it[j],it[i]] = s
                    sim[it[i],it[i]] = np.inf
            bar.print_progress_bar(i)
    
    
    else:
        if dist:
            for i in range(len(it)):
                sim [it[i],:] = calc_dist(data, trsh, time_trsh, id_to_compare=it[i])
        else:
            for i in range(len(it)):
                sim[it[i],:] = 1/calc_dist(data, trsh, time_trsh, id_to_compare=it[i])
                
    bar.print_progress_bar(len(it))    
    return sim


