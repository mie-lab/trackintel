from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np
import pandas as pd
from trackintel.geogr.distances import calculate_distance_matrix
from sklearn.neighbors import NearestNeighbors
import scipy.spatial 
import matplotlib.pyplot as plt
from scipy.spatial.qhull import QhullError

def weights_transition_count(staypoints, adjacency_dict=None):
    """
    Calculate the number of transition between places as graph weights.

    Graphs based on the activity places (trackintel places) can have several
    types of weighted edges. This function calculates the edge weight based
    on the number of transitions of an individual user between places.

    The function requires the staypoints to have a cluster id field (e.g.
    staypoints.as_staypoints.extract_places() was already used.

    Parameters
    ----------
    staypoints : GeoDataFrame

    Returns
    -------
    adjacency_dict : dictionary
            A dictionary of adjacency matrices of type scipy.sparse.coo_matrix
    """
    # todo: check if cluster id is missing
    # todo: check if adjacency matrix is symmetric?

    all_users = staypoints["user_id"].unique()

    staypoints_a = staypoints.sort_values(['user_id', 'started_at'])
    # Deleting staypoints without cluster means that we count non-direct
    # transitions between two clusters e.g., 1 -> -1 -> 2 as direct transitions
    # between two clusters!
    # E.g., 1 -> 2
    staypoints_a = staypoints_a.loc[staypoints_a['place_id'] != -1]

    # count transitions between cluster
    staypoints_a["place_id_end"] = staypoints_a.groupby("user_id"
                                                          )[
                                                    "place_id"].shift(-1)
    try:
        
        counts = staypoints_a.groupby(by=['user_id', 'place_id',
                                          'place_id_end']
                                      ).size().reset_index(name='counts')
    except ValueError:
        # If there are only rows with nans, groupby throws an error but should
        # return an empty dataframe
        counts = pd.DataFrame(columns=['user_id', 'place_id', 'place_id_end',
                                       'counts'])
    # create Adjacency matrix
    adjacency_dict = create_adjacency_matrix_from_counts(counts, all_users,
                                                         adjacency_dict)
    
    

    return adjacency_dict


def create_adjacency_matrix_from_counts(counts, user_list, adjacency_dict):
    """
    Transform transition counts into a adjacency matrix per user.

    The input provides transition counts between places of a user. These
    counts are transformed into a weighted adjacency matrix.

    Parameters
    ----------
    counts : DataFrame
        pandas DataFrame that has at least the columns ['user_id',
        'cluster_id', 'cluster_id_end', 'counts']. Counts represents the
        number of transitions between two places.
    user_list : iterable
        A list of the relevant user_ids, must be a subset of the user_id column
        in counts

    Returns
    -------
    adjacency_dict : dictionary
            A dictionary of adjacency matrices of type scipy.sparse.coo_matrix
    """
    if adjacency_dict is None:
        adjacency_dict = {}

    for user_id_this in user_list:
        counts_user = counts.loc[counts['user_id'] == user_id_this]
        
        row_ix = counts_user['place_id'].values.astype('int')
        col_ix = counts_user['place_id_end'].values.astype('int')
        values = counts_user['counts'].values
             
        if len(values) == 0:
            A =  coo_matrix((0, 0))
            place_id_order = np.asarray([])

        else:

            # ix transformation
            org_ix = np.unique(np.concatenate((row_ix, col_ix)))
            new_ix = np.arange(0,len(org_ix))
            ix_tranformation = dict(zip(org_ix,new_ix))
            ix_backtranformation = dict(zip(new_ix,org_ix))
            
            row_ix = [ix_tranformation[row_ix_this] for row_ix_this in row_ix]
            row_ix = np.asarray(row_ix)
            col_ix = [ix_tranformation[col_ix_this] for col_ix_this in col_ix]
            col_ix = np.asarray(col_ix)
            
            # set shape and create sparse matrix
            max_ix = np.max([np.max(row_ix), np.max(col_ix)]) + 1
            shape = (max_ix, max_ix)
            
            A = coo_matrix((values,(row_ix, col_ix)),shape=shape)
            place_id_order = org_ix
            
        
        if user_id_this not in adjacency_dict:
            adjacency_dict[user_id_this] = {'A': [A],
                                            'place_id_order': [place_id_order],
                                            'edge_name': ['transition_counts']}
        else:
            adjacency_dict[user_id_this]['A'].append(A)
            adjacency_dict[user_id_this]['place_id_order'].append(place_id_order)
            adjacency_dict[user_id_this]['edge_name'].append('transition_counts')

    return adjacency_dict

def weights_n_neighbors(places, n=None, distance_matrix_metric='haversine',adjacency_dict=None):
    """
    Calculate the distance of the n nearest places as graph weights.

    Graphs based on the activity places (trackintel places) can have several
    types of weighted edges. This function calculates the edge weight based
    on the distance to the n closest neighbors (places) of the same user.

    Parameters
    ----------
    places: GeoDataFrame
    
    n: int
    number of nearst places to take into account
    
    distance_matrix_metric: String
    can be 

    Returns
    -------
    distance_matrix_metric: string
        The distance metric used to calculate the distance between places.
        Uses the Trackintel.geogr.distances.calculate_distance_matrix()
        function. Possible metrics are: {'haversine', 'euclidean'} or any 
        mentioned in: 
            https://scikit-learn.org/stable/modules/generated/
            sklearn.metrics.pairwise_distances.html
    """
    # todo: check if cluster id is missing
    # todo: check if adjacency matrix is symmetric?
    # todo: What if n is too large?
    
    all_users = places["user_id"].unique()
    if adjacency_dict is None:
        adjacency_dict = {}
    
    
    for user_id_this in all_users:
        row_ixs = []
        col_ixs = []
        values = []
        
        user_places = places[places["user_id"] == user_id_this].sort_values('place_id')
        
        places_distance_matrix = calculate_distance_matrix(
                user_places,
                dist_metric=distance_matrix_metric)
        org_ixs = user_places['place_id'].values
        
        shape = places_distance_matrix.shape
        place_id_order = org_ixs
        
        
        if n is None or n == 'fconn':
            A = coo_matrix(places_distance_matrix)
            edge_name = 'fconn_distant'
            
        else:
        
            # for every row, keep only the n smallest elements
            for row_ix_this in range(shape[0]):
                row_this = places_distance_matrix[row_ix_this,:]
                
                min_ixs = np.argsort(row_this)[0:n+1] 
                
                col_ixs = col_ixs + list(min_ixs)
                row_ixs = row_ixs + [row_ix_this for x in range(len(min_ixs))]
                values = values + list(row_this[min_ixs])
                
    
            # enforce symmetry: 
            col_ixs_temp = col_ixs.copy()
            col_ixs = col_ixs + row_ixs
            row_ixs = row_ixs + col_ixs_temp
            values = values + values
            
            A = coo_matrix((values,(row_ixs, col_ixs)),shape=shape)
            a = A.todense()
            edge_name = '{}_distant'.format(n)
        
        if user_id_this not in adjacency_dict:
            adjacency_dict[user_id_this] = {'A': [A],
                                            'place_id_order': [place_id_order],
                                            'edge_name': [edge_name]}
        else:
            adjacency_dict[user_id_this]['A'].append(A)
            adjacency_dict[user_id_this]['place_id_order'].append(place_id_order)
            adjacency_dict[user_id_this]['edge_name'].append(edge_name)
        
        
    return adjacency_dict


def generate_activity_graphs(places, adjacency_dict, node_feature_names=[]):
    """
    Generate user specific graphs based on activity locations (places).

    This function creates a networkx graph per user based on the places of
    the user as nodes and a set of (weighted) edges defined in adjacency dict.

    Parameters
    ----------
    places : GeoDataFrame
        Trackintel dataframe of type places
    adjacency_dict : dictionary or list of dictionaries
         A dictionary with adjacendy matrices of type: {user_id:
         scipy.sparse.coo_matrix}.
    edgenames : List
        List of names (stings) given to edges in a multigraph

    Returns
    -------
    G_dict : dictionary
        A dictionary of type: {user_id: networkx graph}.

    """
    # Todo: Enable multigraph input. E.g. adjacency_dict[user_id] = [edges1,
    #  edges2]
    # Todo: Should we do a check if places is really a dataframe of trackintel
    #  type?

    G_dict = {}
    for user_id_this in places['user_id'].unique():
        if user_id_this not in adjacency_dict:
            continue
        
        places_user_view = places.loc[places['user_id'] == user_id_this]
        places_user_view = places_user_view.sort_values('place_id')
        
        G = initialize_multigraph(user_id_this, places_user_view,
                                  node_feature_names)
        G.graph['edge_keys'] = []
        
        A_list = adjacency_dict[user_id_this]['A']
        place_id_order_list = adjacency_dict[user_id_this]['place_id_order']
        edge_name_list = adjacency_dict[user_id_this]['edge_name']
        
        for ix in range(len(A_list)):
            A = A_list[ix]
            a = A.todense()
            place_id_order = place_id_order_list[ix]
            edge_name = edge_name_list[ix]
            # todo: assert place_id_order
        
            G_temp = nx.from_scipy_sparse_matrix(A)
            edge_list = nx.to_edgelist(G_temp)      
            
#           edge_list = [(x[0], x[1], {**x[2], **{'edge_name': edge_name}}) 
            edge_list = [(x[0], x[1], edge_name,
                                {**x[2], **{'edge_name': edge_name}})
                                for x in edge_list]
            
            G.add_edges_from(edge_list, weight='weight')
            G.graph['edge_keys'].append(edge_name)
            
       
        
        
        G_dict[user_id_this] = G

    return G_dict

def weights_delaunay(places, to_crs=None, distance_matrix_metric='haversine',
                        adjacency_dict=None):


    all_users = places["user_id"].unique()
    if adjacency_dict is None:
        adjacency_dict = {}
    
    for user_id_this in all_users:
    
        user_places = places[places["user_id"] == user_id_this].sort_values('place_id')
        org_ixs = user_places['place_id'].values
        place_id_order = org_ixs
        edge_name = 'delaunay'
    
        if to_crs is not None:
            geometry = user_places['center'].to_crs(to_crs)
        else:
            geometry = user_places['center']
            
        # import point data as xy coordinates 
        # nx graph from scipy.spatial.Delaunay:
        # https://groups.google.com/forum/#!topic/networkx-discuss/D7fMmuzVBAw
        points = list(zip(geometry.x,geometry.y))
        # -------------------------------------- 
    
        # make a Delaunay triangulation of the point data 
        try:
            delTri = scipy.spatial.Delaunay(points) 
        
            # create a set for edges that are indexes of the points 
            edges = set() 
            # for each Delaunay triangle 
            for n in range(delTri.nsimplex): 
                # for each edge of the triangle 
                # sort the vertices 
                # (sorting avoids duplicated edges being added to the set) 
                # and add to the edges set 
                edge = sorted([delTri.vertices[n,0], delTri.vertices[n,1]]) 
                edges.add((edge[0], edge[1])) 
                edge = sorted([delTri.vertices[n,0], delTri.vertices[n,2]]) 
                edges.add((edge[0], edge[1])) 
                edge = sorted([delTri.vertices[n,1], delTri.vertices[n,2]]) 
                edges.add((edge[0], edge[1])) 
        
            # add distances to edges
            places_distance_matrix = calculate_distance_matrix(
                        user_places,
                        dist_metric=distance_matrix_metric)
            
            edges = [(u, v, places_distance_matrix[u,v]) for u,v in edges]
            row_ixs, col_ixs, values = map(list, zip(*edges))
            
            # enforce symmetry: 
            col_ixs_temp = col_ixs.copy()
            col_ixs = col_ixs + row_ixs
            row_ixs = row_ixs + col_ixs_temp
            values = values + values
            
            # create adjacency matrix
            shape = places_distance_matrix.shape
            A = coo_matrix((values,(row_ixs, col_ixs)),shape=shape)
        
        except QhullError:    
            A =  coo_matrix((0, 0))
            place_id_order = np.asarray([])
        
        
        if user_id_this not in adjacency_dict:
            adjacency_dict[user_id_this] = {'A': [A],
                                            'place_id_order': [place_id_order],
                                            'edge_name': [edge_name]}
        else:
            adjacency_dict[user_id_this]['A'].append(A)
            adjacency_dict[user_id_this]['place_id_order'].append(place_id_order)
            adjacency_dict[user_id_this]['edge_name'].append(edge_name)

    return adjacency_dict


def initialize_multigraph(user_id_this, places_user_view, node_feature_names):
    
    # create graph
    G = nx.MultiGraph()
    G.graph["user_id"] = user_id_this
    
    # add node information
    node_ids = np.arange(len(places_user_view))
    node_features = places_user_view.loc[:,
            ['place_id','extent', 'center'] + node_feature_names
            ].to_dict('records')

    node_tuple = tuple(zip(node_ids, node_features))
    G.add_nodes_from(node_tuple)
    return G


def nx_coordinate_layout(G):
    """
    Return networkx graph layout based on geographic coordinates.

    Parameters
    ----------
    G : networkx graph
        A networkx graph that was generated based on trackintel places.
        Nodes require the `center` attribute that holds a shapely point
        geometry

    Returns
    -------
    pos : dictionary
        dictionary with node_id as key that holds coordinates for each node

    """
    node_center = nx.get_node_attributes(G, 'center')
    pos = {key: (geometry.x, geometry.y) for key, geometry in
           node_center.items()}

    return pos


def count_places_by_user(staypoints):
    """
    Count how many places per user exist

    Parameters
    ----------
    staypoints : GeoDataFrame

    Returns
    -------
    places_by_user_count : dictionary
        Dictionary of type: {user_id: place_count}. It has the number of
        places per user.

    """
    all_users = staypoints["user_id"].unique()

    # delete invalid places if exist
    staypoints_a = staypoints.loc[staypoints['cluster_id'] != -1]

    # count places by user
    places_by_user_count = {}
    for user_id in all_users:
        unique_places = staypoints_a.loc[
            staypoints_a["user_id"] == user_id, 'cluster_id'].unique()

        places_by_user_count[user_id] = len(unique_places)

    return places_by_user_count
