from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np
import pandas as pd


def weights_transition_count(staypoints):
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
    adjacency_dict = create_adjacency_matrix_from_counts(counts, all_users)

    return adjacency_dict


def create_adjacency_matrix_from_counts(counts, user_list):
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
    adjacency_dict = {}

    for user_id in user_list:
        counts_user = counts.loc[counts['user_id'] == user_id]
        
        row_ix = counts_user['place_id'].values.astype('int')
        col_ix = counts_user['place_id_end'].values.astype('int')
        values = counts_user['counts'].values
             
        if len(values) == 0:
            adjacency_dict[user_id] = {'A': coo_matrix((0, 0)),
                                        'place_id_order': np.asarray([])}

        else:
            # start adjacency matrix at (ij) = (0,0)
#            min_id = np.min([np.min(row_ix), np.min(col_ix)])
#            row_ix = row_ix - min_id
#            col_ix = col_ix - min_id
            
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
            adjacency_dict[user_id] = {'A': coo_matrix((values,
                                              (row_ix, col_ix)),shape=shape),
                                        'place_id_order': org_ix}


    return adjacency_dict


def generate_activity_graphs(places, adjacency_dict):
    """
    Generate user specific graphs based on activity locations (places).

    This function creates a networkx graph per user based on the places of
    the user as nodes and a set of (weighted) edges defined in adjacency dict.

    Parameters
    ----------
    places : GeoDataFrame
        Trackintel dataframe of type places
    adjacency_dict : dictionary
         A dictionary with adjacendy matrices of type: {user_id:
         scipy.sparse.coo_matrix}.

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
            
            
        A = adjacency_dict[user_id_this]['A']
        place_id_order = adjacency_dict[user_id_this]['place_id_order']
        G = nx.from_scipy_sparse_matrix(A)

        # add graph information
        G.graph["user_id"] = user_id_this

        places_user_view = places.loc[places['user_id'] == user_id_this]
        places_user_view = places_user_view.sort_values('place_id')
        
        # add node information
        node_ids = np.arange(len(places_user_view))
        node_features = places_user_view.loc[:,
                ['place_id','extent', 'center']].to_dict('records')

        node_dict = dict(zip(node_ids, node_features))
        nx.set_node_attributes(G, node_dict)

        G_dict[user_id_this] = G

    return G_dict


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
