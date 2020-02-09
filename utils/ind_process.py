import _pickle as cPickle
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys

def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    data =  adj.tocoo().data
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col
    #return (indices, adj.data, adj.shape), adj.row, adj.col, data#adj.data


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    nx_graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx_graph)


    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]


    ################# added section to extract train subgraph ######################
    ids = set(range(labels.shape[0]))
    train_ids = ids.difference(set(list(idx_val) + list(idx_test)))
    # train_edges = [edge for edge in nx_graph.edges() if edge[0] in train_ids and edge[1] in train_ids]
    #
    # adj_train = sparse.dok_matrix((len(ids), len(ids)))
    # for edge in train_edges:
    #     if edge[0] != edge[1]:
    #         adj_train[edge[0], edge[1]] = 1

    nx_train_graph = nx_graph.subgraph(train_ids)
    adj_train = nx.adjacency_matrix(nx_train_graph)

    features = features.todense()

    features_train = features[np.array(list((train_ids)))]

    #mean value
    features_target = construct_traget_neighbors(nx_graph,features)
    features__train_target = features_target[np.array(list((train_ids)))]
    ################################################################################
    #features__train_target = construct_traget_neighbors(nx_train_graph,features_train)

    return adj_train, adj, features_train, features, labels, idx_train, idx_val, idx_test,features__train_target


def construct_traget_neighbors(nx_G, X):
    # construct target neighbor feature matrix
    X_target = np.zeros(X.shape)
    nodes = nx_G.nodes()
    # autoencoder for reconstructing Weighted Average Neighbor
    for node in nodes:
        neighbors = list(nx_G.neighbors(node))
        if len(neighbors) == 0:
            X_target[node] = X[node]
        else:
            temp = np.array(X[node])
            for n in neighbors:
                temp = np.vstack((temp, X[n]))
            temp = np.mean(temp, axis=0)
            X_target[node] = temp
    return X_target  