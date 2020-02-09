import numpy as np
from sklearn.preprocessing import normalize
from scipy import sparse
import itertools

def get_hops(A, K):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The graph represented as a sparse matrix
    K : int
        The maximum hopness to consider.

    Returns
    -------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    """
    hops = {1: A.tolil(), -1: A.tolil()}
    hops[1].setdiag(0)

    for h in range(2, K + 1):
        # compute the next ring
        next_hop = hops[h - 1].dot(A)
        next_hop[next_hop > 0] = 1

        # make sure that we exclude visited n/edges
        for prev_h in range(1, h):
            next_hop -= next_hop.multiply(hops[prev_h])

        next_hop = next_hop.tolil()
        next_hop.setdiag(0)

        hops[h] = next_hop
        hops[-1] += next_hop

    return hops


def sample_last_hop(A, nodes):
    """
    For each node in nodes samples a single node from their last (K-th) neighborhood.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix encoding which nodes belong to any of the 1, 2, ..., K-1, neighborhoods of every node
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N]
        The sampled nodes.
    """
    N = A.shape[0]

    sampled = np.random.randint(0, N, len(nodes))

    nnz = A[nodes, sampled].nonzero()[1]
    while len(nnz) != 0:
        new_sample = np.random.randint(0, N, len(nnz))
        sampled[nnz] = new_sample
        nnz = A[nnz, new_sample].nonzero()[1]

    return sampled


def sample_all_hops(hops, nodes=None):
    """
    For each node in nodes samples a single node from all of their neighborhoods.

    Parameters
    ----------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N, K]
        The sampled nodes.
    """

    N = hops[1].shape[0]

    if nodes is None:
        nodes = np.arange(N)

    return np.vstack((nodes,
                      np.array([[-1 if len(x) == 0 else np.random.choice(x) for x in hops[h].rows[nodes]]
                                for h in hops.keys() if h != -1]),
                      sample_last_hop(hops[-1], nodes)
                      )).T




#@pysnooper.snoop()
def to_triplets(sampled_hops, scale_terms):
    """
    Form all valid triplets (pairwise constraints) from a set of sampled nodes in triplets

    Parameters
    ----------
    sampled_hops : array-like, shape [N, K]
       The sampled nodes.
    scale_terms : dict
        The appropriate up-scaling terms to ensure unbiased estimates for each neighbourhood

    Returns
    -------
    triplets : array-like, shape [?, 3]
       The transformed triplets.
    """
    triplets = []
    triplet_scale_terms = []

    for i, j in itertools.combinations(np.arange(1, sampled_hops.shape[1]), 2):
        triplet = sampled_hops[:, [0] + [i, j]]
        triplet = triplet[(triplet[:, 1] != -1) & (triplet[:, 2] != -1)]
        triplet = triplet[(triplet[:, 0] != triplet[:, 1]) & (triplet[:, 0] != triplet[:, 2])]
        triplets.append(triplet)
        #print(i,'****',j)
        #print(triplet[:, 1])
        #print(triplet[:, 2])
        triplet_scale_terms.append(scale_terms[i][triplet[:, 1]] * scale_terms[j][triplet[:, 2]])

    return np.row_stack(triplets),np.concatenate(triplet_scale_terms)
