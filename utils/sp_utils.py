import scipy.sparse as sp
import sys

# Convert Tensorflow sparse matrix to Numpy sparse matrix
def conver_sparse_tf2np(input):
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])), shape=(input[layer][2][0], input[layer][2][1])) for layer in input]
