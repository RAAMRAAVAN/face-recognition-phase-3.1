# Utils: ram
import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    # check if sparse_mx is coo matrix or not.
    if not sp.isspmatrix_coo(sparse_mx):
        # convert sparse_mx to coo matrix
        sparse_mx = sparse_mx.tocoo()
    # print("sparse_mx",type(sparse_mx),sparse_mx.shape)
    # sparse_mx.row = list of rows. Eg: [  0   0   0 ... 463 464 466]
    # sparse_mx.col = list of columns. Eg: [ 11  37 164 ... 464 465 467]
    # np.vstack = stacks array in sequence vertically. Eg: [[  0   0   0 ... 463 464 466],[ 11  37 164 ... 464 465 467]]
    # coords = np.vstack.transpose() = [[0,11],[0,37],[0,164],......,[466,467]]
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    # values = list of edge weights 
    values = sparse_mx.data
    # shape = shape of adj matrix
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # adj_orig.diagonal() = all diagonal elements. Eg: [0,0,0,.....0], Shape = (1,468), 1 Dimensional
    # adj_orig.diagonal()[np.newaxis, :] = [[0,0,0,....,0]], Shape = (1,468), 2 Dimensional
    # sp.dia_matrix = sparse matrix with diagonal storage
    # sp.dia_matrix((data,offsets),shape) : data = [[0,0,0,...,0]], offsets = [0], and shape = (468,468)
    # subtractiong adj matrix with 468*468 matrix of zeros. ( Remove diagonal elements )
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    # eliminating zeros
    adj.eliminate_zeros()
    # Checking sum of diagonal elements is zero or not
    # adj.todense: for converting it into numpy matrix object
    # if sum of diagonal elements is not zero, it will give assertion error
    assert np.diag(adj.todense()).sum() == 0
    # storing upper triangle of adj martix in adj_triu
    adj_triu = sp.triu(adj)
    # print("edges_3=",adj_triu[0].shape)
    # print("adj_triu=",adj_triu.shape)
    # Converting upper triangle in th form of tuple, and storing in adj_tuple
    adj_tuple = sparse_to_tuple(adj_triu)
    # print("adj_tuple=",len(adj_tuple),adj_tuple)
    # seperating edge list from upper triangle of adjacency matrix and storing it in edges variable
    edges = adj_tuple[0]
    # print("edges=",edges)
    # seperating edge list from adjacency matrix, and storing it in edges_all variable
    edges_all = sparse_to_tuple(adj)[0]
    # print("edges all=",edges_all.shape)
    # print("edges.shape[0]=",edges.shape[0])
    # edges.sape[0] / 10. = 1322 / 10. = 132
    num_test = int(np.floor(edges.shape[0] / 10.))
    # edges.sape[0] / 20. = 1322 / 20. = 66
    num_val = int(np.floor(edges.shape[0] / 20.))
    # all_edges_idx = [0,1,2,...., 1321]
    all_edge_idx = list(range(edges.shape[0]))
    # print("idx=",all_edge_idx)
    # shuffel all_edges_idx
    np.random.shuffle(all_edge_idx)
    # val_edge_idx = first 66 edges from shuffeled list of edges
    val_edge_idx = all_edge_idx[:num_val]
    # test_edges_idx = 132 edges 66th edge from shuffeled list of edges 
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    # test_edges = 132 edges from upper triangle of edge
    test_edges = edges[test_edge_idx]
    # val_edges = 66 edges from upper triangle of edge
    val_edges = edges[val_edge_idx]
    # array = upper triangle of edges
    # delete = np.hstack([test_edge_idx,val_edge_idx]), test_edge_idx = 132 edges,val_edge_idx = 66 edges, so delete 132 + 66 = 198 edges
    # train_edges = np.delete(array, to be deleted, axis) = 1124 edges, removed those edges, which are in test_edge_idx and val_edge_idx
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    # print("train_ edges=", len(train_edges))
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    # test_edges_false = []
    test_edges_false = []
    # 132 entries in test_edges_false
    # excluded diagonal elements 
    # excluded edges present in edges_all
    # test_edges_false = elements with zero value, excluding diagonal
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    # 66 entries in val_edges_false
    # excluded diagonal elements. 
    # excluding val_edges. 66 excluded
    # excluding train_edges. 1124 excluded
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    # identity matrix of shape 1124. shape is equal to the shape of train_edges matrix
    data = np.ones(train_edges.shape[0])
    # print("data=",data.shape)
    # Re-build adj matrix
    # CSR - Compressed Sparse Row. For fast row slicing, faster matrix vector products
    # csr_matrix(arr).count_nonzero() to count non zeros
    # data = 1124 edges
    # row = x-axis of train_edges
    # col = y-axis of train_edges
    # shape = 468 X 468
    # adjacency matrix with only 1124 edges, removed lower triangular values and 198 edges, which are in val and test 
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    # converting adj matrix to coo matrix
    adj = sp.coo_matrix(adj)
    # sp.eye(adj.shape[0]) = Sparse matrix with ones on diagonal
    # adj_ = adj + diagonal vaues = 1 
    adj_ = adj + sp.eye(adj.shape[0])
    # Sum the matrix elements over a given axis
    # calculating sum of each row
    rowsum = np.array(adj_.sum(1))
    # print("rowsum=",rowsum)
    # sp.diags = Construct a sparse matrix from diagonals
    # np.power(rowsum, -0.5) = square root of each element
    # flatten() = Return a copy of the array collapsed into one dimension
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # print("degree=",degree_mat_inv_sqrt)
    # (adj + feature) * pow(D,.5) * pow(D,.5)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    # reconstruct matrix
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    # print("preds",labels_all.shape)
    # print("labels",labels_all)
    # print("labels",type(labels_all))
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score, emb
