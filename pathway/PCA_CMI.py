import numpy as np
import networkx as nx
import itertools
import math
import copy


def pca_cmi(data, new_net_bit, theta, max_order, show=False):
    predicted_graph = nx.DiGraph()
    predicted_graph.add_nodes_from(data.index.to_list())
    for _, row in new_net_bit.iterrows():
        TF = row['TF']
        Gene = row['Gene']
        predicted_graph.add_edge(TF, Gene)
    num_edges = predicted_graph.number_of_edges()
    L = 0
    nochange = False
    data = data.T
    while L < max_order and nochange == False:
        L = L + 1
        predicted_graph, nochange = remove_edges(predicted_graph, data, L, theta)
    if show:
        print("Final Prediction:")
        print("-----------------")
        print("Order : {}".format(L))
        print("Number of edges in the predicted graph : {}".format(predicted_graph.number_of_edges()))
    predicted_adjMatrix = nx.adjacency_matrix(predicted_graph)
    return predicted_adjMatrix, predicted_graph


def remove_edges(predicted_graph, data, L, theta):
    initial_num_edges = predicted_graph.number_of_edges()
    edges = predicted_graph.edges()

    for edge in list(edges):
        neighbors1 = set(predicted_graph.neighbors(edge[0]))
        neighbors2 = set(predicted_graph.neighbors(edge[1]))
        neighbors = neighbors1.intersection(neighbors2)
        nhbrs = copy.deepcopy(sorted(neighbors))
        T = len(nhbrs)
        if (T < L and L != 0) or edge[0] == edge[1]:
            continue
        else:
            x = data[edge[0]].to_numpy()
            if x.ndim == 1:
                x = np.reshape(x, (-1, 1))
            y = data[edge[1]].to_numpy()
            if y.ndim == 1:
                y = np.reshape(y, (-1, 1))
            K = list(itertools.combinations(nhbrs, L))
            if L == 0:
                cmiVal = conditional_mutual_info(x.T, y.T)

                if cmiVal < theta:
                    predicted_graph.remove_edge(edge[0], edge[1])
            else:
                maxCmiVal = 0
                for zgroup in K:
                    XYZunique = len(np.unique(list([edge[0], edge[1], zgroup[0]])))
                    if XYZunique < 3:
                        continue
                    else:
                        z = data[list(zgroup)].to_numpy()
                        if z.ndim == 1:
                            z = np.reshape(z, (-1, 1))
                        cmiVal = conditional_mutual_info(x.T, y.T, z.T)
                    if cmiVal > maxCmiVal:
                        maxCmiVal = cmiVal
                if maxCmiVal < theta:
                    predicted_graph.remove_edge(edge[0], edge[1])
    final_num_edges = predicted_graph.number_of_edges()
    if final_num_edges < initial_num_edges:
        return predicted_graph, False
    return predicted_graph, True



def conditional_mutual_info(X, Y, Z=np.array(1)):
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    if Y.ndim == 1:
        Y = np.reshape(Y, (-1, 1))
    if Z.ndim == 0:
        c1 = np.cov(X)
        if c1.ndim != 0:
            d1 = np.linalg.det(c1)
        else:
            d1 = c1.item()
        c2 = np.cov(Y)
        if c2.ndim != 0:
            d2 = np.linalg.det(c2)
        else:
            d2 = c2.item()
        c3 = np.cov(X, Y)
        if c3.ndim != 0:
            d3 = np.linalg.det(c3)
        else:
            d3 = c3.item()
        cmi = (1 / 2) * np.log((d1 * d2) / d3)
    else:
        if Z.ndim == 1:
            Z = np.reshape(Z, (-1, 1))

        c1 = np.cov(np.concatenate((X, Z), axis=0))
        if c1.ndim != 0:
            d1 = np.linalg.det(c1)
        else:
            d1 = c1.item()
        c2 = np.cov(np.concatenate((Y, Z), axis=0))
        if c2.ndim != 0:
            d2 = np.linalg.det(c2)
        else:
            d2 = c2.item()
        c3 = np.cov(Z)
        if c3.ndim != 0:
            d3 = np.linalg.det(c3)
        else:
            d3 = c3.item()
        c4 = np.cov(np.concatenate((X, Y, Z), axis=0))
        if c4.ndim != 0:
            d4 = np.linalg.det(c4)
        else:
            d4 = c4.item()
        cmi = (1 / 2) * np.log((d1 * d2) / (d3 * d4))
    if math.isinf(cmi):
        cmi = 0
    return cmi



