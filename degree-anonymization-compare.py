import math
import os
import numpy as np
import networkx as nx
import sys

from loguru import logger
from networkx.drawing.tests.test_pylab import plt


def compute_i(d, start, end):
    d_i = d[0]
    res = 0
    for i in range(start, end + 1):
        res += d_i - d[i]

    return res


def compute_I(d):
    d_i = d[0]
    res = 0
    for d_j in d:
        res += d_i - d_j
    return res


def c_merge(d, d1, k):
    res = d1 - d[k] + compute_I(d[k + 1: min(len(d), 2 * k)])
    return res


def c_new(d, k):
    t = d[k:min(len(d), 2 * k - 1)]
    res = compute_I(t)
    return res


def greedy_rec_algorithm(array_degrees, k_degree, pos_init, extension):
    if pos_init + extension >= len(array_degrees) - 1:
        for i in range(pos_init, len(array_degrees)):
            array_degrees[i] = array_degrees[pos_init]
        return array_degrees
    else:
        d1 = array_degrees[pos_init]
        c_merge_cost = c_merge(array_degrees, d1, pos_init + extension)
        c_new_cost = c_new(d, pos_init + extension)

        if c_merge_cost > c_new_cost:
            for i in range(pos_init, pos_init + extension):
                array_degrees[i] = d1
            greedy_rec_algorithm(array_degrees, k_degree, pos_init + extension, k_degree)
        else:
            greedy_rec_algorithm(array_degrees, k_degree, pos_init, extension + 1)


def count_edge_from_Vl(G, tmp, Vl):
    count = 0
    for i in range(0, len(Vl)):
        if (G.has_edge(tmp, Vl[i])):
            count += 1
    return count




def dp_graph_anonymization(array_degrees, k_degree):
    # complete this function
    node_anonimize = []
    cost_anonimize = []
    for i in range(0, len(array_degrees)):
        # caso base
        if i < (2 * k_degree - 1):
            aux_list = []
            aux_list.append(0)
            node_anonimize.append(aux_list)
            cost_anonimize.append(compute_i(array_degrees, 0, i))
        # passo induttivo
        else:
            ind_min = (max(k_degree-1, i - 2 * k_degree-1))
            ind_max = i - k_degree
            aux_cost = cost_anonimize[ind_min] + compute_i(array_degrees, ind_min, ind_max)
            aux_val = ind_min
            for t in range(ind_min + 1, ind_max):
                tmp_cost = cost_anonimize[t] + compute_i(array_degrees, t, ind_max)
                if tmp_cost < aux_cost:
                    aux_cost = tmp_cost
                    aux_val = t
            cost_anonimize.append(aux_cost)
            aux_list = node_anonimize[aux_val + 1].copy()
            aux_list.append(aux_val)
            node_anonimize.append(aux_list)
    # aggiungo l'ultimo indice che corrisponde alla lunghezza dell'array all'array che contiene gli indici della soluzione ottimale
    final_index = node_anonimize[len(array_degrees) - 1]
    final_index.append(len(array_degrees) - 1)
    final_value = array_degrees.copy()

    # creo l'array anonimizzato
    for i in range(0, k_degree + 1):#va a rimediare al problema del +1 per ind
        final_value[i] = array_degrees[0]
    for i in range(1,len(final_index)-1):
        ind = final_index[i] + 1
        for j in range(final_index[i], final_index[i + 1] + 1):
            final_value[j] = array_degrees[ind]
    return final_value


def vector_norm(a):
    count = 0
    for x in a:
        count += x*x
    math.sqrt(count)
    return count


if __name__ == "__main__":

    k_degree = int(sys.argv[1])
    file_graph = str(sys.argv[2])
    l = 10
    #l = int(sys.argv[3])
    G = nx.Graph()
    if os.path.exists(file_graph):
        # if file exist
        with open(file_graph) as f:
            content = f.readlines()
        # read each line
        content = [x.strip() for x in content]
        for line in content:
            # split name inside each line
            names = line.split(",")
            #names = line.split(" ")#if use real dataset
            start_node = names[0]
            if start_node not in G:
                G.add_node(start_node)
            for index in range(1, len(names)):
                node_to_add = names[index]
                if node_to_add not in G:
                    G.add_node(node_to_add)
                G.add_edge(start_node, node_to_add)

    # Degree arrays preparation
    d = [x[1] for x in G.degree()]
    array_index = np.argsort(d)[::-1]
    array_degrees = np.sort(d)[::-1]



    array_degrees_greedy = array_degrees.copy()
    # greedy
    logger.info("Start compute greedy alghoritm")
    greedy_rec_algorithm(array_degrees_greedy, k_degree, 0, k_degree)

    logger.info("Finish compute greedy alghoritm")

    # dp
    logger.info("Start compute dp alghoritm")
    vertex_degree_dp = dp_graph_anonymization(array_degrees.copy(), k_degree)
    logger.info("Finish compute dp alghoritm")

    """
    print(str(k_degree)+" "+ file_graph)
    print("Array of degrees sorted (array_degrees_greedy) : {}".format(array_degrees_greedy))
    print("Array of dp sorted (vertex_degree_dp) : {}".format(vertex_degree_dp))
    print(str("Number of edges of degree alghoritm:") + str(sum(array_degrees)))
    print(str("Number of edges of dp alghoritm:")+str(sum(vertex_degree_dp)))
    print(str("Number of edges of greedy alghoritm:") + str(sum(array_degrees_greedy)))
    """
    f = open('metric_k_anonymization-compare', 'a')
    f.write(str(k_degree)+" ")
    f.close()
    f = open('metric_norm_anonymization-compare', 'a')
    norm_greedy = vector_norm(array_degrees_greedy-array_degrees)
    norm_dp = vector_norm(vertex_degree_dp-array_degrees)
    r = norm_greedy/norm_dp
    f.write(str(r) + " ")
    f.close()