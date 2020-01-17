import numpy as np
import networkx as nx
import sys
from networkx.drawing.tests.test_pylab import plt


def compute_i(d, start, end):
    d_i = d[0]
    res = 0
    for i in range(start, end + 1):
        res += d_i - d[i]
    return res

def count_edge_from_Vl(G, tmp, Vl):
    count = 0
    for i in range(0, len(Vl)):
        if (G.has_edge(tmp, Vl[i])):
            count += 1
    return count


def supergraph(G, array_index, array_degrees, array_degrees_anonymized, l):
    # diff between d~ e d
    graph = G.copy()
    a = array_degrees_anonymized - array_degrees
    # after the creation of vector, must control if it's even otherwise return none
    if sum(a) % 2 == 1:
        return None
    # sort the nodes based on the already ordered vector
    a, array_index = map(list, zip(*sorted(zip(a, array_index), key=lambda x: x[0])))
    # take the l node with largest a(i) values
    Vl = (array_index[-l:])[::-1]
    # consider the l largest a(i) values
    a_Vl = (a[-l:])[::-1]  # (need of this for first sum in the preconditon
    # First Sum
    sum_a_Vl = sum(a_Vl)
    # Second Sum
    sum_Vl = 0
    for i in range(0, len(Vl)):
        tmp = Vl[i]
        count = count_edge_from_Vl(G, tmp, Vl)
        sum_Vl += len(Vl) - 1 - count
        # print(sum_Vl)
    # Third Sum
    sum_VminusVl = 0
    for i in range(0, len(array_index) - l + 1):
        tmp = array_index[i]
        count = count_edge_from_Vl(G, tmp, Vl)
        sum_VminusVl += min(len(Vl) - count, a[i])
    # Check if difference is satisfied
    if sum_a_Vl > sum_Vl + sum_VminusVl:
        return None
    a = a[::-1]
    array_index = array_index[::-1]
    c = 0  # conta gli archi aggiunti. Se sono pari alla meta della somma del vettore a significa che abbiamo aggiunto gli archi giusti
    sum_a = sum(a)
    while True:
        if all(di == 0 for di in a):
            if c == sum_a / 2:
                return graph
            else:
                return None
        v = np.random.choice((np.where(np.array(a) > 0))[0])
        dv = a[v]
        a[v] = 0
        for ind in np.array((np.where(np.array(a) > 0))[0]):
            if not graph.has_edge(array_index[v], array_index[ind]) and a[ind] > 0 and dv > 0 and ind != v:
                graph.add_edge(array_index[v], array_index[ind])
                a[ind] -= 1
                dv -= 1
                c += 1


def construct_graph(tab_index, anonymized_degree):
    graph = nx.Graph()
    if sum(anonymized_degree) % 2 == 1:
        return None

    while True:
        if not all(di >= 0 for di in anonymized_degree):
            return None
        if all(di == 0 for di in anonymized_degree):
            return graph

        v = np.random.choice((np.where(np.array(anonymized_degree) > 0))[0])
        dv = anonymized_degree[v]
        anonymized_degree[v] = 0
        for index in np.argsort(anonymized_degree)[-dv:][::-1]:
            if index == v:
                return None
            if not graph.has_edge(tab_index[v], tab_index[index]):
                graph.add_edge(tab_index[v], tab_index[index])
                anonymized_degree[index] = anonymized_degree[index] - 1


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
            ind_min = (max(k_degree, i - 2 * k_degree)) - 1
            ind_max = i - k_degree
            aux_cost = cost_anonimize[ind_min] + compute_i(array_degrees, ind_min, ind_max)
            aux_val = ind_min
            for t in range(ind_min, ind_max):
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
    for i in range(0, k_degree + 1):
        final_value[i] = array_degrees[0]
    for i in range(1, len(final_index) - 1):
        ind = final_index[i] + 1
        for j in range(final_index[i] + 1, final_index[i + 1] + 1):
            final_value[j] = array_degrees[ind]
    # final_value[k_degree - 1] = final_value[0]
    return final_value


if __name__ == "__main__":
    k_degree = int(sys.argv[1])
    l = 50
    """---------------------------------"""
    G = nx.karate_club_graph()  # this graph is inside the library 'networkx'
    edges_color_G = np.repeat('b', len(G.edges))
    nx.draw_circular(G, with_labels=True, edge_color=edges_color_G)
    plt.savefig("karate.png")
    d = [x[1] for x in G.degree()]
    array_index = np.argsort(d)[::-1]
    array_degree = np.sort(d)[::-1]
    """---------------------------------"""
    # compute the greedy alghoritm for k-degree
    vertex_degree_dp = dp_graph_anonymization(array_degree.copy(), k_degree)
    """---------------------------------"""
    # compute the construct graph
    graph_degree = construct_graph(array_index.copy(), vertex_degree_dp.copy())

    if graph_degree is not None:
        edges_color_construct = np.repeat('b', len(graph_degree.edges))
        plt.clf()
        nx.draw_circular(graph_degree, with_labels=True, edge_color=edges_color_construct)
        plt.savefig("karate_construct.png")

    # compute the supergraph
    superg = supergraph(G.copy(), array_index.copy(), array_degree.copy(), vertex_degree_dp.copy(), l)
    if superg is not None:
        edges_color_superg = np.repeat('r', len(superg.edges))
        for i, edge in enumerate(superg.edges):
            if edge in G.edges:
                edges_color_superg[i] = 'b'
        plt.clf()
        nx.draw_circular(superg, with_labels=True, edge_color=edges_color_superg)
        plt.savefig("karate_superg.png")

