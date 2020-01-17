import os
import numpy as np
import networkx as nx
import sys
import math

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

    graph = G.copy()
    a = array_degrees_anonymized - array_degrees
    # after the creation of vector, must control if it's even otherwise return none
    if sum(a) % 2 == 1:
        print("number of edges is odd")
        return None
    # sort the nodes based on the already ordered vector
    a, array_index = map(list, zip(*sorted(zip(a, array_index), key=lambda x: x[0])))
    # take the l node with largest a(i) values
    Vl = (array_index[-l:])[::-1]
    # consider the l largest a(i) values
    a_Vl = (a[-l:])[::-1]  # (need of this for first sum in the preconditon) and metrics
    # First Sum
    sum_a_Vl = sum(a_Vl)
    # Second Sum
    sum_Vl = 0
    for i in range(0, len(Vl)):
        tmp = Vl[i]
        count = count_edge_from_Vl(G, tmp, Vl)
        sum_Vl += len(Vl) - 1 - count
    # Third Sum
    sum_VminusVl = 0
    for i in range(0, len(array_index) - l + 1):
        tmp = array_index[i]
        count = count_edge_from_Vl(G, tmp, Vl)
        sum_VminusVl += min(len(Vl) - count, a[i])
    # Check if difference is satisfied
    if sum_a_Vl > sum_Vl + sum_VminusVl:
        print("condition for supergraph not satisfied")
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
                print("c not equal to sum_a/2")
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
    return final_value


def create_file_for_adiacent(g, s):
    f = open(s, 'w')
    for nbr, datadict in g.adj.items():
        f.write("Node: " + str(nbr) + "\n")
        c = 0
        f.write("Edge with: ")
        for x in datadict:
            c += 1
            f.write(str(x) + ", ")
        f.write("\n")
        f.write("Count edge: " + str(c) + "\n")
    d = [x[1] for x in g.degree()]
    array_degrees = np.sort(d)[::-1]
    array_i = np.argsort(d)[::-1]
    #f.write("Array of node : {} \n".format(array_i))
    for i in array_degrees:
        f.write(str(i)+ " ")
    f.close()


def create_subgraph_of_node_with_value_different_to_zero_in_vector_a(a, array_of_index, init_graph, superg_graph):
    a, array_of_index = map(list, zip(*sorted(zip(a, array_of_index), key=lambda x: x[0])))
    c = 0
    for i in a:
        if i > 0:
            c += 1
    list_node = array_of_index[-c::][::-1]
    subg = init_graph.subgraph(list_node)
    edges_color_subg = np.repeat('b', len(subg.edges))
    plt.clf()
    nx.draw_circular(subg, with_labels=True, edge_color=edges_color_subg)
    plt.savefig("subgraph_init.png")  # sottografo del grafo iniziale contenete i nodi con a[i] > 0

    subg_of_superg = superg_graph.subgraph(list_node)
    edges_color_subg_of_superg = np.repeat('r', len(subg_of_superg.edges))
    for i, edge in enumerate(subg_of_superg.edges):
        if edge in subg.edges:
            edges_color_subg_of_superg[i] = 'b'
    plt.clf()
    nx.draw_circular(subg_of_superg, with_labels=True, edge_color=edges_color_subg_of_superg)
    plt.savefig("subgraph_supergraph.png")


def vector_norm(a):
    count = 0
    for x in a:
        count += x*x
    math.sqrt(count)
    return count



if __name__ == "__main__":

    k_degree = int(sys.argv[1])
    file_graph = str(sys.argv[2])
    l = 50

    G = nx.Graph()

    if os.path.exists(file_graph):
        # if file exist
        with open(file_graph) as f:
            content = f.readlines()
        # read each line
        content = [x.strip() for x in content]
        for line in content:
            # split name inside each line
            names = line.split(" ")
            start_node = names[0]
            if start_node not in G:
                G.add_node(start_node)
            for index in range(1, len(names)):
                node_to_add = names[index]
                if node_to_add not in G:
                    G.add_node(node_to_add)
                G.add_edge(start_node, node_to_add)
        cc = []
        apl = []
        cc.append(nx.average_clustering(G))
        # Degree arrays preparation
        d = [x[1] for x in G.degree()]
        array_index = np.argsort(d)[::-1]
        array_degrees = np.sort(d)[::-1]
        print(len(G.nodes))
        """--------------Anonymization Phase---------------"""
        #dp anonymization
        vertex_degree_dp = dp_graph_anonymization(array_degrees.copy(), k_degree)
        print(sum(vertex_degree_dp))
        vertex_degree_dp_for_supergraph = vertex_degree_dp.copy()
        """-------------Construct Graph Phase----------------"""
        #
        """-----------------------------"""
        """
        graph_dp = construct_graph(array_index, vertex_degree_dp.copy)
        if graph_dp is not None:
            print("construct dp")
            d = [x[1] for x in graph_dp.degree()]
            array_degrees_dp = np.sort(d)[::-1]
            print(sum(array_degrees_dp))
            #cc[y] = nx.average_clustering(graph_dp) apl[y] = nx.average_shortest_path_length(graph_dp) y += 1 for metrics
        """
        """-----------------------------"""
        # Mapping of initial Graph
        dic = dict()
        k = 0
        for i in G.nodes():
            dic[i] = k
            k += 1
        g = nx.Graph()
        for i, j in G.edges():
            if not g.has_edge(dic[i], dic[j]):
                g.add_edge(dic[i], dic[j])

        """-------------Supergraph Phase----------------"""

        superg = supergraph(g.copy(), array_index, array_degrees, vertex_degree_dp_for_supergraph.copy(), l)
        if superg is not None:
            print("superg dp")
            degree_super = [x[1] for x in superg.degree()]
            array_degrees_sup = np.sort(degree_super)[::-1]
            #print("Array of degrees sorted (array_degrees) : {}".format(array_degrees_sup))
            print(sum(array_degrees_sup))
            #create_file_for_adiacent(superg, './supergraph_dp')
            print(nx.average_clustering(superg))
            #create_subgraph_of_node_with_value_different_to_zero_in_vector_a(vertex_degree_dp_for_supergraph - array_degrees, array_index.copy(), g, superg)
            cc.append(nx.average_clustering(superg))
            vect = vertex_degree_dp - array_degrees
            norm = vector_norm(vect)
            """f = open('metric_norm_trial1000','a')
            f.write(str(norm)+" ")
            f.close()
            f = open('metric_cc_trial1000','a')
            for x in cc:
                f.write(str(x)+" ")
            f.write("\n")
            f.close()
            f = open('metric_k_trial1000','a')
            f.write(str(k_degree)+" ")
            f.close()
            #f = open('metrics_apl_web','a')
            #f.write(str(apl)+"\n")
            #f.close()"""

