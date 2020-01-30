import os
import numpy as np
import networkx as nx
import sys
import math
from networkx.drawing.tests.test_pylab import plt

from objc._objc import NULL

if __name__ == "__main__":

    """---Upload norm ---"""
    file_graph = str(sys.argv[1])

    if os.path.exists(file_graph):
        # if file exist
        with open(file_graph) as f:
            content = f.readlines()
        # read each line
        content = [x.strip() for x in content]
        """for line in content:
            # split name inside each line
            names = line.split(" ")"""
        norm = NULL
        size = len(content)
        if size == 1:
            for line in content:
                norm = line.split(" ")

            """---Upload k-degree---"""
            file_graph = str(sys.argv[2])

            if os.path.exists(file_graph):
            # if file exist
                with open(file_graph) as f:
                    content = f.readlines()
                # read each line
                content = [x.strip() for x in content]
                k_array = NULL
                size = len(content)
                if size == 1:
                    for line in content:
                        k_array = line.split(" ")


                    """---Upload Clustering Coefficent---"""
                    file_graph = str(sys.argv[3])

                    if os.path.exists(file_graph):
                    # if file exist
                        with open(file_graph) as f:
                            content = f.readlines()
                    # read each line
                        content = [x.strip() for x in content]
                        original_cc = []
                        supergraph_cc = []
                    for line in content:
                        value = line.split(" ")
                        if len(value):
                            supergraph_cc.append(float(value.pop()))
                            original_cc.append(float(value.pop()))

                    plt.clf()
                    plt.plot(k_array,original_cc, 'r--',k_array, supergraph_cc, 'g-')
                    plt.ylabel("Clustering Coefficent")
                    plt.xlabel("k_degree")
                    plt.legend(('Original Graph', 'Supergraph'), loc='lower center', shadow=True)
                    plt.title(str(sys.argv[4]))
                    plt.savefig("metric_cc_web.png") #if choose dataset web
                    #plt.savefig("metric_cc_socfb.png")
                    plt.clf()
                    list_norm = []
                    for i in norm:
                        list_norm.append(float(i))
                    list_k_array = []
                    for i in k_array:
                        list_k_array.append(float(i))
                    plt.plot(list_k_array, list_norm, 'r--')
                    plt.ylabel("Norm of vector a")
                    plt.xlabel("k_degree")
                    plt.title(str(sys.argv[4]))
                    plt.savefig("metric_norm_web.png") #if choose dataset web
                    #plt.savefig("metric_norm_socfb.png") #otherwise
