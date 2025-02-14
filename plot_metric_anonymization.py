import os
import sys

from networkx.drawing.tests.test_pylab import plt
from objc._objc import NULL

if __name__ == "__main__":
    file_graph = str(sys.argv[1])

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

            file_graph = str(sys.argv[2])

            if os.path.exists(file_graph):
                # if file exist
                with open(file_graph) as f:
                    content = f.readlines()
                # read each line
                content = [x.strip() for x in content]
                ratio = NULL
                size = len(content)
                if size == 1:
                    for line in content:
                        ratio = line.split(" ")
                    list_ratio = []
                    for i in ratio:
                        list_ratio.append(float(i))
                    list_k_array = []
                    for i in k_array:
                        list_k_array.append(float(i))
                    plt.clf()
                    plt.figure(figsize=(16, 10))
                    plt.ylabel("Ratio")
                    plt.xlabel("k_degree")
                    plt.plot(list_k_array,list_ratio, 'r--')
                    plt.title("Graph friend 1000 10 100")
                    plt.savefig("ratio_fakedataset.png", dpi=120)
