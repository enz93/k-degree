### k-degree anonymity
### Install requirements
```
pip install -r requirements.txt
```
### Usage
```
python k-degree-anonymization.py k_value graph_to_anonymize
python plot_metrics_supergraph.py file_of_array_of_norm file_of_array_of_k-degree_of_supergraph file_of_array_of_clustering_coefficent name_of_dataset
python supergraph-dimonstration.py k-degree
pyhton degree-anonymization-compare.py k-degree file_of_graph_to_anonymize
python plot_metric_anonymization.py file_of_array_of_k-degree_of_degree_anonymization file_of_array_ratio_of_degree_anonymization
```
#### Example
```
python k-degree-anonymization.py 5 Dataset/positivetesting/web-BerkStan-dir/web-BerkStan
python plot_metrics_supergraph.py Metrics/metric_norm_socfb Metrics/metric_k_socfb Metrics/metric_cc_socfb "Socfb-Rice31"
python degree-anonymization-compare.py 10 Dataset/trialdataset/graph_friend_1000_10_100.csv (8 is the max number of degree for an optimal computation of greedy alghoritm with graph_friend_1000_10_100.csv dataset. With real dataset doesn't work degree alghoritm)
```
### Dataset
```
we use different dataset for testing suoergraph alghoritm:
-Zachary karate club network(inside networkx library) - The dataset contains social ties among the members of a university karate club collected by Wayne Zachary in 1977. (Usata per mostrare un immagine dei risultati del programma)
-BERKSTAN-DIR. Hyperlink network of the websites of Berkley and Stanford University. Nodes represent web pages and directed edges represent hyperlinks.(Real Dataset)
-RICE31. A social friendship network extracted from Facebook consisting of people (nodes) with edges representing friendship ties(Real Dataset)
These datasets allow to obtain a positive results.
-MANN-a27 and P-HAT1500-3. supergraph alghoritm always returns null because the condition is not satisfied.
```
### Hints && Our Testing
```
1)Degree Anonymization 

python degree-anonymization-compare.py 3 Dataset/trialdataset/graph_friend_1000_10_100.csv
......
python degree-anonymization-compare.py 8 Dataset/trialdataset/graph_friend_1000_10_100.csv

After this execution, we running

python plot_metric_anonymization.py Metrics/metric_k_anonymization-compare Metrics/metric_norm_anonymization-compare

2)Supergraph Alghoritm

Dataset with high number of nodes and low number of edges

python k-degree-anonymization.py 5 Dataset/positivetesting/web-BerkStan-dir/web-BerkStan
python k-degree-anonymization.py 7 Dataset/positivetesting/web-BerkStan-dir/web-BerkStan
                                27 Dataset/positivetesting/web-BerkStan-dir/web-BerkStan
                                39 Dataset/positivetesting/web-BerkStan-dir/web-BerkStan
                                43 Dataset/positivetesting/web-BerkStan-dir/web-BerkStan
                                47 Dataset/positivetesting/web-BerkStan-dir/web-BerkStan
                                49 Dataset/positivetesting/web-BerkStan-dir/web-BerkStan
                                55 Dataset/positivetesting/web-BerkStan-dir/web-BerkStan
                                57 Dataset/positivetesting/web-BerkStan-dir/web-BerkStan
 with 9,10,11,14,15,16,17,18,20 … 26, 28 … 38 Dataset/web-BerkStan-dir/web-BerkStan number of edges is odd
13 Dataset/web-BerkStan-dir/web-BerkStan NULL

*

--------------

Dataset with low number of nodes and high number of edges


python k-degree-anonymization.py 13 Dataset/positivetesting/socfb-Rice31/socfb-Rice31.mtx
python k-degree-anonymization.py 17 Dataset/positivetesting/socfb-Rice31/socfb-Rice31.mtx
                                19 Dataset/positivetesting/socfb-Rice31/socfb-Rice31.mtx
                                23 Dataset/positivetesting/socfb-Rice31/socfb-Rice31.mtx
                                25 Dataset/positivetesting/socfb-Rice31/socfb-Rice31.mtx
                                27 Dataset/positivetesting/socfb-Rice31/socfb-Rice31.mtx
                                31 Dataset/positivetesting/socfb-Rice31/socfb-Rice31.mtx
                                37 Dataset/positivetesting/socfb-Rice31/socfb-Rice31.mtx
                                39 Dataset/positivetesting/socfb-Rice31/socfb-Rice31.mtx
5 … 12,14…16 Dataset/socfb-Rice31/socfb-Rice31.mtx number of edges is odd

*

*)After these execution, we running
plot_metrcis_supergraph.py Metrics/metric_norm_socfb Metrics/metric_k_socfb Metrics/metric_cc_socfb "Socfb-Rice31" or with "_web"
```