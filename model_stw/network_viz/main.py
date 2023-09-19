# from pyvis.network import Network
# import networkx as nx

# nx_graph = nx.cycle_graph(10)
# nx_graph.nodes[1]['title'] = 'Number 1'
# nx_graph.nodes[1]['group'] = 1
# nx_graph.nodes[3]['title'] = 'I belong to a different group!'
# nx_graph.nodes[3]['group'] = 10
# nx_graph.add_node(20, size=20, title='couple', group=2)
# nx_graph.add_node(21, size=15, title='couple', group=2)
# nx_graph.add_edge(20, 21, weight=5)
# nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
# nt = Network('500px', '500px')
# #pulates the nodes and edges data structures
# result = nt.from_nx(nx_graph)
# nt.show('nx.html', notebook=False)


from pyvis.network import Network
import pandas as pd

got_net = Network(height="100vh", 
                  width="100%", 
                  bgcolor="#222222", 
                  font_color="white")

# set the physics layout of the network
got_net.barnes_hut()
got_data = pd.read_csv("/projects/genomic-ml/da2343/ml_project_1/model_stw/necromass_bacteria_Spearman_source_target.csv")

sources = got_data['source']
targets = got_data['target']
weights = got_data['weight']

edge_data = zip(sources, targets, weights)

for e in edge_data:
                src = e[0]
                dst = e[1]
                w = e[2]

                got_net.add_node(src, src, title=src)
                got_net.add_node(dst, dst, title=dst)
                got_net.add_edge(src, dst, value=w)

neighbor_map = got_net.get_adj_list()

# add neighbor data to node hover data
for node in got_net.nodes:
                node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
                node["value"] = len(neighbor_map[node["id"]])

got_net.show("gameofthrones.html", notebook=False)