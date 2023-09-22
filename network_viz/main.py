from pyvis.network import Network
import pandas as pd
import random


def save_network(stw_df, otu_df, network_name):
    net = Network(height="80vh", 
                    width="100%", 
                    # bgcolor="#222222", 
                    # font_color="white",
                    # font_color="black",
                    select_menu=True,
                    filter_menu = True,
                    cdn_resources = "in_line",
                    )

    # set the physics layout of the network
    net.barnes_hut()

    sources = stw_df['source']
    targets = stw_df['target']
    weights = stw_df['weight']

    edge_data = zip(sources, targets, weights)

    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]
        
        is_src_index_bacteria = src <= 18
        is_dst_index_bacteria = dst <= 18 
        
        # find the name of column from otu_df with index src
        src = otu_df.columns[src]
        dst = otu_df.columns[dst]
        
        # random_color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        # net.add_node(src, src, title=src, color= random_color, size=50)
        # net.add_node(src, src, title=src, color= random_color, size=50)

        if is_src_index_bacteria:
            net.add_node(src, src, title=src, color= "indigo", size=50)
        else:
            net.add_node(src, src, title=src, color= "red", size=50)
            
        if is_dst_index_bacteria:
            net.add_node(dst, dst, title=dst, color= "indigo", size=50)
        else:
            net.add_node(dst, dst, title=dst, color= "red", size=50)
        
        
        if w > 0:
            # net.add_edge(src, dst, value=w , color="#DC8858", weight=abs(w))
            net.add_edge(src, dst, value=w , color="#DC8858", weight=abs(w))
        else:
            net.add_edge(src, dst, value=w , color="#5170AE", weight=abs(w))
        
    neighbor_map = net.get_adj_list()

    # add neighbor data to node hover data
    for node in net.nodes:
        # node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])

    net.show_buttons(filter_=['physics'])
    # rewrite show buttons and change the position of the buttons
    net.show(f"{network_name}.html", notebook=False)
    

stw_otu_list = [
    {
    "network_name" : "necromass_bacteria_fungi_conservative_Spearman_network",
    "stw_path": "necromass_bacteria_fungi_conservative_Spearman_source_target.csv",
    "otu_path": "bacteria_fungi_conservative_power_transformed.csv"    
    },
    {
    "network_name" : "necromass_bacteria_fungi_conservative_Pearson_network",
    "stw_path": "necromass_bacteria_fungi_conservative_Pearson_source_target.csv",
    "otu_path": "bacteria_fungi_conservative_power_transformed.csv"    
    },
    {
    "network_name" : "necromass_bacteria_fungi_conservative_LASSO_network",
    "stw_path": "necromass_bacteria_fungi_conservative_LASSO_source_target.csv",
    "otu_path": "bacteria_fungi_conservative_power_transformed.csv"    
    },
    {
    "network_name" : "necromass_bacteria_fungi_conservative_GGM_network",
    "stw_path": "necromass_bacteria_fungi_conservative_GGM_source_target.csv",
    "otu_path": "bacteria_fungi_conservative_power_transformed.csv"    
    },
    
    # # bacteria only
    # {
    # "network_name" : "necromass_bacteria_conservative_Spearman_network",
    # "stw_path": "necromass_bacteria_conservative_Spearman_source_target.csv",
    # "otu_path": "bacteria_conservative_power_transformed.csv"    
    # },
    # {
    # "network_name" : "necromass_bacteria_conservative_Pearson_network",
    # "stw_path": "necromass_bacteria_conservative_Pearson_source_target.csv",
    # "otu_path": "bacteria_conservative_power_transformed.csv"    
    # },
    # {
    # "network_name" : "necromass_bacteria_conservative_LASSO_network",
    # "stw_path": "necromass_bacteria_conservative_LASSO_source_target.csv",
    # "otu_path": "bacteria_conservative_power_transformed.csv"    
    # },
    # {
    # "network_name" : "necromass_bacteria_conservative_GGM_network",
    # "stw_path": "necromass_bacteria_conservative_GGM_source_target.csv",
    # "otu_path": "bacteria_conservative_power_transformed.csv"    
    # },
    
    # # fungi only
    # {
    # "network_name" : "necromass_fungi_conservative_Spearman_network",
    # "stw_path": "necromass_fungi_conservative_Spearman_source_target.csv",
    # "otu_path": "fungi_conservative_power_transformed.csv"    
    # },
    # {
    # "network_name" : "necromass_fungi_conservative_Pearson_network",
    # "stw_path": "necromass_fungi_conservative_Pearson_source_target.csv",
    # "otu_path": "fungi_conservative_power_transformed.csv"    
    # },
    # {
    # "network_name" : "necromass_fungi_conservative_LASSO_network",
    # "stw_path": "necromass_fungi_conservative_LASSO_source_target.csv",
    # "otu_path": "fungi_conservative_power_transformed.csv"    
    # },
    # {
    # "network_name" : "necromass_fungi_conservative_GGM_network",
    # "stw_path": "necromass_fungi_conservative_GGM_source_target.csv",
    # "otu_path": "fungi_conservative_power_transformed.csv"    
    # }
]

for stw_otu in stw_otu_list:
    stw_df = pd.read_csv(f"/projects/genomic-ml/da2343/ml_project_1/model_stw/{stw_otu['stw_path']}")
    otu_df = pd.read_csv(f"/projects/genomic-ml/da2343/ml_project_1/data/necromass/{stw_otu['otu_path']}")
    
    save_network(stw_df, otu_df, stw_otu["network_name"])
    
    
    
