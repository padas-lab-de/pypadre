import pandas as pd
import numpy as np

def create_csv_from_txt_files(edges_path, labels_path, csv_file):
    #Load dataset
    edge_src = []
    edge_tgt = []
    fl = open(edges_path, 'r')

    for line in fl:
        if ' ' in line:
            a = line.strip('\n').split(' ')
        elif '\t' in line:
            a = line.strip('\n').split('\t')
        edge_src.append(int(a[0]))
        edge_tgt.append(int(a[1]))

    fl.close()

    node_id = []
    node_label = []
    fl = open(labels_path, 'r')

    for line in fl:
        if ' ' in line:
            a = line.strip('\n').split(' ')
        elif '\t' in line:
            a = line.strip('\n').split('\t')
        node_id.append(int(a[0]))
        node_label.append(int(a[1]))

    fl.close()
    label_mapping = { id:node_label[i] for i,id in enumerate(node_id)}

    # TODO validate the dataset creation in case of missing label (not all nodes are src_nodes) Temporary fix.
    #In case not all nodes are connected in the graph or in case not all nodes are present in edge_src.
    nc_nodes = list(set(node_id).difference(set(edge_src) | set(edge_tgt)) | set(node_id).difference(set(edge_src)))

    df = pd.DataFrame([edge_src,edge_tgt]).T
    df[2] = df[0].map(label_mapping)


    for node_id in nc_nodes:
        df = df.append({0:node_id,1:-1,2:label_mapping[node_id]}, ignore_index=True)

    df.to_csv(csv_file)


    return df

if __name__=='__main__':

    # df = create_csv_from_txt_files('data/Cora_edgelist.txt','data/Cora_labels.txt','data/Cora.csv')
    df1 = create_csv_from_txt_files('data/Wiki_edgelist.txt', 'data/wiki_labels.txt', 'data/Wikipedia.csv')
    # df2 = create_csv_from_txt_files('data/CiteSeer.edges','data/CiteSeer_labels.txt','data/CiteSeer.csv')