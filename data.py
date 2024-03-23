from torch_geometric.datasets import Planetoid
import os
import torch
from torch_geometric.utils import to_dense_adj, degree
# from torch.diag import diag_embed

def download_datasets():
    dataset_name = 'Cora'
    print(f"Starting to download {dataset_name} dataset...")
    dataset = Planetoid(root='./data', name=dataset_name)
    torch.save(dataset, os.path.join('./data', dataset_name + '.pt'))
    print(f"{dataset_name} dataset downloaded successfully.")

def preprocess_datasets():
    dataset_name = 'Cora'
    dataset = Planetoid(root='./data', name=dataset_name)
    data_list = []
    for i in range(len(dataset)):
        data = dataset[i]
        data.A = to_dense_adj(data.edge_index).squeeze(0)  # 邻接矩阵
        data.H = data.x.float()  # 节点特征
        degree_vector = degree(data.edge_index[0])  # 度向量
        data.D = torch.diag(degree_vector)  # 度矩阵
        data_list.append(data)

    torch.save(data_list, os.path.join('./data', dataset_name + '_preprocessed.pt'))
    print(f"{dataset_name} dataset preprocessed successfully.")
    print(data.A.shape, data.H.shape, data.D.shape)
    print(f"Adjacency Martirx {data.A}")
    print(f"Node Feature {data.H}")
    print(f"Degree Martirx {data.D}")

if __name__ == "__main__":
    download_datasets()
    preprocess_datasets()