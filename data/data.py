from ogb.graphproppred import PygGraphPropPredDataset
import os
import torch

def download_datasets():
    datasets = ['ogbg-molhiv', 'ogbg-molpcba']
    for dataset_name in datasets:
        print(f"Starting to download {dataset_name} dataset...")
        dataset = PygGraphPropPredDataset(name=dataset_name)
        torch.save(dataset, os.path.join('data', dataset_name + '.pt'))
        print(f"{dataset_name} dataset downloaded successfully.")

def preprocess_datasets(dataset_name):
    print(f"Starting to preprocess {dataset_name} dataset...")
    dataset = torch.load(os.path.join('data', dataset_name + '.pt'))
    data_list = []
    for i in range(len(dataset)):
        data = dataset[i]
        data_list.append(data)
    torch.save(data_list, os.path.join('data', dataset_name + '_preprocessed.pt'))
    print(f"{dataset_name} dataset preprocessed successfully.")

def view_dataset_features():
    datasets = ['ogbg-molhiv', 'ogbg-molpcba']
    for dataset_name in datasets:
        print(f"Loading {dataset_name} dataset...")
        if os.path.exists(os.path.join('data', dataset_name + '_preprocessed.pt')):
            data_list = torch.load(os.path.join('data', dataset_name + '_preprocessed.pt'))
            print(f"Features of the first graph in {dataset_name} dataset:")
            first_graph = data_list[0]
            for key, value in first_graph.items():
                if isinstance(value, int):
                    print(f"{key}: {value}")
        else:
            print(f"The file {dataset_name}_preprocessed.pt does not exist.")

def get_dataset_info():
    datasets = ['ogbg-molhiv', 'ogbg-molpcba']
    for dataset_name in datasets:
        print(f"Loading {dataset_name} dataset...")
        if os.path.exists(os.path.join('data', dataset_name + '.pt')):
            dataset = torch.load(os.path.join('data', dataset_name + '.pt'))
            num_nodes = sum([data.num_nodes for data in dataset])
            num_edges = sum([data.edge_index.shape[1] for data in dataset])
            num_classes = dataset.num_classes
            num_features = dataset.num_features
            print(f"{dataset_name} dataset:")
            print(f"Number of nodes: {num_nodes}")
            print(f"Number of edges: {num_edges}")
            print(f"Number of classes: {num_classes}")
            print(f"Number of features: {num_features}")
        else:
            print(f"The file {dataset_name}.pt does not exist.")

if __name__ == "__main__":
    download_datasets()
    preprocess_datasets()
    view_dataset_features()
    get_dataset_info()