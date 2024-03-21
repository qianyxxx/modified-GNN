import torch
from torch_geometric.data import DataLoader
from model import GCN
from data.data import preprocess_datasets

def train():
    datasets = ['ogbg-molhiv', 'ogbg-molpcba']
    for dataset_name in datasets:
        print(f"Loading {dataset_name} dataset...")
        data_list = preprocess_datasets(dataset_name)
        train_loader = DataLoader(data_list, batch_size=32, shuffle=True)

        num_features = data_list[0].num_features
        num_classes = len(set([data.y for data in data_list]))

        model = GCN(num_features, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        for epoch in range(200):
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data)
                loss = torch.nn.functional.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()