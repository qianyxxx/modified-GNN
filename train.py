# train.py
import torch
from model import GCN
from utils import normalize_adj
import torch.nn.functional as F

def train():
    dataset = torch.load('./data/Cora_preprocessed.pt')
    data = dataset[0]
    adj = normalize_adj(data.A)
    features = data.H
    labels = data.y
    idx_train = data.train_mask
    idx_val = data.val_mask
    idx_test = data.test_mask

    model = GCN(nfeat=features.shape[1],
                nhid=16,
                nclass=labels.max().item() + 1,
                dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            acc_val = accuracy(output[idx_val], labels[idx_val])
        print(f'Epoch: {epoch+1}, Loss: {loss_train.item()}, Validation Accuracy: {acc_val.item()}')

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

def test():
    dataset = torch.load('./data/Cora_preprocessed.pt')
    data = dataset[0]
    adj = normalize_adj(data.A)
    features = data.H
    labels = data.y
    idx_test = data.test_mask

    model = GCN(nfeat=features.shape[1],
                nhid=16,
                nclass=labels.max().item() + 1,
                dropout=0.5)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
    print(f'Test Accuracy: {acc_test.item()}')

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

if __name__ == "__main__":
    train()
    test()