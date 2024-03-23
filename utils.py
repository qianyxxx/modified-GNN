# utils.py
import torch
from scipy.sparse import coo_matrix

# def encode_onehot(labels):
#     classes = set(labels)
#     classes_dict = {c: torch.tensor([i == c for i in classes], dtype=torch.float32) for c in classes}
#     labels_onehot = torch.stack([classes_dict[c] for c in labels], dim=0)
#     return labels_onehot

# def sparse_to_torch_sparse(m):
#     coo = coo_matrix(m)
#     values = coo.data
#     indices = np.vstack((coo.row, coo.col))
#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = coo.shape
#     return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def normalize_adj(adj):
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return adj.matmul(d_mat_inv_sqrt).transpose(0,1).matmul(d_mat_inv_sqrt)