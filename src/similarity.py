import torch
from sklearn.neighbors import NearestNeighbors


def compute_similarity_matrix(partition, matrix, alpha, reduction="gmean", threshold=0., normalize=True):
    """
    The format of matrix is: (n_samples, features)
    The feature space is "partitioned" according to partition.
    Submat is of shape (n_samples, sub_features), i.e.,submat is the restriction on each subset of the partition.
    """
    submat_list = []
    for subset in partition:
        idx = torch.LongTensor(subset)
        submat_list.append(matrix[:, idx])
    submat_list = torch.stack(submat_list)  # shape: (nb_submat, n_samples, card(subset))
    if normalize:
        submat_list = torch.nn.functional.normalize(submat_list, dim=2, p=2)
    tmp = torch.bmm(submat_list, torch.transpose(submat_list, 1, 2).conj())  # shape: (nb_submat, n_samples, n_samples)
    tmp = torch.abs(tmp)
    if normalize:
        tmp = torch.clamp(torch.abs(tmp), min=0., max=1.)
    tmp = tmp.pow(alpha)
    if reduction == "sum":
        tmp = tmp.sum(dim=0)
    elif reduction == "mean":
        tmp = tmp.mean(dim=0)
    elif reduction == "prod":
        tmp = tmp.prod(dim=0)
    elif reduction == "gmean":
        tmp = torch.exp(torch.mean(torch.log(tmp), dim=0))
    else:
        raise NotImplementedError
    if threshold:
        m = torch.nn.Threshold(threshold, 0)
        return m(tmp)
    return tmp


def keep_k_nonzero_entries_per_col(adjacency, k):
    (m, n) = adjacency.shape
    topk = torch.zeros_like(adjacency)
    for i in range(m):
        vals, idx = adjacency[i].topk(k)
        topk[i][idx] = vals
    return topk


def nearest_neighbors_graph(similarity, n_neighbors):
    X = similarity.cpu().numpy()
    estimator = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed").fit(X)
    connectivity = estimator.kneighbors_graph(X=X, mode="connectivity")
    affinity = 0.5 * (connectivity + connectivity.T)
    affinity = affinity.todense()
    return torch.from_numpy(affinity)