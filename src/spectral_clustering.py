import numpy as np
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from torch_kmeans import ConstrainedKMeans
import torch

from src.similarity import nearest_neighbors_graph


def unnormalized_laplacian(adjacency):
    degree_mat = torch.diag(torch.sum(adjacency, dim=1))
    return degree_mat - adjacency


def spectral_features(adjacency, nb_clusters):
    laplacian = unnormalized_laplacian(adjacency)
    _, eigvecs = torch.linalg.eigh(laplacian)
    return eigvecs[:, 1:nb_clusters+1]


def unnormalized_spectral_clustering(adjacency, nb_clusters):
    features = spectral_features(adjacency, nb_clusters)
    features = features.cpu().numpy()
    km = KMeans(n_clusters=nb_clusters)
    km.fit(features)
    return km.labels_


def constrained_unnormalized_spectral_clustering(adjacency, nb_clusters, size_min, size_max):
    # begin = time.time()
    features = spectral_features(adjacency, nb_clusters)
    # print(time.time() - begin)

    # Implementation cpu only
    features = features.cpu().numpy()
    km = KMeansConstrained(n_clusters=nb_clusters, size_max=size_max)
    km.fit(features)
    labels = km.labels_
    return labels


def _torch_constrained_unnormalized_spectral_clustering(adjacency, nb_clusters, size):
    """ WARNING it does not work """
    # Implementation GPU
    features = spectral_features(adjacency, nb_clusters).unsqueeze(0)
    km = ConstrainedKMeans(n_clusters=nb_clusters, init_method='k-means++', )
    w = torch.ones(features.shape[:-1], device=features.get_device()) / size
    result = km(features, k=nb_clusters, weights=w)
    labels = result.labels
    return labels.cpu().numpy()


def recover_partition_with_spectral_clustering(similarity, nb_clusters, size_cluster=None, nearest_neighbors=None):
    """ get the partition obtained after spectral clustering """
    if nearest_neighbors:
        similarity = nearest_neighbors_graph(similarity, nearest_neighbors)
    if size_cluster is None:
        labels = unnormalized_spectral_clustering(similarity, nb_clusters=nb_clusters)
    else:
        labels = constrained_unnormalized_spectral_clustering(
            similarity,
            nb_clusters=nb_clusters,
            size_min=size_cluster,
            size_max=size_cluster
        )
    recovered_partition = []
    for i in range(nb_clusters):
        idx = np.where(labels == i)[0]
        assert len(idx) == size_cluster
        recovered_partition.append(idx.tolist())
    recovered_partition = sorted(recovered_partition, key=lambda x: x[0])
    assert len(recovered_partition) == nb_clusters
    return recovered_partition
