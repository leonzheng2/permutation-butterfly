import random

import numpy as np

import src.cluster_tree
from src.debfly.tree import get_monarch_sizes_from_tree


def _sample_with_remove(seq, n_sample):
    sample = [seq.pop(random.randrange(len(seq))) for _ in range(n_sample)]
    return seq, sorted(sample)


def sample_partition(n_subset, card):
    """ sample randomly a partition with a given number of subsets of a given cardinality """
    n = n_subset * card
    partition = []
    seq = list(range(n))
    for i in range(n_subset):
        seq, subset = _sample_with_remove(seq, card)
        partition.append(subset)
    return sorted(partition, key=lambda x: x[0])


def canonical_row_partition_monarch(diag, row):
    """ diag = supscript[1] and row = subscript[0] of left factor in monarch factorization """
    partition = [list(diag * np.arange(row) + i) for i in range(diag)]
    return partition


def canonical_col_partition_monarch(nb_blocks, col):
    partition = [list(np.arange(col) + i * col) for i in range(nb_blocks)]
    return partition


def check_col_partition_is_monarch_compatible(col_partition, tree):
    """ check that the column partition is compatible for the considered monarch problem """
    output_size, middle_size, input_size, nb_blocks = get_monarch_sizes_from_tree(tree)
    col_partition_np = np.array(col_partition)
    assert col_partition_np.shape[0] == nb_blocks
    assert col_partition_np.shape[1] == input_size // nb_blocks


def check_row_partition_is_monarch_compatible(row_partition, tree):
    """ check that the row partition is compatible for the considered monarch problem """
    output_size, middle_size, input_size, nb_blocks = get_monarch_sizes_from_tree(tree)
    row_partition_np = np.array(row_partition)
    assert row_partition_np.shape[0] == middle_size // nb_blocks
    assert row_partition_np.shape[1] == nb_blocks * output_size // middle_size


def get_butterfly_partitioning(row_cluster_tree, col_cluster_tree):
    """ get the partitions that give low rank blocks in a butterfly matrix (without permutations) """
    assert isinstance(row_cluster_tree, src.cluster_tree.ClusterTree)
    assert isinstance(col_cluster_tree, src.cluster_tree.ClusterTree)
    assert len(row_cluster_tree.node) == len(col_cluster_tree.node)
    size = len(row_cluster_tree.node)
    log_n = int(np.log2(size))
    assert 2**log_n == size
    list_matrix_partitioning = []
    for ell in range(1, log_n):
        row_partition = row_cluster_tree.nodes_at_level(log_n - ell)
        row_partition = sorted(row_partition, key=lambda x: min(x))
        col_partition = col_cluster_tree.nodes_at_level(ell)
        col_partition = sorted(col_partition, key=lambda x: min(x))
        list_matrix_partitioning.append([row_partition, col_partition])
    return list_matrix_partitioning


def partition_is_valid(partition):
    """ check that a partition is valid """
    for subset in partition:
        if len(subset) != len(partition[0]):
            return False
    size = len(partition) * len(partition[0])
    return set(range(size)) == set().union(*[set(subset) for subset in partition])


if __name__ == '__main__':
    print(partition_is_valid(sample_partition(2, 4)))

    print(partition_is_valid([[1, 2], [2, 3]]))
    print(partition_is_valid([[1, 2], [0, 3]]))
    print(partition_is_valid([[1, 2], [3]]))
