import torch
import numpy as np

import src.debfly.tree
import src.partition
import src.debfly.product

from src.cluster_tree import ClusterTree


def col_permute(matrix, permutation):
    """ permute columns of matrix according to permutation """
    assert len(matrix.shape) == 2
    if not isinstance(permutation, torch.LongTensor):
        permutation = torch.LongTensor(permutation)
    assert len(permutation.shape) == 1
    return matrix[:, permutation]


def row_permute(matrix, permutation):
    """ permute rows of matrix according to permutation """
    assert len(matrix.shape) == 2
    if not isinstance(permutation, torch.LongTensor):
        permutation = torch.LongTensor(permutation)
    assert len(permutation.shape) == 1
    return matrix[permutation, :]


def permutation_to_matrix(permutation):
    """ permutation is a tuple """
    n = len(permutation)
    return col_permute(torch.eye(n), permutation)


def inverse_permutation(permutation):
    """ compute the inverse permutation """
    permutation = np.array(permutation)
    inverse_perm = np.argsort(permutation)
    return tuple(inverse_perm)


def composition(a, b):
    """ permute according to a, then permute according to b """
    if not isinstance(a, torch.LongTensor):
        a = torch.LongTensor(a)
    if not isinstance(b, torch.LongTensor):
        b = torch.LongTensor(b)
    return a[b]


def perm_col_partition_to_monarch_canonical(col_partition, tree):
    """ return permutation to go from col_partition to monarch canonical """
    src.partition.check_col_partition_is_monarch_compatible(col_partition, tree)
    p = torch.LongTensor(col_partition)
    p = p.view(-1)
    return p


def _perm_row_canonical_to_block(diag, row):
    p = torch.LongTensor(src.partition.canonical_row_partition_monarch(diag, row))
    return p.view(-1)


def perm_row_partition_to_monarch_canonical(row_partition, tree):
    """ permutation to go from row_partition to monarch canoncial """
    src.partition.check_row_partition_is_monarch_compatible(row_partition, tree)
    diag = len(row_partition)
    row = len(row_partition[0])

    p1 = torch.LongTensor(row_partition)
    p1 = p1.view(-1)
    perm = composition(p1, inverse_permutation(_perm_row_canonical_to_block(diag, row)))
    return perm


def perm_row_tree_to_butterfly_canonical(row_cluster_tree):
    """ permutation to go from the row cluster tree to the canonical butterfly row cluster tree """
    assert isinstance(row_cluster_tree, ClusterTree)
    size = len(row_cluster_tree.node)
    log_n = int(np.log2(size))
    assert size == 2**log_n
    leaves = np.squeeze(np.array(row_cluster_tree.nodes_at_level(log_n)), axis=1).tolist()

    canonical_tree = src.cluster_tree.even_odd_split_cluster_tree(list(range(size)))
    canonical_leaves = np.squeeze(np.array(canonical_tree.nodes_at_level(log_n)), axis=1).tolist()

    return composition(leaves, inverse_permutation(canonical_leaves))


def perm_col_tree_to_butterfly_canonical(col_cluster_tree):
    """ permutation to go from the column cluster tree to the canonical butterfly column cluster tree """
    assert isinstance(col_cluster_tree, ClusterTree)
    size = len(col_cluster_tree.node)
    log_n = int(np.log2(size))
    assert size == 2 ** log_n
    leaves = np.squeeze(np.array(col_cluster_tree.nodes_at_level(log_n)), axis=1).tolist()

    canonical_tree = src.cluster_tree.middle_split_cluster_tree(list(range(size)))
    canonical_leaves = np.squeeze(np.array(canonical_tree.nodes_at_level(log_n)), axis=1).tolist()

    return composition(leaves, inverse_permutation(canonical_leaves))


def perm_DFT(num_factors):
    result = []
    size = 2 ** num_factors
    for i in range(num_factors):
        if i == 0:
            continue
        for j in range(3):
            z = perm_type(i + 1, j)
            result.append(np.kron(np.identity(2 ** (num_factors - 1 - i)), z))
    return result


def perm_type(i, type):
    """
    Type 0 is c in paper. Type 1 is b in paper. Type 2 is a in paper.
    :param i:
    :param type:
    :return:
    """
    size = 2 ** i
    result = np.zeros((size,size))
    if type == 0:
        result[np.arange(size//2), np.arange(size//2)] = 1
        result[size//2 + np.arange(size//2), size - 1 - np.arange(size//2)] = 1
    elif type == 1:
        result[size // 2 - 1 - np.arange(size//2), np.arange(size//2)] = 1
        result[size // 2 + np.arange(size//2), size//2 + np.arange(size//2)] = 1
    else:
        result[np.arange(size//2), np.arange(size//2) * 2] = 1
        result[size//2 + np.arange(size//2), np.arange(size//2) * 2 + 1] = 1
    return result


def bit_reversal_permutation(log_n):
    perm_mat = src.debfly.product.matrix_product(perm_DFT(log_n))
    return perm_mat


if __name__ == '__main__':
    p = (3, 2, 0, 1)
    mat = torch.randn(4, 4)
    print(mat @ permutation_to_matrix(p) == col_permute(mat, p))
    print(permutation_to_matrix(p).t() @ mat == row_permute(mat, p))
    print(inverse_permutation(p))
    print(col_permute(col_permute(mat, inverse_permutation(p)), p) == mat)
    print(col_permute(col_permute(mat, p), inverse_permutation(p)) == mat)
    print(row_permute(row_permute(mat, inverse_permutation(p)), p) == mat)
    print(row_permute(row_permute(mat, p), inverse_permutation(p)) == mat)

    p1 = (0, 1, 3, 2)
    p2 = (0, 2, 1, 3)
    p1_then_p2 = composition(p1, p2)
    print(p1_then_p2)

    print(col_permute(col_permute(mat, p1), p2) == col_permute(mat, p1_then_p2))
