from einops import rearrange
import torch

from .low_rank import best_rank_one_approximation
from .tree import Node


def dense_to_pre_low_rank_projection(matrix, b1, c1):
    return rearrange(matrix, 'a d (b1 b2) (c1 c2) -> (a c1) (b2 d) b1 c2', b1=b1, c1=c1)


def left_to_twiddle(left, c1):
    return rearrange(left, '(a c1) d b c -> a d b (c c1)', c1=c1)


def right_to_twiddle(right, b2):
    return rearrange(right, 'a (b2 d) b c -> a d (b b2) c', b2=b2)


def tree_bfly_factorization_old(target, tree, level=0, stop_level=None, left_normalization=False):
    """
    Hierarchical factorization of a target tensor using a given tree
    Input:
    target: A tensor of order 4 of size (1, 1, m, n) (if you want to factorize A, please reshape A first)
    tree: balanced or comb trees
    level: the depth of the current factorization
    Output:
    A list of twiddles (factors)
    """
    assert isinstance(tree, Node)
    if stop_level is not None:
        if level >= stop_level:
            return [target]
    if tree.is_leaf():
        return [target]

    # calculate the best approximation
    left_factor, right_factor = best_rank_one_approximation(
        dense_to_pre_low_rank_projection(target, tree.left.core_size[0], tree.left.core_size[1]),
        # left_normalization=left_normalization
    )

    # reshape the factor to normal form
    left_factor = left_to_twiddle(left_factor, tree.left.core_size[1])
    right_factor = right_to_twiddle(right_factor, tree.right.core_size[0])

    # recursion
    left_factorization = tree_bfly_factorization_old(left_factor, tree.left, level=level + 1, stop_level=stop_level)
    right_factorization = tree_bfly_factorization_old(right_factor, tree.right, level=level + 1, stop_level=stop_level)

    # return all the factors
    return left_factorization + right_factorization


def tree_bfly_factorization(target, tree, level=0):
    if tree.is_leaf():
        return [target]
    target = rearrange(target, 'a d (b1 b2) (c1 c2) -> (a c1) (b2 d) b1 c2', b1=tree.left.core_size[0],
                       c1=tree.left.core_size[1])
    # calculate the best approximation
    left_factor, right_factor = low_rank_project(target)

    # reshape the factor to normal form
    left_factor = rearrange(left_factor, '(a c1) d b c -> a d b (c c1)', c1=tree.left.core_size[1])
    right_factor = rearrange(right_factor, 'a (b2 d) b c -> a d (b b2) c', b2=tree.right.core_size[0])

    # recursion
    left_factorization = tree_bfly_factorization(left_factor, tree.left, level=level + 1)
    right_factorization = tree_bfly_factorization(right_factor, tree.right, level=level + 1)

    # return all the factors
    return left_factorization + right_factorization


def low_rank_project(M, rank = 1):
    """Supports batches of matrices as well.
    """
    # time_list = []
    # start = time.time()
    U, S, Vt = torch.linalg.svd(M)
    # time_list.append(time.time() - start)
    # start = time.time()
    S_sqrt = S[..., :rank].sqrt()
    # time_list.append(time.time() - start)
    # start = time.time()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    # time_list.append(time.time() - start)
    # start = time.time()
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    # time_list.append(time.time() - start)
    return U, Vt
