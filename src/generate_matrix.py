import torch
import numpy as np

from src.debfly.product import matrix_product
from src.debfly.support import bfly_supp
from src.debfly.tree import Node, square_pot_butterfly_tree
from src.debfly.product import twiddle_list_to_dense

from scipy.stats import ortho_group, unitary_group


def _fixed_supp_random_matrix(supp, mean=1, std=0.5):
    m = torch.distributions.bernoulli.Bernoulli(probs=0.5 * torch.ones_like(supp))
    supp = supp * (m.sample() * 2 - 1)
    return supp * (torch.randn(supp.shape[0], supp.shape[1]) * std + mean)


def random_bfly_matrix(log_n, mean=1, std=0.5):
    """generate a butterfly matrix where the butterfly factors have random entries"""
    list_supp = [bfly_supp(i, i, log_n) for i in range(1, log_n + 1)]
    list_fs_random_mat = [
        _fixed_supp_random_matrix(supp, mean, std) for supp in list_supp
    ]
    return matrix_product(list_fs_random_mat)


def random_parker_butterfly(log_n, device="cpu", norm=1.0, complex_value=False):
    """generate a random orthogonal butterfly matrix"""
    tree = square_pot_butterfly_tree(log_n, "unbalanced")
    supscript_list, subscript_list = tree.supscript_subscript_of_leaves()
    t_list = []
    for supscript, subscript in zip(supscript_list, subscript_list):
        assert subscript[0] == subscript[1] == 2
        if not complex_value:
            random_orth = norm * ortho_group.rvs(2, supscript[0] * supscript[1])
            twiddle = torch.from_numpy(random_orth).to(device)
        else:
            random_unit = norm * unitary_group.rvs(2, supscript[0] * supscript[1])
            twiddle = torch.from_numpy(random_unit).to(device)
        twiddle = twiddle.reshape(
            supscript[0], supscript[1], subscript[0], subscript[1]
        )
        t_list.append(twiddle)
    return twiddle_list_to_dense(t_list).squeeze()


def random_debfly_matrix(tree, device="cpu", complex_value=False):
    """compute a random butterfly matrix that admits a factorization described by tree"""
    assert isinstance(tree, Node)
    supscript_list, subscript_list = tree.supscript_subscript_of_leaves()
    t_list = []
    for supscript, subscript in zip(supscript_list, subscript_list):
        if complex_value:
            twiddle = torch.randn(
                supscript[0],
                supscript[1],
                subscript[0],
                subscript[1],
                device=device,
                dtype=torch.complex128,
            )
            twiddle += 1.0j * torch.randn(
                supscript[0], supscript[1], subscript[0], subscript[1], device=device
            )
        else:
            twiddle = torch.randn(
                supscript[0],
                supscript[1],
                subscript[0],
                subscript[1],
                device=device,
                dtype=torch.float64,
            )
        t_list.append(twiddle)
    return twiddle_list_to_dense(t_list).squeeze()


def rank_one_family_partitioning(partitioning, dim):
    """
    Generate a family of rank-one vectors. The family is partitioned according to partitioning.
    In each subset of the partition, vectors are in the same span.
    Vectors are of dimension dim.
    """
    n_subsets = len(partitioning)
    card = len(partitioning[0])
    all_vectors = torch.empty(n_subsets * card, dim)
    for subset in partitioning:
        assert len(subset) == card
        random_vectors = torch.outer(torch.randn(card), torch.randn(dim))
        idx = torch.LongTensor(subset)
        all_vectors[idx, :] = random_vectors
    return all_vectors


def random_haar_butterfly(log_n):
    if log_n == 0:
        return torch.tensor([[1.0]])
    theta = 2 * np.pi * np.random.rand()
    rotation = torch.tensor(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    return torch.kron(rotation, random_haar_butterfly(log_n - 1))


def concat_2_by_2_blocks(blocks):
    return torch.cat(
        [
            torch.cat([blocks[0][0], blocks[0][1]], dim=1),
            torch.cat([blocks[1][0], blocks[1][1]], dim=1),
        ],
        dim=0,
    )


def random_diagonal_butterfly(log_n, simple=False):
    if log_n == 0:
        return torch.tensor([[1.0]])
    n = 2**log_n
    theta = torch.tensor(2 * np.pi * np.random.rand(n // 2), dtype=torch.float)
    if simple:
        butterfly = random_diagonal_butterfly(log_n - 1, simple=simple)
        return concat_2_by_2_blocks(
            [
                [
                    torch.cos(theta).unsqueeze(1) * butterfly,
                    torch.sin(theta).unsqueeze(1) * butterfly,
                ],
                [
                    -torch.sin(theta).unsqueeze(1) * butterfly,
                    torch.cos(theta).unsqueeze(1) * butterfly,
                ],
            ]
        )
    butterfly_1 = random_diagonal_butterfly(log_n - 1, simple=simple)
    butterfly_2 = random_diagonal_butterfly(log_n - 1, simple=simple)
    return concat_2_by_2_blocks(
        [
            [
                torch.cos(theta).unsqueeze(1) * butterfly_1,
                torch.sin(theta).unsqueeze(1) * butterfly_2,
            ],
            [
                -torch.sin(theta).unsqueeze(1) * butterfly_1,
                torch.cos(theta).unsqueeze(1) * butterfly_2,
            ],
        ]
    )


if __name__ == "__main__":
    n = 3
    a = random_parker_butterfly(n)
    print((a.t() @ a - torch.eye(2**n)).sum())
    print((a @ a.t() - torch.eye(2**n)).sum())
