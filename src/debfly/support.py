import numpy as np
import torch


def partial_prod_deformable_butterfly_supports(supscript, subscript, low, high):
    """
    Closed form expression of partial matrix_product of butterfly supports. We name S_L, ..., S_1 the butterfly supports of
    size 2^L, represented as binary matrices. Then, the method computes the partial matrix_product S_{high-1} ... S_low.
    :param supscript: list of sizes of factors
    :param subscript: list of sizes of blocks
    :param low: int
    :param high: int, excluded
    :return: numpy array, binary matrix
    """
    size_id_begin = supscript[low][0]
    size_id_end = supscript[high-1][1]
    size_one_middle_h = 1
    size_one_middle_w = 1
    for i in range(low, high):
        b, c = subscript[i]
        size_one_middle_h *= b
        size_one_middle_w *= c
    return np.kron(np.eye(size_id_begin), np.kron(np.ones((size_one_middle_h, size_one_middle_w)), np.eye(size_id_end)))


def bfly_supp(p, q, J):
    """ Get binary matrices describing the supports of butterfly factors """
    assert 1 <= p <= q <= J
    size = 2**(q-p+1)
    V_p_q = torch.kron(torch.ones((size, size)), torch.eye(2**(J-q)))
    return torch.kron(torch.eye(2**(p-1)), V_p_q)