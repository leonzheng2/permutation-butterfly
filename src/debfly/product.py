from einops import rearrange
import torch


def twiddle_mul_twiddle(l_twiddle, r_twiddle):
    """
    Compute the product of two compatible twiddles
    Input:
    l_twiddle, r_twiddle: two tensors of order 4
    Output: a tensor of order 4 (a twiddle)
    """
    a1, d1, b1, c1 = l_twiddle.size()
    a2, d2, b2, c2 = r_twiddle.size()
    l_twiddle = rearrange(l_twiddle, 'a1 d1 b1 (c c1) -> (a1 c1) d1 b1 c', c1=c1)
    r_twiddle = rearrange(r_twiddle, 'a2 d2 (b b2) c2 -> a2 (b2 d2) b c2', b2=b2)
    result = torch.matmul(l_twiddle, r_twiddle)
    result = rearrange(result, '(a c1) (b2 d) b1 c2 -> a d (b1 b2) (c1 c2)', c1=c1, b2=b2)
    return result


def twiddle_list_to_dense(twiddle_list):
    """
    Compute the product of a sequence of twiddles in O(n^2).
    Input:
    twiddle_list: list of tensor of size (a, d, b, c)
    Output: A tensor of shape (1, 1, m, n) (if the twiddle_list has proper size)
    """
    if twiddle_list == 1:
        return twiddle_list[0].squeeze()
    result = twiddle_list[0]
    for twiddle in twiddle_list[1:]:
        result = twiddle_mul_twiddle(result, twiddle)
    return result


def matrix_product(list_matrix):
    matrix = list_matrix[0]
    for i in range(1, len(list_matrix)):
        matrix = matrix @ list_matrix[i]
    return matrix
