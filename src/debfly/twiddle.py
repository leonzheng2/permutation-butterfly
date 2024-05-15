import torch


def twiddle_to_matrix(twiddle):
    """
    Compute the matrix corresponding to the a twiddle. WARNING: Can be very time consuming
    Input:
    twiddle: a tensor of order 4 (twiddle)
    Output:
    A matrix (tensor of order 2)
    """
    blocks = []
    a = twiddle.size()[0]
    d = twiddle.size()[1]
    b = twiddle.size()[2]
    c = twiddle.size()[3]
    for block in twiddle:
        sub_blocks = [[torch.diag(block[:,i,j]) for i in range(b)] for j in range(c)]
        sub_intermediate_blocks = [torch.cat(sb, dim = 0) for sb in sub_blocks]
        resulted_block = torch.cat(sub_intermediate_blocks, dim = 1)
        blocks.append(resulted_block)
    result = torch.zeros(a * d * b, a * d * c)
    for i in range(a):
        result[d * b * i:d * b * (i + 1), d * c * i: d * c * (i+1)] = blocks[i]
    return result


def extract_diagonal_blocks(twiddle):
    a, d, b, c = twiddle.shape
    assert d == 1
    return [twiddle[i, 0] for i in range(a)]