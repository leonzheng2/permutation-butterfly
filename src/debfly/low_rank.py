import torch
from einops import rearrange


def torch_svd(A):
    if A.dtype == torch.complex64 or A.dtype == torch.complex128:
        B = torch.matmul(A, A.mH)
    else:
        B = torch.matmul(A, A.transpose(-1, -2))

    sq_S, U = torch.linalg.eigh(B)
    U = U[...,-1:]
    S = torch.sqrt(sq_S[...,-1:])
    if A.dtype == torch.complex64 or A.dtype == torch.complex128:
        S_times_Vh = torch.matmul(U.mH, A)
    else:
        S_times_Vh = torch.matmul(U.transpose(-1,-2), A)
    Vh = S_times_Vh / S.unsqueeze(-1)
    return U, S, Vh


def best_rank_one_approximation(M):
    """
    Compute best rank one approximation based on SVD
    Supports batches of matrices as well.
    """
    U, S, Vt = torch_svd(M)
    S_sqrt = S.sqrt()
    U = U * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt
    return U, Vt
