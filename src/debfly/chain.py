import numpy as np


def check_deformable_butterfly(supscript, subscript):
    """ check if a butterfly chain is compatible """
    if len(supscript) != len(subscript):
        raise Exception("Lengths of subscript and subscript are not equal")
    length = len(subscript)
    curr = 1
    for index in reversed(range(length)):
        sup = supscript[index]
        sub = subscript[index]
        if len(sup) != 2:
            raise Exception("Length of subscript element is expected to be equal to 2")
        if len(sub) != 2:
            raise Exception("Length of subscript element is expected to be equal to 2")
        a, d = sup
        b, c = sub
        p = a * b * d
        q = a * c * d
        if curr != d:
            raise Exception("d condition failed")
        curr *= b
        if index < length - 1 and q != supscript[index + 1][0] * supscript[index + 1][1] * subscript[index + 1][0]:
            raise Exception("Matrix multiplication condition failed")


def reduce_chain_to_two_factors(supscript, subscript, split):
    """ given a butterfly factorization with several factors,
    compute the corresponding monarch chain (with two factors)"""
    assert len(supscript) == len(subscript)
    n_factors = len(supscript)
    if n_factors == 2:
        return supscript, subscript
    assert 1 <= split <= n_factors - 1
    supscript_monarch = [
        (supscript[0][0], supscript[split - 1][1]),
        (supscript[split][0], supscript[-1][1])
    ]
    subscript_monarch = [
        (np.prod([b for (b, c) in subscript[:split]]), np.prod([c for (b, c) in subscript[:split]])),
        (np.prod([b for (b, c) in subscript[split:]]), np.prod([c for (b, c) in subscript[split:]]))
    ]
    check_deformable_butterfly(supscript_monarch, subscript_monarch)

    return supscript_monarch, subscript_monarch
