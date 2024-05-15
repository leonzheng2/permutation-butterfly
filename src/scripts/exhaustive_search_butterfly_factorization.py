import itertools

import src.solver
import src.cluster_tree
import src.debfly.tree
import torch
import numpy as np
import src.generate_matrix
from pathlib import Path

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="results/exhaustive_search_butterfly_factorization")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    n = args.size

    log_n = int(np.log2(n))

    butterfly_tree = src.debfly.tree.square_pot_butterfly_tree(log_n, "unbalanced")
    target = src.generate_matrix.random_debfly_matrix(butterfly_tree).cuda()
    # target = torch.from_numpy(1.0 * scipy.linalg.hadamard(n)).cuda()
    # target = torch.from_numpy(scipy.linalg.dft(n)).cuda()
    print(target)

    all_cluster_tree_list = src.cluster_tree.all_cluster_tree(list(range(n)))

    error_list = []
    for row_tree, col_tree in itertools.product(all_cluster_tree_list, all_cluster_tree_list):
        approx = src.solver.solve_permuted_butterfly(target, butterfly_tree, col_tree, row_tree)
        rel_err = torch.norm(approx - target) / torch.norm(target)
        if rel_err < 1e-8:
            print(rel_err)
            print(row_tree.print())
            print(col_tree.print())
        error_list.append(rel_err.item())
        np.save(save_dir / f"size={args.size}-error_list.npy", np.array(error_list))
    print(error_list)
