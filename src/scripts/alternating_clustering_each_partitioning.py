"""
Script to test alternate clustering: one spectral clustering to find row partition, and another to find col partition.
Iterate on each partitioning independently for a same target matrix.
"""


import argparse
import numpy as np
import torch
from pathlib import Path
import pandas as pd
import scipy.linalg
import scipy.special

import src.debfly.tree
import src.generate_matrix
import src.perm
import src.cluster_tree
import src.partition
import src.solver
import src.similarity
import time

from pygsp import graphs


def main(seed, args):
    size = args.size
    log_n = int(np.log2(size))
    noise_level = args.noise_level
    alpha_list = args.alpha
    output_size = size
    middle_size = size
    input_size = size

    # Define target matrix
    if args.target_type == "random_parker_butterfly":
        target = src.generate_matrix.random_parker_butterfly(
            log_n, complex_value=args.complex_value
        )
    elif args.target_type == "dft":
        bit_reversal = torch.from_numpy(src.perm.bit_reversal_permutation(log_n)).to(
            dtype=torch.complex128
        )
        dft_permuted = torch.from_numpy(scipy.linalg.dft(2**log_n))
        target = dft_permuted @ bit_reversal
    else:
        raise NotImplementedError

    # Add noise on target matrix
    noise = torch.randn_like(target)
    target += noise_level * torch.norm(target) / torch.norm(noise) * noise
    if args.cuda:
        target = target.cuda()

    # Randomly permute rows and columns of target matrix
    true_row_cluster_tree = src.cluster_tree.random_cluster_tree(range(size))
    true_row_perm = src.perm.inverse_permutation(
        src.perm.perm_row_tree_to_butterfly_canonical(true_row_cluster_tree)
    )
    target = src.perm.row_permute(target, true_row_perm)
    true_col_cluster_tree = src.cluster_tree.random_cluster_tree(range(size))
    true_col_perm = src.perm.inverse_permutation(
        src.perm.perm_col_tree_to_butterfly_canonical(true_col_cluster_tree)
    )
    target = src.perm.col_permute(target, true_col_perm)

    # Save results
    results_correlation = pd.DataFrame()
    results_random = pd.DataFrame()

    for ell in range(1, log_n):
        print(f"============= seed = {seed}, ell = {ell} ==============")
        nb_blocks = 2**ell

        tree = src.debfly.tree.rectangle_monarch_tree(output_size, middle_size, input_size, nb_blocks)

        # Monarch approximation when partition is known
        # true_col_partition = src.partition.canonical_col_partition_monarch(nb_blocks=nb_blocks,
        #                                                                    col=input_size // nb_blocks)
        # true_row_partition = src.partition.canonical_row_partition_monarch(diag=middle_size // nb_blocks,
        #                                                                    row=nb_blocks * output_size // middle_size)

        # original_approx = src.solver.solve_permuted_monarch(target, tree, true_col_partition, true_row_partition)
        original_approx = src.solver.solve_general_deblfy_given_permutations(target, tree, src.perm.inverse_permutation(true_col_perm), src.perm.inverse_permutation(true_row_perm))
        original_rel_err = torch.norm(target - original_approx) / torch.norm(target)

        # Solve monarch approximation with recovered partition
        for alpha in alpha_list:
            for guess in range(1, args.n_guesses + 1):
                begin = time.time()
                min_rel_err, recovered_row_partition, recovered_col_partition = src.solver.solve_monarch_with_permutation(
                    target, tree, args.n_iter, alpha,
                    random_init=args.random_init,
                    verbose=args.verbose,
                    reduction=args.reduction,
                    n_neighbors_times_size_cluster=args.n_neighbors_times_size_cluster
                )
                recovered_approx = src.solver.solve_permuted_monarch(target, tree,
                                                                     recovered_col_partition, recovered_row_partition)
                recovered_rel_err = torch.norm(target - recovered_approx) / torch.norm(target)
                used_time = time.time() - begin
                print(f"Correlation similarity, alpha={alpha} ({guess}/{args.n_guesses}): {recovered_rel_err / original_rel_err} (original={original_rel_err}). Time: {used_time}")
                metrics_1 = {
                    # Target matrix
                    "seed": seed,
                    "target_type": args.target_type,
                    "size": size,
                    "noise_level": noise_level,
                    "complex_value": args.complex_value,
                    # Alternate spectral cluster params
                    "random_init": args.random_init,
                    "alpha": alpha,
                    "n_iter": args.n_iter,
                    "reduction": args.reduction,
                    "guess": guess,
                    "n_neighbors_times_size_cluster": args.n_neighbors_times_size_cluster,
                    # Results
                    "ell": ell,
                    "original_err": original_rel_err.item(),
                    "recovered_err": recovered_rel_err.item(),
                    "ratio": recovered_rel_err.item() / original_rel_err.item(),
                    "time": used_time
                }
                results_correlation = pd.concat([results_correlation, pd.DataFrame([metrics_1])])

        # Solve monarch approximation with random guesses of partitions
        for guess in range(1, args.n_random_partition + 1):
            row_partition = src.partition.sample_partition(
                n_subset=middle_size // nb_blocks,
                card=nb_blocks * output_size // middle_size
            )
            col_partition = src.partition.sample_partition(
                n_subset=nb_blocks,
                card=input_size // nb_blocks
            )
            random_approx = src.solver.solve_permuted_monarch(target, tree, col_partition, row_partition)
            random_rel_err = torch.norm(target - random_approx) / torch.norm(target)
            metrics_2 = {
                # Target matrix
                "seed": seed,
                "target_type": args.target_type,
                "size": size,
                "noise_level": noise_level,
                "complex_value": args.complex_value,
                # Random partition guesses
                "guess": guess,
                # Results
                "ell": ell,
                "original_err": original_rel_err.item(),
                "random_rel_err": random_rel_err.item(),
                "ratio": random_rel_err.item() / original_rel_err.item()
            }
            results_random = pd.concat([results_random, pd.DataFrame([metrics_2])])

        if len(results_random) > 0:
            print(f"Random partitions, min/avg/max over {args.n_random_partition}: {results_random['random_rel_err'].min():.4f} {results_random['random_rel_err'].mean():.4f} {results_random['random_rel_err'].max():.4f}")

    return results_correlation, results_random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Target matrix
    parser.add_argument("--target_type", type=str, default="dft")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--complex_value", action="store_true")
    # Random partition guesses
    parser.add_argument("--n_random_partition", type=int, default=1)
    # Alternate spectral cluster params
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--alpha", type=float, nargs='+', default=[1e-2, 1e-1, 1, 1e1, 1e2, 1e3])  # 1e-2, 1e-1, 1, 1e1, 1e2, 1e3
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--reduction", type=str, default="sum")
    parser.add_argument("--n_guesses", type=int, default=5)
    parser.add_argument("--n_neighbors_times_size_cluster", type=float, default=None)
    # Misc
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save_dir", type=str)
    # Repeat experiments
    parser.add_argument("--repeat", type=int, default=1)
    arguments = parser.parse_args()

    print(arguments)
    if arguments.save_dir:
        save_dir = Path(arguments.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    else:
        save_dir = None
    correlation_csv = pd.DataFrame()
    random_csv = pd.DataFrame()
    for seed in range(1, arguments.repeat + 1):
        begin = time.time()
        df1, df2 = main(seed, arguments)
        print(f"Time of the experiment: {time.time() - begin}")
        correlation_csv = pd.concat([correlation_csv, df1])
        random_csv = pd.concat([random_csv, df2])
        if save_dir:
            correlation_csv.to_csv(save_dir / "correlation.csv")
            random_csv.to_csv(save_dir / "random.csv")
        print(correlation_csv)
        print(random_csv)
