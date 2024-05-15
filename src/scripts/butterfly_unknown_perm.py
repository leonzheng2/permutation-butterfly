import argparse
import numpy as np
import pandas as pd
import scipy.linalg
import torch
from pathlib import Path

from src.utils import is_power_of_two
import src.generate_matrix
import src.solver
import src.cluster_tree
import src.debfly.tree
import src.partition
import src.perm


def one_experiment(args):
    size = args.size
    assert is_power_of_two(size)
    log_n = int(np.log2(size))

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
    target += args.noise_level * torch.norm(target) / torch.norm(noise) * noise
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

    # Compute approximation error of butterfly factorization when permutations are known
    print("\nCompare with the case where permutations are known:")
    butterfly_tree = src.debfly.tree.square_pot_butterfly_tree(log_n, style="balanced")
    butterfly_approx = src.solver.solve_permuted_butterfly(
        target, butterfly_tree, true_col_cluster_tree, true_row_cluster_tree
    )
    original_bfly_rel_err = torch.norm(target - butterfly_approx) / torch.norm(target)
    print(f"Butterfly error with known permutation: {original_bfly_rel_err}")

    # Expected monarch error
    original_monarch_rel_err = []
    list_matrix_partitions = src.partition.get_butterfly_partitioning(
        true_row_cluster_tree, true_col_cluster_tree
    )
    for ell in range(1, log_n):
        row_partition, col_partition = list_matrix_partitions[ell - 1]
        monarch_tree = src.debfly.tree.square_pot_monarch_tree(log_n, ell)
        monarch_approx = src.solver.solve_permuted_monarch(
            target, monarch_tree, col_partition, row_partition
        )
        monarch_rel_err = torch.norm(target - monarch_approx) / torch.norm(target)
        print(f"Monarch error for ell={ell} with known permutation: {monarch_rel_err}")
        original_monarch_rel_err.append(monarch_rel_err.item())

    # Butterfly factorization without knowing permutations.
    print(
        "\nApplying the proposed heuristic for butterfly factorization with permutations:"
    )
    success, results = src.solver.solve_square_butterfly_with_permutation(
        target,
        alpha_list=args.alpha,
        n_reinitialize=args.n_guesses,
        n_alternate_iter=args.n_iter,
        reduction=args.reduction,
    )
    if not success:
        print("Recovery failed")
        return original_bfly_rel_err, original_monarch_rel_err, success, None, None

    approx, row_cluster_tree, col_cluster_tree, recovered_monarch_rel_err = results
    recovered_bfly_rel_err = torch.norm(target - approx) / torch.norm(target)
    print(f"Final approximation error of the heuristic: {recovered_bfly_rel_err:.4f}")
    return (
        original_bfly_rel_err,
        original_monarch_rel_err,
        success,
        recovered_bfly_rel_err,
        recovered_monarch_rel_err,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Target matrix
    parser.add_argument("--target_type", type=str, default="random_parker_butterfly")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--complex_value", action="store_true")
    # Parameters for butterfly factorization
    parser.add_argument(
        "--alpha", type=float, nargs="+", default=[1e-2, 1e-1, 1, 1e1, 1e2]
    )
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--reduction", type=str, default="sum")
    parser.add_argument("--n_guesses", type=int, default=1)
    # Misc
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save_dir", type=str)
    arguments = parser.parse_args()

    print(arguments)
    if arguments.save_dir:
        save_dir = Path(arguments.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    else:
        save_dir = None
    results_csv = pd.DataFrame()

    # Perform one experiment
    (
        original_bfly_rel_err,
        original_monarch_rel_err,
        success,
        recovered_bfly_rel_err,
        recovered_monarch_rel_err,
    ) = one_experiment(arguments)
    original_bfly_rel_err = original_bfly_rel_err.item()
    if success:
        recovered_bfly_rel_err = recovered_bfly_rel_err.item()
        ratio = recovered_bfly_rel_err / original_bfly_rel_err
    else:
        ratio = None

    # Save matrics
    metrics_1 = {
        # Target matrix
        "target_type": arguments.target_type,
        "size": arguments.size,
        "noise_level": arguments.noise_level,
        "complex_value": arguments.complex_value,
        # Alternate spectral cluster params
        "alpha": arguments.alpha,
        "n_iter": arguments.n_iter,
        "reduction": arguments.reduction,
        "guess": arguments.n_guesses,
        # Results
        "original_monarch_rel_err": original_monarch_rel_err,
        "recovered_monarch_rel_err": recovered_monarch_rel_err,
        "original_butterfly_err": original_bfly_rel_err,
        "recovered_butterfly_err": recovered_bfly_rel_err,
        "success": success,
        "ratio": ratio,
    }
    results_csv = pd.concat([results_csv, pd.DataFrame([metrics_1])])
    print(results_csv)
    if save_dir:
        results_csv.to_csv(save_dir / "results.csv")
