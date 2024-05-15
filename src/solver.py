import numpy as np

import src.debfly.factorization
import src.debfly.product
import src.debfly.tree
import src.perm
import src.partition
import src.similarity
import src.cluster_tree
import torch
import time

import src.spectral_clustering


def solve_permuted_butterfly(target, factorization_tree, col_tree, row_tree, return_factors=False):
    col_perm = src.perm.perm_col_tree_to_butterfly_canonical(col_tree)
    row_perm = src.perm.perm_row_tree_to_butterfly_canonical(row_tree)
    return solve_general_deblfy_given_permutations(target, factorization_tree, col_perm, row_perm, return_factors)


def solve_general_deblfy_given_permutations(target, factorization_tree, col_perm, row_perm, return_factors=False):
    # permute to canonical row and col tree of butterfly
    perm_target = src.perm.col_permute(target, col_perm)
    perm_target = src.perm.row_permute(perm_target, row_perm)
    # factorization with classical cluster trees
    t_list = src.debfly.factorization.tree_bfly_factorization(perm_target.unsqueeze(0).unsqueeze(0), factorization_tree)
    approx = src.debfly.product.twiddle_list_to_dense(t_list)
    approx = approx.squeeze()
    # return approximation into the original partition
    approx = src.perm.row_permute(approx, src.perm.inverse_permutation(row_perm))
    approx = src.perm.col_permute(approx, src.perm.inverse_permutation(col_perm))
    if not return_factors:
        return approx
    return approx, t_list


def solve_permuted_monarch(target, factorization_tree, col_partition, row_partition, return_factors=False):
    col_perm = src.perm.perm_col_partition_to_monarch_canonical(col_partition, factorization_tree)
    row_perm = src.perm.perm_row_partition_to_monarch_canonical(row_partition, factorization_tree)
    return solve_general_deblfy_given_permutations(target, factorization_tree, col_perm, row_perm, return_factors)


def solve_monarch_with_permutation(target, tree, n_iter, alpha, reduction, random_init=True, verbose=False, n_neighbors_times_size_cluster=None):
    output_size, middle_size, input_size, nb_blocks = src.debfly.tree.get_monarch_sizes_from_tree(tree)

    # Initialize at random partitioning guess and compute error
    if random_init:
        row_partition = src.partition.sample_partition(
            n_subset=middle_size // nb_blocks,
            card=nb_blocks * output_size // middle_size
        )
        col_partition = src.partition.sample_partition(
            n_subset=nb_blocks,
            card=input_size // nb_blocks
        )
    else:
        similarity = src.similarity.compute_similarity_matrix([list(range(input_size))], target, alpha, reduction=reduction)
        row_partition = src.spectral_clustering.recover_partition_with_spectral_clustering(
            similarity,
            nb_clusters=middle_size // nb_blocks,
            size_cluster=nb_blocks * output_size // middle_size
        )

        similarity = src.similarity.compute_similarity_matrix([list(range(output_size))], target.t(), alpha, reduction=reduction)
        col_partition = src.spectral_clustering.recover_partition_with_spectral_clustering(
            similarity,
            nb_clusters=nb_blocks,
            size_cluster=input_size // nb_blocks
        )

    current_approx_err = src.solver.solve_permuted_monarch(target, tree, col_partition, row_partition)
    current_rel_err = torch.norm(target - current_approx_err) / torch.norm(target)
    if verbose:
        print(f"Initial: {current_rel_err:.4f}")
        # print(row_partition)
        # print(col_partition)

    min_rel_err = current_rel_err
    min_row_partition = row_partition
    min_col_partition = col_partition

    for i in range(1, n_iter + 1):
        # Fix column partition and find row partitioning
        similarity = src.similarity.compute_similarity_matrix(col_partition, target, alpha, reduction=reduction)
        if n_neighbors_times_size_cluster is not None:
            n_neighbors = int(n_neighbors_times_size_cluster * len(row_partition[0]))
            similarity = src.similarity.nearest_neighbors_graph(similarity, n_neighbors)
        row_partition = src.spectral_clustering.recover_partition_with_spectral_clustering(
            similarity,
            nb_clusters=len(row_partition),
            size_cluster=len(row_partition[0]),
            nearest_neighbors=None,
        )
        current_approx_err = src.solver.solve_permuted_monarch(target, tree, col_partition, row_partition)
        current_rel_err = torch.norm(target - current_approx_err) / torch.norm(target)
        if verbose:
            off_diag_sim = similarity - torch.diag(torch.diag(similarity))
            print(f"Iter {i}, fixing col_partition: {current_rel_err:.4f}. Min/avg/max similarity: {off_diag_sim.min().item():.4f}, {off_diag_sim.mean().item():.4f}, {off_diag_sim.max().item():.4f}")
            # print(row_partition)
            # print(col_partition)
        if current_rel_err < min_rel_err:
            min_rel_err = current_rel_err
            min_row_partition = row_partition
            min_col_partition = col_partition

        # Fix row partitioning and find column partition
        similarity = src.similarity.compute_similarity_matrix(row_partition, target.t(), alpha, reduction=reduction)
        if n_neighbors_times_size_cluster is not None:
            n_neighbors = int(n_neighbors_times_size_cluster * len(col_partition[0]))
            similarity = src.similarity.nearest_neighbors_graph(similarity, n_neighbors)
        col_partition = src.spectral_clustering.recover_partition_with_spectral_clustering(
            similarity,
            nb_clusters=len(col_partition),
            size_cluster=len(col_partition[0])
        )
        current_approx_err = src.solver.solve_permuted_monarch(target, tree, col_partition, row_partition)
        current_rel_err = torch.norm(target - current_approx_err) / torch.norm(target)
        if verbose:
            off_diag_sim = similarity - torch.diag(torch.diag(similarity))
            print(f"Iter {i}, fixing row_partition: {current_rel_err:.4f}. Min/avg/max similarity: {off_diag_sim.min().item():.4f}, {off_diag_sim.mean().item():.4f}, {off_diag_sim.max().item():.4f}")
            # print(row_partition)
            # print(col_partition)
        if current_rel_err < min_rel_err:
            min_rel_err = current_rel_err
            min_row_partition = row_partition
            min_col_partition = col_partition

    return min_rel_err, min_row_partition, min_col_partition


def solve_square_butterfly_with_permutation(target, alpha_list, n_reinitialize, n_alternate_iter, reduction,
                                            random_init=True, verbose=False, n_neighbors_times_size_cluster=None):
    size = target.shape[0]
    assert size == target.shape[1]
    log_n = int(np.log2(size))
    assert 2**log_n == size

    # Step 1: find each partitioning for each ell
    list_matrix_partitions = []
    list_monarch_err = []
    for ell in range(1, log_n):
        list_partitions = []
        list_errors = []
        nb_blocks = 2**ell
        monarch_tree = src.debfly.tree.rectangle_monarch_tree(size, size, size, nb_blocks)
        begin = time.time()
        for alpha in alpha_list:
            for seed in range(1, n_reinitialize + 1):
                rel_err, row_partition, col_partition = src.solver.solve_monarch_with_permutation(
                    target, monarch_tree, n_alternate_iter, alpha,
                    random_init=random_init,
                    verbose=verbose,
                    reduction=reduction,
                    n_neighbors_times_size_cluster=n_neighbors_times_size_cluster
                )
                list_partitions.append([row_partition, col_partition])
                list_errors.append(rel_err.item())
                # print(f"ell={ell}, alpha={alpha} ({seed}/{n_reinitialize}): time={used_time:.1f}, error={rel_err}")
        min_idx = np.argmin(list_errors)
        used_time = time.time() - begin
        print(f"ell={ell}: time={used_time:.1f}, error={list_errors[min_idx]}")
        list_matrix_partitions.append(list_partitions[min_idx])
        list_monarch_err.append(list_errors[min_idx])

    # Step 2: can we fuse all the recovered partitions?
    list_row_partitions = [matrix_partitions[0] for matrix_partitions in list_matrix_partitions]
    valid_row, row_cluster_tree = src.cluster_tree.cluster_tree_from_partition(list_row_partitions)
    if not valid_row:
        print("Row cluster tree is not valid")
        return False, None
    print("\nReconstructed row cluster tree:")
    print(row_cluster_tree.print())

    list_col_partitions = [matrix_partitions[1] for matrix_partitions in list_matrix_partitions]
    valid_col, col_cluster_tree = src.cluster_tree.cluster_tree_from_partition(list_col_partitions)
    if not valid_col:
        print("Column cluster tree is not valid")
        return False, None
    print("\nReconstructed column cluster tree:")
    print(col_cluster_tree.print())

    # Step 3: solve butterfly with these recovered cluster trees
    butterfly_tree = src.debfly.tree.square_pot_butterfly_tree(log_n, style="balanced")
    approx, t_list = solve_permuted_butterfly(target, butterfly_tree, col_cluster_tree, row_cluster_tree, return_factors=True)
    return True, (approx, row_cluster_tree, col_cluster_tree, list_monarch_err)
