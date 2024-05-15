import argparse
from pathlib import Path
import itertools

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concatenation-df", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--noise-level", type=float, nargs='+', default=[0.0, 0.01, 0.03, 0.1])
    parser.add_argument("--size", type=int, nargs='+', default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--n-instances", type=int, default=20)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    target_list = ["dft", "random_parker_butterfly"]

    if args.concatenation_df is None:
        # extract df
        results_df = pd.DataFrame()
        for target_matrix, size, noise_level, instance in itertools.product(target_list, args.size, args.noise_level,
                                                                            list(range(args.n_instances))):
            df = pd.read_csv(
                args.results_dir / f"target_type={target_matrix}-size={size}-noise_level={noise_level}/instance={instance}/results.csv")
            df["instance"] = [instance]
            results_df = pd.concat([results_df, df])
        results_df.to_csv(args.results_dir / f"concatenation.csv")

    else:
        results_df = pd.read_csv(args.concatenation_df)

    print(len(results_df))
    print(results_df)


    for target_matrix in target_list:
        fig, ax = plt.subplots(figsize=(6, 3))

        # extract target matrix
        target_df = results_df[results_df['target_type'] == target_matrix]
        error_df = target_df[['size', 'noise_level', 'recovered_butterfly_err']].dropna()
        mean_error_df = error_df.groupby(['size', 'noise_level']).mean()
        mean_error_df = mean_error_df.rename(columns={"recovered_butterfly_err": "mean_recovered_butterfly_err"})
        std_error_df = error_df.groupby(['size', 'noise_level']).std()
        std_error_df = std_error_df.rename(columns={"recovered_butterfly_err": "std_recovered_butterfly_err"})
        error_df = mean_error_df.join(std_error_df)

        # plot error vs matrix size for each noise level
        for noise_level in [0.01, 0.03, 0.1]:
            error_df_local = error_df.loc[(slice(None), noise_level), :]
            sizes = error_df_local.index.levels[0]
            ax.errorbar(sizes, error_df_local["mean_recovered_butterfly_err"] / noise_level, yerr=error_df_local["std_recovered_butterfly_err"],
                        label=f"$\epsilon={noise_level}$", marker='x', capsize=2)
            ax.grid()
            ax.legend()
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(plt.ScalarFormatter())
            ax.set_xticks(sizes)
            ax.set_xlabel("Matrix size $n$")
            ax.set_ylabel("Relative error divided by $\epsilon$")
            ax.set_xlim(3, 160)
            ax.set_ylim(0.4, 1.0)

        plt.tight_layout()
        plt.savefig(args.results_dir / f"error_vs_matrix_size-target_matrix={target_matrix}.pdf")
        plt.show()


    # success rate
    # success_rate = results_df[['target_type', 'size', 'noise_level', 'success']].groupby(
    #     ['target_type', 'size', 'noise_level']).mean() * 100
    # print(success_rate)

    # plot success rate for each target matrix. success_rate vs. matrix size. several plots for each noise level
    # fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    # for id_ax, target_matrix in enumerate(target_list):
    #     for noise_level in args.noise_level:
    #         success_rate_local = success_rate.loc[target_matrix, :, noise_level]
    #         ax[id_ax].plot(success_rate_local.index.get_level_values('size'), success_rate_local,
    #                        label=f"$\epsilon={noise_level}$", marker='.')
    #     ax[id_ax].set_title(f"DFT" if target_matrix == "dft" else "Random orthogonal butterfly")
    #     ax[id_ax].set_xlabel("Matrix size n")
    #     ax[id_ax].set_ylabel("Success rate (%)")
    #     ax[id_ax].legend()
    #
    #     # grid
    #     ax[id_ax].grid()
    #
    #     # set log scale base 2 for x axis and show integer in base 10 for ticks label
    #     ax[id_ax].set_xscale("log", base=2)
    #     ax[id_ax].xaxis.set_major_formatter(plt.ScalarFormatter())
    #     ax[id_ax].set_xticks(success_rate_local.index.get_level_values('size'))
    # plt.tight_layout()
    # plt.show()
    #

    # fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharey=True)

    # width = 0.1  # the width of the bars
    # for id_ax, target_matrix in enumerate(target_list):
    #     for i, noise_level in enumerate(args.noise_level):
    #         success_rate_local = success_rate.loc[target_matrix, :, noise_level]
    #         positions = np.arange(len(success_rate_local.index.get_level_values('size')))
    #         ax[id_ax].bar(positions + i * width, np.array(success_rate_local.values).reshape(-1), width,
    #                       label=f"$\epsilon={noise_level}$")
    #
    #     ax[id_ax].set_title("DFT" if target_matrix == "dft" else "Random orthogonal butterfly")
    #     ax[id_ax].set_xlabel("Matrix size n")
    #     ax[id_ax].set_ylabel("Success rate (%)")
    #
    #     if target_matrix == "dft":
    #         ax[id_ax].legend()
    #
    #     # grid
    #     ax[id_ax].grid()
    #
    #     # Since we're using bar plots, consider linear or categorical labels for clarity
    #     ax[id_ax].set_xticks(positions + width / len(args.noise_level))
    #     ax[id_ax].set_xticklabels(success_rate_local.index.get_level_values('size'))
    #
    # plt.tight_layout()
    # plt.show()
    #
    # width = 0.1  # the width of the bars
    # num_bars = 4  # the number of bars for each x-tick
    #
    # for target_matrix in target_list:
    #     fig, ax = plt.subplots(figsize=(7, 3))
    #
    #     # Calculate offset to center groups: Subtract half of the total width of all bars in a group
    #     offset = (width * num_bars) / 2
    #
    #     for i, noise_level in enumerate(args.noise_level):
    #         success_rate_local = success_rate.loc[target_matrix, :, noise_level]
    #         # Assume `sizes` contains the distinct sizes, corresponding to x-ticks
    #         sizes = np.array(success_rate_local.index.get_level_values('size').unique())
    #         num_sizes = len(sizes)
    #         positions = np.arange(num_sizes)  # Base positions for each group of bars
    #
    #         # Adjust position for each bar to center the group
    #         bar_positions = positions - offset + (i * width) + (width / 2)
    #         ax.bar(bar_positions, np.array(success_rate_local.values).reshape(-1), width, label=f"$\epsilon={noise_level}$")
    #
    #     # ax.set_title("DFT" if target_matrix == "dft" else "Random orthogonal butterfly")
    #     ax.set_xlabel("Matrix size n")
    #     ax.set_ylabel("Success rate (%)")
    #
    #     if target_matrix == "dft":
    #         ax.legend()
    #
    #     # Grid
    #     ax.grid()
    #
    #     # Set x-ticks and labels. Now the labels should be centered under each group of bars
    #     ax.set_xticks(positions)
    #     ax.set_xticklabels(sizes)
    #
    #     plt.tight_layout()
    #     plt.savefig(args.results_dir / f"success_rate-target_matrix={target_matrix}.pdf")
    #     plt.show()
