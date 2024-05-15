import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--seed", type=int, default=5)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    correlation_df = pd.read_csv(args.results_dir / 'correlation.csv')
    random_df = pd.read_csv(args.results_dir / 'random.csv')

    # restrict df to seed
    correlation_df = correlation_df[correlation_df['seed'] == args.seed]
    random_df = random_df[random_df['seed'] == args.seed]

    max_ell = correlation_df['ell'].max()
    assert max_ell == 6

    noise_level = correlation_df["noise_level"].iloc[0]

    fig, ax = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(8, 6))

    for ell in range(1, max_ell + 1):
        id_row = (ell - 1) // 3
        id_col = (ell - 1) % 3

        # extract dataframe for which ell = ell
        correlation_df_ell = correlation_df[correlation_df['ell'] == ell]
        random_df_ell = random_df[random_df['ell'] == ell]
        print(correlation_df_ell)

        min_random_error = random_df_ell["random_rel_err"].min()
        assert correlation_df_ell["original_err"].std() < 1e-8, f"{correlation_df_ell['original_err'].std()}"
        orignal_err = correlation_df_ell["original_err"].median()

        median_error_correlation = correlation_df_ell[["alpha", "recovered_err"]].groupby("alpha").median()
        median_error_correlation.rename(columns={"recovered_err": "median_recovered_err"}, inplace=True)
        max_error_correlation = correlation_df_ell[["alpha", "recovered_err"]].groupby("alpha").max()
        max_error_correlation.rename(columns={"recovered_err": "max_recovered_err"}, inplace=True)
        min_error_correlation = correlation_df_ell[["alpha", "recovered_err"]].groupby("alpha").min()
        min_error_correlation.rename(columns={"recovered_err": "min_recovered_err"}, inplace=True)
        # std_error_correlation = correlation_df_ell[["alpha", "recovered_err"]].groupby("alpha").std()
        # std_error_correlation.rename(columns={"recovered_err": "std_recovered_err"}, inplace=True)
        error_correlation = median_error_correlation.join(max_error_correlation).join(min_error_correlation)

        print(error_correlation)
        print(min_random_error, orignal_err)

        ax[id_row][id_col].errorbar(error_correlation.index, error_correlation["median_recovered_err"],
                                    (error_correlation["median_recovered_err"] - error_correlation["min_recovered_err"],
                                     error_correlation["max_recovered_err"] - error_correlation["median_recovered_err"]),
                                    marker="x", capsize=3, color="tab:orange")
        # horizontal line for min_random_error
        ax[id_row][id_col].axhline(min_random_error, color="tab:green", linestyle="dotted")
        # horizontal line for orignal_err
        ax[id_row][id_col].axhline(orignal_err, color="tab:blue", linestyle="--")

        # add grid
        ax[id_row][id_col].grid()

        # log scale on xaxis
        ax[id_row][id_col].set_xscale("log")

        # set title
        ax[id_row][id_col].set_title(f"$\ell$={ell}, $\epsilon={noise_level}$")

        # ylabel: Relative error
        if id_col == 0:
            ax[id_row][id_col].set_ylabel("Relative error")

        # xlabel: $\alpha$
        ax[id_row][id_col].set_xlabel("$\\alpha$")

        # set ylim
        ax[id_row][id_col].set_ylim(0, 0.85)

    plt.tight_layout()
    plt.savefig(args.results_dir / f"seed={args.seed}.pdf")
    plt.show()
