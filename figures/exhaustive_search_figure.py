import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=Path)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    error = np.load(args.results_file)
    print(len(error))
    print(min(error))

    # plot histogram
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(error, bins=100, color='tab:blue', alpha=0.9, zorder=2, edgecolor='grey', linewidth=0.1)
    ax.set_xlabel('Relative approximaion error')

    ax.set_xlim(-0.02, np.max(error) + 0.02)

    ax.grid(zorder=0)

    # set log scale on y axis
    ax.set_yscale('log')
    ax.set_ylabel('Number of occurrences')

    plt.tight_layout()
    plt.savefig(args.results_file.with_suffix('.pdf'))
    plt.show()
