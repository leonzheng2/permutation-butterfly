# Code for reproducibility - Butterfly factorization by algorithmic identification of rank-one blocks

Léon Zheng, Gilles Puy, Elisa Riccietti, Patrick Pérez, Rémi Gribonval. [Factorisation butterfly par
identification algorithmique de blocs de rang un](https://inria.hal.science/hal-04145743). XXIXème Colloque Francophone de Traitement du
Signal et des Images, Aug 2023, Grenoble, France. hal-04145743

## Experiment 1: exhaustive search of permutations for butterfly factorization 
Reproduce Figure 2.

* Target matrix: random butterfly matrix.
* Different sizes are possible, choose e.g. size = 4 or 8.
* Python script: ``src/scripts/exhaustive_search_butterfly_factorization.py``.
* Bash script: ``experiments/exhaustive_search_butterfly_factorization.sh``.
* Script for plotting figures: ``figures/exhaustive_search_figure.py``.

## Experiment 2: complete pipeline of the heuristic
Reproduce Section 5 (Table 1, Figure 4).

* Target matrix: either DFT or random orthogonal butterfly matrix.
* Different sizes, different noise levels are possible.
* Python script: ``src/scripts/butterfly_unknown_perm.py``.
* Bash scripts: ``experiments/dft_butterfly_unknown_perm.sh`` and ``experiments/parker_butterfly_unknown_perm.sh``.
* Script for plotting figures: ``figures/butterfly_unknown_perm_figure.py``.

## Experiment 3: alternating clustering of rows and columns to find each partition
Reproduce Figure 3.

* Target matrix: either DFT or random orthogonal butterfly matrix.
* Different sizes, different noise levels are possible.
* Python script: ``src/scripts/alternating_clustering_each_partitioning.py``.
* Bash scripts: ``experiments/alternating_clustering_each_partitioning.sh``.
* Script for plotting figures: ``figures/alternating_clustering_figure.py``.

## Citation

If you use this code, please cite our paper:

```
@article{zheng2023butterfly,
  title={Butterfly factorization by algorithmic identification of rank-one blocks},
  author={Zheng, L{\'e}on and Puy, Gilles and Riccietti, Elisa and P{\'e}rez, Patrick and Gribonval, R{\'e}mi},
  journal={arXiv preprint arXiv:2307.00820},
  year={2023}
}
```