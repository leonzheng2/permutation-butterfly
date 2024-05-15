from setuptools import setup, find_packages

setup(
    name="butterfly-permutation",
    packages=find_packages(),
    install_requires=[
        "k-means-constrained",
        "torch-kmeans",
        "einops",
        "Pyarrow",
        # "pygsp",
        # "POT",
        # "progressbar2",
        # "spams"
    ]
)
