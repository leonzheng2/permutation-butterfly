import random
import itertools
import scipy.special

import src.partition

from src.utils import is_power_of_two


class ClusterTree:
    """ class for cluster tree """

    def __init__(self, index_set):
        self.node = tuple(sorted(index_set))
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None

    def add_children(self, left_tree, right_tree):
        assert self.is_leaf()
        assert isinstance(left_tree, ClusterTree)
        assert isinstance(right_tree, ClusterTree)
        assert set(self.node) == set(left_tree.node).union(set(right_tree.node))
        self.left = left_tree
        self.right = right_tree

    def nodes_at_level(self, level, current_level=0):
        if self.is_leaf() or current_level == level:
            return [list(self.node)]
        return self.left.nodes_at_level(level, current_level=current_level + 1) + self.right.nodes_at_level(level, current_level=current_level + 1)

    def print(self, level=0):
        ret = "\t" * level + str(self) + "\n"
        for child in [self.left, self.right]:
            if child:
                ret += child.print(level + 1)
        return ret

    def __str__(self):
        return f"{self.node}"


def random_equal_split(subset):
    subset = set(subset)
    assert len(subset) % 2 == 0
    subset_1 = random.sample(sorted(subset), len(subset) // 2)
    subset_2 = subset.difference(subset_1)
    two_subsets = sorted([subset_1, subset_2], key=lambda x: min(x))
    return two_subsets[0], two_subsets[1]


def all_equal_split(subset):
    subset = set(subset)
    assert len(subset) % 2 == 0
    equal_split_list = []
    for subset_1 in itertools.combinations(subset, len(subset) // 2):
        subset_1 = set(subset_1)
        subset_2 = subset.difference(subset_1)
        two_subsets = sorted([subset_1, subset_2], key=lambda x: min(x))
        equal_split_list.append(two_subsets)
    # for i in range(len(equal_split_list)):
    #     equal_split_list[i] = sorted(equal_split_list[i], key=lambda x: min(x))
    length = len(equal_split_list) // 2
    return equal_split_list[:length]


def random_cluster_tree(subset):
    """ create a random cluster tree """
    tree = ClusterTree(subset)
    if len(subset) == 1:
        return tree
    left_subset, right_subset = random_equal_split(subset)
    left_tree = random_cluster_tree(left_subset)
    right_tree = random_cluster_tree(right_subset)
    tree.add_children(left_tree, right_tree)
    return tree


def all_cluster_tree(subset):
    """ enumerate all cluster tree """
    if len(subset) == 1:
        return [ClusterTree(subset)]
    list_all_cluster_tree = []
    for (left_subset, right_subset) in all_equal_split(subset):
        list_all_cluster_tree_left = all_cluster_tree(left_subset)
        list_all_cluster_tree_right = all_cluster_tree(right_subset)
        for (left_tree, right_tree) in itertools.product(list_all_cluster_tree_left, list_all_cluster_tree_right):
            tree = ClusterTree(subset)
            tree.add_children(left_tree, right_tree)
            list_all_cluster_tree.append(tree)
    return list_all_cluster_tree


def middle_split_cluster_tree(subset):
    tree = ClusterTree(subset)
    if len(subset) == 1:
        return tree
    assert len(subset) % 2 == 0
    left_subset = tree.node[:len(subset) // 2]
    left_tree = middle_split_cluster_tree(left_subset)
    right_subset = tree.node[len(subset) // 2:]
    right_tree = middle_split_cluster_tree(right_subset)
    tree.add_children(left_tree, right_tree)
    return tree


def even_odd_split_cluster_tree(subset):
    tree = ClusterTree(subset)
    if len(subset) == 1:
        return tree
    assert len(subset) % 2 == 0
    left_subset = tree.node[::2]
    left_tree = even_odd_split_cluster_tree(left_subset)
    right_subset = tree.node[1::2]
    right_tree = even_odd_split_cluster_tree(right_subset)
    tree.add_children(left_tree, right_tree)
    return tree


def denombrement_cluster_tree(N):
    assert N % 2 == 0
    if N == 2:
        return 1
    return scipy.special.binom(N, N//2) * 0.5 * denombrement_cluster_tree(N // 2)**2


def list_parent_trees_from_children_trees(list_parent_nodes, list_children_tree):
    assert src.partition.partition_is_valid(list_parent_nodes)
    children_partition = [list(tree.node) for tree in list_children_tree]
    assert src.partition.partition_is_valid(children_partition)

    list_parent_nodes = sorted(list_parent_nodes, key=lambda x: min(x))
    list_children_tree = sorted(list_children_tree, key=lambda x: min(x.node))
    list_parent_tree = []
    for parent_node in list_parent_nodes:
        parent_node = set(parent_node)
        children = []
        for child_tree in list_children_tree:
            assert isinstance(child_tree, ClusterTree)
            if set(child_tree.node).issubset(parent_node):
                children.append(child_tree)
        if len(children) != 2:
            return False, None
        children = sorted(children, key=lambda tree: min(tree.node))
        parent_tree = ClusterTree(parent_node)
        parent_tree.add_children(children[0], children[1])
        list_parent_tree.append(parent_tree)

    list_parent_tree = sorted(list_parent_tree, key=lambda x: min(x.node))
    return True, list_parent_tree


def cluster_tree_from_partition(list_partition):
    list_partition = sorted(list_partition, key=lambda x: len(x))
    for i, partition in enumerate(list_partition):
        assert is_power_of_two(len(partition))
        assert src.partition.partition_is_valid(partition)
        if i == 0:
            assert len(partition) == 2
        else:
            assert len(list_partition[i]) == 2 * len(list_partition[i-1])

    size = len(list_partition[0]) * len(list_partition[0][0])
    list_parent_tree = [ClusterTree([i]) for i in range(size)]
    for ell in range(len(list_partition) - 1, -1, -1):
        list_parents_nodes = list_partition[ell]
        list_children_tree = list_parent_tree
        valid, list_parent_tree = list_parent_trees_from_children_trees(list_parents_nodes, list_children_tree)
        if not valid:
            return False, None
        # list_parents_nodes = list_partition[ell]
    assert len(list_parent_tree) == 2
    tree = ClusterTree(list(range(size)))
    tree.add_children(list_parent_tree[0], list_parent_tree[1])
    return True, tree


if __name__ == '__main__':
    tree = random_cluster_tree(set(range(8)))
    print(tree.print())

    for level in [0, 1, 2, 3]:
        print(tree.nodes_at_level(level))

    print(all_equal_split([0, 1, 2, 3, 4, 5]))

    list_all_cluster_tree = all_cluster_tree([0, 1, 2, 3, 4, 5, 6, 7])
    for tree in list_all_cluster_tree:
        print(tree.print())

    print(len(list_all_cluster_tree))
    N = 32
    print(denombrement_cluster_tree(N))
    print(len(all_cluster_tree(list(range(N)))))

    import numpy as np
    n = 32
    log_n = int(np.log2(n))
    subset = list(range(n))
    print(even_odd_split_cluster_tree(subset).print())
    print(even_odd_split_cluster_tree(subset).nodes_at_level(log_n))
    print(middle_split_cluster_tree(subset).print())
    print(middle_split_cluster_tree(subset).nodes_at_level(log_n))

