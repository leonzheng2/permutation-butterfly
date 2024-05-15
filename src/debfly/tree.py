from .support import partial_prod_deformable_butterfly_supports
from .chain import check_deformable_butterfly


class Node:
    """
    Implementation of a tree.
    """
    def __init__(self, supscript, subscript):
        """
        :param supscript: int
        :param subscript: int
        """
        self.supscript = supscript
        self.subscript = subscript
        self.left = None  # Empty left child
        self.right = None  # Empty right child
        self.first_size = supscript[0][0]
        self.last_size = supscript[-1][-1]
        self.core_size = [1, 1]
        for (b, c) in subscript:
            self.core_size[0] *= b
            self.core_size[1] *= c

    def support(self):
        return partial_prod_deformable_butterfly_supports(self.supscript, self.subscript, 0, len(self.supscript))

    def print_tree(self, level=0):
        """
        Method to print tree.
        :param level: int
        :return: str
        """
        ret = "\t" * level + str(self) + "\n"
        for child in [self.left, self.right]:
            if child:
                ret += child.print_tree(level + 1)
        return ret

    def __str__(self):
        return f"[{self.first_size}; {self.core_size}; {self.last_size}]"

    def is_leaf(self):
        return self.left is None and self.right is None

    def supscript_subscript_of_leaves(self, level=0, stop_level=None):
        if stop_level is not None and level >= stop_level:
            return [(self.first_size, self.last_size)], [(self.core_size[0], self.core_size[1])]
        if self.is_leaf():
            return [(self.first_size, self.last_size)], [(self.core_size[0], self.core_size[1])]
        assert isinstance(self.left, Node) and isinstance(self.right, Node)
        l_supscript, l_subscript = self.left.supscript_subscript_of_leaves(level=level + 1, stop_level=stop_level)
        r_supcript, r_subscript = self.right.supscript_subscript_of_leaves(level=level + 1, stop_level=stop_level)
        return l_supscript + r_supcript, l_subscript + r_subscript


def generate_tree(supscript, subscript, level=0, style="balanced", stop_level=None) -> Node:
    """
    Generate a balanced tree, with root's value {low, ..., high - 1}. num_factors corresponds to log_2(n), where
    n is the size of the matrix to factorize.
    :param supscript: int
    :param subscript: int
    :param level: int
    :param style: ["balanced", "unbalanced"]
    :return: Node object
    """
    if level == 0:
        check_deformable_butterfly(supscript, subscript)
    root = Node(supscript, subscript)
    length = len(supscript)
    if stop_level is not None and level >= stop_level:
        return root
    if length > 1:
        if style == "balanced":
            split_index = length // 2
        elif style == "unbalanced":
            split_index = 1
        else:
            raise NotImplementedError
        root.left = generate_tree(supscript[:split_index], subscript[:split_index],
                                  level=level + 1, style=style, stop_level=stop_level)
        root.right = generate_tree(supscript[split_index:], subscript[split_index:],
                                   level=level + 1, style=style, stop_level=stop_level)
    return root


def square_pot_butterfly_tree(log_n, style):
    """ tree for butterfly decomposition """
    supscript = [(2 ** i, 2 ** (log_n - i - 1)) for i in range(log_n)]
    subscript = [(2, 2)] * log_n
    root = generate_tree(supscript, subscript, style=style)
    return root


def square_pot_monarch_tree(log_n, split):
    """ tree for monarch decomposition """
    assert 1 <= split <= log_n - 1
    nb_blocks = 2**split
    n = 2**log_n
    return rectangle_monarch_tree(n, n, n, nb_blocks)


def rectangle_monarch_tree(output_size, middle_size, input_size, nb_blocks):
    """
    nb_blocks is the number of blocks in the right factor
    middle_size is the output size after the right factor / input size before the left factor
    """
    assert middle_size % nb_blocks == 0
    assert input_size % nb_blocks == 0
    assert (nb_blocks * output_size) % middle_size == 0
    supscript = [(1, middle_size // nb_blocks), (nb_blocks, 1)]
    subscript = [(nb_blocks * output_size // middle_size, nb_blocks),
                 (middle_size // nb_blocks, input_size // nb_blocks)]
    root = generate_tree(supscript, subscript, style="balanced")
    return root


def get_monarch_sizes_from_tree(tree):
    """ helper function to get parameters describing monarch factorization """
    assert tree.left.is_leaf()
    assert tree.right.is_leaf()
    (a_left, d_left) = tree.left.supscript[0]
    (b_left, c_left) = tree.left.subscript[0]
    (a_right, d_right) = tree.right.supscript[0]
    (b_right, c_right) = tree.right.subscript[0]

    nb_blocks = a_right
    middle_size = a_right * b_right * d_right
    output_size = a_left * d_left * b_left
    input_size = a_right * d_right * c_right

    return output_size, middle_size, input_size, nb_blocks