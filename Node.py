class Node:
    """A node object is the base unit of the decision tree.

    It can hold up to two branches containing child nodes,
    named left and right. It contains the feature (designated by the column id) that gave the best split using the
    gini index and the predicated label based on based on the proportions of a class at a node."""

    def __init__(self, predicted_label, samples):
        self.predicted_label = predicted_label
        self.samples = samples
        self.feature_column = 0
        self.split_value = 0
        self.left = None
        self.right = None
        self.gini_index = None
        self.current_depth = None
