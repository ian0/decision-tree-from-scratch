from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from Node import Node

class CartDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    A decision tree model for classification

    It currently only supports continuous features.  It supports multi-class labels and assumes the labels appear
    in the last column of the data.  It consists of a top down hierarchy of nodes.
    """

    def __init__(self, max_depth=None, min_samples_leaf=1):
        """
        :param max_depth: The maximum depth of the tree
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.number_of_features = None

    def fit(self, X, y):
        """
        Build a decision tree from the training set

        :param X:the training features of the dataset
        :param y: the training labels of the dataset
        """
        self.number_of_features = X.shape[1]
        self.tree = self.build_tree(X, y)
        return self

    @staticmethod
    def split_on_best_feature(X, y, split_column, split_value):
        """
        Split the feature data and labels into two subsets, selecting the feature column to be used, using the
        split value as a threshold.

        :param X: the 2 D numpy  array holding the features dataset
        :param y: the 1 D numpy array holding the labels.
        :param split_column: corresponds to all th values for a feature
        :param split_value: the threshold value to split on
        :return: two pairs of branches, left and right, holding feature and labels

        >>> f = np.array([[5.1, 3.5, 1.4, 0.2],[4.9, 3.0, 1.4, 0.2],
        ... [6.5,2.8,4.6,1.5],[5.7,2.8,4.5,1.3]])
        >>> l = np.array([0,0,1,1])
        >>> sc = 3
        >>> sv = 0.8
        >>> CartDecisionTreeClassifier.split_on_best_feature(f,l,sc,sv)
        (array([[5.1, 3.5, 1.4, 0.2],
               [4.9, 3. , 1.4, 0.2]]), array([0, 0]), array([[6.5, 2.8, 4.6, 1.5],
               [5.7, 2.8, 4.5, 1.3]]), array([1, 1]))
        """
        indices_left = X[:, split_column] <= split_value
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]
        return X_left, y_left, X_right, y_right

    @staticmethod
    def find_feature_midpoint(column_values, split_value):
        """
        Split on a value mid way between two feature values to increase the chances
        of finding a good boundary

        :param column_values: The values in a feature column
        :param split_value: The feature value that returned the best gini index
        :return: the mid point between the split value and the feature value
        >>> y = np.array([1,2,3,4,5,6,7,8])
        >>> sv = 4
        >>> CartDecisionTreeClassifier.find_feature_midpoint(y, sv)
        4.5
        """
        index = column_values.tolist().index(split_value)
        if index < len(column_values):
            next_value = column_values[index + 1]
            best_split_value = (split_value + next_value) / 2
        else:
            best_split_value = split_value
        return best_split_value

    @staticmethod
    def calculate_gini_index(y):
        """
        The gini index is used to measure the impurity of a set

        :param y: the label values
        :return: the gini index
        >>> t = np.array([0,0,0,1,1,3,3,3,3,3])
        >>> CartDecisionTreeClassifier.calculate_gini_index(t)
        0.62
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - sum(probabilities ** 2)
        return gini

    def calculate_gini_gain(self, y_left, y_right):
        """
        The gini gain is a measure of the  likelihood of an incorrect classification of a
        new instance of a random variable, if that new instance were randomly classified
        according to the distribution of class labels from the data set.

        :param y_left: the set of labels below the threshold
        :param y_right: the set of labels afove the threshold
        :return: the measure of gini gain
        >>> ll = np.array([0,0,0,0])
        >>> lr = np.array([1,1,2,2,2,2])
        >>> dtc = CartDecisionTreeClassifier()
        >>> dtc.calculate_gini_gain(ll, lr)
        0.4444444444444444
        """
        return self.calculate_gini_index(y_left) + self.calculate_gini_index(y_right)

    def find_best_split_value_for_feature(self, X, y, split_column):
        """
        Find the best split value for a given feature

        :param X: the features of the dataset
        :param y: the labels in the dataset
        :param split_column: the column number
        :return: return a tuple contains the best split value and the gini gain
        >>> f = np.array([[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[6.2,2.9,4.3,1.3],
        ... [5.0,2.3,3.3,1.0],[6.3,3.3,6.0,2.5], [4.7,3.2,1.3,0.2]])
        >>> l = np.array([0,0,0,1,1,2])
        >>> sc = 3
        >>> dtc = CartDecisionTreeClassifier()
        >>> dtc.find_best_split_value_for_feature(f, l, sc)
        (1.9, 0.5599999999999999)
        """
        gini_gain = 999
        X_values = X[:, split_column]
        unique_values = np.unique(X_values)

        for i in range(0, len(unique_values)):
            split_value = unique_values[i]
            X_left, y_left, X_right, y_right = \
                self.split_on_best_feature(X, y, split_column, split_value)
            temp_gain = self.calculate_gini_gain(y_left, y_right)
            if temp_gain <= gini_gain:
                gini_gain = temp_gain
                best_split_value = split_value
        normalized_split_value = self.find_feature_midpoint(unique_values, best_split_value)
        return normalized_split_value, gini_gain

    def find_best_feature(self, features, labels):
        """
        Find the feature by looping through the columns of the dataset and finding the one
        that returns the best gini gain score

        :param features: the features of the dataset
        :param labels: the labels in the dataset
        :return: return a tuple containing the best feature and the best split value for
        that feature
        >>> f = np.array([[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[6.2,2.9,4.3,1.3],
        ... [5.0,2.3,3.3,1.0],[6.3,3.3,6.0,2.5], [4.7,3.2,1.3,0.2]])
        >>> l = np.array([0,0,0,1,1,2])
        >>> dtc = CartDecisionTreeClassifier()
        >>> dtc.find_best_feature(f, l)
        (2, 1.35, 0.48)
        """
        gini_index = 999
        _, num_columns = features.shape
        for column_index in range(0, num_columns):
            split_value, temp_gi = self.find_best_split_value_for_feature(features, labels, column_index)
            if temp_gi <= gini_index:
                gini_index = temp_gi
                best_split_value = split_value
                best_column = column_index

        return best_column, best_split_value, gini_index

    @staticmethod
    def predict_label(labels):
        """
        Find the label with the highest number of values in a subset
        :param labels: an array of labels
        :return: the most frequently occuring label
        >>> l = np.array([0,0,0,1,1,2])
        >>> CartDecisionTreeClassifier.predict_label(l)
        0
        """
        unique_labels, counts_unique_labels = np.unique(labels, return_counts=True)
        index = counts_unique_labels.argmax()
        predicted_label = unique_labels[index]
        return predicted_label

    @staticmethod
    def is_leaf(labels):
        """
        If all the labels are of the same type this is a leaf node and no futher processing is
        required
        :param labels: labels: an array of labels
        :return: a boolean
        >>> l = np.array([0,0,0,0,0,0])
        >>> CartDecisionTreeClassifier.is_leaf(l)
        True
        """
        unique_classes = np.unique(labels)
        if len(unique_classes) == 1:
            return True
        else:
            return False

    def build_tree(self, X, y, counter=0):
        """
        Recursively build tree, splitting on the feature and value that increases information purity each time.
        Check for depth and the number of class labels in each node.

        :param X:the training features of the dataset
        :param y: the training labels of the dataset
        :param counter: Records the current depth of the tree
        :return: a node is returned for the current depth containing child branches if any
        along with the predicted label, the feature column used for the prediction and the
        spit value on the feature column
        For the doctest dont print the left and right branches
        >>> f = np.array([[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[6.2,2.9,4.3,1.3],
        ... [5.0,2.3,3.3,1.0],[6.3,3.3,6.0,2.5], [4.7,3.2,1.3,0.2]])
        >>> l = np.array([0,0,0,1,1,2])
        >>> dtc = CartDecisionTreeClassifier()
        >>> test_node = dtc.build_tree(f, l)
        >>> print(test_node.predicted_label, test_node.feature_column, test_node.split_value)
        0 2 1.35
        """
        if (self.is_leaf(y)) or (counter == self.max_depth) or \
                (len(y) < self.min_samples_leaf):
            predicted_label = self.predict_label(y)
            node = Node(predicted_label=predicted_label, samples=len(y))
            return node

        else:
            counter += 1
            predicted_label = self.predict_label(y)
            node = Node(predicted_label=predicted_label, samples=len(y))
            node.current_depth = counter
            split_column, split_value, gini_index = self.find_best_feature(X, y)
            X_left, y_left, X_right, y_right = \
                self.split_on_best_feature(X, y, split_column, split_value)

            node.feature_column = split_column
            node.split_value = split_value
            node.gini_index = gini_index

            # the recursive bit...
            left_branch = self.build_tree(X_left, y_left, counter)
            right_branch = self.build_tree(X_right, y_right, counter)

            if left_branch == right_branch:
                node = left_branch
            else:
                node.left = left_branch
                node.right = right_branch

            return node

    def predict(self, y):
        """
        The predict method runs on the test set.  Iterate through the test feature set.  For each row, traverse
        the nodes of the tree with the test feature values and return the predicted class label as per
        the created model

        :param y: The testing features of the dataset
        :return: return the predicted label
        """
        predicted_label = []
        for row in y:
            node = self.tree
            while node.left:
                if row[node.feature_column] < node.split_value:
                    node = node.left
                else:
                    node = node.right
            predicted_label.append(node.predicted_label)
        return np.array(predicted_label)

    def feature_importance(self):
        # todo: implement this class to use sklearn feature selection
        pass
