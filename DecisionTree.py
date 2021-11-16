import sys

from pandas.errors import EmptyDataError
from CartDecisionTreeClassifier import CartDecisionTreeClassifier
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import math
import seaborn as sn

def create_dataframe(args):
    """
    Create a datadframe from the file read in
    :param args:
    :return:
    """
    filename = args.file
    if args.Transpose:
        try:
            df = pd.read_csv(filename, sep="\t", header=None).transpose()
        except EmptyDataError:
            print("The file is empty, please load another")
            sys.exit()
    else:
        try:
            df = pd.read_csv(filename)
        except EmptyDataError:
            print("The file is empty, please load another")
            sys.exit()
    df = df.drop(df.columns[0], axis=1)
    return df

def read_arguments():
    """
    Read in a file to run the classifer on
    :return:
    """
    parser = argparse.ArgumentParser(description="Run a DecisionTreeClassifier on a file")
    parser.add_argument('-f', '--file', help='Usage: python DecisionTree.py -f hazelnuts.txt -T', required=True)
    parser.add_argument('-T', '--Transpose', action='store_true', default=False, help='Add if file needs to be transposed.')
    parser.set_defaults(func=create_dataframe)
    try:
        args = parser.parse_args()
        print(args)
        return args
    except ValueError as val_err:
        print(val_err)
        sys.exit(0)
    except Exception as err:
        print(err)
        parser.print_help()
        sys.exit(0)


def fit_classifier(clf, tree_params, cv, X_train, y_train):
    """

    :param clf: The classifier
    :param tree_params: parameters used for tuning
    :param cv: sklear croos fold
    :param X_train: training features
    :param y_train: training labesl
    :return: sklearn GridSearchCV
    """
    grid_search = GridSearchCV(clf, param_grid=tree_params, cv=cv)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    training_score = grid_search.score(X_train, y_train)
    return grid_search, best_params, training_score


def write_results_to_file(dataset):
    dataset.to_csv("cart_results.csv")


def get_train_size(num):
    """
    We want to have an equal number of rows in all folds
    :param num: the length of the training set
    :return: the rounded one third size of the training set
    """
    a = (num / 100) * 66
    return int((math.ceil(a / 10.0)) * 10.0)


def print_grid_scores(clf):
    print("Cross Validation scores for best GridSearchCV in Cart DecisionTree are:")
    for i in range(0, 9):
        val = 'split' + str(i) + '_test_score'
        print("score ", i, ": ", clf.cv_results_[val][clf.best_index_])


def main():
    print("Processing....")
    args = read_arguments()
    df = create_dataframe(args)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle the rows

    X = df[df.columns[:-1]].values
    X = X.astype('float64')
    y = df[df.columns[-1]].values

    test_size = len(X) - get_train_size(len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    tree_para = {
        'max_depth': np.arange(1, 7)
    }

    cross_validation = KFold(n_splits=10)
    clf = CartDecisionTreeClassifier()
    cart_gs_clf, cart_params, cart_score = fit_classifier(clf, tree_para, cross_validation, X_train, y_train)
    cart_prediction = cart_gs_clf.predict(X_test)
    max_features = len(np.unique(y))

    skl_clf = DecisionTreeClassifier(criterion='gini')
    skl_tree_para = {
        'criterion': ['gini', 'entropy'],
        'max_depth': np.arange(1, 7),
        'max_features': np.arange(1, max_features)
    }

    skl_gs_clf, dt_params, dt_score = fit_classifier(skl_clf, skl_tree_para, cross_validation, X_train, y_train)
    skl_prediction = skl_gs_clf.predict(X_test)

    print_grid_scores(cart_gs_clf)
    print("---------------------------------------------------------------")
    print("accuracy_score against test set for Cart DecisionTree implementation: %.2f" %
          (metrics.accuracy_score(y_test, cart_prediction) * 100), "%")
    print(classification_report(y_test, cart_prediction))

    print("accuracy_score against test set for sklearn decisiontree with gini is: %.2f" %
          (metrics.accuracy_score(y_test, skl_prediction) * 100), "%")
    print("---------------------------------------------------------------")
    results = np.column_stack((cart_prediction, y_test))
    dataset = pd.DataFrame({'Predicted': results[:, 0], 'Actual': results[:, 1]})
    print(classification_report(y_test, skl_prediction))

    confusion_df = pd.crosstab(dataset['Actual'], dataset['Predicted'],
                               rownames=['Actual'], colnames=['Predicted'])
    sns_plot = sn.heatmap(confusion_df, annot=True, vmin=-1, cmap='coolwarm')
    bottom, top = sns_plot.get_ylim()
    sns_plot.set_ylim(bottom + 0.2, top - 0.2)
    figure = sns_plot.get_figure()
    figure.savefig("confusion-matrix.jpg")
    write_results_to_file(dataset)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
