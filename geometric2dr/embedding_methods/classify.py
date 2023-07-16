"""Module containing various functions for classification (on top of the learned embeddings)
mainly useful for providing convenience functions on common benchmark classification methods

"""
import json

# Sklearn SVC (for "fair" comparison with existing methods)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV

from random import randint
import numpy as np
import logging

from .utils import get_files, get_class_labels


def linear_svm_classify(X_train, X_test, Y_train, Y_test):
    """Utility function for quickly performing Scikit Learn GridSearchCV over a linear SVM
    with 10 fold CrossVal given the train test splits

    Parameters
    ----------
    X_train : numpy ndarray
            training feature vectors
    X_test : numpy ndarray
            testing feature vectors
    Y_train : numpy ndarray
            training set labels
    Y_test : numpy ndarray
            test set labels

    Returns
    -------
    tuple
            tuple with accuracy, precision, recall, fbeta_score as applicable
    """
    params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    if len(set(Y_train)) == 2:
        classifier = GridSearchCV(
            LinearSVC(max_iter=100000000),
            params,
            cv=10,
            scoring="f1",
            verbose=1,
            n_jobs=-1,
        )
    else:
        classifier = GridSearchCV(
            LinearSVC(max_iter=100000000),
            params,
            cv=10,
            scoring="f1_weighted",
            verbose=1,
            n_jobs=-1,
        )
    classifier.fit(X_train, Y_train)
    logging.info("best classifier models hyperparameters", classifier.best_params_)

    Y_pred = classifier.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    logging.info("Linear SVM accuracy: {}".format(acc))

    report = classification_report(Y_test, Y_pred)
    logging.info(report)

    precision, recall, fbeta_score, support = precision_recall_fscore_support(
        Y_test, Y_pred
    )

    return (acc, precision, recall, fbeta_score)


def rbf_svm_classify(X_train, X_test, Y_train, Y_test):
    """Utility function for quickly performing Scikit Learn
    GridSearchCV over a rbf kernel SVM with 10 fold CrossVal
    given the train test splits

    Parameters
    ----------
    X_train : numpy ndarray
            training feature vectors
    X_test : numpy ndarray
            testing feature vectors
    Y_train : numpy ndarray
            training set labels
    Y_test : numpy ndarray
            test set labels

    Returns
    -------
    tuple
            tuple with accuracy, precision, recall, fbeta_score as applicable
    """

    params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    if len(set(Y_train)) == 2:
        classifier = GridSearchCV(
            SVC(gamma="scale"), params, cv=10, scoring="f1", verbose=1, n_jobs=-1
        )
    else:
        classifier = GridSearchCV(
            SVC(gamma="scale"),
            params,
            cv=10,
            scoring="f1_weighted",
            verbose=1,
            n_jobs=-1,
        )
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(
        Y_test, Y_pred
    )

    return (acc, precision, recall, fbeta_score)


def perform_classification(corpus_dir, extension, embedding_fname, class_labels_fname):
    """Perform classification over the graph files of dataset given they have corresponding
    embeddings in the saved embedding file and class labels

    Parameters
    ----------
    corpus_dir : str
            folder containing graphdoc files
    extension : str
            extension of the graphdoc files
    embedding_fname : str
            file containing embeddings
    class_labels_fname : str
            files containing labels of each graph

    Returns
    -------
    tuple
            tuple with accuracy, precision, recall, fbeta_score as applicable

    """

    wlk_files = get_files(corpus_dir, extension)

    Y = np.array(get_class_labels(wlk_files, class_labels_fname))
    logging.info("Y (label) matrix shape: {}".format(Y.shape))

    seed = randint(0, 1000)
    with open(embedding_fname, "r") as fh:
        graph_embedding_dict = json.load(fh)
    X = np.array([graph_embedding_dict[fname] for fname in wlk_files])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=seed
    )
    logging.info(
        "Training and Test Matrix Shapes: {}. {}. {}. {} ".format(
            X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
        )
    )

    scores = rbf_svm_classify(X_train, X_test, Y_train, Y_test)
    return scores


def cross_val_accuracy(
    corpus_dir, extension, embedding_fname, class_labels_fname, cv=10, mode=None
):
    """
    Performs 10 (default) fold cross validation, returns the mean accuracy and associated
    standard deviation

    Parameters
    ----------
    corpus_dir : str
            folder containing graphdoc files
    extension : str
            extension of the graphdoc files
    embedding_fname : str
            file containing embeddings
    class_labels_fname : str
            files containing labels of each graph
    cv : int
            integer stating number of folds and therefore experiments to carry out

    Returns
    -------
    tuple : (acc, std)
            tuple containing the mean accuracies of performing 10 fold cross validation 10 times.
            This gives a better picture of usual performance expected performance in a Monte
            Carlo fashion instead of presenting just best performance.

    """
    # our accuracies
    acc_results = []
    wlk_files = get_files(corpus_dir, extension)
    Y = np.array(get_class_labels(wlk_files, class_labels_fname))

    for i in range(cv):
        seed = randint(0, 1000)
        with open(embedding_fname, "r") as fh:
            graph_embedding_dict = json.load(fh)
        X = np.array([graph_embedding_dict[fname] for fname in wlk_files])

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.1, random_state=seed
        )

        if mode == "linear":
            scores = linear_svm_classify(X_train, X_test, Y_train, Y_test)
        else:
            scores = rbf_svm_classify(X_train, X_test, Y_train, Y_test)

        acc_results.append(scores[0])

    return np.mean(acc_results), np.std(acc_results)


def cross_val_accuracy_rbf_bag_of_words(P, y_ids, cv=10):
    r"""cv times Monte Carlo experimentation of 10 fold cross validation, used on
    given dataset matrix returns overall mean accuracy and associated standard deviation.
    Terminology and method name will be updated in future version to address overloading
    term and generalizability of function.

    Parameters
    ----------
    P : numpy ndarray
            a obs x num_features matrix showing dataset
    y_ids : numpy ndarray
            numpy 1 x obs array of class labels for the rows of `P`
    cv : int (default=10)
            overloaded term of monte carlo restarts of the SVM evaluation over 10 fold CV

    Returns
    -------
    tuple : (acc, std)
            tuple containing the mean accuracies of performing 10 fold cross validation 10 times.
            This gives a better picture of usual performance expected performance in a Monte
            Carlo fashion instead of presenting just best performance.

    """
    acc_results = []
    Y = np.array(y_ids)
    seeds = range(cv)
    for i in range(cv):
        seed = seeds[i]
        X_train, X_test, Y_train, Y_test = train_test_split(
            P, Y, test_size=0.1, random_state=seed
        )
        scores = rbf_svm_classify(X_train, X_test, Y_train, Y_test)
        acc_results.append(scores[0])
    return np.mean(acc_results), np.std(acc_results)
