"""
Unit tests for classify

"""
import numpy as np
from sklearn import datasets
from unittest import TestCase
from geometric2dr.embedding_methods.classify import *

class TestClassify(TestCase):
    def setUp(self) -> None:
        X, y = datasets.load_iris(return_X_y=True)
        n_sample = len(X)
        np.random.seed(0)
        order = np.random.permutation(n_sample)
        X = X[order]
        y = y[order].astype(float)

        self.X = X
        self.y = y

        self.X_train = X[: int(0.9 * n_sample)]
        self.y_train = y[: int(0.9 * n_sample)]
        self.X_test = X[int(0.9 * n_sample) :]
        self.y_test = y[int(0.9 * n_sample) :]

    def test_linear_svm_classify(self) -> None:
        acc, precision, recall, fbeta_score = linear_svm_classify(self.X_train, self.X_test, self.y_train, self.y_test)
        assert acc > 0

    def test_rbf_svm_classify(self) -> None:
        acc, precision, recall, fbeta_score = rbf_svm_classify(self.X_train, self.X_test, self.y_train, self.y_test)
        assert acc > 0

    def test_perform_classification(self) -> None:
        corpus_dir = "tests/test_data/MUTAG/"
        extension = ".wld2" 
        embedding_fname = "tests/test_data/Graph2Vec_Embeddings.json" 
        class_labels_fname = "tests/test_data/MUTAG.Labels"
        acc, precision, recall, fbeta_score = perform_classification(corpus_dir, extension, embedding_fname, class_labels_fname)
        assert acc > 0

    def test_cross_val_accuracy(self) -> None:
        corpus_dir = "tests/test_data/MUTAG/"
        extension = ".wld2" 
        embedding_fname = "tests/test_data/Graph2Vec_Embeddings.json" 
        class_labels_fname = "tests/test_data/MUTAG.Labels"
        meanacc, stdacc = cross_val_accuracy(corpus_dir, extension, embedding_fname, class_labels_fname)
        assert meanacc > 0

    def test_cross_val_accuracy_rbf_bag_of_words(self) -> None:
        meanacc, stdacc = cross_val_accuracy_rbf_bag_of_words(self.X, self.y)
        assert meanacc > 0