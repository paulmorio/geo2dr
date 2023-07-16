""" Unit tests for all the data readers

"""

import numpy as np
from torch.utils.data import DataLoader
from unittest import TestCase
from geometric2dr.embedding_methods.skipgram_data_reader import (
    SkipgramCorpus,
    InMemorySkipgramCorpus,
)
from geometric2dr.embedding_methods.pvdbow_data_reader import (
    PVDBOWCorpus,
    PVDBOWInMemoryCorpus,
)
from geometric2dr.embedding_methods.pvdm_data_reader import PVDMCorpus
from geometric2dr.embedding_methods.cbow_data_reader import CbowCorpus

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################


class TestSkipgramCorpus(TestCase):
    def setUp(self) -> None:
        corpus_dir = "tests/test_data/MUTAG/"
        extension = ".wld2"
        max_files = 0
        min_count = 0
        window_size = 1
        self.corpus = SkipgramCorpus(
            corpus_dir, extension, max_files, min_count, window_size
        )
        # Should run scan_and_load_corpus, initTableDiscards, initTableNegatives

    def test_scan_and_load_corpus(self) -> None:
        self.corpus.scan_and_load_corpus()
        assert len(self.corpus.graph_fname_list) == 188
        assert len(self.corpus.graph_ids_for_batch_traversal) == 188

    def test_add_file(self) -> None:
        assert self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2") is None
        self.corpus.graph_fname_list.remove("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 187
        self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 188

    def test_initTableDiscards(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0

        self.corpus.initTableDiscards()
        assert isinstance(self.corpus.discards, np.ndarray)

    def test_initTableNegatives(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0

        self.corpus.scan_and_load_corpus()
        self.corpus.initTableDiscards()
        self.corpus.initTableNegatives()
        assert isinstance(self.corpus.negatives, np.ndarray)

    def test_getNegatives(self) -> None:
        response = self.corpus.getNegatives(1, 10)
        assert response.shape == (10,)

    def test_len(self) -> None:
        response = self.corpus.__len__()
        assert response > 0

    def test_get_item(self) -> None:
        dataloader = DataLoader(
            self.corpus,
            batch_size=20,
            shuffle=False,
            num_workers=1,
            collate_fn=self.corpus.collate,
        )
        pos, con, neg = next(iter(dataloader))
        assert pos.shape == (20,)
        assert con.shape == (20,)
        assert neg.shape == (20, 10)


class TestInMemorySkipgramCorpus(TestCase):
    def setUp(self) -> None:
        corpus_dir = "tests/test_data/MUTAG/"
        extension = ".wld2"
        max_files = 0
        min_count = 0
        window_size = 1
        self.corpus = InMemorySkipgramCorpus(
            corpus_dir, extension, max_files, min_count, window_size
        )
        # Should run scan_and_load_corpus, initTableDiscards, initTableNegatives, preload_corpus

    def test_scan_and_load_corpus(self) -> None:
        self.corpus.scan_and_load_corpus()
        assert len(self.corpus.graph_fname_list) == 188
        assert len(self.corpus.graph_ids_for_batch_traversal) == 188

    def test_add_file(self) -> None:
        assert self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2") is None
        self.corpus.graph_fname_list.remove("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 187
        self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 188

    def test_initTableDiscards(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0
        self.corpus.initTableDiscards()
        assert isinstance(self.corpus.discards, np.ndarray)

    def test_initTableNegatives(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0
        self.corpus.scan_and_load_corpus()
        self.corpus.initTableDiscards()
        self.corpus.initTableNegatives()
        assert isinstance(self.corpus.negatives, np.ndarray)

    def test_getNegatives(self) -> None:
        response = self.corpus.getNegatives(1, 10)
        assert response.shape == (10,)

    def test_len(self) -> None:
        response = self.corpus.__len__()
        assert response > 0

    # TODO
    def test_get_item(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0
        self.corpus.scan_and_load_corpus()
        self.corpus.initTableDiscards()
        self.corpus.initTableNegatives()
        self.corpus.preload_corpus()
        dataloader = DataLoader(
            self.corpus,
            batch_size=20,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.corpus.collate,
        )
        a = next(iter(dataloader))
        assert True


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################


class TestPVDBOWCorpus(TestCase):
    def setUp(self) -> None:
        corpus_dir = "tests/test_data/MUTAG/"
        extension = ".wld2"
        max_files = 0
        min_count = 0
        self.corpus = PVDBOWCorpus(corpus_dir, extension, max_files, min_count)
        # Should run scan_and_load_corpus, initTableDiscards, initTableNegatives

    def test_scan_and_load_corpus(self) -> None:
        self.corpus.scan_and_load_corpus()
        assert len(self.corpus.graph_fname_list) == 188
        assert len(self.corpus.graph_ids_for_batch_traversal) == 188

    def test_add_file(self) -> None:
        assert self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2") is None
        self.corpus.graph_fname_list.remove("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 187
        self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 188

    def test_initTableDiscards(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0

        self.corpus.initTableDiscards()
        assert isinstance(self.corpus.discards, np.ndarray)

    def test_initTableNegatives(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0

        self.corpus.scan_and_load_corpus()
        self.corpus.initTableDiscards()
        self.corpus.initTableNegatives()
        assert isinstance(self.corpus.negatives, np.ndarray)

    def test_getNegatives(self) -> None:
        response = self.corpus.getNegatives(1, 10)
        assert response.shape == (10,)

    def test_len(self) -> None:
        response = self.corpus.__len__()
        assert response > 0

    def test_get_item(self) -> None:
        dataloader = DataLoader(
            self.corpus,
            batch_size=20,
            shuffle=False,
            num_workers=1,
            collate_fn=self.corpus.collate,
        )
        pos, con, neg = next(iter(dataloader))
        assert pos.shape == (20,)
        assert con.shape == (20,)
        assert neg.shape == (20, 10)


class TestPVDBOWInMemoryCorpus(TestCase):
    def setUp(self) -> None:
        corpus_dir = "tests/test_data/MUTAG/"
        extension = ".wld2"
        max_files = 0
        min_count = 0
        window_size = 1
        self.corpus = PVDBOWInMemoryCorpus(corpus_dir, extension, max_files, min_count)
        # Should run scan_and_load_corpus, initTableDiscards, initTableNegatives, preload_corpus

    def test_scan_and_load_corpus(self) -> None:
        self.corpus.scan_and_load_corpus()
        assert len(self.corpus.graph_fname_list) == 188
        assert len(self.corpus.graph_ids_for_batch_traversal) == 188

    def test_add_file(self) -> None:
        assert self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2") is None
        self.corpus.graph_fname_list.remove("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 187
        self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 188

    def test_initTableDiscards(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0
        self.corpus.initTableDiscards()
        assert isinstance(self.corpus.discards, np.ndarray)

    def test_initTableNegatives(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0
        self.corpus.scan_and_load_corpus()
        self.corpus.initTableDiscards()
        self.corpus.initTableNegatives()
        assert isinstance(self.corpus.negatives, np.ndarray)

    def test_getNegatives(self) -> None:
        response = self.corpus.getNegatives(1, 10)
        assert response.shape == (10,)

    def test_len(self) -> None:
        response = self.corpus.__len__()
        assert response > 0

    # TODO
    def test_get_item(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0
        self.corpus.scan_and_load_corpus()
        self.corpus.initTableDiscards()
        self.corpus.initTableNegatives()
        self.corpus.pre_load_corpus()
        dataloader = DataLoader(
            self.corpus,
            batch_size=20,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.corpus.collate,
        )
        a = next(iter(dataloader))
        assert True


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################


class TestCBOWCorpus(TestCase):
    def setUp(self) -> None:
        corpus_dir = "tests/test_data/MUTAG/"
        extension = ".wld2"
        max_files = 0
        min_count = 0
        window_size = 1
        self.corpus = CbowCorpus(
            corpus_dir, extension, max_files, min_count, window_size
        )
        # Should run scan_and_load_corpus, initTableDiscards, initTableNegatives

    def test_scan_and_load_corpus(self) -> None:
        self.corpus.scan_and_load_corpus()
        assert len(self.corpus.graph_fname_list) == 188
        assert len(self.corpus.graph_ids_for_batch_traversal) == 188

    def test_add_file(self) -> None:
        assert self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2") is None
        self.corpus.graph_fname_list.remove("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 187
        self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 188

    def test_initTableDiscards(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0

        self.corpus.initTableDiscards()
        assert isinstance(self.corpus.discards, np.ndarray)

    def test_initTableNegatives(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0

        self.corpus.scan_and_load_corpus()
        self.corpus.initTableDiscards()
        self.corpus.initTableNegatives()
        assert isinstance(self.corpus.negatives, np.ndarray)

    def test_getNegatives(self) -> None:
        response = self.corpus.getNegatives(1, 10)
        assert response.shape == (10,)

    def test_len(self) -> None:
        response = self.corpus.__len__()
        assert response > 0

    def test_get_item(self) -> None:
        dataloader = DataLoader(
            self.corpus,
            batch_size=20,
            shuffle=False,
            num_workers=1,
            collate_fn=self.corpus.collate,
        )
        pos, con, neg = next(iter(dataloader))
        assert pos.shape == (20,)
        assert con.shape == (20, 1)
        assert neg.shape == (20, 10)


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################


class TestPVDMCorpus(TestCase):
    def setUp(self) -> None:
        corpus_dir = "tests/test_data/MUTAG/"
        extension = ".wld2"
        max_files = 0
        min_count = 0
        window_size = 1
        self.corpus = PVDMCorpus(
            corpus_dir, extension, max_files, min_count, window_size
        )
        # Should run scan_and_load_corpus, initTableDiscards, initTableNegatives

    def test_scan_and_load_corpus(self) -> None:
        self.corpus.scan_and_load_corpus()
        assert len(self.corpus.graph_fname_list) == 188
        assert len(self.corpus.graph_ids_for_batch_traversal) == 188

    def test_add_file(self) -> None:
        assert self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2") is None
        self.corpus.graph_fname_list.remove("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 187
        self.corpus.add_file("tests/test_data/MUTAG/1.gexf.wld2")
        assert len(self.corpus.graph_fname_list) == 188

    def test_initTableDiscards(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0

        self.corpus.initTableDiscards()
        assert isinstance(self.corpus.discards, np.ndarray)

    def test_initTableNegatives(self) -> None:
        self.corpus.negatives = []
        self.corpus.discards = []
        self.corpus.negpos = 0

        self.corpus.scan_and_load_corpus()
        self.corpus.initTableDiscards()
        self.corpus.initTableNegatives()
        assert isinstance(self.corpus.negatives, np.ndarray)

    def test_getNegatives(self) -> None:
        response = self.corpus.getNegatives(1, 10)
        assert response.shape == (10,)

    def test_len(self) -> None:
        response = self.corpus.__len__()
        assert response > 0

    def test_get_item(self) -> None:
        dataloader = DataLoader(
            self.corpus,
            batch_size=20,
            shuffle=False,
            num_workers=1,
            collate_fn=self.corpus.collate,
        )
        pos, pos_target_subgraph, con, neg = next(iter(dataloader))
        assert pos.shape == (20,)
        assert con.shape == (20, 1)
        assert neg.shape == (20, 10)
