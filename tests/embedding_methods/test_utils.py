"""
Unit tests for embedding utils

"""
import requests
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
from unittest import TestCase
from geometric2dr.embedding_methods.utils import *

from geometric2dr.embedding_methods.skipgram_data_reader import InMemorySkipgramCorpus
from geometric2dr.decomposition.shortest_path_patterns import sp_corpus


class TestUtils(TestCase):
    def setUp(self) -> None:
        self.dname = "tests/test_data/MUTAG"
        self.extension_gexf = ".gexf"
        self.extension_rw = ".rw"

        mutagzip_url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip"
        # Download file in tmp if not already present
        if not os.path.exists("tests/test_data/dortmund_data/MUTAG.zip"):
            Path("tests/test_data/dortmund_data/").mkdir(parents=True, exist_ok=True)
            r = requests.get(mutagzip_url)
            z = ZipFile(BytesIO(r.content))
            z.extractall("tests/test_data/dortmund_data/")

    def test_get_files(self) -> None:
        all_files = get_files(self.dname, ".gexf", max_files=0)
        assert len(all_files) == 188
        all_files = get_files(self.dname, ".gexf", 20)
        assert len(all_files) == 20

    # TODO
    def test_save_graph_embeddings(self) -> None:
        pass

    # TODO
    def test_save_subgraph_embeddings(self) -> None:
        pass

    def test_get_class_labels(self) -> None:
        all_files = get_files(self.dname, ".gexf", max_files=0)
        labels = get_class_labels(all_files, "tests/test_data/MUTAG.Labels")
        assert len(all_files) == len(labels)

    def test_get_class_labels_tuples(self) -> None:
        all_files = get_files(self.dname, ".gexf", max_files=0)
        labels = get_class_labels(all_files, "tests/test_data/MUTAG.Labels")
        label_tuples = get_class_labels_tuples(
            all_files, "tests/test_data/MUTAG.Labels"
        )
        assert len(all_files) == len(labels) and len(all_files) == len(label_tuples)

    def test_get_kernel_matrix_row_idx_with_class(self) -> None:
        all_files = get_files(self.dname, ".gexf", max_files=0)
        labels_fname = "tests/test_data/MUTAG.Labels"
        corpus_dir = self.dname
        corpus, vocabulary, prob_map, num_graphs, graph_map = sp_corpus(corpus_dir)
        skipgram_corpus = InMemorySkipgramCorpus(
            corpus_dir, extension=".spp", window_size=2
        )
        skipgram_corpus.scan_and_load_corpus()
        assert skipgram_corpus.num_graphs == 188

        # Actual function test
        kernel_row_x_id, kernel_row_y_id = get_kernel_matrix_row_idx_with_class(
            skipgram_corpus, ".spp", all_files, labels_fname
        )
        assert len(kernel_row_x_id) == len(kernel_row_y_id)
