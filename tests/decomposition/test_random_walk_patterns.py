"""Unit tests for random walk patterns

"""

from pathlib import Path
import os
import requests
from io import BytesIO
from zipfile import ZipFile
from unittest import TestCase
import numpy as np
import random

# Random seeds from reproduction
random.seed(2018)
np.random.seed(2018)

# Module
from geometric2dr.decomposition.random_walk_patterns import *


class TestRandomWalkPatterns(TestCase):
    """Tests for the random walk patterns module"""

    def setUp(self) -> None:
        mutagzip_url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip"
        # Download file in tmp if not already present
        if not os.path.exists("tests/test_data/dortmund_data/MUTAG.zip"):
            Path("tests/test_data/dortmund_data/").mkdir(parents=True, exist_ok=True)
            r = requests.get(mutagzip_url)
            z = ZipFile(BytesIO(r.content))
            z.extractall("tests/test_data/dortmund_data/")

        self.graph_file_handle = "tests/test_data/MUTAG/1.gexf"
        self.corpus_dir = "tests/test_data/MUTAG/"

    def test_load_graph(self) -> None:
        graph, adj_matrix = load_graph(self.graph_file_handle)
        assert graph.number_of_nodes() == 17
        assert adj_matrix.shape[0] == 17
        return None

    def test_create_random_walk_graph(self) -> None:
        g, _ = load_graph(self.graph_file_handle)
        assert g.number_of_nodes() == 17
        rw = create_random_walk_graph(g)
        assert g.number_of_edges() == 19
        assert rw.number_of_edges() == 38
        return None

    def test_random_step_node(self) -> None:
        g, _ = load_graph(self.graph_file_handle)
        assert g.number_of_nodes() == 17
        rw = create_random_walk_graph(g)
        assert g.number_of_edges() == 19
        assert rw.number_of_edges() == 38

        rws = []
        for i, node in enumerate(rw):
            rws.append(random_step_node(rw, node))
        assert len(rws) == 17
        return None

    def test_random_walk_with_label_nodes(self) -> None:
        g, _ = load_graph(self.graph_file_handle)
        assert g.number_of_nodes() == 17
        rw = create_random_walk_graph(g)
        random_walk_instance = random_walk_with_label_nodes(g, rw, "1", 2)
        assert len(random_walk_instance) == 2 or len(random_walk_instance) == 3
        return None

    def test_random_walks_graph(self) -> None:
        g, _ = load_graph(self.graph_file_handle)
        assert g.number_of_nodes() == 17
        walks = random_walks_graph(g, 2, 1)
        assert len(walks) == 17
        assert ["(0, 0, 0)"] in walks
        return None

    # TODO
    def test_save_rw_doc(self) -> None:
        pass

    def test_rw_corpus(self) -> None:
        corpus, vocabulary, prob_map, num_graphs, graph_map = rw_corpus(
            self.corpus_dir, walk_length=2, neighborhood_size=2, saving_graph_docs=False
        )
        assert len(corpus) == 188
        assert len(vocabulary) >= 3
        assert "0" in vocabulary
        assert len(prob_map) == 188
        assert num_graphs == 188
        assert len(graph_map) == 188
        return None
