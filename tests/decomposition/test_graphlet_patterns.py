"""Unit tests for graphlet patterns

"""

from pathlib import Path
import os
import requests
from io import BytesIO
from zipfile import ZipFile
from unittest import TestCase
import numpy as np
import random

# from geometric2dr.decomposition.graphlet_patterns import *
from geometric2dr.decomposition.random_walk_patterns import *

class TestGraphlletPatterns(TestCase):
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

    #TODO
    def test_get_maps(self) -> None:
        pass

    #TODO
    def test_get_graphlet(self) -> None:
        pass

    #TODO
    def test_graphlet_corpus(self) -> None:
        pass

    #TODO 
    def test_save_graphlet_document(self) -> None:
        pass