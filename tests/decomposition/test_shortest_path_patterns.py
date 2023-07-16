"""Unit tests for shortest path patterns

"""

# Standard libraries

from pathlib import Path
import os
import requests
from io import BytesIO
from zipfile import ZipFile
from unittest import TestCase
import numpy as np
import random

import sys
import os
import glob
import pickle
import itertools
import random
from collections import defaultdict
from time import time
from tqdm import tqdm

# 3rd party
import numpy as np
import networkx as nx

# Random seeds from Yanardag et al.
random.seed(314124)
np.random.seed(2312312)

# Module
from geometric2dr.decomposition.shortest_path_patterns import *


class TestShortestPathPatterns(TestCase):
    """Tests for the shortest path patterns module"""

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

    # TODO
    def test_save_sp_doc(self) -> None:
        pass

    def test_sp_corpus(self) -> None:
        corpus_dir = self.corpus_dir
        corpus, vocabulary, prob_map, num_graphs, graph_map = sp_corpus(corpus_dir)
        assert len(corpus) == 188
        assert len(vocabulary) >= 3
        assert "0_0_0" in vocabulary
        assert len(prob_map) == 188
        assert num_graphs == 188
        assert len(graph_map) == 188
