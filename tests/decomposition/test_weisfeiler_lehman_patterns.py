"""Unit tests for rooted subgraph (WL) patterns

"""

from pathlib import Path
import os
import requests
from io import BytesIO
from zipfile import ZipFile
from unittest import TestCase
import numpy as np
import random

# Modules
from geometric2dr.decomposition.weisfeiler_lehman_patterns import *
from geometric2dr.decomposition.utils import *
from geometric2dr.decomposition.random_walk_patterns import load_graph

class TestWeisfeilerLehmanPatterns(TestCase):
    """Tests for the WL patterns module"""

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

    def test_initial_relabel(self) -> None:
        graph, adj_matrix = load_graph(self.graph_file_handle)
        g = initial_relabel(graph, node_label_attr_name='Label')
        assert 'relabel' in g.nodes['1']
        assert len(g.nodes()) == 17
        return None
    
    def test_wl_relabel(self) -> None:
        graph, adj_matrix = load_graph(self.graph_file_handle)
        g = initial_relabel(graph, node_label_attr_name='Label')
        wlgraph = wl_relabel(g, 1)
        assert 0 in graph.nodes['1']['relabel'].keys()
        assert 1 in graph.nodes['1']['relabel'].keys()

    #TODO
    def test_save_wl_doc(self) -> None:
        pass

    def test_wl_corpus(self) -> None:
        fnames = get_files(self.corpus_dir, ".gexf")
        max_h = 2
        node_label_attr_name = 'Label'

        corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(fnames, max_h, node_label_attr_name)
        assert len(corpus) == 188
        assert len(vocabulary) >= 3
        assert len(prob_map) == 188
        assert num_graphs == 188
        assert len(graph_map) == 188

    


    