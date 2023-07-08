"""Unit tests for anonymous walk patterns

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
from geometric2dr.decomposition.anonymous_walk_patterns import *

class TestAnonymousWalkPatterns(TestCase):

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

    def test_all_paths(self) -> None:
        paths = {}
        awe_length = 2
        keep_last = True
        new_paths = all_paths(paths, steps=awe_length, keep_last=keep_last)
        assert len(new_paths)==2
        assert len(new_paths[0])==3

    def test_all_paths_edges(self) -> None:
        paths = {}
        awe_length = 2
        keep_last = True
        new_paths = all_paths(paths, steps=awe_length, keep_last=keep_last)
        assert len(new_paths)==2
        assert len(new_paths[0])==3

    def test_all_paths_nodes(self) -> None:
        paths = {}
        awe_length = 2
        keep_last = True
        new_paths = all_paths(paths, steps=awe_length, keep_last=keep_last)
        assert len(new_paths)==2
        assert len(new_paths[0])==3

    def test_all_paths_edges_nodes(self) -> None:
        paths = {}
        awe_length = 2
        keep_last = True
        new_paths = all_paths(paths, steps=awe_length, keep_last=keep_last)
        assert len(new_paths)==2
        assert len(new_paths[0])==3

    def test_create_random_walk_graph(self) -> None:
        g, _ = load_graph(self.graph_file_handle)
        assert g.number_of_nodes() == 17
        rw = create_random_walk_graph(g)
        assert g.number_of_edges() == 19
        assert rw.number_of_edges() == 38

    def test_random_step_node(self):
        g, _ = load_graph(self.graph_file_handle)
        rw = create_random_walk_graph(g)
        aws = []
        for i, node in enumerate(rw):
            aws.append(random_step_node(rw, node))
        assert len(aws) == 17

    def test_random_walk_node(self):
        g, _ = load_graph(self.graph_file_handle)
        rw = create_random_walk_graph(g)
        aws = []
        for i, node in enumerate(rw):
            aws.append(random_walk_node(rw, node,steps=2))
        assert len(aws) == 17
        assert all([len(x) == 3 for x in aws]) # ensure all walks are the correct length

        g, _ = load_graph(self.graph_file_handle)
        rw = create_random_walk_graph(g)
        aws = []
        for i, node in enumerate(rw):
            aws.append(random_walk_node(rw, node,steps=3))
        assert len(aws) == 17
        assert all([len(x) == 4 for x in aws]) # ensure all walks are the correct length

    def test_random_walk_with_label_edges(self) -> None:
        # g, _ = load_graph(self.graph_file_handle)
        # rw = create_random_walk_graph(g)
        # aws = []
        # for i, node in enumerate(rw):
        #     aws.append(random_walk_with_label_edges(g, rw, node, steps=3))
        pass
    

    def test_random_walk_with_label_edges_nodes(self) -> None:
        # g, _ = load_graph(self.graph_file_handle)
        # rw = create_random_walk_graph(g)
        # aws = []
        # for i, node in enumerate(rw):
        #     aws.append(random_walk_with_label_edges_nodes(g, rw, node, steps=3))
        pass

    def test_anonymous_walk_node(self) -> None:
        g, _ = load_graph(self.graph_file_handle)
        rw = create_random_walk_graph(g)
        aws = []
        for i, node in enumerate(rw):
            aws.append(anonymous_walk_node(g, rw, node, steps=2, label_setting=None))
        assert len(aws) == 17
        assert all([len(x) == 3 for x in aws]) # ensure all walks are the correct length

    def test_anonymous_walks(self) -> None:
        paths = dict()
        awe_length = 2
        all_paths(paths, awe_length, keep_last=True)
        walk_ids = dict()
        print(paths)
        for i, path in enumerate(paths[awe_length]):
            walk_ids[tuple(path)] = i
        g, _ = load_graph(self.graph_file_handle)
        label_setting = None
        aws = anonymous_walks(g, 2, walk_ids=walk_ids, awe_length=2, label_setting=label_setting)
        assert len(aws) == 17
        assert all([len(x) == 2 for x in aws])

    def test_awe_corpus(self) -> None:
        awe_length = 2
        corpus, vocabulary, prob_map, num_graphs, graph_map = awe_corpus(self.corpus_dir, awe_length=awe_length, label_setting=None, saving_graph_docs=True)
        assert len(corpus) == 188
        assert len(vocabulary) == 2
        assert '0' in vocabulary
        assert len(prob_map) == 188
        assert all([len(prob_map[i]) == 2 for i in range(1,189)])
        assert num_graphs == 188
        assert len(graph_map) == 188

