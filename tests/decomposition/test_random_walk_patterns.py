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

    def test_create_random_walk_graph(self) -> None:
        g, _ = load_graph(self.graph_file_handle)
        assert g.number_of_nodes() == 17
        rw = create_random_walk_graph(g)
        assert g.number_of_edges() == 19
        assert rw.number_of_edges() == 38

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

    def test_random_walk_with_label_nodes(self) -> None:
        

# def random_step_node(rw_graph, node):
# 	'''Moves one step from the current according to probabilities of 
# 	outgoing edges. Return next node.

# 	'''

# 	r = random.uniform(0, 1)
# 	low = 0
# 	for v in rw_graph[node]:
# 		p = rw_graph[node][v]['weight']
# 		if r <= low + p:
# 			return v
# 		low += p

# def random_walk_with_label_nodes(graph, rw_graph, node, steps):
# 	'''Creates anonymous walk from a node for arbitrary steps with usage of node labels.
# 	Returns a tuple with consequent nodes.

# 	'''
	
# 	d = dict()
# 	count = 0
# 	pattern = []
# 	for i in range(steps + 1):
# 		label = graph.nodes[node]['Label']
# 		if label not in d:
# 			d[label] = count
# 			count += 1
# 		pattern.append(d[label])
# 		node = random_step_node(rw_graph, node)
# 	return tuple(pattern)