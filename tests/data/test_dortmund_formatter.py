"""Unit tests for data/dortmund_formatter.py

"""
# General imports
from pathlib import Path
import os
import requests
from io import BytesIO
from zipfile import ZipFile

import glob

from unittest import TestCase

# Module import
from geometric2dr.data.dortmund_formatter import *


def test_DortmundGexf():
    mutagzip_url = (
        "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip"
    )
    # Download file in tmp if not already present
    if not os.path.exists("tests/test_data/dortmund_data/MUTAG.zip"):
        Path("tests/test_data/dortmund_data/").mkdir(parents=True, exist_ok=True)
        r = requests.get(mutagzip_url)
        z = ZipFile(BytesIO(r.content))
        z.extractall("tests/test_data/dortmund_data/")

    # Instantiate and format the dataset
    dataset = "MUTAG"
    path_to_dataset = "tests/test_data/dortmund_data/"
    relative_output_directory = "tests/test_data/"
    gexifier = DortmundGexf(dataset, path_to_dataset, relative_output_directory)
    gexifier.format_dataset()

    assert os.path.isdir(os.path.join(relative_output_directory, dataset))

    # Test the number of gexf files == num graphs in dataset
    num_files = len(
        glob.glob1(os.path.join(relative_output_directory, dataset), "*.gexf")
    )
    assert num_files == 188


class TestDortmundGexf(TestCase):
    """Class containing unit tests on DortmundGexf"""

    def setUp(self) -> None:
        """Download and extract the MUTAG data if not available and instantiate"""
        mutagzip_url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip"
        # Download file in tmp if not already present
        if not os.path.exists("test_data/dortmund_data/MUTAG.zip"):
            Path("tests/test_data/dortmund_data/").mkdir(parents=True, exist_ok=True)
            r = requests.get(mutagzip_url)
            z = ZipFile(BytesIO(r.content))
            z.extractall("tests/test_data/dortmund_data/")

        dataset = "MUTAG"
        path_to_dataset = "tests/test_data/dortmund_data/"
        relative_output_directory = "tests/test_data/"
        self.dg = DortmundGexf(dataset, path_to_dataset, relative_output_directory)

    def test_format_output(self) -> None:
        self.dg.format_dataset()
        assert os.path.isdir(
            os.path.join(self.dg.output_dir_for_graph_files, self.dg.dataset)
        )

    def test_format_duplicate(self) -> None:
        self.dg.format_dataset()
        assert os.path.isdir(
            os.path.join(self.dg.output_dir_for_graph_files, self.dg.dataset)
        )

    def test_number_graphs(self) -> None:
        num_files = len(
            glob.glob1(
                os.path.join(self.dg.output_dir_for_graph_files, self.dg.dataset),
                "*.gexf",
            )
        )
        assert num_files == 188
