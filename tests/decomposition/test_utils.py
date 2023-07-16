"""
Unit tests for decomposition utils

"""

from pathlib import Path
import os
import requests
from io import BytesIO
from zipfile import ZipFile

from geometric2dr.decomposition.utils import *


def test_get_files() -> None:
    mutagzip_url = (
        "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip"
    )
    # Download file in tmp if not already present
    if not os.path.exists("tests/test_data/dortmund_data/MUTAG.zip"):
        Path("tests/test_data/dortmund_data/").mkdir(parents=True, exist_ok=True)
        r = requests.get(mutagzip_url)
        z = ZipFile(BytesIO(r.content))
        z.extractall("tests/test_data/dortmund_data/")

    corpus_dir = "tests/test_data/MUTAG/"

    all_files = get_files(corpus_dir, ".gexf", max_files=0)
    assert len(all_files) == 188
    all_files = get_files(corpus_dir, ".gexf", 20)
    assert len(all_files) == 20
