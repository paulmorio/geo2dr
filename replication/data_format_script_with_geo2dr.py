"""
Quick script to transform datasets from TU Dortmund graph kernel repo
to format compatible with geo2dr
"""

from geometric2dr.data import DortmundGexf

# Inputs
dataset = "MUTAG"
directory_with_dataset = "data/dortmund_data/"
relative_output_directory = "data/"

gexifier = DortmundGexf(dataset, directory_with_dataset, relative_output_directory)
gexifier.format_dataset()
