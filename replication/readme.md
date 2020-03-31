# Overview of replication folder and how to replicate results

Essentially:
- Download dataset from Kersting et al. at https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets into data/dortmund_data
- Transform dataset to Geo2DR Gexf format using data_format_script_with_geo2dr.py
- Use any of the scripts based on the method you want to replicate and adjust the script to use the dataset downloaded.

## Contents 

- `data_format_script_with_geo2dr.py` : a script for formatting datasets from Kersting et al. graph classification benchmark repository https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
- `dgk_gk_all.py` : DGK-GK results replication script
- `dgk_sp_all.py` : DGK-SP results replication script
- `dgk_wl_all.py` : DGK-WL results replication script
- `graph2vec_all.py` : Graph2vec results replication script
- `awe_all.py` : AWE-DD results replication script
- `graphlet_mle_kernel.py` : Graphlet kernel results replication script
- `shortest_path_mle_kernel.py` : Shortest path kernel results replication script
- `wl_mle_kernel.py` : WL kernel results replication script
- `anon_walk_mle_kernel.py` : Anonymous walk kernel results replication script
- `gridsearch_svmrbf_method_dataset.py` : Script to run the SVMs on all of the output embeddings of Graph2Vec and AWE-DD. Presents an ordered table of results and the hyperparameter settings used to achieve them. 
- `summary_results.py` : Script to collect the results of the DGK-WL, -SP, -GK methods. Presents an ordered table of results and the hyperparameter settings used to achieve them
- `runtime_analysis_graph2vec_geo2dr.py` : Runtime analysis script running all different models
- `runtime_dgk_gensim.py` : Runtime analysis script running models interfacing with gensim library (requires gensim to be installed obviously)
- `reverse_gexf_to_dgk_format.py` : script to transform Geo2DR compliant gexf datasets into format used by Yanardag and Vishwanathan DGK implementation
- `readme.md` : this file

## Instructions

### Basic set up
The first step is to download the desired dataset from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets . Unzip this dataset into replication/data/dortmund_data/ where an example using the MUTAG dataset is included. Afterwards, one can modify the script `data_format_script_with_geo2dr.py` to format and preprocess the dataset into GEXF format saved in replication/data/ this is where all of the other scripts expect the data to be.

Each of the scripts have to be lightly modified at the top of the file to refer to the desired dataset. So in `data_format_script_with_geo2dr.py`

```python
from geometric2dr.data import DortmundGexf

# Inputs
dataset = "MUTAG" # <---------- CHANGE TO DOWNLOADED DATASET NAME
directory_with_dataset = "data/dortmund_data/"
relative_output_directory = "data/" #<---------- THE FORMATTED DATASET WILL BE PUT HERE WITH THE SAME NAME WITH A .Labels FILE

gexifier = DortmundGexf(dataset, directory_with_dataset, relative_output_directory)
gexifier.format_dataset()
```

Then run any of the result replication scripts (wl_mle_kernel.py, graph2vec_all.py, etc.) as in the contents above, whilst modifying the dataset argument within the script at the top. Typically this is right under the appropriate comment:

```python
# ... Imports

# Input data paths
dataset = "MUTAG" # <---------- CHANGE THIS TO DOWNLOADED AND FORMATTED DATASET
corpus_data_dir = "data/" + dataset

# ... Code

```

### Some notes

- In all of the kernel method scripts the result is directly printed on the terminal. 
- For the DGK based methods, each of the results of ten fold cross validation per run on the hyperparameter settings are saved in individual files within the directory describing the method and the dataset (such as DGK_WL_Performance_MUTAG for the DGK-WL method on the MUTAG dataset). The `summary_results.py` summarises all of the Monte Carlo averages of the 10 fold cross validations and presents them within a Pandas table (ordered by performance and printed on terminal). 
- For Graph2Vec and AWE-DD, `graph2vec_all.py` and `awe_all.py` produce graph-level embeddings saved in appropriately named directories denoting method and dataset such as `Graph2vec_Embeddings_MUTAG`. Within the directory are json files whose file names describe the hyperparameter settings used to obtain them. The script `gridsearch_svmrbf_method_dataset.py` runs the 10 fold CV with the SVM on all of the embeddings and records the Monte Carlo average of the aggregated runs on the same hyperparameter settings, returning a Pandas table (also printed on terminal).