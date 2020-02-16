# Data formatting

The various modules and classes of geometric2dr expect graph datasets to be stored on the machine in the form of individual graphs in the GEXF (Graph Exchange XML Format) (denoted by the `.gexf` extension) collected under a folder with the dataset's name. Additionally geometric2dr expects a seperate `.Labels` plain-text file which contains space delimited classifications of the filename of a graph and its classification.

In summary one would typically have the setup
```
some_dir/dataset_name/graph_name.gexf : folder containing individual gexf files of each graph.
some_dir/dataset_name.Labels : a plain-text file with each line containing a graph's file_name and its classification
```

Modules and classes under the `geometric2dr.data` help transform some common formats employed by dataset repositories such as the TU-Dortmund Graph Kernel Datasets collection into the format described above. An example of this is the `DortmundGexf` class under `geometric2dr.data.data_formatter`, which can be modified to fit your own dataset.

As the graph learning increases in popularity and different graph data repositories employ standardised formats, more formatting and transformation tools will be added to the library.

## DortmundGexf
`DortmundGexf` is a class which transforms the benchmark graph classification datasets in the format hosted on https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets 

**DortmundGexf(dataset, path_to_dataset, output_dir_for_graph_files)**

**Parameters**

- `dataset` (str): name of the folder containing the data
- `path_to_dataset` (str): path to directory containing the dataset directory
- `output_dir_for_graph_files` (str): path where the dataset in gexf format will be stored

**Return**
`DortmundGexf` object

**DortmundGexf.format_dataset()**
Transforms the dataset in path_to_dataset into the format:

```
some_dir/dataset_name/graph_name.gexf : folder containing individual gexf files of each graph.
some_dir/dataset_name.Labels : a plain-text file with each line containing a graph's file_name and its classification
```

into the path defined by `output_dir_for_graph_files`

### Example
```python
gexifier = DortmundGexf("MUTAG", "dortmund_data/", "/tmp/")
gexifier.format_dataset()
```