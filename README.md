## *Geo2DR*: A library for learning distributed representations of graphs.

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![PyPI](https://img.shields.io/pypi/v/geometric2dr)
![PyPI - License](https://img.shields.io/pypi/l/geometric2dr)

**[Documentation](https://geo2dr.readthedocs.io/en/latest/)** | **[Paper](https://arxiv.org/abs/2003.05926)** | **[PyPI Page](https://pypi.org/project/geometric2dr/)**

TL;DR Geo2DR is a library for rapidly ***constructing*** methods capable of learning distributed representations of graphs.

## Overview
This library consists of various data prorcessing, graph factorization and embedding algorithms built around a common conceptual framework to enable quick construction of systems capable of [learning distributed representations of graphs](https://geo2dr.readthedocs.io/en/latest/getting_started/understandingdrg.html). The emphasis is not on providing single-line API like calls to specific methods, but providing the various building blocks necessary to construct (or reconstruct) new methods for learning distributed representations of graphs. We hope this encourages exploration of methods using the distributive hypothesis as an inductive bias to learn vector space models of graphs in an unsupervised manner.

To encourage users we have numerous examples recreating existing methods such as Deep Graph Kernels (Yanardag and Vishwanathan, 2015), Graph2Vec (Narayanan et al., 2017) and Anonymous Walk Embeddings (AWE) (Ivanov and Burnaev, 2018) as well as kernel methods with Geo2DR. This library utilises PyTorch to maximise the utilisation of CUDA enabled GPUs for learning, but will also run efficiently on CPUs if a GPU is not available. Geo2DR's modules can also be used to interface with other libraries such as Gephi and NetworkX for analysing data or with Gensim for learning distributed representations.

### Concept
Popular sytems such as Deep Graph Kernels (Yanardag and Vishwanathan, 2015), Graph2Vec (Narayanan et al., 2017), and Anonymous Walk Embeddings (AWE) (Ivanov and Burnaev, 2018) are all methods which learn distributed representations of arbitrary sized graphs. Such systems can largely be characterised by the choice of:

1. **Induced Substructure Pattern:** Each graph in the dataset of graphs is reduced to a set of induced substructure patterns according to some decomposition algorithm. An example is inducing all the rooted subgraphs of a graph using a byproduct of the Weisfeiler-Lehman graph isomorphism test (Shervashidze et al., 2011) as done in Graph2Vec. The set of unique substructure patterns induced across the set of input graphs defines a common "vocabulary" that can describe a graph in relation to another graph based on the induced subgraphs patterns. 
2. **Neural Embedding Method using Distributive Hypothesis:** The distributive hypothesis (Harris, 1954) posits that words which are used and exist within the same context have similar semantic meanings. In a similar way we may define that a graph is *contextualized* by the substructure patterns producing a new corpus dataset of (target-graph, context-substructure_pattern) pairs. Embedding methods which exploit the distributive hypothesis such as skipgram (Mikolov et al., 2014) can then be used to learn fixed-size distributed vector embeddings of each substructure pattern or graph in an unsupervised manner.

![banner](docs/source/getting_started/geo_v3.png)

Once the distributed vector representations are learned for each of the graphs in a dataset. The graph embeddings may be used in any downstream application such as graph classification, regression, etc.

Geo2DR implements a variety of graph decomposition algorithms for inducing discrete substructure patterns and learning models that exploit the distributive hypothesis. It also contains tools for data processing of popular graph datasets, corpus construction, and optimization aids. This enables the quick recreation of existing systems such as Graph2Vec or AWE but also the creation of new combinations leading to new(!) systems capable of learning distributed representations. This enables deeper studies into how we can build better representations of graphs and more reliable comparative analyses on the same codebase. 

The following substructure induction algorithms are implemented

- Weisfeiler-Lehman rooted subgraph decomposition
- Random walks
- Anonymous walks
- Graphlets (currently support graphlet pattern matching of size 2-8 with PyNauty)
- Shortest paths

The following embedding systems are included
- Skipgram with negative sampling
- PV-DBOW with negative sampling
- PV-DM with negative sampling
- GloVe (Coming soon)

The following methods are currently implemented in the `examples` directory
- Graph2Vec from Narayanan et al. [Graph2Vec: Learning Distributed Representations of Graphs](https://arxiv.org/abs/1707.05005) (2017 International Workshop on Mining and Learning with Graphs)
- AWE-DD from from Ivanov and Burnaev [Anonymous Walk Embeddings](https://arxiv.org/abs/1805.11921) (ICML 2018)
- Deep GK from Yanardag and Vishwanathan [Deep Graph Kernels](https://dl.acm.org/citation.cfm?id=2783417) (KDD 2015)
- Deep SP from Yanardag and Vishwanathan [Deep Graph Kernels](https://dl.acm.org/citation.cfm?id=2783417) (KDD 2015)
- Deep WL from Yanardag and Vishwanathan [Deep Graph Kernels](https://dl.acm.org/citation.cfm?id=2783417) (KDD 2015)
- MLE graph kernels to showcase different induced substructure patterns.  

Care was taken so that modules can act independently. This was done so that users can use their own implementations for decompositions/learning algorithms even with different learning frameworks and implementations to allow more freedom and focus on improving what is important to them.

## Installation
We recommend following the installation procedure within a virtual environment.

### Installing dependencies: PyNauty and Pytorch
Geo2DR has two main dependencies that need to be installed prior to installing Geo2DR. PyNauty for the fast graphlet pattern matching, and PyTorch for our neural embedding methods and optimization thereof.

#### PyNauty dependencies
To build PyNauty the following additional components are needed:

- Python 3.6+
- The most recent version of Nauty.
- An ANSI C compiler.

In theory, pynauty should work on all platforms where Python is available and Nauty’s source can be complied. The instructions below are for Linux/Unix environment.

#### Pynauty installation
Download pynauty’s sources as a [compressed tar file](https://web.cs.dal.ca/~peter/software/pynauty/pynauty-0.6.0.tar.gz) and unpack it. That should create a directory like pynauty-X.Y.Z/ containing the source files.

Download the most recent sources of [nauty](http://pallini.di.uniroma1.it/nauty26r12.tar.gz) from Nauty. That file should be something like nautyXYZ.tar.gz.

Change to the directory pynauty-X.Y.Z/ replacing X.Y.Z with the actual version of pynauty:

```bash
cd pynauty-X.Y.Z
```

Unpack nautyXYZ.tar.gz inside the pynauty-X.Y.Z/ directory. Create a symbolic link named nauty to nauty’s source directory replacing XYZ with the actual version of Nauty:

```bash
ln -s nautyXYZ nauty
```

To build pynauty use the command:

```bash
make pynauty
```
To install pynauty to the virtual environment call the following in the pynauty folder whilst the virtual environment is activated

```bash
make virtenv-ins 
```

### Installing PyTorch
Pytorch is installed based on the available hardware of the machine (GPU or CPU) please follow the appropriate pip installation on the official PyTorch website.

## Installing Geo2DR

### (Stable Release) From PyPI
This installation procedure refers to installation from the python package index. This version is stable but may lack the features included currently in this Github repository.

```bash
pip install geometric2dr
```


### (Dev Release) From source
Geo2DR can be installed into the virtualenvironment from the source folder to get its latest features. If ones wishes to simply use Geo2DR modules one can use the standard pip source install from the project folder containing `setup.py`

```bash
pip install .
```

If you wish to modify some of the source code in `geometric2dr` for your own task, you can make a developer install. It is slightly slower, but immediately takes on any changes into the source code.

```bash
pip install -e .
```

### FAQ:
1. Why decompose graphs into different substructure patterns?

- *This is because the useful properties of graphs in different datasets may be better captured using specialised decompositions. There is no one-size fits all. There is no free sandwich. The cake is a lie.*  

2. How is this different from other libraries for learning representations of graphs, such as PyTorch Geometric or Spektral?

- *The systems produced here learn distributed representations of graphs. These representations are built by taking the perspective that graphs are composed of discrete substructures which characterise the graph in relation to the patterns found in other graphs of a collection under observation. The distributed vector representations are learned by exploiting the distributive hypothesis. Most of representations of graphs learned using systems such as created in PyTorch Geometric rely on a series of spectral/spatial graph convolutions to create node representations which are pooled in various ways to form graph level representations. The foundation of these algorithms can be found in the message-passing+pooling paradigm whereas our foundation is the distributive hypothesis. Hence this library is different and complementary to other existing libraries, and as far as I am aware this is the first, especially for creating your own methods.*

3. Hey my model for learning distributed representations is not implemented in the examples!

- *Thanks for telling us. Given the overloaded excitement in machine learning it is impossible to keep up with all the papers. Please open an issue and we will try to get it implemented as soon as possible. Similarly the intention of this library is to enable creation of existing/novel models quickly. Maybe you could make a pull request and contribute!*

4. Do you accept contributions?

- *Yes contributions are always very welcome. A contributions guideline will be made available in due time, the author is also still learning to navigate open source collaboration on GitHub*

## Cite

If this toolkit or any of the examples of pre-existing methods were useful, please cite the original authors and consider citing the library.

```
@inproceedings{geometric2dr,
    title={Learning distributed representations of graphs with Geo2DR},
    author={Paul Scherer and Pietro Lio},
    booktitle={ICML Workshop on Graph Representation Learning and Beyond},
    year={2020},
}
```
