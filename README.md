# Geo2DR

## *Geo2DR*: A library for learning distributed representations of graphs/manifolds.

This library consists of various graph factorization and embedding algorithms built around a common framework to enable quick construction of systems capable of learning distributed representations of graphs. This library utilises PyTorch to maximise the utilisation of CUDA enabled GPUs for learning, but will also run on CPUs if a GPU is not available. 

Popular sytems such as Deep Graph Kernels (Yanardag and Vishwanathan, 2015), Graph2Vec (Narayanan et al., 2017), and Anonymous Walk Embeddings (AWE) (Ivanov and Burnaev, 2018) are all methods which learn distributed representations of arbitrary sized graphs. Such systems can be characterised by a common pipeline described below.

1. **Decomposition of graphs into descriptive substructures:** Each graph in the dataset of graphs is reduced to a set of induced substructure patterns according to some reduction algorithm. An example is finding all the rooted subgraphs of a graph using the Weisfeiler-Lehman algorithm (Shervashidze et al., 2011) as done in Graph2Vec, or shortest paths as in Deep Graph Kernels to name a few. The union of the sets of substructure patterns induced in each graph across the dataset defines a common "vocabulary" that can describe a graph in relation to another graph based on the induced subgraphs patterns. 
2. **Learning distributed vector representations:** The distributive hypothesis (Harris, 1954) posits that words which are used and exist within the same context have similar semantic meanings. In a similar way we may define that a graph is *contextualized* by the substructure patterns producing a new dataset of (graph, induced_subgraph_pattern) pairs. Embedding methods which exploit the distributive hypothesis such as skipgram (Mikolov et al., 2014) can then be used to learn fixed-size distributed vector embeddings of each graph in an unsupervised manner.

Once the distributed vector representations are learned for each of the graphs in a dataset. The graph embeddings may be used in any downstream application such as graph classification, regression, etc.

Geo2DR implements a variety of graph reduction algorithms (such as Weisfeiler-Lehman, anonymous walks, graphlets) and learning models which exploits the distributive hypothesis (such as skipgram with noise contrastive sampling, PV-DM). This enables the quick recreation of existing systems such as Graph2Vec or AWE but also the creation of new combinations leading to new(!) systems capable of learning distributed representations. This enables deeper studies into how we can build better representations of graphs and more reliable comparative analyses on the same codebase. 

Of course care was taken so that modules can act independently so that users can use their own implementations for decompositions/learning algorithms even with different learning frameworks and implementations to allow more freedom.

The following substructure induction algorithms are implemented

- Weisfeiler-Lehman rooted subgraph decomposition
- Anonymous walks
- Graphlets (currently support graphlets of size 2-8)
- Shortest paths

The following embedding systems are included
- Skipgram with negative sampling
- PV-DBOW with negative sampling
- PV-DM with negative sampling

The following methods are currently implemented in the examples ()
- Graph2Vec from Narayanan et al. [Graph2Vec: Learning Distributed Representations of Graphs](https://arxiv.org/abs/1707.05005) (2017 International Workshop on Mining and Learning with Graphs)
- AWE-DD from from Ivanov and Burnaev [Anonymous Walk Embeddings](https://arxiv.org/abs/1805.11921) (ICML 2018)
- Deep GK from Yanardag and Vishwanathan [Deep Graph Kernels](https://dl.acm.org/citation.cfm?id=2783417) (KDD 2015)
- Deep SP from Yanardag and Vishwanathan [Deep Graph Kernels](https://dl.acm.org/citation.cfm?id=2783417) (KDD 2015)
- Deep WL from Yanardag and Vishwanathan [Deep Graph Kernels](https://dl.acm.org/citation.cfm?id=2783417) (KDD 2015)
- MLE graph kernels to showcase different induced substructure patterns.  


<!-- On top of this Geo2DR benefits from different classes handling the corpus datasets and dataloading into the learning models. These give the option of loading datasets from files in the hard drive or loading a corpus into memory for significant speedup in the learning process. -->

## Installation
We recommend following the installation procedure within a virtual environment.

### Installing dependencies: PyNauty and Pytorch
In order to extract graphlets in an efficient manner Geo2DR uses PyNauty (a python wrapper to the Nauty C++ library) and requires its installation prior to installation of Geo2DR. If you are not interested in inducing graphlets you may skip the installation of PyNauty and move to installing PyTorch. 

#### Dependencies
To build pynauty the following components are needed:

- Python 2.7 or Python 3.5
- The most recent version of Nauty.
- An ANSI C compiler.

#### Building of Nauty and linking with PyNauty
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

Pynauty can be built both for Python 2.7 or Python 3.5.

To build pynauty use the command (make sure to do this within the virtualenvironment):

```bash
make pynauty
```

That takes care compiling the necessary object files from nauty’s source and compiling the pynauty Python extension module. To run all the tests coming with the package type:

```bash
make tests
```

##### Installation of PyNauty
To install pynauty to the virtual environment call the following in the pynauty folder whilst the virtual environment is activated

```bash
make virtenv-ins 
```

To uninstall simply use the corresponding make command

```bash
make virtenv-unins
```

Please note, the install/unistall procedures use pip.

### Installing PyTorch
Pytorch is installed based on the available hardware of the machine (GPU or CPU) please follow the appropriate pip installation on the official PyTorch website.

### Installing Geo2DR

Geo2DR can be installed into the virtualenvironment from the source folder. There are two main options:

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

- *The systems produced here learn distributed representations of graphs. These representations are built by taking the perspective that graphs are composed of discrete substructures which characterise the graph in relation to the patterns found in other graphs of a collection under observation. The distributed vector representations are learned by exploiting the distributive hypothesis. Most of representations of graphs learned using systems such as created in PyTorch Geometric rely on a series of spectral/spatial graph convolutions to create node representations which are pooled in various ways to form graph level representations. The foundation of these algorithms can be found in the message-passing+pooling paradigm whereas our foundation is the distributive hypothesis. Hence this library is different and complementary to other existing libraries, and as far as I am aware this is the first.*

3. Hey my model for learning distributed representations is not implemented in the examples!

- *Thanks for telling us. Given the overloaded excitement in machine learning it is impossible to keep up with all the papers. Please open an issue and we will try to get it implemented as soon as possible. Similarly the intention of this library is to enable creation of existing/novel models quickly.*

4. Do you accept contributions?

- *Yes contributions are very welcome. A contributions guideline will be made available in due time, the author is also still learning to navigate open source collaboration on GitHub*