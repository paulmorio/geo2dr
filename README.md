# geo2dr
Geo2DR: A framework for learning distributed representations of graphs/manifolds.

It consists of various methods built around a general pipeline for building distributed representations of graphs in an unsupervised fashion. Popular methods such as Deep Graph Kernels (Yanardag and Vishwanathan, 2015), Graph2Vec (Narayanan et al., 2017), and Anonymous Walk Embeddings (Ivanov and Burnaev) can be found to function within a common underlying framework which can be described in two steps.

1. **Decomposition of graphs into descriptive substructures:** Each graph in the dataset of graphs is reduced to a set of induced substructure patterns according to some reduction algorithm. An example is finding all the rooted subgraphs of a graph using the Weisfeiler-Lehman algorithm (Shervashidze et al., 2011), other examples of substructure can be induced graphlets, walks, and paths to name a few. The union of the substructure patterns of each graph across the dataset defines a common "vocabulary" that can describe a graph in relation to another graph based on the induced subgraphs patterns of each. 
2. **Learning distributed vector representations:** The distributive hypothesis (Harris, 1954) posits that words which are used and exist within the same context have similar semantic meanings. In a similar way we may define that a graph is *contextualized* by the substructure patterns producing a new dataset of (graph, induced_subgraph_pattern) pairs. Embedding methods which exploit the distributive hypothesis such as skipgram (Mikolov et al., 2014) can then be used to learn fixed-size distributed vector embeddings of each graph in an unsupervised manner.

Once the distributed vector representations are learned for each of the graphs in a dataset. The graph embeddings may be used in any downstream application such as graph classification, regression, etc.

Geo2DR implements a variety of graph reduction and vocabulary building algorithms (such as Weisfeiler-Lehman, anonymous walks, graphlets) and learning models which exploits the distributive hypothesis (such as skipgram, skipgram+negative sampling, GLoVe). This enables the quick creation of systems capable of learning distributed representations of graphs.

## FAQ:
1. Why decompose graphs into different substructure patterns?

- *This is because the useful properties of graphs in different datasets may be better captured using specialised decompositions. There is no one-size fits all.*   
