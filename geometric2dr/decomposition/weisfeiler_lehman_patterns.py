"""Functions for inducing Weisfeiler Lehman graph decomposition algorithm via node relabeling
as described in Shervashidze et al. [3]_.

Based on the implementation available in the original source code of Graph2Vec [4]_
and adapted for Geo2DR https://github.com/MLDroid/graph2vec_tf which has no license

"""

# Author: Paul Scherer 2019.


import os
import glob
from time import time
import networkx as nx

# Global label_to_compressed_label_map in its initial empty state 
label_to_compressed_label_map = {}

# Function to get the node label (ignore the WLk_h id before it) as an int
get_int_node_label = lambda l: int(l.split('+')[-1])

def initial_relabel(graph, node_label_attr_name="Label"): 
    """The initial relabeling of the graphs in the dataset. 

    Taking the attributed label of the node as stated by the attr 
    in the gexf file, it gives new labels to these node types
    (without regard for neighbours) hence it really is just a relabeling
    of the initial labels into our own "0+<newlabel>" format (to use 
    with WL relabeling scheme after

    Parameters
    ----------
    graph : networkx graph
        a networkx graph
    node_label_attr_name : str
        string literal of the attribute name used as a label in the 
        gexf file NB: "label" in the gexf refers to nodeID "Label"
        refers to the dataset node label
    
    Returns
    -------
    graph : networkx graph
        the same nx graph but with a new "relabel" attribute with
        the 0th wlk-h entry label

    """
    global label_to_compressed_label_map # This is the global WL hash function for compresssed labels

    nx.convert_node_labels_to_integers(graph, first_label=0)
    for node in graph.nodes(): graph.nodes[node]['relabel'] = {} # make a dictionary attribute

    # Check for previous labelings otherwise we relabel
    for node in graph.nodes():
        try:
            label = graph.nodes[node][node_label_attr_name]
        except:
            # no node label in node_label_attr_name that is specifically pulled from the gexf file so
            graph.nodes[node]['relabel'][0] = '0+0'
            continue

        if not label in label_to_compressed_label_map:
            # if no label start with 1 and increment every time a new node label is seen
            compressed_label = len(label_to_compressed_label_map)+1
            label_to_compressed_label_map[label] = compressed_label
            graph.nodes[node]['relabel'][0] = '0+' + str(compressed_label)
        else:
            # if it already has a label we just keep the same label
            graph.nodes[node]['relabel'][0] = '0+' + str(label_to_compressed_label_map[label])

    return graph

def wl_relabel(graph, it):
    """Runs an iteration of the WL relabeling algorithm, onto
    the graph using the global label_to_compressed_label_map
    
    Parameters
    ----------
    graph : networkx graph
        a networkx graph from the dataset which we want to relabel 
        (has had to be initally relabeled, ie have the graph.nodes.[node]['relabel'] attribute)
    it : int
        an int, signifiying iteration in the WL relabeling algorithm.

    Returns
    -------
    graph : networkx graph
        the input nx graph with more labels in the "relabel" attribute

    """
    global label_to_compressed_label_map # This is the global hash function for compression

    prev_iter = it - 1
    for node in graph.nodes():
        prev_iter_node_label = get_int_node_label(graph.nodes[node]['relabel'][prev_iter]) # just a int ("1") in first it 0
        node_label = [prev_iter_node_label]
        neighbours = list(nx.all_neighbors(graph, node))
        neighbourhood_label = sorted([get_int_node_label(graph.nodes[nei]['relabel'][prev_iter]) for nei in neighbours])
        node_neighbourhood_label = tuple(node_label + neighbourhood_label)
        

        if not node_neighbourhood_label in label_to_compressed_label_map:
            compressed_label = len(label_to_compressed_label_map)+1
            label_to_compressed_label_map[node_neighbourhood_label] = compressed_label
            graph.nodes[node]['relabel'][it] = str(it) + "+" + str(compressed_label)
        else:
            graph.nodes[node]['relabel'][it] = str(it) + "+" + str(label_to_compressed_label_map[node_neighbourhood_label])

    return graph


def save_wl_doc(fname,max_h,graph=None):
    """Saves the induced rooted subgraph patterns 

    Saves the subgraph sentences in format <center> <context> <context> ....
    In other words we are saving the relabelings of node from the WL algorithm 
    into a text document which can be fed into our skipgram architecture
    
    Parameters
    ----------
    fname : str
        path/filename of the graph
    max_h : int
        highest_iteration wlk_h specified (ie depth of rooted subgraph)
    graph : networkx graph
        the nx graph of the filename

    Returns
    -------
    None : None 
        The rooted subgraph patterns are saved into a text file in the 
        format <center> <context> <context> <context> ....

    """

    open_fname = fname + '.wld' + str(max_h)

    # # no need to write if it already exists
    # if os.path.isfile(open_fname):
    #     return

    # otherwise we write into the file
    with open(open_fname,'w') as fh:
        for n,d in graph.nodes(data=True):
            for it in range(0, max_h+1):
                try:
                    center = d['relabel'][it]
                except:
                    continue
                # neis_labels_prev_deg = []
                # neis_labels_next_deg = []

                # if it != 0:
                #     neis_labels_prev_deg = list(set([graph.nodes[nei]['relabel'][it-1] for nei in nx.all_neighbors(graph, n)]))
                #     neis_labels_prev_deg.sort()
                NeisLabelsSameDeg = list(set([graph.nodes[nei]['relabel'][it] for nei in nx.all_neighbors(graph,n)])) # neighbours  on iteration it basically
                # if it != max_h:
                #     neis_labels_next_deg = list(set([graph.nodes[nei]['relabel'][it+1] for nei in nx.all_neighbors(graph,n)]))
                #     neis_labels_next_deg.sort()

                # nei_list = NeisLabelsSameDeg + neis_labels_prev_deg + neis_labels_next_deg
                nei_list = NeisLabelsSameDeg
                nei_list = ' '.join(nei_list)

                sentence = center + ' ' + nei_list
                print(sentence, file=fh)
                

def wl_corpus(fnames, max_h, node_label_attr_name='Label'):
    """Induce rooted subgraph patterns using the WL node relabeling 
    algorith given list gexf files and save corresponding graph files.

    Given a set of graphs from the dataset, a maximum h for WL, and 
    the label attribute name used in the gexf files we initially relabel
    the original labels into a compliant relabeling (caesar shift for 
    ease) then perform max-h iterations of the WL relabeling algorithm 
    (1979) to create new labels which are compressed versions of the 
    rooted subgraphs for each node in the graph. These are all present
    in the nx graph objects's nodes as attributes, with the original 
    label being 'Label' and our subsequent relabelings in the "relabel"
    attribute
    
    The main use case is for the user to input a path containing individual
    graphs of the dataset in gexf format.
    The decomposition algorithm will induce substructure patterns for graphs
    recording the dataset/"global" vocabulary of patterns within a dictionary.
    The graph and its associated patterns (by IDs given through our hash function)
    are saved into a  <graphid>.wldr<depth> file which contains a line delimited
    list of all the substructure pattern ids.

    Parameters
    ----------
    fnames : list
        list of gexf file paths for the graphs in the dataset
    max_h : int
        the maximum depth of rooted subgraph pattern to induce across 
        the dataset of graphs
    node_label_attr_name : str
        string literal of the attribute name used as a label in the 
        gexf file NB: "label" in the gexf refers to nodeID "Label"
        refers to the dataset node label

    Returns
    -------
    corpus : list of lists of str
        a list of lists, with each inner list containing all the rooted subgraph patterns in one graph of the dataset
    vocabulary : list
        a set of the unique rooted subgraph pattern ids
    prob_map : dict
        a map {gidx: {wl_pattern: normalized_prob}} of normalized probabilities of a rooted subgraph pattern appearing in a graph based on counts made in generation
    num_graphs : int
        the number of graphs in the dataset
    graph_map : dict
        a map {gidx: {wl_pattern: count}} of the number of times a certain rooted subgraph pattern appeared in a graph for each graph gidx in the dataset

    None : None
        The rooted subgraph patterns are also saved into a text file for each graph in 'fnames' in the 
        format <center> <context> <context> <context> ....

    """
    global label_to_compressed_label_map
    compressed_labels_map_list = [] # list of compressed labels maps that can be used to go backwards

    # Read each graph as a networkx graph
    graphs = [nx.read_gexf(fname) for fname in fnames]
    assert len(graphs) > 0, "fnames parameter does not contain valid .gexf files"
    print ('#... Loaded all the graphs')

    # Do an initial relabeling of each nxgraph g in graphs
    graphs = [initial_relabel(g, node_label_attr_name) for g in graphs]
    print ('#... initial relabeling done in')

    # Perform the Weisfeiler-Lehman Relabeling Process for h iterations (up to h depth rooted subgraphs)
    for it in range(1, max_h + 1):
        t0 = time()
        compressed_labels_map_list.append(label_to_compressed_label_map)
        label_to_compressed_label_map = {}
        graphs = [wl_relabel(g, it) for g in graphs]
        print ('WL iteration {} done in {} sec.'.format(it, round(time() - t0, 2)))
        print ('num of WL rooted subgraphs in iter {} is {}'.format(it, len(label_to_compressed_label_map)))

    # Save the patterns into graph documents
    for fname, g in zip(fnames, graphs):
        save_wl_doc(fname, max_h, g)

    # Match return signatures of other decomposition algorithms
    corpus = []
    vocabulary = set()
    graph_map = {}

    for fname, g in zip(fnames, graphs):
        gidx = int((os.path.basename(fname)).replace(".gexf", ""))
        tmp_corpus = []
        count_map = {}
        for n, d in g.nodes(data=True):
            for it in range(0, max_h+1):
                try:
                    pattern_at_node = d['relabel'][it]
                    vocabulary.add(pattern_at_node)
                    tmp_corpus.append(pattern_at_node)
                    count_map[pattern_at_node] = count_map.get(pattern_at_node, 0) + 1
                except:
                    continue      

                NeisLabelsSameDeg = list(set([g.nodes[nei]['relabel'][it] for nei in nx.all_neighbors(g,n)]))
                for nei_pattern in NeisLabelsSameDeg:
                    vocabulary.add(nei_pattern)
                    tmp_corpus.append(nei_pattern)
                    count_map[nei_pattern] = count_map.get(nei_pattern, 0) + 1
        
        corpus.append(tmp_corpus)
        graph_map[gidx] = count_map

    # Normalise the probabilities of a graphlet in a graph.
    prob_map = {gidx: {graphlet: count/float(sum(graphlets.values())) \
        for graphlet, count in graphlets.items()} for gidx, graphlets in graph_map.items()}
    num_graphs = len(prob_map)          

    return corpus, vocabulary, prob_map, num_graphs, graph_map

# Manual test
if __name__ == "__main__":
    ip_folder ="../data/dortmund_gexf/MUTAG"
    max_h = 2

    all_files = sorted(glob.glob(os.path.join(ip_folder, '*gexf')))
    print("Loaded %s files in total" % (str(len(all_files))))

    corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(all_files, max_h)