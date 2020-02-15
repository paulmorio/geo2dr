"""
Weisfeiler Lehman graph decomposition algorithm via node relabeling
(Shershavidze et al) with VocabBuilder and GraphDoc outputs.

The main use case is for the user to input a path containing individual
graphs of the dataset in gexf format.
The decomposition algorithm will induce substructure patterns for graphs
recording the dataset/"global" vocabulary of patterns within a dictionary.
The graph and its associated patterns (by IDs given through our hash function)
are saved into a  <graphid>.wldr<depth> file which contains a line delimited
list of all the substructure pattern ids.

Author: Paul Scherer 2019.
"""

import os
import glob
from time import time
import networkx as nx

# Global label_to_compressed_label_map in its initial empty state 
label_to_compressed_label_map = {}

# Function to get the node label (ignore the WLk_h id before it) as an int
get_int_node_label = lambda l: int(l.split('+')[-1])

def initial_relabel(graph, node_label_attr_name="Label"): 
    """
    The initial relabeling of the graphs in the dataset. Taking the attributed label of the node as stated
    by the attr in the gexf file, it gives new labels to these node types (without regard for neighbours)
    hence it really is just a relabeling of the initial labels into our own "0+<newlabel>" format (to use 
    with WL relabeling scheme after

    :param graph: a nx graph
    :param node_label_attr_name: string literal of the attribute name used as a label in the gexf file
                                NB: "label" in the gexf refers to nodeID "Label" refers to the dataset node label
    :return graph: the same nx graph but with a new "relabel" attribute with the 0th wlk-h entry label
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
    """
    Runs an iteration of the WL relabeling algorithm, onto the graph using the global 
    label_to_compressed_label_map
    
    :param graph: an nx graph from the dataset which we want to relabel (has had to be initally relabeled, ie have the
                    graph.nodes.[node]['relabel'] attribute)
    :param it: an int, signifiying iteration in the WL relabeling algorithm.
    :return graph: the input nx graph with more labels in the "relabel" attribute
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
    """
    Dumps the subgraph sentences in format <center> <context in it> <context in it-1> <context in it+1>
    In other words we are saving the relabelings of node from the WL algo into a text document
    which can be fed into our skipgram architecture

    :param fname: path/filename of the graph
    :param max_h: highest_iteration wlk_h specified (ie depth of rooted subgraph)
    :param graph: the nx graph of the filename
    :return None: we save text into a file
    """
    open_fname = fname + '.wld' + str(max_h)

    # no need to write if it already exists
    if os.path.isfile(open_fname):
        return

    # otherwise we write into the file
    with open(open_fname,'w') as fh:
        for n,d in graph.nodes(data=True):
            for it in range(0, max_h+1):
                try:
                    center = d['relabel'][it]
                except:
                    continue
                neis_labels_prev_deg = []
                neis_labels_next_deg = []

                if it != 0:
                    neis_labels_prev_deg = list(set([graph.nodes[nei]['relabel'][it-1] for nei in nx.all_neighbors(graph, n)]))
                    neis_labels_prev_deg.sort()
                NeisLabelsSameDeg = list(set([graph.nodes[nei]['relabel'][it] for nei in nx.all_neighbors(graph,n)])) # neighbours  on iteration it basically
                if it != max_h:
                    neis_labels_next_deg = list(set([graph.nodes[nei]['relabel'][it+1] for nei in nx.all_neighbors(graph,n)]))
                    neis_labels_next_deg.sort()

                nei_list = NeisLabelsSameDeg + neis_labels_prev_deg + neis_labels_next_deg
                nei_list = ' '.join (nei_list)

                sentence = center + ' ' + nei_list
                print(sentence, file=fh)
                

def wlk_relabeled_corpus(fnames, max_h, node_label_attr_name='Label'):
    """
    Given a set of graphs from the dataset, a maximum h for WL, and the label attribute name used in 
    the gexf files we initially relabel the original labels into a compliant relabeling (caesar shift for ease)
    then perform max-h iterations of the WL relabeling algorithm (1979) to create new labels which are compressed
    versions of the rooted subgraphs for each node in the graph. These are all present in the nx graph objects's nodes
    as attributes, with the original label being 'Label' and our subsequent relabelings in the "relabel" attribute
    """

    global label_to_compressed_label_map
    compressed_labels_map_list = [] # list of compressed labels maps that can be used to go backwards

    # Read each graph as a networkx graph
    t0 = time()
    graphs = [nx.read_gexf(fname) for fname in fnames]
    print ('loaded all graphs in {} sec'.format(round(time() - t0, 2)))

    # Do an initial relabeling of each nxgraph g in graphs
    t0 = time()
    graphs = [initial_relabel(g,node_label_attr_name) for g in graphs]
    print ('initial relabeling done in {} sec'.format(round(time() - t0, 2)))

    # Perform the Weisfeiler-Lehman Relabeling Process for h iterations (up to h depth rooted subgraphs)
    for it in range(1, max_h + 1):
        t0 = time()
        compressed_labels_map_list.append(label_to_compressed_label_map)
        label_to_compressed_label_map = {}
        graphs = [wl_relabel(g, it) for g in graphs]
        print ('WL iteration {} done in {} sec.'.format(it, round(time() - t0, 2)))
        print ('num of WL rooted subgraphs in iter {} is {}'.format(it, len(label_to_compressed_label_map)))

    t0 = time()
    for fname, g in zip(fnames, graphs):
        save_wl_doc(fname, max_h, g)
    print ('dumped sentences in {} sec.'.format(round(time() - t0, 2)))
    return graphs

# Test
def main():
    ip_folder = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/dortmund_gexf/MUTAG"
    max_h = 2

    all_files = sorted(glob.glob(os.path.join(ip_folder, '*gexf')))
    print("Loaded %s files in total" % (str(len(all_files))))

    # Relabel nodes using WL algorithm and generate graph documents
    graphs = wlk_relabeled_corpus(all_files, max_h)
    return graphs

if __name__ == "__main__":
    graphs = main()

