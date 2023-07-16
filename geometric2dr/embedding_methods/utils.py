"""General purpose utilities for I/O

Currently Includes:
Functions for getting all files in a directory with a given extension
Saving graph embeddings into a JSON format
Generating a dictionary matching graph files with classification labels

"""

import os
import json


def get_files(dname, extension, max_files=0):
    """Returns a list of strings which are all the files with
    the given extension in a sorted manner

    Parameters
    ----------
    dname : str
        directory with files
    extension : str
        string denoting which extension should be matched in search for files
    max_files : int (default=0)
        the maximum number of files to get, the default of 0 means all files

    Returns
    -------
    all_files : list
        list of all files matching extension inside the directory dname

    """
    all_files = [
        os.path.join(dname, f) for f in os.listdir(dname) if f.endswith(extension)
    ]

    for (
        root,
        dirs,
        files,
    ) in os.walk(dname):
        for f in files:
            if f.endswith(extension):
                all_files.append(os.path.join(root, f))

    # no duplicates
    all_files = list(set(all_files))
    all_files.sort()
    if max_files:
        return (all_files)[:max_files]
    else:
        return all_files


def save_graph_embeddings(corpus, final_embeddings, opfname):
    """Saves the trained embeddings of a corpus into a dictionary
    and saves this into a json file on the path given by opfname

    Parameters
    ----------
    corpus : corpus
        any corpus class such as `PVDBOWCorpus`
    final_embeddings : numpy ndarray
        matrix of target embeddings to be saved
    opfname : str
        path to file where embeddings should be saved in json format (extension optional in Unix)

    Returns
    -------
    None
        embeddings will be saved into path denoted by `opfname`
    """

    dict_to_save = {}
    for i in range(len(final_embeddings)):
        graph_fname = corpus._id_to_graph_name_map[i]
        graph_embedding = final_embeddings[i, :].tolist()
        dict_to_save[graph_fname] = graph_embedding

    with open(opfname, "w") as filehandler:
        json.dump(dict_to_save, filehandler, indent=4)


def save_subgraph_embeddings(corpus, final_embeddings, opfname):
    """Save the embeddings along with a map to the patterns and the corpus

    Parameters
    ----------
    corpus : corpus
        a corpus class such as SkipgramCorpus
    final_embeddings : numpy ndarray
        matrix of target embeddings to be saved
    opfname : str
        path to file where embeddings should be saved in json format

    Returns
    -------
    None
        embeddings will be saved into path denoted by `opfname`

    """
    dict_to_save = {}
    for i in range(len(final_embeddings)):
        subgraph_name = corpus._id_to_subgraph_map[i]
        subgraph_embedding = final_embeddings[i, :].tolist()
        dict_to_save[subgraph_name] = subgraph_embedding

    with open(opfname, "w") as filehandler:
        json.dump(dict_to_save, filehandler, indent=4)


def get_class_labels(graph_files, class_labels_fname):
    """Given the list of graph files (as in get_files) and
    path of the associated class labels returns the list
    of labels associated with each graph file in graph_files

    Parameters
    ----------
    graph_files : list
        list of paths to graph_files
    class_labels_fname : str
        path to class labels file (.Labels typically) with file names in `graph_files`

    Returns
    -------
    labels : list
        list of class labels for corresponding to graph files in `graph_files`

    """
    graph_to_class_label_map = {
        l.split()[0].split(".")[0]: int(l.split()[1].strip())
        for l in open(class_labels_fname)
    }
    labels = [
        graph_to_class_label_map[os.path.basename(g).split(".")[0]] for g in graph_files
    ]
    return labels


def get_class_labels_tuples(graph_files, class_labels_fname):
    """Returns list of tuples associating each of the graph files
    to their classification labels

    Parameters
    ----------
    graph_files : list
        list of paths to graph_files
    class_labels_fname : str
        path to class labels file (.Labels typically) with file names in `graph_files`

    Returns
    -------
    labels : list
        list of tuples (base_name_of_graph_file, class_label)

    """
    graph_to_class_label_map = {
        l.split()[0].split(".")[0]: int(l.split()[1].strip())
        for l in open(class_labels_fname)
    }
    labels = []
    for g in graph_files:
        g_num = os.path.basename(g).split(".")[0]
        labels.append(
            (int(g_num), graph_to_class_label_map[os.path.basename(g).split(".")[0]])
        )
    return labels


def get_kernel_matrix_row_idx_with_class(
    corpus, extension, graph_files, class_labels_fname
):
    """Returns two arrays, the first is an list of integers each referencing a row in
    a kernel matrix and thereby a kernel vector corresponding to one of the graphs
    in the dataset, the second is a list of class labels whose value is the classification
    of the graph in the same index of the first

    Parameters
    ----------
    corpus : corpus
        a corpus instance (such as SkipgramCorpus)
    extension : str
        extension of graph document under study
    graph_files : list
        list of paths to graph file
    class_labels_fname : str
        path to graph class label file

    Returns
    -------
    tuple
        kernel_row_x_id, kernel_row_y_id. The first is an list of integers each referencing a row in
        a kernel matrix and thereby a kernel vector corresponding to one of the graphs
        in the dataset, the second is a list of class labels whose value is the classification
        of the graph in the same index of the first


    """
    graph_id_to_class_tuples = []
    graph_to_class_label_map = {
        l.split()[0]: int(l.split()[1].strip()) for l in open(class_labels_fname)
    }
    for graph_fname in graph_files:
        basename = os.path.basename(graph_fname)
        clabel = graph_to_class_label_map[basename]
        gidx = corpus._graph_name_to_id_map[graph_fname + extension]
        graph_id_to_class_tuples.append((gidx, clabel))

    graph_id_to_class_tuples.sort(key=lambda tup: tup[0])
    kernel_row_x_id, kernel_row_y_id = zip(*graph_id_to_class_tuples)

    return kernel_row_x_id, kernel_row_y_id
