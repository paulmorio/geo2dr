"""
General purpose utilities for I/O for decomposition
"""

import os

def get_files(dname, extension, max_files=0):
    """
    Returns a list of strings which are all the files with 
    the given extension in a sorted manner
    """
    all_files = [os.path.join(dname, f) for f in os.listdir(dname) if f.endswith(extension)]

    for root, dirs, files, in os.walk(dname):
        for f in files:
            if f.endswith(extension):
                all_files.append(os.path.join(root,f))

    # no duplicates
    all_files = list(set(all_files))
    all_files.sort()
    if (max_files):
        return(all_files)[:max_files]
    else:
        return all_files