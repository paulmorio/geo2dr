Installation
============

The current version of Geo2DR or more specifically ``geometric2dr`` relies on some functionalities in other packages which need to be installed in advance. These packages can be installed after geometric2dr but either way are necessary for the 

.. note::
    We do not recommend installation as root user on your system python.
    Please set up a virtual environment with a virtualenv. This can be done with a command such as ``virtualenv -p python3 venv`` which creates a new ``venv`` which contains its self-contained python3 instance. The rest of this document assumes a virtual environment is being used and is activated.

``geometric2dr``'s two main dependencies are PyNauty and PyTorch. PyNauty is used for efficient hash or certificate generation of topological structures used in some decomposition algorithms. PyTorch is the machine learning and numerical framework we use to define our neural language models and related utilities.

Installing PyNauty dependency
-----------------------------

PyNauty is reliant on Nauty, a C library, and therefore requires some building and symbolic linking as part of its installation we have tried to condense this here into as few steps as possible. To build Pynauty for Geo2DR one needs:

* Python 3.6+
* The most recent version of Nauty
* An ANSI C compiler

Download PyNauty's `compressed tar sources <https://web.cs.dal.ca/~peter/software/pynauty/pynauty-0.6.0.tar.gz>`_ and unpack it. That should create a directory like pynauty-X.Y.Z/ containing the source files. 

Also download the most recent sources of `nauty <http://pallini.di.uniroma1.it/nauty26r12.tar.gz>`_. That file should be something like nautyXYZ.tar.gz.

Change to the directory pynauty-X.Y.Z/ replacing X.Y.Z with the actual version of pynauty:

.. code-block:: none

	cd pynauty-X.Y.Z

Unpack nautyXYZ.tar.gz inside the pynauty-X.Y.Z/ directory. Create a symbolic link named nauty to nauty’s source directory replacing XYZ with the actual version of Nauty and build it:

.. code-block:: none

	ln -s nautyXYZ nauty
	make pynauty

To install pynauty to the virtual environment call the following in the pynauty folder whilst the virtual environment is activated.

.. code-block:: none

	make virtenv-ins

To uninstall simply use the corresponding make command

.. code-block:: none

	make virtenv-unins

Installing PyTorch dependency
-----------------------------

Pytorch is installed based on the available hardware of the machine (GPU or CPU) please follow the appropriate pip installation on the official PyTorch website.

Installing Geo2DR
-----------------

geometric2dr can be installed either from Pypi or the sources from the GitHub repository. Pypi installation is done through pip.

.. code-block:: none

	pip install geometric2dr

Installation from sources
^^^^^^^^^^^^^^^^^^^^^^^^^

Geo2DR can be installed into the virtualenvironment from the source folder to get its latest features. If ones wishes to simply use Geo2DR modules one can use the standard pip source install from the project root folder containing `setup.py`

.. code-block:: none

	pip install .

If you wish to modify some of the source code in `geometric2dr` for your own task, you can make a developer install. It is slightly slower, but immediately takes on any changes into the source code.

.. code-block:: none

	pip install -e .

