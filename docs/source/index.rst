:github_url: https://github.com/paulmorio/geo2dr

Welcome to Geo2DR's Docs!
==================================

**Geo2DR** is a Python library for learning distributed representations of graphs. Here, embeddings of *substructure patterns* (walks, graphlets, trees, etc.) and *whole graphs* are learned by exploiting the distributive hypothesis used in statistical language modelling. It attempts to make the various algorithms used for inducing discrete structures, creating corpi, training neural language models available under a simple easy to use library. 

.. note::
   This is an early-stage project and a lot more documentation is planned to be included (you can view the raw rst file to see the upcoming tutorials and reference materials to be included in the coming weeks). Its quite tough going at it alone!

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting Started:

   getting_started/installation
   getting_started/understandingdrg
   getting_started/introduction
   getting_started/dataformatting
   getting_started/decomposition
   getting_started/learning
   getting_started/downstream
   license

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/geometric2dr.data
   modules/geometric2dr.decomposition
   modules/geometric2dr.embedding_methods

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

If the code or specific examples of existing systems is useful to your work, please cite the original works and also consider citing Geo2DR.

.. code-block:: latex

   @misc{geometric2dr,
         title={Learning distributed representations of graphs with Geo2DR},
         author={Paul Scherer and Pietro Lio},
         year={2020},
         eprint={2003.05926},
         archivePrefix={arXiv},
         primaryClass={cs.LG}
   }