Downstream applications of embeddings
=====================================

After the  are learned in an unsupervised manner with the neural embedding method. The distributed vector representations of graphs (or substructures) are amenable to any downstream application. 

If using one of the `trainer` tools as seen in the Graph2Vec example on the repository, one can use the `give_target_embeddings()` method after training the embedding method (here a skipgram model) to get a `numpy.ndarray` of the embeddings. 

.. code-block:: python

	trainer = InMemoryTrainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, output_fh=output_embedding_fh,
	                  emb_dimension=32, batch_size=128, epochs=250, initial_lr=0.1,
	                  min_count=min_count_patterns)
	trainer.train()

	target_embeddings = trainer.skipgram.give_target_embeddings()

Practical notes on comparing methods of distributed representations of graphs.
------------------------------------------------------------------------------

Some practical notes to consider when you create a new method for learning distributed representations of graphs and you wish to compare the distributed representations against existing methods.