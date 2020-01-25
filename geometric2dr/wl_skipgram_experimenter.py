"""
Experiment script which runs sets up and builds distributed representations of graphs using 
the WL decomposition algorithm at different hyperparameter values.

Then we build the distributed representations using the skipgram with different:
- embedding_dimensions
- batch_size
- epochs
- initial_learning rate (fixed for now)
- min_count (on number of subgraph patterns)

Author: Paul Scherer
"""
import embedding_methods.utils as utils
from decomposition.weisfeiler_lehman_patterns import wlk_relabeled_corpus
from embedding_methods.classify import cross_val_accuracy
from embedding_methods.trainer import Trainer

min_counts = [1, 2]
wl_degrees = [1, 2, 3]
embedding_dimensions = [8, 16, 32, 64, 128]
batch_size = 128
runs = 5
epochs = 100
initial_lr = 0.001
embedding_folder = "embeddings"

dataset = "MUTAG"
corpus_dir = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/dortmund_gexf/" + dataset

for wl_h in wl_degrees:
    extension = ".wld" + str(wl_h)
    for embedding_dimension in embedding_dimensions:
        for min_count in min_counts:
            for run in range(runs):
                # Set the output file name for the embeddings
                output_fh = str.join("_", [dataset, "WL", str(wl_h), "embd", str(embedding_dimension), "epochs", str(
                    epochs), "bs", str(batch_size), "minCount", str(min_count), "run", str(run)])
                output_fh += ".json"
                output_fh = embedding_folder + "/" + output_fh

                # Run the decomposition algorithm
                graph_files = utils.get_files(corpus_dir, ".gexf", max_files=0)
                print("###.... Loaded %s files in total" %
                      (str(len(graph_files))))
                print(output_fh)
                # This takes the gexf graphfiles and produces .wld<wl_h> graph document files
                wlk_relabeled_corpus(graph_files, wl_h)

                # Use the graph documents in the directory to create a trainer which handles creation of datasets/corpus/dataloaders
                # and the skipgram model.
                trainer = Trainer(corpus_dir=corpus_dir, extension=extension, max_files=0, output_fh=output_fh,
                                  emb_dimension=embedding_dimension, batch_size=batch_size, epochs=epochs, initial_lr=initial_lr,
                                  min_count=min_count)
                trainer.train()

                # Classification if needed
                final_embeddings = trainer.skipgram.give_target_embeddings()
                graph_files = trainer.corpus.graph_fname_list
                class_labels_fname = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/MUTAG.Labels"
                embedding_fname = trainer.output_fh
                classify_scores = cross_val_accuracy(corpus_dir, trainer.corpus.extension, embedding_fname, class_labels_fname)
                mean_acc, std_dev = classify_scores
                print("Mean accuracy using 10 cross fold accuracy: %s with std %s" % (
                    mean_acc, std_dev))
