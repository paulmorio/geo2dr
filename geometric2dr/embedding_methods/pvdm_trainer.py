"""
A trainer class which faciliates training of the embedding methods by the set hyperparameters.

Author: Paul Scherer 2020
"""
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal
from .pvdm_data_reader import PVDMCorpus
from .pvdm import PVDM
from .utils import save_graph_embeddings

# For testing
from .classify import perform_classification, cross_val_accuracy


class PVDM_Trainer(object):
    """Handles corpus construction, CBOW initialization and training.

    Paramaters
    ----------
    corpus_dir : str
            path to directory containing graph files
    extension : str
            extension used in graph documents produced after decomposition stage
    max_files : int
            the maximum number of graph files to consider, default of 0 uses all files
    window_size : int
            the number of cooccuring context subgraph patterns to use
    output_fh : str
            the path to the file where embeddings should be saved
    emb_dimension : int (default=128)
            the desired dimension of the embeddings
    batch_size : int (default=32)
            the desired batch size
    epochs : int (default=100)
            the desired number of epochs for which the network should be trained
    initial_lr : float (default=1e-3)
            the initial learning rate
    min_count : int (default=1)
            the minimum number of times a pattern should occur across the dataset to
            be considered part of the substructure pattern vocabulary

    Returns
    -------
    self : PVDM_Trainer
            A PVDM_Trainer instance
    """

    def __init__(
        self,
        corpus_dir,
        extension,
        max_files,
        window_size,
        output_fh,
        emb_dimension=128,
        batch_size=32,
        epochs=100,
        initial_lr=1e-3,
        min_count=1,
    ):
        self.corpus = PVDMCorpus(
            corpus_dir, extension, max_files, min_count, window_size
        )
        self.dataloader = DataLoader(
            self.corpus,
            batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.corpus.collate,
        )

        self.corpus_dir = corpus_dir
        self.extension = extension
        self.max_files = max_files
        self.output_fh = output_fh
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.min_count = min_count
        self.window_size = window_size

        self.num_targets = self.corpus.num_graphs
        self.vocab_size = self.corpus.num_subgraphs

        self.pvdm = PVDM(self.num_targets, self.vocab_size, self.emb_dimension)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.pvdm.cuda()
        else:
            self.device = torch.device("cpu")

    def train(self):
        """Train the network with the settings used to initialise the PVDM_Trainer"""

        for epoch in range(self.epochs):
            print("### Epoch: " + str(epoch))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adagrad(
                self.pvdm.parameters(), lr=self.initial_lr, lr_decay=0.00001
            )

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                if len(sample_batched[0]) > 1:
                    pos_target_graph = sample_batched[0].to(self.device)
                    pos_target_subgraph = sample_batched[1].to(self.device)
                    pos_contexts_for_subgraph_target = sample_batched[2].to(self.device)
                    pos_negatives = sample_batched[3].to(self.device)

                    optimizer.zero_grad()
                    loss = self.pvdm.forward(
                        pos_target_graph,
                        pos_target_subgraph,
                        pos_contexts_for_subgraph_target,
                        pos_negatives,
                    )
                    loss.backward()

                    # log_probs = self.pvdm(pos_target_graph, pos_target_subgraph, pos_contexts_for_subgraph_target, pos_negatives)
                    # print(log_probs)
                    # print("Most Likely Class %s " % ([np.argmax(x.detach().numpy()) for x in log_probs] ) )
                    # print("True Class %s " % (pos_target_subgraph.detach().numpy()))
                    # loss = criterion(log_probs, torch.tensor(pos_target_subgraph, dtype=torch.long))

                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
            print(" Loss: " + str(running_loss))

        final_embeddings = self.pvdm.target_embeddings.weight.cpu().data.numpy()
        save_graph_embeddings(self.corpus, final_embeddings, self.output_fh)


# Some test code
if __name__ == "__main__":
    corpus_dir = "../data/dortmund_gexf/MUTAG"  # A needed parameter
    extension = ".awe_8_nodes"
    output_file = "PVDMEmbeddings.json"  # A needed parameter
    emb_dimension = 64  # A needed parameter
    batch_size = 32  # A needed parameter
    epochs = 250  # A needed parameter
    initial_lr = 0.00001  # A needed parameter
    min_count = 0  # A needed parameter
    window_size = 5

    trainer = PVDM_Trainer(
        corpus_dir=corpus_dir,
        extension=extension,
        max_files=0,
        window_size=window_size,
        output_fh=output_file,
        emb_dimension=emb_dimension,
        batch_size=batch_size,
        epochs=epochs,
        initial_lr=initial_lr,
        min_count=min_count,
    )
    trainer.train()

    final_embeddings = trainer.pvdm.give_target_embeddings()
    graph_files = trainer.corpus.graph_fname_list
    class_labels_fname = "../data/MUTAG.Labels"
    embedding_fname = trainer.output_fh
    classify_scores = cross_val_accuracy(
        corpus_dir, trainer.corpus.extension, embedding_fname, class_labels_fname
    )
    mean_acc, std_dev = classify_scores
    print(
        "Mean accuracy using 10 cross fold accuracy: %s with std %s"
        % (mean_acc, std_dev)
    )
