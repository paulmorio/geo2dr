"""
Module containining class definitions of trainers for skipgram models, 
which are partly used by Deep Graph Kernels

Author: Paul Scherer
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal
from .skipgram_data_reader import SkipgramCorpus, InMemorySkipgramCorpus
from .skipgram import Skipgram
from .utils import save_subgraph_embeddings


class Trainer(object):
    """Handles corpus construction (hard drive version), skipgram initialization and training.

    Parameters
    ----------
    corpus_dir : str
            path to directory containing graph files
    extension : str
            extension used in graph documents produced after decomposition stage
    max_files : int
            the maximum number of graph files to consider, default of 0 uses all files
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
    self : Trainer
            A Trainer instance

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
        self.corpus = SkipgramCorpus(
            corpus_dir, extension, max_files, min_count, window_size
        )
        self.dataloader = DataLoader(
            self.corpus,
            batch_size,
            shuffle=False,
            num_workers=4,
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

        self.num_targets = (
            self.corpus.num_subgraphs
        )  # the special feature here is that we are learning subgraph reps
        self.vocab_size = self.corpus.num_subgraphs

        self.skipgram = Skipgram(self.num_targets, self.vocab_size, self.emb_dimension)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.skipgram.cuda()
        else:
            self.device = torch.device("cpu")

    def train(self):
        """Train the network with the settings used to initialise the Trainer"""
        for epoch in range(self.epochs):
            print("### Epoch: " + str(epoch))
            optimizer = optim.Adagrad(self.skipgram.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(self.dataloader)
            )

            running_loss = 0.0
            for sample_batched in tqdm(self.dataloader):
                if len(sample_batched[0]) > 1:
                    pos_target = sample_batched[0].to(self.device)
                    pos_context = sample_batched[1].to(self.device)
                    neg_context = sample_batched[2].to(self.device)

                    optimizer.zero_grad()
                    loss = self.skipgram.forward(
                        pos_target, pos_context, neg_context
                    )  # the loss is integrated into the forward function
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
            print(" Loss: " + str(running_loss))

        final_embeddings = self.skipgram.target_embeddings.weight.cpu().data.numpy()
        save_subgraph_embeddings(self.corpus, final_embeddings, self.output_fh)
        return final_embeddings


class InMemoryTrainer(object):
    """Handles corpus construction (in-memory version), PVDBOW initialization and training.

    Paramaters
    ----------
    corpus_dir : str
            path to directory containing graph files
    extension : str
            extension used in graph documents produced after decomposition stage
    max_files : int
            the maximum number of graph files to consider, default of 0 uses all files
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
    self : InMemoryTrainer
            A trainer instance which has the dataset stored in memory for fast access
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
        self.corpus = InMemorySkipgramCorpus(
            corpus_dir, extension, max_files, min_count, window_size
        )
        self.dataloader = DataLoader(
            self.corpus,
            batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
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

        self.num_targets = (
            self.corpus.num_subgraphs
        )  # the special feature here is that we are learning subgraph reps
        self.vocab_size = self.corpus.num_subgraphs

        self.skipgram = Skipgram(self.num_targets, self.vocab_size, self.emb_dimension)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.skipgram.cuda()
        else:
            self.device = torch.device("cpu")

    def train(self):
        """Train the network with the settings used to initialise the Trainer"""
        for epoch in range(self.epochs):
            print("### Epoch: " + str(epoch))
            optimizer = optim.Adagrad(self.skipgram.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(self.dataloader)
            )

            running_loss = 0.0
            for sample_batched in tqdm(self.dataloader):
                if len(sample_batched[0]) > 1:
                    pos_target = sample_batched[0].to(self.device)
                    pos_context = sample_batched[1].to(self.device)
                    neg_context = sample_batched[2].to(self.device)

                    optimizer.zero_grad()
                    loss = self.skipgram.forward(
                        pos_target, pos_context, neg_context
                    )  # the loss is integrated into the forward function
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
            print(" Loss: " + str(running_loss))

        final_embeddings = self.skipgram.target_embeddings.weight.cpu().data.numpy()
        save_subgraph_embeddings(self.corpus, final_embeddings, self.output_fh)
        return final_embeddings


if __name__ == "__main__":
    pass
