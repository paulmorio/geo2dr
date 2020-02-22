"""
Module containining class definitions of trainers for cbow models, 
which are partly used by Deep Graph Kernels

Author: Paul Scherer
"""


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal 
from .cbow_data_reader import CbowCorpus
from .cbow import Cbow
from .utils import save_subgraph_embeddings

class Trainer(object):
	def __init__(self, corpus_dir, extension, max_files, window_size, output_fh, emb_dimension=128, batch_size=32, epochs=100, initial_lr=1e-3, min_count=1):
		self.corpus = CbowCorpus(corpus_dir, extension, max_files, min_count, window_size)
		self.dataloader = DataLoader(self.corpus, batch_size, shuffle=False, num_workers=0, collate_fn = self.corpus.collate)

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

		self.num_targets = self.corpus.num_subgraphs # the special feature here is that we are learning subgraph reps
		self.vocab_size = self.corpus.num_subgraphs

		self.cbow = Cbow(self.num_targets, self.vocab_size, self.emb_dimension)

		if torch.cuda.is_available():
			self.device = torch.device("cuda")
			self.cbow.cuda()
		else:
			self.device = torch.device("cpu")

	def train(self):
		for epoch in range(self.epochs):
			print("### Epoch: " + str(epoch))
			optimizer = optim.Adagrad(self.cbow.parameters(), lr=self.initial_lr, lr_decay=0.00001)

			running_loss = 0.0
			for i, sample_batched in enumerate(tqdm(self.dataloader)):

				if len(sample_batched[0]) > 1:
					pos_target = sample_batched[0].to(self.device)
					pos_context = sample_batched[1].to(self.device)
					neg_context = sample_batched[2].to(self.device)

					optimizer.zero_grad()
					loss = self.cbow.forward(pos_target, pos_context, neg_context) # the loss is integrated into the forward function
					loss.backward()
					optimizer.step()

					running_loss = running_loss * 0.9 + loss.item() * 0.1
			print(" Loss: " + str(running_loss))

		final_embeddings = self.cbow.target_embeddings.weight.cpu().data.numpy()
		save_subgraph_embeddings(self.corpus, final_embeddings, self.output_fh)
		return(final_embeddings)