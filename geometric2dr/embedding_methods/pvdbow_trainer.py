"""Module containining class definitions of trainers for pvdbow models [6]_, 
which are partly used by Deep Graph Kernels [2]_

Author: Paul Scherer
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal 
from .pvdbow_data_reader import PVDBOWCorpus, PVDBOWInMemoryCorpus
from .skipgram import Skipgram
from .utils import save_graph_embeddings

# # For testing
# from embedding_methods.classify import perform_classification, cross_val_accuracy

class Trainer(object):
	"""Handles corpus construction (hard drive version), PVDBOW initialization and training.

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
	self : Trainer
		A Trainer instance
	"""

	def __init__(self, corpus_dir, extension, max_files, output_fh, emb_dimension=128, batch_size=32, epochs=100, initial_lr=1e-3, min_count=1):
		self.corpus = PVDBOWCorpus(corpus_dir, extension, max_files, min_count)
		self.dataloader = DataLoader(self.corpus, batch_size, shuffle=False, num_workers=4, collate_fn = self.corpus.collate)
		
		self.corpus_dir = corpus_dir
		self.extension = extension
		self.max_files = max_files
		self.output_fh = output_fh
		self.emb_dimension = emb_dimension
		self.batch_size = batch_size
		self.epochs = epochs
		self.initial_lr = initial_lr
		self.min_count = min_count

		self.num_targets = self.corpus.num_graphs
		self.vocab_size = self.corpus.num_subgraphs

		self.skipgram = Skipgram(self.num_targets, self.vocab_size, self.emb_dimension)

		if torch.cuda.is_available():
			self.device = torch.device("cuda")
			self.skipgram.cuda()
		else:
			self.device = torch.device("cpu")

	def train(self):
		"""Train the network with the settings used to initialise the Trainer
		
		"""
		for epoch in range(self.epochs):
			print("### Epoch: " + str(epoch))
			optimizer = optim.SparseAdam(self.skipgram.parameters(), lr=self.initial_lr)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

			running_loss = 0.0
			for sample_batched in tqdm(self.dataloader):

				if len(sample_batched[0]) > 1:
					pos_target = sample_batched[0].to(self.device)
					pos_context = sample_batched[1].to(self.device)
					neg_context = sample_batched[2].to(self.device)

					optimizer.zero_grad()
					loss = self.skipgram.forward(pos_target, pos_context, neg_context) # the loss is integrated into the forward function
					loss.backward()
					optimizer.step()
					scheduler.step()

					running_loss = running_loss * 0.9 + loss.item() * 0.1
			print(" Loss: " + str(running_loss))

		final_embeddings = self.skipgram.target_embeddings.weight.cpu().data.numpy()
		save_graph_embeddings(self.corpus, final_embeddings, self.output_fh)


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
	def __init__(self, corpus_dir, extension, max_files, output_fh, emb_dimension=128, batch_size=32, epochs=100, initial_lr=1e-3, min_count=1):
		self.corpus = PVDBOWInMemoryCorpus(corpus_dir, extension, max_files, min_count)
		self.dataloader = DataLoader(self.corpus, batch_size, shuffle=False, num_workers=4, collate_fn = self.corpus.collate)
		
		self.corpus_dir = corpus_dir
		self.extension = extension
		self.max_files = max_files
		self.output_fh = output_fh
		self.emb_dimension = emb_dimension
		self.batch_size = batch_size
		self.epochs = epochs
		self.initial_lr = initial_lr
		self.min_count = min_count

		self.num_targets = self.corpus.num_graphs
		self.vocab_size = self.corpus.num_subgraphs

		self.skipgram = Skipgram(self.num_targets, self.vocab_size, self.emb_dimension)

		if torch.cuda.is_available():
			self.device = torch.device("cuda")
			self.skipgram.cuda()
		else:
			self.device = torch.device("cpu")

	def train(self):
		"""Train the network with the settings used to initialise the Trainer
		
		"""
		for epoch in range(self.epochs):
			print("### Epoch: " + str(epoch))
			optimizer = optim.SGD(self.skipgram.parameters(), lr=self.initial_lr)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

			running_loss = 0.0
			for i, sample_batched in enumerate(self.dataloader):

				if len(sample_batched[0]) > 1:
					pos_target = sample_batched[0].to(self.device)
					pos_context = sample_batched[1].to(self.device)
					neg_context = sample_batched[2].to(self.device)

					optimizer.zero_grad()
					loss = self.skipgram.forward(pos_target, pos_context, neg_context) # the loss is integrated into the forward function
					loss.backward()
					optimizer.step()
					scheduler.step()

					running_loss = running_loss * 0.9 + loss.item() * 0.1
			print(" Loss: " + str(running_loss))

		final_embeddings = self.skipgram.target_embeddings.weight.cpu().data.numpy()
		save_graph_embeddings(self.corpus, final_embeddings, self.output_fh)

# Some test code
if __name__ == '__main__':

	corpus_dir = "../data/dortmund_gexf/MUTAG" # A needed parameter
	extension = ".wld2"
	output_file = "Embeddings.json" # A needed parameter
	emb_dimension = 64 # A needed parameter
	batch_size = 16 # A needed parameter
	epochs = 500 # A needed parameter
	initial_lr = 0.001 # A needed parameter
	min_count= 0 # A needed parameter

	trainer = InMemoryTrainer(corpus_dir=corpus_dir, extension=extension, max_files=0, 
					output_fh=output_file, emb_dimension=emb_dimension, batch_size=batch_size,
					epochs=epochs, initial_lr=initial_lr, min_count=min_count)
	trainer.train()

	final_embeddings = trainer.skipgram.give_target_embeddings()
	graph_files = trainer.corpus.graph_fname_list
