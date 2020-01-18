"""
Training utility class. Contains the trainer class which trains various embedding learning methods.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# internal
from data_reader import DataReader, GraphCorpusDataset
from skipgram import Skipgram

class Trainer(object):
	"""A class for handling the training of various embedding methods"""
	def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, epochs=3, initial_lr=1e-3, min_count=2):
		self.data = DataReader(input_file, min_count)
		dataset = GraphCorpusDataset(self.data) # we can also set the context setting here
		self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate)

		self.output_file_name = output_file
		self.num_targets = len(self.data.graph2id)
		self.emb_dimension = emb_dimension
		self.batch_size = batch_size
		self.epochs = epochs
		self.initial_lr = initial_lr
		self.skipgram = Skipgram(self.num_targets, self.emb_dimension)

		self.use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda", if self.use_cuda else "cpu")
		if self.use_cuda:
			self.skipgram.cuda()

	def train(self):
		for epoch in range(self.epochs):
			print("### Epoch: " + str(epoch))
			optimizer = optim.SparseAdam(self.skipgram.parameters(), lr=self.initial_lr)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

			running_loss = 0.0
			for i, sample_batched in enumerate(tqdm(self.dataloader)):

				if len(sample_batched[0]) > 1:
					pos_target = sample_batched[0].to(self.device)
					pos_context = sample_batched[1].to(self.device)
					neg_context = sample_batched[2].to(self.device)

					scheduler.step()
					optimizer.zero_grad()
					loss = self.skipgram.forward(pos_target, pos_context, neg_context) # the loss is integrated into the forward function
					loss.backward()
					optimizer=step()

					running_loss = running_loss * 0.9 + loss.item() * 0.1
					if i>0 and i%500==0:
						print(" Loss: " + str(running_loss))

			self.skipgram.save_embedding(self.data.id2graph, self.output_file_name)

if __name__ == '__main__':
	g2dr = trainer(input_file='wl_graph_mutag.gcorpus', output_file="mutag_test_embeddings.vec")
	g2dr.train()