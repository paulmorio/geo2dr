"""
A class which faciliates training of the embedding methods by the set hyperparameters.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal 
from embedding_methods.data_reader import Corpus
from embedding_methods.skipgram import Skipgram
from embedding_methods.utils import get_files, get_class_labels, get_class_labels_tuples, save_graph_embeddings

# For testing
from embedding_methods.classify import perform_classification, cross_val_accuracy

class Trainer(object):
	def __init__(self, corpus_dir, extension, max_files, output_fh, emb_dimension=128, batch_size=32, epochs=100, initial_lr=1e-3, min_count=1):
		self.corpus = Corpus(corpus_dir, extension, max_files, min_count)
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

		self.num_targets = self.corpus.num_graphs
		self.vocab_size = self.corpus.num_subgraphs

		self.skipgram = Skipgram(self.num_targets, self.vocab_size, self.emb_dimension)

		if torch.cuda.is_available():
			self.device = torch.device("cuda")
			self.skipgram.cuda()
		else:
			self.device = torch.device("cpu")

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

					optimizer.zero_grad()
					loss = self.skipgram.forward(pos_target, pos_context, neg_context) # the loss is integrated into the forward function
					loss.backward()
					optimizer.step()
					scheduler.step()

					running_loss = running_loss * 0.9 + loss.item() * 0.1
					if i>0 and i%500==0:
						print(" Loss: " + str(running_loss))

		final_embeddings = self.skipgram.target_embeddings.weight.cpu().data.numpy()
		save_graph_embeddings(self.corpus, final_embeddings, self.output_fh)

# Some test code
if __name__ == '__main__':

	corpus_dir = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/dortmund_gexf/MUTAG" # A needed parameter
	extension = ".wld2"
	output_file = "Embeddings.json" # A needed parameter
	emb_dimension = 64 # A needed parameter
	batch_size = 16 # A needed parameter
	epochs = 500 # A needed parameter
	initial_lr = 0.001 # A needed parameter
	min_count= 0 # A needed parameter

	trainer = Trainer(corpus_dir=corpus_dir, extension=extension, max_files=0, 
					output_fh=output_file, emb_dimension=emb_dimension, batch_size=batch_size,
					epochs=epochs, initial_lr=initial_lr, min_count=min_count)
	trainer.train()

	final_embeddings = trainer.skipgram.give_target_embeddings()
	graph_files = trainer.corpus.graph_fname_list
	class_labels_fname = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/MUTAG.Labels"
	embedding_fname = trainer.output_fh
	classify_scores = cross_val_accuracy(corpus_dir, trainer.corpus.extension, embedding_fname, class_labels_fname)
	mean_acc, std_dev = classify_scores
	print("Mean accuracy using 10 cross fold accuracy: %s with std %s" % (mean_acc, std_dev))

