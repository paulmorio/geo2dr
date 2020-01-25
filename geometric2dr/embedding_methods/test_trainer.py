"""
A development trainer
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# internal
from data_reader import Corpus
from skipgram import Skipgram
from utils import get_files, get_class_labels, get_class_labels_tuples, save_graph_embeddings


# For the classification bit
from classify import perform_classification, cross_val_accuracy



corpus_dir = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/dortmund_gexf/MUTAG" # A needed parameter
corpus = Corpus(corpus_dir, extension=".wld2", max_files=0, min_count=0)
dataloader = DataLoader(corpus, batch_size=4, shuffle=False, num_workers=0, collate_fn=corpus.collate)


output_file = "Embeddings.testfile" # A needed parameter
output_file_name = output_file
num_targets = corpus.num_graphs
vocab_size = corpus.num_subgraphs
emb_dimension = 100 # A needed parameter
batch_size = 256 # A needed parameter
epochs = 1000 # A needed parameter
initial_lr = 0.001 # A needed parameter
skipgram = Skipgram(num_targets, vocab_size, emb_dimension)

if torch.cuda.is_available():
	device = torch.device("cuda")
	skipgram.cuda()
else:
	device = torch.device("cpu")

for epoch in range(epochs):
	print("### Epoch: " + str(epoch))
	optimizer = optim.SparseAdam(skipgram.parameters(), lr=initial_lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

	running_loss = 0.0
	for i, sample_batched in enumerate(tqdm(dataloader)):
	# for i, sample_batched in enumerate(dataloader):

		if len(sample_batched[0]) > 1:
			pos_target = sample_batched[0].to(device)
			pos_context = sample_batched[1].to(device)
			neg_context = sample_batched[2].to(device)

			optimizer.zero_grad()
			loss = skipgram.forward(pos_target, pos_context, neg_context) # the loss is integrated into the forward function
			loss.backward()
			optimizer.step()
			scheduler.step()

			running_loss = running_loss * 0.9 + loss.item() * 0.1
			if i>0 and i%500==0:
				print(" Loss: " + str(running_loss))

# skipgram.save_embedding(corpus._id_to_graph_name_map, output_file_name)

final_embeddings = skipgram.target_embeddings.weight.cpu().data.numpy()
save_graph_embeddings(corpus, final_embeddings, "jsonEmbeddings.json")


### Downstream classification.
graph_files = corpus.graph_fname_list
class_labels_fname = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/MUTAG.Labels"
graph_and_class_tuples = get_class_labels_tuples(graph_files, class_labels_fname)

embedding_fname = "jsonEmbeddings.json"
classify_scores = cross_val_accuracy(corpus_dir, corpus.extension, embedding_fname, class_labels_fname)
mean_acc, std_dev = classify_scores
print("Mean accuracy using 10 cross fold accuracy: %s with std %s" % (mean_acc, std_dev))