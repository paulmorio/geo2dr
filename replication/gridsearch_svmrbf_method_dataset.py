"""
Script to collect read the embeddings and peform a 10 fold cross validation SVM
over 10 MonteCarlo iterations to get an average and standard deviation of how 
each method performs against a dataset empirically.

It will record the scores into an pandas table and output this as a CSV
"""
import os
import numpy as np
import pandas
import geometric2dr.embedding_methods.utils as utils
from geometric2dr.embedding_methods.classify import cross_val_accuracy

# Setup parameters
method = "awe"
dataset = "MUTAG"
data_path = "data/"
corpus_data_dir = "data/" + dataset
class_labels_fh = data_path + dataset + ".Labels"


if method == "graph2vec":
	embeddings_folder = "Graph2vec_Embeddings_" + dataset
	csv_fh = method + "_" + dataset + "_results.csv"
	csv_fh_avg = method + "_" + dataset + "fullCV_results.csv"
	# Dataframe setup
	data = []
	header = ["dataset", "wl_depth", "embedding_dimension", "batch_size", "epochs", "initial_lr", "run", "accuracy", "std"]

	# if method == "graph2vec"
	embedding_files = utils.get_files(embeddings_folder, "", max_files=0)



	for embedding_file in embedding_files:
		embedding_file_basename = os.path.basename(embedding_file)
		dataset, wl_depth, embedding_dimension, batch_size, epochs, initial_lr, run = embedding_file_basename.split("_")
		print(embedding_file_basename.strip().split("_"))
		extension = ".wld" + wl_depth


		classify_scores = cross_val_accuracy(corpus_dir=corpus_data_dir, extension=extension, embedding_fname=embedding_file, class_labels_fname=class_labels_fh)
		mean_acc, std_dev = classify_scores
		print("Mean accuracy using 10 cross fold accuracy: %s with std %s" % (mean_acc, std_dev))

		data_input = [dataset, wl_depth, embedding_dimension, batch_size, epochs, initial_lr, run, mean_acc, std_dev]
		data.append(data_input)

	df = pandas.DataFrame(data, columns=header)
	df.to_csv(csv_fh)


	# Summarized table over runs
	datasets = list(df.dataset.unique())
	wl_depths = list(df.wl_depth.unique())
	embedding_dimensions = list(df.embedding_dimension.unique())
	batch_sizes = list(df.batch_size.unique())
	num_epochs = list(df.epochs.unique())
	initial_lrs = list(df.initial_lr.unique())

	# Table set up
	avg_data = []
	avg_header = ["dataset", "wl_depth", "embedding_dimension", "batch_size", "epochs", "initial_lr", "accuracy", "std", "cvs"]

	for dataset in datasets:
		for wl_depth in wl_depths:
			for embedding_dimension in embedding_dimensions:
				for batch_size in batch_sizes:
					for epochs in num_epochs:
						for initial_lr in initial_lrs:
							# subset table we are interested in
							subtable = df.loc[(df['dataset'] == dataset) & (df['wl_depth'] == wl_depth) & 
											(df['embedding_dimension'] == embedding_dimension) & (df['batch_size'] == batch_size) &
											(df['epochs'] == epochs) & (df['initial_lr'] == initial_lr)]
							cvs = len(subtable.accuracy)
							run_acc = np.mean(subtable.accuracy)
							run_std = np.std(subtable.accuracy)
							data_input = [dataset, wl_depth, embedding_dimension, batch_size, epochs, initial_lr, run_acc, run_std, cvs]
							avg_data.append(data_input)

	avg_df = pandas.DataFrame(avg_data, columns=avg_header)
	avg_df.to_csv(csv_fh_avg)


	# Sort the table in terms of accuracy
	sorted_df = avg_df.sort_values(by='accuracy', ascending=False)
	print(sorted_df)

if method=="awe":
	embeddings_folder = "AWE_Embeddings_" + dataset
	csv_fh = method + "_" + dataset + "_results.csv"
	csv_fh_avg = method + "_" + dataset + "fullCV_results.csv"
	# Dataframe setup
	data = []
	header = ["dataset", "aw_length", "embedding_dimension", "batch_size", "epochs", "initial_lr", "window_size", "run", "accuracy", "std"]

	# if method == "graph2vec"
	embedding_files = utils.get_files(embeddings_folder, "", max_files=0)



	for embedding_file in embedding_files:
		embedding_file_basename = os.path.basename(embedding_file)
		dataset, aw_length, embedding_dimension, batch_size, epochs, initial_lr, window_size, run = embedding_file_basename.split("_")
		print(embedding_file_basename.strip().split("_"))
		extension = ".awe_" + str(aw_length) + "_" + "nodes"


		classify_scores = cross_val_accuracy(corpus_dir=corpus_data_dir, extension=extension, embedding_fname=embedding_file, class_labels_fname=class_labels_fh)
		mean_acc, std_dev = classify_scores
		print("Mean accuracy using 10 cross fold accuracy: %s with std %s" % (mean_acc, std_dev))

		data_input = [dataset, aw_length, embedding_dimension, batch_size, epochs, initial_lr, window_size, run, mean_acc, std_dev]
		data.append(data_input)

	df = pandas.DataFrame(data, columns=header)
	df.to_csv(csv_fh)


	# Summarized table over runs
	datasets = list(df.dataset.unique())
	aw_lengths = list(df.aw_length.unique())
	embedding_dimensions = list(df.embedding_dimension.unique())
	batch_sizes = list(df.batch_size.unique())
	num_epochs = list(df.epochs.unique())
	initial_lrs = list(df.initial_lr.unique())
	window_sizes = list(df.window_size.unique())


	# Table set up
	avg_data = []
	avg_header = ["dataset", "aw_length", "embedding_dimension", "batch_size", "epochs", "initial_lr", "window_size", "accuracy", "std", "cvs"]

	for dataset in datasets:
		for aw_length in aw_lengths:
			for embedding_dimension in embedding_dimensions:
				for batch_size in batch_sizes:
					for epochs in num_epochs:
						for initial_lr in initial_lrs:
							for window_size in window_sizes:
								# subset table we are interested in
								subtable = df.loc[(df['dataset'] == dataset) & (df['aw_length'] == aw_length) & 
												(df['embedding_dimension'] == embedding_dimension) & (df['batch_size'] == batch_size) &
												(df['epochs'] == epochs) & (df['initial_lr'] == initial_lr) & (df["window_size"] == window_size)]
								cvs = len(subtable.accuracy)
								run_acc = np.mean(subtable.accuracy)
								run_std = np.std(subtable.accuracy)
								data_input = [dataset, aw_length, embedding_dimension, batch_size, epochs, initial_lr, window_size, run_acc, run_std, cvs]
								avg_data.append(data_input)

	avg_df = pandas.DataFrame(avg_data, columns=avg_header)
	avg_df.to_csv(csv_fh_avg)


	# Sort the table in terms of accuracy
	sorted_df = avg_df.sort_values(by='accuracy', ascending=False)
	print(sorted_df)



