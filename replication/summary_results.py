"""
Script for summarising performance scores for model on dataset.
Primarily used for the outputs of the DGK models which
save 10 fold accuracy scores 
"""

import os
import numpy as np
import pandas
import geometric2dr.embedding_methods.utils as utils

# Setup parameters
perf_folder = "DGK_WL_Performance_MUTAG"
dgk, mode, performance, dataset = perf_folder.strip().split("_")



if mode == "GK":
	# Dataframe setup
	data = []
	header = ["dataset", "num_graphlet", "sample_size", "emb_dimension", "batch_size", "epochs", "run", "accuracy", "std"]


	perf_files = utils.get_files(perf_folder, "", max_files=0)

	for perf_file in perf_files:
		perf_file_basename = os.path.basename(perf_file)
		dataset, num_graphlet, sample_size, emb_dimension, batch_size, epochs, run  = perf_file_basename.strip().split("_")
		print(perf_file_basename.strip().split("_"))

		# Get the accuracy and the std in it
		with open(perf_file, "r") as fh:
			lines = fh.readlines()
			for line in lines:
				mean_acc, std = line.strip().split(",")
				mean_acc = float(mean_acc)
				std = float(std)

		data_input = [dataset, num_graphlet, sample_size, emb_dimension, batch_size, epochs, run, mean_acc, std]
		data.append(data_input)

	df = pandas.DataFrame(data, columns=header)
	print(df)

	# Summarized table over runs
	datasets = list(df.dataset.unique())
	num_graphlets = list(df.num_graphlet.unique())
	sample_sizes = list(df.sample_size.unique())
	embedding_dimensions = list(df.emb_dimension.unique())
	batch_sizes = list(df.batch_size.unique())
	num_epochs = list(df.epochs.unique())

	# Table set up
	avg_data = []
	avg_header = ["dataset", "num_graphlet", "sample_size", "emb_dimension", "batch_size", "epochs", "accuracy", "std", "cvs"]

	for dataset in datasets:
		for num_graphlet in num_graphlets:
			for embedding_dimension in embedding_dimensions:
				for batch_size in batch_sizes:
					for epochs in num_epochs:
						# subset table we are interested in
						subtable = df.loc[(df['dataset'] == dataset) & (df['num_graphlet'] == num_graphlet) & (df['sample_size'] == sample_size) &
										(df['emb_dimension'] == embedding_dimension) & (df['batch_size'] == batch_size) &
										(df['epochs'] == epochs)]
						cvs = len(subtable.accuracy)
						run_acc = np.mean(subtable.accuracy)
						run_std = np.std(subtable.accuracy)
						data_input = [dataset, num_graphlet, sample_size, embedding_dimension, batch_size, epochs, run_acc, run_std, cvs]
						avg_data.append(data_input)

	avg_df = pandas.DataFrame(avg_data, columns=avg_header)

	# Sort the table in terms of accuracy
	sorted_df = avg_df.sort_values(by='accuracy', ascending=False)
	print(sorted_df)

if mode == "SP":
	# Dataframe setup
	data = []
	header = ["dataset", "method", "emb_dimension", "batch_size", "epochs", "lr", "run", "accuracy", "std"]

	perf_files = utils.get_files(perf_folder, "", max_files=0)
	for perf_file in perf_files:
		perf_file_basename = os.path.basename(perf_file)
		dataset, method, emb_dimension, batch_size, epochs, lr, run  = perf_file_basename.strip().split("_")
		print(perf_file_basename.strip().split("_"))

		# Get the accuracy and the std in it
		with open(perf_file, "r") as fh:
			lines = fh.readlines()
			for line in lines:
				mean_acc, std = line.strip().split(",")
				mean_acc = float(mean_acc)
				std = float(std)

		data_input = [dataset, method, emb_dimension, batch_size, epochs, lr, run, mean_acc, std]
		data.append(data_input)

	df = pandas.DataFrame(data, columns=header)
	print(df)

	# Summarized table over runs
	datasets = list(df.dataset.unique())
	methods = list(df.method.unique())
	emb_dimensions = list(df.emb_dimension.unique())
	batch_sizes = list(df.batch_size.unique())
	num_epochs = list(df.epochs.unique())
	initial_lrs = list(df.lr.unique())

	# Table set up
	avg_data = []
	avg_header = ["dataset", "method", "emb_dimension", "batch_size", "epochs", "lr", "accuracy", "std", "cvs"]

	for dataset in datasets:
		for method in methods:
			for emb_dimension in emb_dimensions:
				for batch_size in batch_sizes:
					for epochs in num_epochs:
						for lr in initial_lrs:
							subtable = df.loc[(df['dataset'] == dataset) & (df['method'] == method) & (df['lr'] == lr) &
											(df['emb_dimension'] == emb_dimension) & (df['batch_size'] == batch_size) &
											(df['epochs'] == epochs)]

							cvs = len(subtable.accuracy)
							run_acc = np.mean(subtable.accuracy)
							run_std = np.std(subtable.accuracy)
							data_input = [dataset, method, emb_dimension, batch_size, epochs, lr, run_acc, run_std, cvs]
							avg_data.append(data_input)

	avg_df = pandas.DataFrame(avg_data, columns=avg_header)

	# Sort the table in terms of accuracy
	sorted_df = avg_df.sort_values(by='accuracy', ascending=False)
	print(sorted_df)

if mode == "WL":
	# Dataframe setup
	data = []
	header = ["dataset", "wl_depth", "emb_dimension", "batch_size", "epochs", "lr", "run", "accuracy", "std"]


	perf_files = utils.get_files(perf_folder, "", max_files=0)
	for perf_file in perf_files:
		perf_file_basename = os.path.basename(perf_file)
		dataset, wl_depth, emb_dimension, batch_size, epochs, lr, run  = perf_file_basename.strip().split("_")
		print(perf_file_basename.strip().split("_"))

		# Get the accuracy and the std in it
		with open(perf_file, "r") as fh:
			lines = fh.readlines()
			for line in lines:
				mean_acc, std = line.strip().split(",")
				mean_acc = float(mean_acc)
				std = float(std)

		data_input = [dataset, wl_depth, emb_dimension, batch_size, epochs, lr, run, mean_acc, std]
		data.append(data_input)

	df = pandas.DataFrame(data, columns=header)
	print(df)

	# Summarized table over runs
	datasets = list(df.dataset.unique())
	wl_depths = list(df.wl_depth.unique())
	emb_dimensions = list(df.emb_dimension.unique())
	batch_sizes = list(df.batch_size.unique())
	num_epochs = list(df.epochs.unique())
	initial_lrs = list(df.lr.unique())

	# Table set up
	avg_data = []
	avg_header = ["dataset", "wl_depth", "emb_dimension", "batch_size", "epochs", "lr", "accuracy", "std", "cvs"]

	for dataset in datasets:
		for wl_depth in wl_depths:
			for emb_dimension in emb_dimensions:
				for batch_size in batch_sizes:
					for epochs in num_epochs:
						for lr in initial_lrs:
							subtable = df.loc[(df['dataset'] == dataset) & (df['wl_depth'] == wl_depth) & (df['lr'] == lr) &
											(df['emb_dimension'] == emb_dimension) & (df['batch_size'] == batch_size) &
											(df['epochs'] == epochs)]

							cvs = len(subtable.accuracy)
							run_acc = np.mean(subtable.accuracy)
							run_std = np.std(subtable.accuracy)
							data_input = [dataset, wl_depth, emb_dimension, batch_size, epochs, lr, run_acc, run_std, cvs]
							avg_data.append(data_input)

	avg_df = pandas.DataFrame(avg_data, columns=avg_header)

	# Sort the table in terms of accuracy
	sorted_df = avg_df.sort_values(by='accuracy', ascending=False)
	print(sorted_df)

else:
	print("NO MODE MATCH")