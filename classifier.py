#!/usr/bin/env python3

""" classify swiss german dialects """

import csv
import random
import re
import pandas as pd
import numpy as np
import argparse
import codecs
from importlib import reload

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Basic scikit features
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold

# Measuring things and feature selection:
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Classifiers without meta estimators:
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.neighbors import NearestCentroid # Mostly because it sounds really cool and seems useful in general
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # multi-layer perceptron

# Meta estimators:
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

parser = argparse.ArgumentParser(description='Classify Swiss Dialects')
parser.add_argument('trainfile', help='the csv file with the train data')
parser.add_argument('testfile', help='the csv file with the test data')
parser.add_argument('resultfile', help='the filename to store the results')

arguments = parser.parse_args()
trainfile = arguments.trainfile
testfile = arguments.testfile
resultfile = arguments.resultfile

word_regex = '(?u)(?:\b)?\w*(?:\b)?'

# s
class DataFrameColumnExtracter(TransformerMixin):
	def __init__(self, column):
		self.column = column

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		return X[self.column]

def read_csv(filename):
	data = pd.read_csv(filename,encoding='utf-8')
	return data

def write_scores(filename, predictions):
	predictions = [(i + 1, j) for i, j in enumerate(predictions)]
	with open(filename, 'w') as resultof:
		csv_writer = csv.writer(resultof, delimiter=",", lineterminator='\n')
		csv_writer.writerow(['Id', 'Prediction'])
		for id_, pred in predictions:
			csv_writer.writerow([id_, pred.strip()])

def grid_search(pipeline, par, train_data):
	"""
	Fine tune the parameters par with regards to the model in pipeline.
	Returns the classifier with the best parameter combination.
	Displays the possible parameters that can be used in par when executed.
	"""
	# Example for parameters: par={"count_vectorizer__ngram_range": [(2, 3), (1, 3)]}

	print("###### [Grid search]", pipeline.named_steps)
	print("[Grid search]Supported Parameters:")
	print(pipeline.get_params().keys())

	train_x = train_data['Text'].values
	train_y = train_data['Label'].values

	gs = GridSearchCV(pipeline, par, n_jobs=3, cv=10, verbose=True)
	gs.fit(train_x, train_y)

	print("[Grid search] Cross validation finished.")

	print("[Grid search] Best parameters:")
	best_parameters = gs.best_estimator_.get_params()
	for param_name in sorted(par.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

	return gs.best_estimator_


# takes panda dataframe
# gets training data, returns n-best calgary tokens
def calgary(data_in):
	# contains tuples of the form (category, sentence)
	category_text = [(c, s) for c, s in zip(data_in['Label'].values, data_in['Text'].values)]
	# tokenize sentences
	category_tokens = []
	for elem in category_text:
		tokens = elem[1].split(" ")
		category_tokens.append((elem[0], tokens))

	# structure: {category: freq}
	term_freq = {}
	# structure: {(category, token):freq}
	term_freq_per_category = {}
	term_count = 0
	term_count_per_category = {'BE': 0, 'BS': 0, 'LU': 0, 'ZH': 0}

	for cat, text in category_tokens:
		for token in text:
			if token in term_freq.keys():
				term_freq[token] += 1
			else:
				term_freq[token] = 1
			if (cat, token) in term_freq_per_category.keys():
				term_freq_per_category[(cat, token)] += 1
			else:
				term_freq_per_category[(cat, token)] = 1

			term_count += 1
			term_count_per_category[cat] += 1

	# structure: [(calgary value, tok)]
	output = []

	print(term_count_per_category)
	for tok, freq in term_freq.items():
		if freq > 2:
			# lol sorry für ds statement
			# max(probability t given category: termfrequency in category/total amount of terms in category)
			oberer_bruch = max(
				(get_term_freq_per_cat(term_freq_per_category, 'BE', tok) / term_count_per_category['BE']),
				(get_term_freq_per_cat(term_freq_per_category, 'BS', tok) / term_count_per_category['BS']),
				(get_term_freq_per_cat(term_freq_per_category, 'LU', tok) / term_count_per_category['LU']),
				(get_term_freq_per_cat(term_freq_per_category, 'ZH', tok) / term_count_per_category['ZH']))
			# probability term: termfrequency/total amount of terms
			unterer_bruch = freq / term_count
			output.append((oberer_bruch / unterer_bruch, tok))

	sorted_output = sorted(output, reverse=True)
	# returns 50 best calgary tokens
	return ([tok for val, tok in sorted_output[:199]])
	# # contains tuples of the form (category, sentence)
	# category_text = [(c, s) for c, s in zip(data_in['Label'].values, data_in['Text'].values)]
	# # tokenize sentences
	# category_tokens = []
	# for elem in category_text:
	# 	tokens = elem[1].split(" ")
	# 	category_tokens.append((elem[0], tokens))
	#
	# # structure: {category: freq}
	# term_freq = {}
	# # structure: {(category, token):freq}
	# term_freq_per_category = {}
	# term_count = 0
	# term_count_per_category = {'BE': 0, 'BS': 0, 'LU': 0, 'ZH': 0}
	#
	# for cat, text in category_tokens:
	# 	for token in text:
	# 		if token in term_freq.keys():
	# 			term_freq[token] += 1
	# 		else:
	# 			term_freq[token] = 1
	# 		if (cat, token) in term_freq_per_category.keys():
	# 			term_freq_per_category[(cat, token)] += 1
	# 		else:
	# 			term_freq_per_category[(cat, token)] = 1
	#
	# 		term_count += 1
	# 		term_count_per_category[cat] += 1
	#
	# # structure: [(calgary value, tok)]
	# output = []
	#
	# print(term_count_per_category)
	# for tok, freq in term_freq.items():
	# 	if freq > 2:
	# 		# lol sorry für ds statement
	# 		# max(probability t given category: termfrequency in category/total amount of terms in category)
	# 		oberer_bruch = max(
	# 			(get_term_freq_per_cat(term_freq_per_category, 'BE', tok) / term_count_per_category['BE']),
	# 			(get_term_freq_per_cat(term_freq_per_category, 'BS', tok) / term_count_per_category['BS']),
	# 			(get_term_freq_per_cat(term_freq_per_category, 'LU', tok) / term_count_per_category['LU']),
	# 			(get_term_freq_per_cat(term_freq_per_category, 'ZH', tok) / term_count_per_category['ZH']))
	# 		# probability term: termfrequency/total amount of terms
	# 		unterer_bruch = freq / term_count
	# 		output.append((oberer_bruch / unterer_bruch, tok))
	#
	# sorted_output = sorted(output, reverse=True)
	# # returns 50 best calgary tokens
	# return ([tok for val, tok in sorted_output[:999]])

# takes panda dataframe
# gets training data, returns n-best calgary tokens
def calgary_ngram(data_in, ngram):
	# contains tuples of the form (category, sentence)
	category_text = [(c, s) for c, s in zip(data_in['Label'].values, data_in['Text'].values)]
	# split up sentences in n-grams (including whitespace)
	category_tokens = []
	for elem in category_text:
		tokens = []
		# creates bi-/tri-/four-/etc-grams
		# http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
		tup = zip(*[list(elem[1])[i:] for i in range(ngram)])
		for e in tup:
			tokens.append(''.join(e))
		category_tokens.append((elem[0], tokens))

	# structure: {category: freq}
	term_freq = {}
	# structure: {(category, token):freq}
	term_freq_per_category = {}
	term_count = 0
	term_count_per_category = {'BE': 0, 'BS': 0, 'LU': 0, 'ZH': 0}

	for cat, text in category_tokens:
		for token in text:
			if token in term_freq.keys():
				term_freq[token] += 1
			else:
				term_freq[token] = 1
			if (cat, token) in term_freq_per_category.keys():
				term_freq_per_category[(cat, token)] += 1
			else:
				term_freq_per_category[(cat, token)] = 1

			term_count += 1
			term_count_per_category[cat] += 1

	# structure: [(calgary value, tok)]
	output = []

	print(term_count_per_category)
	for tok, freq in term_freq.items():
		if freq > 2:
			# lol sorry für ds statement
			# max(probability t given category: termfrequency in category/total amount of terms in category)
			oberer_bruch = max(
				(get_term_freq_per_cat(term_freq_per_category, 'BE', tok) / term_count_per_category['BE']),
				(get_term_freq_per_cat(term_freq_per_category, 'BS', tok) / term_count_per_category['BS']),
				(get_term_freq_per_cat(term_freq_per_category, 'LU', tok) / term_count_per_category['LU']),
				(get_term_freq_per_cat(term_freq_per_category, 'ZH', tok) / term_count_per_category['ZH']))
			# probability term: termfrequency/total amount of terms
			unterer_bruch = freq / term_count
			output.append((oberer_bruch / unterer_bruch, tok))

	sorted_output = sorted(output, reverse=True)
	# returns 25 most significant n-grams
	return ([tok for val, tok in sorted_output[:199]])

	# Alter Code
	# # contains tuples of the form (category, sentence)
	# category_text = [(c, s) for c, s in zip(data_in['Label'].values, data_in['Text'].values)]
	# # split up sentences in n-grams (including whitespace)
	# category_tokens = []
	# for elem in category_text:
	# 	tokens = []
	# 	# creates bi-/tri-/four-/etc-grams
	# 	# http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
	# 	tup = zip(*[list(elem[1])[i:] for i in range(ngram)])
	# 	for e in tup:
	# 		tokens.append(''.join(e))
	# 	category_tokens.append((elem[0], tokens))
	#
	# # structure: {category: freq}
	# term_freq = {}
	# # structure: {(category, token):freq}
	# term_freq_per_category = {}
	# term_count = 0
	# term_count_per_category = {'BE': 0, 'BS': 0, 'LU': 0, 'ZH': 0}
	#
	# for cat, text in category_tokens:
	# 	for token in text:
	# 		if token in term_freq.keys():
	# 			term_freq[token] += 1
	# 		else:
	# 			term_freq[token] = 1
	# 		if (cat, token) in term_freq_per_category.keys():
	# 			term_freq_per_category[(cat, token)] += 1
	# 		else:
	# 			term_freq_per_category[(cat, token)] = 1
	#
	# 		term_count += 1
	# 		term_count_per_category[cat] += 1
	#
	# # structure: [(calgary value, tok)]
	# output = []
	#
	# print(term_count_per_category)
	# for tok, freq in term_freq.items():
	# 	if freq > 2:
	# 		# lol sorry für ds statement
	# 		# max(probability t given category: termfrequency in category/total amount of terms in category)
	# 		oberer_bruch = max(
	# 			(get_term_freq_per_cat(term_freq_per_category, 'BE', tok) / term_count_per_category['BE']),
	# 			(get_term_freq_per_cat(term_freq_per_category, 'BS', tok) / term_count_per_category['BS']),
	# 			(get_term_freq_per_cat(term_freq_per_category, 'LU', tok) / term_count_per_category['LU']),
	# 			(get_term_freq_per_cat(term_freq_per_category, 'ZH', tok) / term_count_per_category['ZH']))
	# 		# probability term: termfrequency/total amount of terms
	# 		unterer_bruch = freq / term_count
	# 		output.append((oberer_bruch / unterer_bruch, tok))
	#
	# sorted_output = sorted(output, reverse=True)
	# # returns 25 most significant n-grams
	# return ([tok for val, tok in sorted_output[:199]])

def average_word_length(sentence_in):
	sum = 0.0
	count = 0
	for word in sentence_in.split(sep=" "):
		sum += len(word)
		count += 1
	return (sum / count)

def get_list_of_double_vocals():
	single_vocals = ['ö','ä','ü','ì','ò','è','a','e','i','o','u']
	double_vocals = []
	for char1 in single_vocals:
		for char2 in single_vocals:
			double_vocals.append(''+char1+char2)
	return double_vocals

def get_term_freq_per_cat(dict, cat, token):
	if (cat, token) in dict.keys():
		return dict[(cat, token)]
	else:
		return 0

# takes sentence and calgary-list and returns all the words that match from the list
def map_calgary(sentence, c_list):
	output = []
	for tok in c_list:
		if re.search(tok, sentence):
			output.append(tok)

	return (" ").join(output)

#function that creates subpipelines for transformer 
def create_subpipeline(name,vectorizer,subpipeline_name,columname):
	return (subpipeline_name,Pipeline([
		('selector',DataFrameColumnExtracter(columname)),
		(name,vectorizer)]))


#function to append new columns with features to the pandas dataframe
def append_feature_columns(train_data_transformed,test_data_transformed,function,columname,calgary_tokens):
	train_map = train_data_transformed.copy()
	if function == map_calgary:
		train_map['Text'] = train_map['Text'].apply(function, c_list=calgary_tokens)
	# in order to apply functions with no arguments
	else:
		train_map['Text'] = train_map['Text'].apply(function)

	train_map = train_map.rename(columns={'Text': columname})
	train_data_transformed = train_data_transformed.join(train_map[columname])

	test_map = test_data_transformed.copy()
	# in order to apply functions with no arguments
	if function == map_calgary:
		test_map['Text'] = test_map['Text'].apply(function, c_list=calgary_tokens)
	else:
		test_map['Text'] = test_map['Text'].apply(function)

	test_map = test_map.rename(columns={'Text': columname})
	test_data_transformed = test_data_transformed.join(test_map[columname])

	return train_data_transformed, test_data_transformed

def classify(train_data, test_data):
	# transformer for feature union, thanks to data frame column extractor it can be applied to a column of the dataframe
	transformer = [
		# create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches'),
		# create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_averagewordlength', 'averagewordlength'), # Seems to be noise
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_text', 'Text'),
		create_subpipeline('count_vec', TfidfVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2,2), analyzer='char'), 'subpipeline_countvocals', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarybimatches', 'calgarybimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarytrimatches', 'calgarytrimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfourmatches', 'calgaryfourmatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfivematches', 'calgaryfivematches')

	]

	transformer2 = [ # To test changes to transformer
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches'),
		create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_averagewordlength', 'averagewordlength'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_text', 'Text'),
		create_subpipeline('count_vec',
						   TfidfVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2, 2), analyzer='char'),
						   'subpipeline_countvocals', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarybimatches', 'calgarybimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarytrimatches', 'calgarytrimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfourmatches', 'calgaryfourmatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfivematches', 'calgaryfivematches')
	]

	transformer3 = [ # only count of words and chars
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches'),
		create_subpipeline('tfidf', TfidfVectorizer(ngram_range=(2,6), analyzer='char'), 'subpipeline_text_char', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_text_words', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarybimatches', 'calgarybimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarytrimatches', 'calgarytrimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfourmatches', 'calgaryfourmatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfivematches', 'calgaryfivematches')
	]

	transformer_n_grams = [
		create_subpipeline('count_vec', CountVectorizer(ngram_range=(2,2), analyzer='word', token_pattern=word_regex), 'subpipeline_word_n_grams', 'Text'),
		create_subpipeline('count_vec', CountVectorizer(ngram_range=(2,5), analyzer='char'), 'subpipeline_char_n_grams', 'Text'),
		create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_averagewordlength', 'averagewordlength'),
		create_subpipeline('count_vec', CountVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2, 2)), 'subpipeline_countvocals', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_text', 'Text')
	]

	transformer_mlp = [
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches'),
		# create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_averagewordlength', 'averagewordlength'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_text', 'Text'),
		create_subpipeline('count_vec', TfidfVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2, 2), analyzer='char'), 'subpipeline_countvocals', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarybimatches', 'calgarybimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarytrimatches', 'calgarytrimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfourmatches', 'calgaryfourmatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfivematches', 'calgaryfivematches')
	]

	# Preparing potential pipelines
	pipeline_Multinomial = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True))
	])

	pipeline_Multinomial2 = Pipeline([
		('union', FeatureUnion(transformer_list=transformer2)),
		('clf', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True))
	])

	pipeline_KNeighbors = Pipeline([
			('union', FeatureUnion(transformer_list = transformer)),
			('clf', KNeighborsClassifier(n_neighbors = 15))
			])

	pipeline_MLP = Pipeline([
		('union', FeatureUnion(transformer_list=transformer_mlp)),
		('clf', MLPClassifier(solver='adam', activation='logistic', max_iter=300))
	])

	pipeline_ridge = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', RidgeClassifier())
	])

	pipeline_ridge_cv = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', RidgeClassifierCV())
	])
	pipeline_nearest_centroid = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', NearestCentroid())
	])
	pipeline_bernoulliNB = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', BernoulliNB())
	])
	pipeline_one_v_one = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', OneVsOneClassifier(estimator=MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)))
	])
	pipeline_one_v_rest = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', OneVsRestClassifier(estimator=MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)))
	])

	# Evaluate pipelines
	evaluate(train_data, pipeline_Multinomial, 'MultinomialNB')
	# evaluate(train_data, pipeline_Multinomial2, 'MultinomialNB2')
	# evaluate(train_data, pipeline_MLP, 'MLP')
	# evaluate(train_data, pipeline_KNeighbors, 'KNN')
	#
	# evaluate(train_data, pipeline_ridge, 'Ridge')
	# evaluate(train_data, pipeline_ridge_cv, 'RidgeCV')
	# evaluate(train_data, pipeline_nearest_centroid, 'NearestCentroid')
	# evaluate(train_data, pipeline_nearest_centroid_2, 'NearestCentroid')
	# evaluate(train_data, pipeline_bernoulliNB, 'BernoulliNB')
	# evaluate(train_data, pipeline_one_v_one, 'One v. one, chosen estimator MultinomialNB')
	# evaluate(train_data, pipeline_one_v_rest, 'One v. rest, chosen estimator MultinomialNB')

	#train_text = train_data['Text'].values
	#train_y = train_data['Label'].values
	#print(test_data)
	#im test file von der web site hat es einen whitespace vor 'Text'
	#test_text = test_data['Text'].values

	#UM MIT TESTDATA ZU ARBEITEN:
	pipeline = pipeline_Multinomial
	train_y = train_data['Label'].values.astype(str)
	train_text = train_data

	test_text = test_data

	print(train_data)
	print(test_data)
	pipeline.fit(train_data,train_y)
	predictions = pipeline.predict(test_text)
	print(predictions)

	for i in range(0,len(predictions)):
		print(predictions[i], test_text['Text'].iloc[i])


	return predictions

def evaluate(train_data, pipeline, name: str):

	print(name+ ':')

	# UNCOMMENT UM nur mit train data zu arbeiten
	k_fold = KFold(n_splits=3)
	for train_indices, test_indices in k_fold.split(train_data):
		train_text = train_data.iloc[train_indices]
		train_y = train_data.iloc[train_indices]['Label'].values.astype(str)
		train_text.drop('Label', axis=1)

		test_text = train_data.iloc[test_indices]
		test_y = train_data.iloc[test_indices]['Label'].values.astype(str)
		test_text.drop('Label', axis=1)

		pipeline.fit(train_text, train_y)

		prediction = pipeline.predict(test_text)
		print('\t'+str(accuracy_score(test_y, prediction)))


def visualize(train_data):
	print('... Adding plots.')

	# Plot for average word length
	# Note: For some reason this has to happen before the calgari plots or the plot will be wrong.
	sns.boxplot(x='Label', y='averagewordlength', hue=None, data=train_data)
	plt.savefig('plots/averagewordlength_plot.pdf')
	print('\tAdded averagewordlength_plot.pdf')

	# Plots for calgari
	calgari_labels = ['calgarybimatches', 'calgarytrimatches', 'calgaryfourmatches', 'calgaryfivematches']
	plt.figure(figsize=(100,10))
	for label in calgari_labels:
		# gather data for calgary n grams
		visualize_calgary_n_grams = pd.DataFrame()
		for index, row in train_data.iterrows():
			for n_gram in row[label].strip().split(' '):
				visualize_calgary_n_grams = visualize_calgary_n_grams.append(
					{label: n_gram, 'Label': row['Label']}, ignore_index=True)
		# create and save plot
		sns.countplot(x=label, hue='Label', data=visualize_calgary_n_grams)
		plt.savefig('plots/'+label+'_plot.pdf')
		print('\tAdded '+label+'_plot.pdf')
	print('Done adding plots.')

def main():
	train_data = read_csv(trainfile)  # train_data should not be changed, so it will only contain the original text
	test_data = read_csv(testfile)  # test_data should not be changed, so it will only contain the original text

	calgary_tokens = calgary(train_data)

	# The following data frames contain the features to be trained on:
	train_data_transformed = read_csv(trainfile)
	test_data_transformed = read_csv(testfile)

	# create a list of lists with tokens to be evaluated 

	token_lists = [
		# (calgary(train_data),'calgarymatches'),
	(calgary_ngram(train_data,2),'calgarybimatches'),
	(calgary_ngram(train_data,3),'calgarytrimatches'),
	(calgary_ngram(train_data,4),'calgaryfourmatches'),
	(calgary_ngram(train_data,5),'calgaryfivematches')
	]

	print('...adding features')
	for (token_list, columname) in token_lists:
		train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed,
																			   test_data_transformed, map_calgary,
																			   columname, token_list)

	train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed,
																		   test_data_transformed, average_word_length,
																		   'averagewordlength', calgary_tokens=None)

	print('Done adding features.')

	# print(test_data_transformed)
	# print(train_data_transformed)

	# print some of the topmost data to show created features and their values
	# print(test_data_transformed.head(30))
	# print("------")
	# print(train_data_transformed.head(30))

	# Visualization of features
	visualize(train_data_transformed)


	# print(train_data_transformed['calgarytrimatches'].head(10))

	# Classify
	# train_data.drop('Id',axis=1)
	# print(list(test_data_transformed))
	# print(list(train_data_transformed))
	# predictions = classify(train_data_transformed, test_data_transformed)

	# TODO: apply map_calgary(sentence, calgary_tokens) for each sentence in panda df and add result to new column

	# write_scores(resultfile, predictions)

# cross_validate(train_data=train_data, k=3)

# grid_search(
#	par={
#		"count_vectorizer__ngram_range": [(1, 4), (1,6), (1,8)]
#		# "count_vectorizer__analyzer": ['char', 'char_wb']
#		# "count_vectorizer__stop_words": [[], ['uf, in, aber, a']], # ohni isch besser lol
		# "count_vectorizer__token_pattern": [word_regex, '(?u)\\b\\B\\B+\\b'] # Umlute sind scho no dr Hit
#		# "count_vectorizer__max_features": [None, 1000, 100] # None
#	},
#	pipeline= Pipeline([
#		('count_vectorizer', CountVectorizer(analyzer="char", token_pattern='(?u)\\b\[\wöäüÖÄÜìòè]\[\wöäüÖÄÜìòè]+\\b')),
#		('classifier', MultinomialNB())
#	]),
#	train_data=train_data
# )

if __name__ == '__main__':
	main()