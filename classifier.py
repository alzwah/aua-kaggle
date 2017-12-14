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
from sklearn.preprocessing import FunctionTransformer

# Measuring things, feature selection, scaling:
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

# Classifiers without meta estimators:
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.neighbors import NearestCentroid # Mostly because it sounds really cool and seems useful in general
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # multi-layer perceptron

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

# Meta estimators:
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

parser = argparse.ArgumentParser(description='Classify Swiss Dialects')
parser.add_argument('trainfile', help='the csv file with the train data')
parser.add_argument('testfile', help='the csv file with the test data')
parser.add_argument('resultfile', help='the filename to store the results')

arguments = parser.parse_args()
trainfile = arguments.trainfile
testfile = arguments.testfile
resultfile = arguments.resultfile


class DataFrameColumnExtracter(TransformerMixin):
	def __init__(self, column):
		self.column = column

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		return X[self.column]

	# To support grid search:
	def get_params(self):
		return None


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

def grid_search(transformer, param_grid, train_data, estimator):
	"""
	Fine tune the parameters param_grid with regards to the model in pipeline.
	Displays the possible parameters that can be used in param_grid when executed.
	"""
	# Example for parameters: { 'solver': ['adam', 'lbfgs'], 'activation': ['logistic', 'relu'] }
	train_x = train_data.copy()
	train_x.drop('Label', axis=1)
	train_y = train_data.copy()['Label']

	gs = GridSearchCV(
		estimator=estimator,
		param_grid=param_grid,
		n_jobs=-1,
		verbose=True
	)

	print("[Grid search]Supported Parameters:")
	print(gs.estimator.get_params().keys())

	pipeline = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('gs', gs)
	])
	pipeline.fit(train_x, train_y)

	print("[Grid search] Cross validation finished.")
	print("[Grid search] Best parameters:")
	best_parameters = gs.best_estimator_.get_params()
	for param_name in sorted(param_grid.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))


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
	
	
#returns a list of tokens that only appear in one category
def unique_tokens(data_in):
    # contains tuples of the form (category, sentence)
	category_text = [(c, s) for c, s in zip(data_in['Label'].values, data_in['Text'].values)]
    # build dict {category: [tokens]}
	category_tokens = {'BE': [], 'BS': [], 'LU': [], 'ZH': []}
	for elem in category_text:
		tokens = elem[1].split(" ")
		category_tokens[elem[0]].extend(tokens)
	
	BE = set(category_tokens['BE'])
	BS = set(category_tokens['BS'])
	LU = set(category_tokens['LU'])
	ZH = set(category_tokens['ZH'])

	return {'BE': (BE - BS - LU - ZH), 'BS': (BS - BE - LU - ZH), 'LU': (LU - BE - BS - ZH), 'ZH': (ZH - BE - BS - LU)}


def apply_unique_tokens(sentence, word_list):
	output = []

	for word in re.findall('\w+', sentence):
		if word in word_list:
			output.append(word)
	return ' '.join(output)

	# return
	# for word in word_list:
	# 	if re.search('\w+', sentence):
	# 		output.append(word)
	# return ' '.join(output)


#returns a list of tokens that appear in 3/4 of categories (e.g. appear in BS and LU and ZH but NOT BE)
def unique_missing_tokens(data_in):
    # contains tuples of the form (category, sentence)
	category_text = [(c, s) for c, s in zip(data_in['Label'].values, data_in['Text'].values)]
    # build dict {category: [tokens]}
	category_tokens = {'BE': [], 'BS': [], 'LU': [], 'ZH': []}
	for elem in category_text:
		tokens = elem[1].split(" ")
		category_tokens[elem[0]].extend(tokens)
		
	BE = set(category_tokens['BE'])
	BS = set(category_tokens['BS'])
	LU = set(category_tokens['LU'])
	ZH = set(category_tokens['ZH'])
	
	missing_tokens = []
	print((BE)-(BS & LU & ZH))
	missing_tokens.extend(list((BE)-(BS & LU & ZH)))
	print((BS)-(BE & LU & ZH))
	missing_tokens.extend(list((BS) -(BE & LU & ZH)))
	print((LU)-(BS & BE & ZH))
	missing_tokens.extend(list((LU) - (BE & BS & ZH)))
	print((ZH)-(BS & LU & BE))
	missing_tokens.extend(list((ZH) - (BE & BS & LU)))
	
	return missing_tokens
	

def average_word_length(sentence_in):
	""" Calculate the average word length in a sentence. """
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


# takes sentence and calgary-list and returns all the substrings that match character sequences from the list
def map_calgary(sentence, c_list):
	output = []
	for tok in c_list:
		if re.search(tok, sentence):
			output.append(tok)

	return (" ").join(output)

# takes sentence and calgary-list and returns all the words that match words from the list
def map_calgary_words(sentence, c_list):
	output = []
	for tok in c_list:
		if re.search('(\W'+tok+'\W|^'+tok+'\W|\W'+tok+'$)', sentence):
			output.append(tok)
	return (" ").join(output)


# function that creates subpipelines for transformer
def create_subpipeline(name,vectorizer,subpipeline_name,columname):
	return (subpipeline_name,Pipeline([
		('selector',DataFrameColumnExtracter(columname)),
		(name,vectorizer)]))


# function to append new columns with features to the pandas dataframe
def append_feature_columns(train_data_transformed, test_data_transformed, function, columname, function_argument):
	# uncomment when using with all data
	train_data_transformed = train_data_transformed.rename(columns={' Text':'Text'})
	test_data_transformed = test_data_transformed.rename(columns = {' Text':'Text'})
	train_map = train_data_transformed.copy()
	if function == map_calgary:
		train_map['Text'] = train_map['Text'].apply(function, c_list=function_argument)
	elif function == map_calgary_words:
		train_map['Text'] = train_map['Text'].apply(function, c_list=function_argument)
	elif function == apply_unique_tokens:
		train_map['Text'] = train_map['Text'].apply(function, word_list=function_argument)
	# in order to apply functions with no arguments
	else:
		train_map['Text'] = train_map['Text'].apply(function)

	train_map = train_map.rename(columns={'Text': columname})
	train_data_transformed = train_data_transformed.join(train_map[columname])

	test_map = test_data_transformed.copy()
	# in order to apply functions with no arguments
	if function == map_calgary:
		test_map['Text'] = test_map['Text'].apply(function, c_list=function_argument)
	elif function == map_calgary_words:
		test_map['Text'] = test_map['Text'].apply(function, c_list=function_argument)
	elif function == apply_unique_tokens:
		test_map['Text'] = test_map['Text'].apply(function, word_list=function_argument)
	else:
		test_map['Text'] = test_map['Text'].apply(function)

	test_map = test_map.rename(columns={'Text': columname})
	test_data_transformed = test_data_transformed.join(test_map[columname])

	return train_data_transformed, test_data_transformed


def classify(train_data, test_data):
	# transformer for feature union, thanks to data frame column extractor it can be applied to a column of the dataframe
	transformer = [
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches_exact_match'),
		# create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_averagewordlength', 'averagewordlength'), # Seems to be noise
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 1)), 'subpipeline_text_words', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 1)), 'subpipeline_text_chars', 'Text'),
		# create_subpipeline('count_vec', TfidfVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2,2), analyzer='char'), 'subpipeline_countvocals', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarybimatches', 'calgarybimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarytrimatches', 'calgarytrimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfourmatches', 'calgaryfourmatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfivematches', 'calgaryfivematches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_unique_word_matches_BE', 'unique_BE'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_unique_word_matches_BS', 'unique_BS'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_unique_word_matches_LU', 'unique_LU'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_unique_word_matches_ZH', 'unique_ZH')
	]

	transformer2 = [ # To test changes to transformer
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches_exact_match'),
		# create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_averagewordlength', 'averagewordlength'), # Seems to be noise
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 1)), 'subpipeline_text_words',
						   'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 1)), 'subpipeline_text_chars',
						   'Text'),
		# create_subpipeline('count_vec', TfidfVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2,2), analyzer='char'), 'subpipeline_countvocals', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarybimatches', 'calgarybimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarytrimatches', 'calgarytrimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfourmatches', 'calgaryfourmatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfivematches', 'calgaryfivematches')
	]

	transformer3 = [ # To test changes to transformer
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches_exact_match'),
		# create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_averagewordlength', 'averagewordlength'), # Seems to be noise
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 1)), 'subpipeline_text_words',
						   'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 1)), 'subpipeline_text_chars',
						   'Text'),
		# create_subpipeline('count_vec', TfidfVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2,2), analyzer='char'), 'subpipeline_countvocals', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_unique_word_matches_BE', 'unique_BE'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_unique_word_matches_BS', 'unique_BS'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_unique_word_matches_LU', 'unique_LU'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_unique_word_matches_ZH', 'unique_ZH')
	]

	transformer_n_grams = [
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 1)), 'subpipeline_text_words', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 1)), 'subpipeline_text_chars', 'Text'),
		create_subpipeline('tfidf', CountVectorizer(ngram_range=(2,2), analyzer='word'), 'subpipeline_word_n_grams', 'Text'),
		create_subpipeline('tfidf', CountVectorizer(ngram_range=(2,5), analyzer='char'), 'subpipeline_char_n_grams', 'Text'),
		create_subpipeline('count_vec', CountVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2, 2)), 'subpipeline_countvocals', 'Text'),
	]

	transformer_mlp = [
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches_exact_match'),
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 1)), 'subpipeline_text_words', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 1)), 'subpipeline_text_chars', 'Text'),
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
	pipeline_MLP2 = Pipeline([
		('union', FeatureUnion(transformer_list=transformer_mlp)),
		('clf', MLPClassifier(solver='adam', activation='logistic', max_iter=300, alpha=9.9999999999999995e-07))
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
	pipeline_bernoulliNB2 = Pipeline([
		('union', FeatureUnion(transformer_list=transformer2)),
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
	pipeline_decision_tree = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', DecisionTreeClassifier())
	])

	pipeline_svc = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', SVC())
	])
	pipeline_linear_svc = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', LinearSVC())
	])
	pipeline_linear_svc2 = Pipeline([
		('union', FeatureUnion(transformer_list=transformer3)),
		('clf', LinearSVC())
	])

	pipeline_logistic_regression = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', LogisticRegression())
	])
	pipeline_sgd_classifier = Pipeline([
		('union', FeatureUnion(transformer_list=transformer3)),
		('clf', SGDClassifier(max_iter=5, loss='log', n_jobs=-1))
	])
	pipeline_sgd_classifier2 = Pipeline([
		('union', FeatureUnion(transformer_list=transformer2)),
		('clf', SGDClassifier(max_iter=5, loss='log', n_jobs=-1))
	])

	pipeline_passive_agressive = Pipeline([
		('union', FeatureUnion(transformer_list=transformer3)),
		('clf', PassiveAggressiveClassifier(max_iter=5, average=True))
	])
	pipeline_passive_agressive2 = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', PassiveAggressiveClassifier(max_iter=5, average=True))
	])

	pipeline_voting_classifier = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', VotingClassifier(estimators=[
			('MultinomialNB', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
			('MLP', MLPClassifier(solver='adam', activation='logistic', max_iter=300)),
			], voting='soft', weights=[1.5, 1], n_jobs=-1)
		)
	])
	pipeline_voting_classifier2 = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', VotingClassifier(estimators=[
			('MultinomialNB', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
			('MLP', MLPClassifier(solver='adam', activation='logistic', max_iter=300,  alpha=9.9999999999999995e-07)),
			], voting='soft', weights=[2, 1], n_jobs=-1)
		)
	])

	pipeline_ada_boost_classifier = Pipeline([
		('union', FeatureUnion(transformer_list=transformer)),
		('clf', AdaBoostClassifier(
			base_estimator=MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
		))
	])



	# Evaluate pipelines
	# evaluate(train_data, pipeline_Multinomial, 'MultinomialNB')
	# evaluate(train_data, pipeline_Multinomial2, 'MultinomialNB2')
	# evaluate(train_data, pipeline_MLP, 'MLP')
	# evaluate(train_data, pipeline_MLP2, 'MLP2')
	# evaluate(train_data, pipeline_KNeighbors, 'KNN')
	#
	# evaluate(train_data, pipeline_ridge, 'Ridge')
	# evaluate(train_data, pipeline_ridge_cv, 'RidgeCV')
	# evaluate(train_data, pipeline_nearest_centroid, 'NearestCentroid')
	# evaluate(train_data, pipeline_nearest_centroid_2, 'NearestCentroid')
	evaluate(train_data, pipeline_bernoulliNB, 'BernoulliNB')
	evaluate(train_data, pipeline_bernoulliNB2, 'BernoulliNB')
	# evaluate(train_data, pipeline_one_v_one, 'One v. one, chosen estimator MultinomialNB')
	# evaluate(train_data, pipeline_one_v_rest, 'One v. rest, chosen estimator MultinomialNB')
	# evaluate(train_data, pipeline_decision_tree, 'Decision tree') # ca. 62%
	# evaluate(train_data, pipeline_svc, 'SVC') # ca. 26%
	evaluate(train_data, pipeline_linear_svc, 'Linear SVC')
	evaluate(train_data, pipeline_linear_svc2, 'Linear SVC')
	# evaluate(train_data, pipeline_logistic_regression, 'Logistic regression')
	evaluate(train_data, pipeline_sgd_classifier, 'SGD')
	evaluate(train_data, pipeline_sgd_classifier2, 'SGD')
	evaluate(train_data, pipeline_passive_agressive, 'Passive agressive')
	evaluate(train_data, pipeline_passive_agressive2, 'Passive agressive')
	# evaluate(train_data, pipeline_voting_classifier, 'Voting classifier')
	# evaluate(train_data, pipeline_voting_classifier2, 'Voting classifier 2')
	# evaluate(train_data, pipeline_ada_boost_classifier, 'Ada')

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

	pipeline.fit(train_data,train_y)
	predictions = pipeline.predict(test_text)
	# print(predictions)
	#
	# for i in range(0,len(predictions)):
	# 	print(predictions[i], test_text['Text'].iloc[i])


	return predictions

def evaluate(train_data, pipeline, name: str):

	print(name+ ':')

	sum = 0.0
	n_splits = 7
	k_fold = KFold(n_splits=n_splits)
	for train_indices, test_indices in k_fold.split(train_data):
		train_text = train_data.iloc[train_indices]
		train_y = train_data.iloc[train_indices]['Label'].values.astype(str)
		train_text.drop('Label', axis=1)

		test_text = train_data.iloc[test_indices]
		test_y = train_data.iloc[test_indices]['Label'].values.astype(str)
		test_text.drop('Label', axis=1)

		pipeline.fit(train_text, train_y)

		prediction = pipeline.predict(test_text)
		accuracy = accuracy_score(test_y, prediction)
		print('\t'+str(accuracy))
		sum += accuracy
	print('Average:', sum/n_splits)


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


def print_features(train_data, test_data):
	# print some of the topmost data to show created features and their values
	print('Train data:')
	print(train_data.head(30))
	print("------")
	print('Test data:')
	print(test_data.head(30))


def main():
	train_data = read_csv(trainfile)  # train_data should not be changed, so it will only contain the original text
	test_data = read_csv(testfile)  # test_data should not be changed, so it will only contain the original text

	calgary_tokens = calgary(train_data)

	# The following data frames contain the features to be trained on:
	train_data_transformed = read_csv(trainfile)
	test_data_transformed = read_csv(testfile)

	# create a list of lists with tokens to be evaluated
	token_lists = [
		(calgary(train_data), 'calgarymatches'),
		(calgary_ngram(train_data,2),'calgarybimatches'),
		(calgary_ngram(train_data,3),'calgarytrimatches'),
		(calgary_ngram(train_data,4),'calgaryfourmatches'),
		(calgary_ngram(train_data,5),'calgaryfivematches')
	]

	print('...adding features')

	for (token_list, columname) in token_lists:
		train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, map_calgary, columname, token_list)

	train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, map_calgary_words, 'calgarymatches_exact_match', calgary_tokens)

	unique_token_list = unique_tokens(train_data)
	train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, apply_unique_tokens, 'unique_BE', unique_token_list['BE'])
	train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, apply_unique_tokens, 'unique_BS', unique_token_list['BS'])
	train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, apply_unique_tokens, 'unique_LU', unique_token_list['LU'])
	train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, apply_unique_tokens, 'unique_ZH', unique_token_list['ZH'])

	train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, average_word_length, 'averagewordlength', function_argument=None)

	print('Done adding features.')

	# Print subset of features to console
	print_features(train_data_transformed, test_data_transformed)

	# Create plots for train data
	# visualize(train_data_transformed)

	# Classify
	predictions = classify(train_data_transformed, test_data_transformed)
	write_scores(resultfile, predictions)

	# Perform grid search for a given transformer
	# grid_search(
	# 	transformer = [
	# 		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches_exact_match'),
	# 		create_subpipeline('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 1)), 'subpipeline_text_words', 'Text'),
	# 		create_subpipeline('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 1)), 'subpipeline_text_chars', 'Text'),
	# 		create_subpipeline('count_vec', TfidfVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2, 2), analyzer='char'), 'subpipeline_countvocals', 'Text'),
	# 		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarybimatches', 'calgarybimatches'),
	# 		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarytrimatches', 'calgarytrimatches'),
	# 		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfourmatches', 'calgaryfourmatches'),
	# 		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfivematches', 'calgaryfivematches')
	# 	],
	# 	train_data=train_data_transformed,
	# 	param_grid={
	# 		'solver': ['adam', 'lbfgs'],
	# 		'activation': ['logistic', 'relu']
	# 	},
	# 	estimator=MLPClassifier()
	# )

if __name__ == '__main__':
	main()