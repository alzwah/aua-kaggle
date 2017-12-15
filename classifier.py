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
#import seaborn as sns
#import matplotlib.pyplot as plt

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

#https://stackoverflow.com/questions/34710281/use-featureunion-in-scikit-learn-to-combine-two-pandas-columns-for-tfidf
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
		
		
		
		
		
		
#############################HELPER METHODS###############################


def average_word_length(sentence_in):
	""" Calculate the average word length in a sentence. """
	sum = 0.0
	count = 0
	for word in sentence_in.split(sep=" "):
		sum += len(word)
		count += 1
	return (sum / count)




def get_term_freq_per_cat(dict, cat, token):
	if (cat, token) in dict.keys():
		return dict[(cat, token)]
	else:
		return 0



#############################CALGARY###############################


def calgary(data_in: pd.DataFrame) -> list:
	'''
	:param data_in: pandas dataframe containing training data
	:return: The 200 best calgary tokens
	'''

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
	return [tok for val, tok in sorted_output[:199]]


def calgary_ngram(data_in: pd.DataFrame, ngram: int) -> list:
	'''
	:param data_in: pandas dataframe containing training data
	:param ngram: length of n-gram
	:return: The 200 best calgary tokens
	'''
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
	return [tok for val, tok in sorted_output[:199]]


def map_calgary(sentence: str, calgari_list: list) -> str:
	'''
	:param sentence: Sentence to be searched.
	:param calgari_list: List of calgari-tokens that should be matched
	:return: all the substrings that match elements from the calgari_list
	'''
	output = []
	for tok in calgari_list:
		if re.search(tok, sentence):
			output.append(tok)

	return (" ").join(output)


def map_calgary_words(sentence: str, calgari_list) -> str:
	'''
	:param sentence: Sentence to be searched.
	:param calgari_list: List of calgari-tokens that should be matched
	:return: all the words that match elements from the calgari_list
	'''
	output = []
	for tok in calgari_list:
		if re.search('(\W'+tok+'\W|^'+tok+'\W|\W'+tok+'$)', sentence):
			output.append(tok)
	return (" ").join(output)
	
	
#############################BIGRAMS###############################


def list_of_bigrams(data_in: pd.DataFrame):
	'''
		Generate a list of all bigrams with their frequencies.
	:param data_in: Dataframe containing the text and its label
	:return: Dictionary containing each Dialect as key, and a list of bigrams, sorted by their frequency in that dialect.
	'''

	# contains tuples of the form (category, sentence)
	category_text = [(c, s) for c, s in zip(data_in['Label'].values, data_in['Text'].values)]

	# build dict {category: [tokens]}
	category_tokens = {'BE': {}, 'BS': {}, 'LU': {}, 'ZH': {}}
	for elem in category_text:
		# count frequencies for all occuring bigrams
		for bigram in zip(*[list(elem[1][i:]) for i in range(2)]):
			dict = category_tokens[elem[0]]
			key = bigram[0]+bigram[1]
			if key in dict:
				dict[key] = dict[key]+1
			else:
				dict[key] = 1

	Labels = ['BE', 'BS', 'LU', 'ZH']
	category_lists = {'BE': [], 'BS': [], 'LU': [], 'ZH': []}
	for label in Labels:
		category_lists[label] = sorted(category_tokens[label].keys(), key= lambda k: category_tokens[label][k])
	return category_lists


def get_bigram_frequency_list(sentence: str) -> str:
	'''
	:param sentence: The sentence to be searched through.
	:return: List of bigrams sorted by how frequent they are in the sentence.
	'''
	bigram_dict = {}
	for bigram in zip(*[sentence[i:] for i in range(2)]):
		key = bigram[0] + bigram[1]
		if key in bigram_dict:
			bigram_dict[key] = bigram_dict[key] + 1
		else:
			bigram_dict[key] = 1
	bigram_list = sorted(bigram_dict.keys(), key= lambda k: bigram_dict[k])
	return bigram_list


def apply_bigram_frequency(sentence: str, bigram_list: list) -> float:
	'''
		Calculate error for bigrams in a sentence compared to a lists of bigrams inspired by Canvar & Trenkle 1994, N-Gram-Based Text Categorization
	:param sentence: The sentence to be searched.
	:param bigram_list: List of bigrams ordered by some frequency.
	:return: Represents the error.
	'''
	sentence_bigrams = get_bigram_frequency_list(sentence)
	if len(sentence_bigrams) == 0:
		return "0"
	out_of_place = 0
	max_value = len(bigram_list)
	max_index = len(bigram_list)

	for s_index, sentence_bigram in enumerate(sentence_bigrams):
		found = False
		for bigram_index, bigram in enumerate(bigram_list[0: max_index]):
			if bigram != sentence_bigram:
				continue
			else:
				out_of_place += abs(s_index-bigram_index)
				found = True
				break
		if not found:
			out_of_place += max_value

	result = out_of_place/len(sentence)

	return str(result)
	
	
def get_list_of_double_vocals() -> list:
	'''
	:return: list of all possible combinations of two vocals.
	'''
	single_vocals = ['ö','ä','ü','ì','ò','è','a','e','i','o','u']
	double_vocals = []
	for char1 in single_vocals:
		for char2 in single_vocals:
			double_vocals.append(''+char1+char2)
	return double_vocals



#############################UNIQUE TOKENS PER CATEGORY###############################

def unique_tokens(data_in: pd.DataFrame):
	'''
	:param data_in: Dataframe containing the text and its label
	:return: Set of tokens that only appear in one category
	'''
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


def apply_unique_tokens(sentence: str, word_list: list) -> str:
	'''

	:param sentence:
	:param word_list:
	:return:
	'''
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
def unique_missing_tokens(data_in: pd.DataFrame) -> list:
    '''
    Collects all the tokens that are present in 3 out of 4 categories
    :param data_in: Dataframe containing the text and its label 
    :return: List of tokens that are only NOT present in one category
    '''
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
	missing_tokens.extend(list((BE)-(BS & LU & ZH)))
	missing_tokens.extend(list((BS) -(BE & LU & ZH)))
	missing_tokens.extend(list((LU) - (BE & BS & ZH)))
	missing_tokens.extend(list((ZH) - (BE & BS & LU)))
	
	return missing_tokens
	


#############################PROCESSING###############################


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
		train_map['Text'] = train_map['Text'].apply(function, calgari_list=function_argument)
	elif function == map_calgary_words:
		train_map['Text'] = train_map['Text'].apply(function, calgari_list=function_argument)
	elif function == apply_unique_tokens:
		train_map['Text'] = train_map['Text'].apply(function, word_list=function_argument)
	elif function == apply_bigram_frequency:
		train_map['Text'] = train_map['Text'].apply(function, bigram_list=function_argument)
	# in order to apply functions with no arguments
	else:
		train_map['Text'] = train_map['Text'].apply(function)

	train_map = train_map.rename(columns={'Text': columname})
	train_data_transformed = train_data_transformed.join(train_map[columname])

	test_map = test_data_transformed.copy()
	# in order to apply functions with no arguments
	if function == map_calgary:
		test_map['Text'] = test_map['Text'].apply(function, calgari_list=function_argument)
	elif function == map_calgary_words:
		test_map['Text'] = test_map['Text'].apply(function, calgari_list=function_argument)
	elif function == apply_unique_tokens:
		test_map['Text'] = test_map['Text'].apply(function, word_list=function_argument)
	elif function == apply_bigram_frequency:
		test_map['Text'] = test_map['Text'].apply(function, bigram_list=function_argument)
	else:
		test_map['Text'] = test_map['Text'].apply(function)

	test_map = test_map.rename(columns={'Text': columname})
	test_data_transformed = test_data_transformed.join(test_map[columname])

	return train_data_transformed, test_data_transformed


def classify(train_data: pd.DataFrame, test_data: pd.DataFrame,resultfile: str):
    '''
    Classifies with feature pipelines and returns predictions
    :param train_data: Training dataframe containing the text and its label 
    :param test_data: Test dataframe containing the text and its label 
    :param resultfile: string pointing to the CSV file where the results will be written
    :return: list of predictions
    '''
	# transformer for feature union, thanks to data frame column extractor it can be applied to a column of the dataframe
	transformer_all = [
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
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_unique_word_matches_ZH', 'unique_ZH'),
		#create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_bigram_frequency_BE', 'bigram_frequency_BE'),
		#create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_bigram_frequency_BS', 'bigram_frequency_BS'),
		#create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_bigram_frequency_LU', 'bigram_frequency_LU'),
		#create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_bigram_frequency_ZH', 'bigram_frequency_ZH')
	]

	transformer_calgari = [ # To test changes to transformer
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches_exact_match'),
		# create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_averagewordlength', 'averagewordlength'), # Seems to be noise
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 1)), 'subpipeline_text_words','Text'),
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 1)), 'subpipeline_text_chars','Text'),
		# create_subpipeline('count_vec', TfidfVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2,2), analyzer='char'), 'subpipeline_countvocals', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarybimatches', 'calgarybimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarytrimatches', 'calgarytrimatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfourmatches', 'calgaryfourmatches'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgaryfivematches', 'calgaryfivematches')
	]

	transformer_unique = [ # To test changes to transformer
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgary', 'calgarymatches_exact_match'),
		# create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_averagewordlength', 'averagewordlength'), # Seems to be noise
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 1)), 'subpipeline_text_words','Text'),
		create_subpipeline('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 1)), 'subpipeline_text_chars','Text'),
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
	
	transformer_dial_big = [
		create_subpipeline('tfidf', CountVectorizer(ngram_range=(2,2), analyzer='char'), 'subpipeline_char_n_grams', 'Text'),
		create_subpipeline('count_vec', TfidfVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2, 2), analyzer='char'), 'subpipeline_countvocals', 'Text'),
		create_subpipeline('count_vec', CountVectorizer(vocabulary=get_list_of_double_vocals(), ngram_range=(2, 2)), 'subpipeline_countvocals_2', 'Text'),
		create_subpipeline('tfidf', TfidfVectorizer(), 'subpipeline_calgarybimatches', 'calgarybimatches'),
		create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_bigram_frequency_BE', 'bigram_frequency_BE'),
		create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_bigram_frequency_BS', 'bigram_frequency_BS'),
		create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_bigram_frequency_LU', 'bigram_frequency_LU'),
		create_subpipeline('count_vec', CountVectorizer(), 'subpipeline_bigram_frequency_ZH', 'bigram_frequency_ZH')
	]


	# pipeline_voting_classifier = Pipeline([
	# 	('union', FeatureUnion(transformer_list=transformer_all)),
	# 	('clf', VotingClassifier(estimators=[
	# 		('MultinomialNB', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
	# 		('MLP', MLPClassifier(solver='adam', activation='logistic', max_iter=300)),
	# 		], voting='soft', weights=[1.5, 1], n_jobs=-1)
	# 	)

	
	

	
	pipeline_voting_classifier_hard = Pipeline([
	 	('union', FeatureUnion(transformer_list=transformer_all)),
	 	#('select_features',SelectKBest(k=10000)),
	 	('clf', VotingClassifier(estimators=[
	 		('MultinomialNB', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
	 		('MultinomialNB_2',MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
	 		('MultinomialNB_3',MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
	 		('MultinomialNB_4',MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
	 		('MultinomialNB_5',MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
	 		('MultinomialNB_6',MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
	 		('MultinomialNB_7',MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
	 		('MultinomialNB_8',MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)),
	 		('MLP', MLPClassifier(solver='adam', activation='logistic', max_iter=300)),
	 		('Linear SVC', LinearSVC()),
	 		('Passive agressive', PassiveAggressiveClassifier(max_iter=5, average=True))
	 	], voting='hard', n_jobs=-1)
	 	 )
	 ])


	# Evaluate pipelines
	# evaluate(train_data, pipeline_voting_classifier, 'Voting classifier')
	# evaluate(train_data, pipeline_voting_classifier_hard, 'Voting hard')
	s

	#UM MIT TESTDATA ZU ARBEITEN:
	pipeline = pipeline_voting_classifier_hard
	train_y = train_data['Label'].values.astype(str)
	train_text = train_data
	
	test_text = test_data
	print('...fitting')
	pipeline.fit(train_data,train_y)
	print('...predicting')
	predictions = pipeline.predict(test_text)
	# print(predictions)
	#
	# for i in range(0,len(predictions)):
	# 	print(predictions[i], test_text['Text'].iloc[i])
	write_scores(resultfile, predictions)

	return predictions

# function to evaluate only on train set
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



	return prediction 

# function to plot data to get a better idea of the features
def visualize(train_data):
	print('... Adding plots.')

	sns.set(style="whitegrid", palette="muted")

	# Plot for average word length
	# Note: For some reason this has to happen before the calgari plots or the plot will be wrong.
	sns.boxplot(x='Label', y='averagewordlength', hue=None, data=train_data)
	plt.savefig('plots/averagewordlength_plot.pdf')
	print('\tAdded averagewordlength_plot.pdf')

	# try:
	# 	sns.pairplot(
	# 		data=train_data,
	# 		x_vars=['bigram_frequency_BE','bigram_frequency_BS','bigram_frequency_LU','bigram_frequency_ZH'],
	# 		y_vars=['Label'],
	# 	)
	# 	plt.savefig('plots/bigram_frequency_plot.pdf')
	# 	print('\tAdded bigram_frequency_plot.pdf')
	# except ValueError:
	# 	pass


	# # Plots for calgari
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

	# # create a list of lists with tokens to be evaluated
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

	bigrams = list_of_bigrams(train_data)
	# train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, apply_bigram_frequency, 'bigram_frequency_BE', function_argument=bigrams['BE'])
	# train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, apply_bigram_frequency, 'bigram_frequency_BS', function_argument=bigrams['BS'])
	# train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, apply_bigram_frequency, 'bigram_frequency_LU', function_argument=bigrams['LU'])
	# train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, apply_bigram_frequency, 'bigram_frequency_ZH', function_argument=bigrams['ZH'])
	print('Done adding features.')

	# Print subset of features to console
	# print_features(train_data_transformed, test_data_transformed)

	# Create plots for train data
	# visualize(train_data_transformed)

	# Classify
	print('...classification started')
	# test_data = test_data.rename({' Text':'Text'})
	predictions = classify(train_data_transformed, test_data_transformed,resultfile)

	print('...writing results')

	write_scores(resultfile, predictions)
	print('done!')

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