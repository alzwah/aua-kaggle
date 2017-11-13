#!/usr/bin/env python3

""" classify swiss german dialects """

import csv
import random
import pandas as pd
import numpy as np
import argparse
import codecs

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier # multi-layer perceptron
from sklearn.model_selection import KFold # for cross-validation
from sklearn.feature_selection import mutual_info_classif


parser = argparse.ArgumentParser(description= 'Classify Swiss Dialects')
parser.add_argument('trainfile', help='the csv file with the train data')
parser.add_argument('testfile', help = 'the csv file with the test data')
parser.add_argument('resultfile', help='the filename to store the results')

arguments = parser.parse_args()
trainfile = arguments.trainfile
testfile = arguments.testfile
resultfile = arguments.resultfile


def read_csv(filename):
	data = pd.read_csv(filename,encoding='latin-1')
	return data

def write_scores(filename,predictions):
	predictions = [(i+1,j) for i,j in enumerate(predictions)]
	with open(filename,'w') as resultof:
		csv_writer = csv.writer(resultof,delimiter=",")
		csv_writer.writerow(['Id','Prediction'])
		for id_,pred in predictions:
			csv_writer.writerow([id_,pred])

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

def cross_validate(train_data, k):
	pipelines = [
		(Pipeline([
			('count_vectorizer', TfidfVectorizer()),
			('classifier', MultinomialNB())
		]), "MultinomialNB, Tfidf vectorizer"),
		(Pipeline([
			('count_vectorizer', CountVectorizer(
				analyzer="char_wb",
				token_pattern='(?u)\\b\[\wöäüÖÄÜìòè]\[\wöäüÖÄÜìòè]+\\b',
				ngram_range=(1,4),
			)),
			('classifier', MultinomialNB())
		]), "MultinomialNB, count vectorizer")
	]

	scores = {}
	for pipeline, name in pipelines:
		scores[name] = 0

	# train models with 10-fold cross-validation for performance comparison
	kf = KFold(n_splits=k, shuffle=False)
	folds = 0
	# Split data into train and test set
	for train, test in kf.split(train_data['Text'].values,train_data['Label'] ):
		folds += 1
		print("Fold:", folds)

		x_train = train_data['Text'].values[train]
		x_test = train_data['Text'].values[test]
		y_train = train_data['Label'].values[train]
		y_test = train_data['Label'].values[test]

		for pipeline, name in pipelines:

			pipeline.fit(x_train, y_train)
			predictions = pipeline.predict(x_test)

			new_score = accuracy_score(y_true=y_test, y_pred=predictions)

			scores[name] = scores[name] + new_score
			print(name + ": ", new_score, scores[name]/folds)
		print()

	# create pretty output
	for pipeline, name in pipelines:
		print(name+":", "\taverage accuracy:", scores[name]/folds)

def classify(train_data,test_data):

	# pipeline = Pipeline([
	# 	('count_vectorizer', TfidfVectorizer()),
	# 	('classifier', MultinomialNB())
	# 	])

	pipeline = Pipeline([
		('count_vectorizer', CountVectorizer(
			analyzer="char_wb",
			token_pattern='(?u)\\b\[\wöäüÖÄÜìòè]\[\wöäüÖÄÜìòè]+\\b',
			ngram_range=(1,4),
		)),
		('classifier', MultinomialNB())
	])

	train_text = train_data['Text'].values
	train_y = train_data['Label'].values
	print(test_data)
	#im test file von der web site hat es einen whitespace vor 'Text'
	test_text = test_data['Text'].values
	
	pipeline.fit(train_text,train_y)
	predictions = pipeline.predict(test_text)

	for i in range(0,len(predictions)):
		print(predictions[i], test_text[i])

	#print(accuracy_score(test_y,predictions))

	return predictions			
def main():
	train_data = read_csv(trainfile)
	test_data = read_csv(testfile)

	# predictions = classify(train_data,test_data)
	# write_scores(resultfile,predictions)

	# cross_validate(train_data=train_data, k=3)

	grid_search(
		par={
			"count_vectorizer__ngram_range": [(1, 4), (1,6), (1,8)]
			# "count_vectorizer__analyzer": ['char', 'char_wb']
			# "count_vectorizer__stop_words": [[], ['uf, in, aber, a']], # ohni isch besser lol
			# "count_vectorizer__token_pattern": ['(?u)\\b\[\wöäüÖÄÜìòè]\[\wöäüÖÄÜìòè]+\\b', '(?u)\\b\\B\\B+\\b'] # Umlute sind scho no dr Hit
			# "count_vectorizer__max_features": [None, 1000, 100] # None
		},
		pipeline= Pipeline([
			('count_vectorizer', CountVectorizer(analyzer="char", token_pattern='(?u)\\b\[\wöäüÖÄÜìòè]\[\wöäüÖÄÜìòè]+\\b')),
			('classifier', MultinomialNB())
		]),
		train_data=train_data
	)


if __name__ == '__main__':
	main()