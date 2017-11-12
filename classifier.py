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

def classify(train_data,test_data):

	pipeline = Pipeline([
		('count_vectorizer', TfidfVectorizer()),
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

	predictions = classify(train_data,test_data)
	write_scores(resultfile,predictions)




if __name__ == '__main__':
	main()