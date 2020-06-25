#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import re
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from math import log

docs = [
	'먹고 싶은 사과',
	'먹고 싶은 바나나',
	'길고 노란 바나나 바나나',
	'저는 과일이 좋아요'
]

vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()
N = len(docs)

def tf(t,d):
	return d.count(t)

def idf(t):
	df = 0
	for doc in docs:
		df += t in doc
	return log(N/(df+1))

def tfidf(t,d):
	return tf(t,d)*idf(t)

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('forproject.html')

@app.route('/forproject2', methods=['POST'])
def forproject2():
	error = None

	if request.method == 'POST':
		myname = request.form['name']
		result = []
		for i in range(N):
			result.append([])
			d = docs[i]
			for j in range(len(vocab)):
				t = vocab[j]
				result[-1].append(tf(t,d))

		tf_ = pd.DataFrame(result, columns = vocab)

		result2 = []
		for j in range(len(vocab)):
			t = vocab[j]
			result2.append(idf(t))

		idf_ = pd.DataFrame(result2, index = vocab, columns = ["IDF"])
		idf_ = idf_.sort_values(by=['IDF'], axis = 0, ascending=False)

		result3 = []
		for i in range(N):
			result3.append([])
			d = docs[i]
			for j in range(len(vocab)):
				t = vocab[j]

				result3[-1].append(tfidf(t,d))

		tfidf_ = pd.DataFrame(result3, columns = vocab)
		
		return render_template('forproject2.html', tables=[tfidf_.to_html(classes='data')], titles=tfidf_.columns.values)

if __name__ == '__main__':
	app.run(debug=True)
