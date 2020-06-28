#!/usr/bin/python3
#-*- coding: utf-8 -*-

import sys, re, requests, math, nltk, numpy, time, operator, os
nltk.download('stopwords')
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, session
from numpy import dot
from numpy.linalg import norm
from math import log
from elasticsearch import Elasticsearch
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
from urllib.request import urlopen
from urllib.request import HTTPError

app = Flask(__name__)

es_host="127.0.0.1"
es_port="9200"

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/analysis_text', methods=['GET', 'POST'])
def analysis_text():
	es = Elasticsearch([{'host':es_host, 'port':es_port}], timeout=30)
	if request.method =='POST':
		url = []
		words = []

		if (request.form['url_one'] != ""):
			url.append(request.form['url_one'])
			crawling(url[0],0,es)
			words.append(es.get(index='data', doc_type='word', id=0)['_source'].get('num'))
			num = 1
		return render_template('analysis.html', num=num, url=url, words=words)

@app.route('/analysis_file', methods=['GET', 'POST'])
def analysis_file():
	es = Elasticsearch([{'host':es_host, 'port':es_port}], timeout=30)
	if request.method =='POST':
		url = []
		words = []
		if (request.files['file'] != ""):
			num = 0
			file = './urls.txt'
			print(os.path.isfile(file))
			if (os.path.isfile(file)):
				os.remove(file)
			f = request.files['file']
			f.save(secure_filename(f.filename))
			f_off = open('urls.txt', 'r')
			while True:
				line = f.readline()
				if not line:
					break
				url.append(line[:len(line)-2])
				crawling(url[num],num,es)
				words.append(es.get(index='data', doc_type='word', id=num)['_source'].get('num'))
				num += 1

		return render_template('analysis.html', num=num, url=url, words=words)

@app.route('/analysis_tfidf', methods=['GET', 'POST'])
def analysis_tfidf():
	es = Elasticsearch([{'host':es_host, 'port':es_port}], timeout=30)
	if request.method =='POST':
		num = request.form['num']
		url = request.form['url']
		words = request.form['words']
		
		return render_template('analysis.html', num=num, url=url, words=words)

@app.route('/analysis_cossim', methods=['GET', 'POST'])
def analysis_cossim():
	es = Elasticsearch([{'host':es_host, 'port':es_port}], timeout=30)
	if request.method =='POST':
		num = request.form['num']
		url = request.form['url']
		words = request.form['words']
		
		return render_template('analysis.html', num=num, url=url, words=words)

@app.route('/tfidf')
def popupTfidf():
	return render_template('tfidf.html')

@app.route('/cossim')
def popupCossim():
	return render_template('cossim.html')

def crawling(url,id_,es):
	words = []
	freq = []
	swlist = []
	res = 0
	n = 0
	try:
		url_ = urlopen(url)
	except HTTPError as e:
		print(e)
		res = 1
	except ValueError as e:
		print(e)
		res = 1
	else:
		page = requests.get(url)
		soup = BeautifulSoup(page.content, "html.parser")
		p = soup.find_all('p')
		for sw in stopwords.words("english"):
			swlist.append(sw)
		for i in range(len(p)):
			p_split = p[i].text.split()
			for j in range(len(p_split)):
				p_split[j] = re.sub('[-=+,#/\?:^$.@*\"~&%!\\|\(\)\[\]\<\>`\']', '', p_split[j])
				p_split[j] = p_split[j].lower()
				k = 0
				if p_split[j] not in swlist:
					while (1):
						if (k == n):
							words.append(p_split[j])
							freq.append(1)
							n += 1
							break
						elif (words[k] == p_split[j]):
							freq[k] += 1
							break
						k += 1
		body_={ "url":url, "num":n, "words":words, "frequencies":freq, "result":res }
		es.index(index='data', doc_type='word', id=id_, body=body_)

def compute_tf(list1,count):
	tf_d = []
	i = 0
	for wordcnt in list1:
		tf_d.append(wordcnt / count)
		i+=1

	return tf_d

def compute_idf(n,es):

	bow = set()
	idf_d = []

	for i in range(0,n):
		res = es.get(index='data', doc_type='word', id=i)['_source'].get('words')
		for word in res:
			bow.add(word)

	j = 0
	for t in bow:
		cnt = 0
		for i in range(0,n):
			res = es.get(index='data', doc_type='word', id=i)['_source'].get('words')
			if t in res:
				cnt+=1
		idf_d.append(log(n/cnt))
		j+=1

	return idf_d

def compute_tfidf(list_,count,n,es):
	tf = []
	idf = []
	res = []
	tf = compute_tf(list_,count)
	idf = compute_idf(n,es)

	if (n==1):
		for i in range(0,count):
			res.append(tf[i])
	else:
		for i in range(0,count):
			res.append(tf[i]*idf[i])

	e = es.get(index='data', doc_type='word', id=id_)['_source']
	e['tfidf'] = res
	es.index(index='data', doc_type='word', id=id_, body=e)
	return res

def compute_top10(id_,n,es):
	start = time.time()
	freq = es.get(index='data', doc_type='word', id=id_)['_source'].get('frequencies')
	length = len(freq)
	word = es.get(index='data', doc_type='word', id=id_)['_source'].get('words')
	top = []
	tfidf = {}

	compute_tfidf(freq,length,n,es)

	for i in range(0,length):
		tfidf[word[i]] = freq[i]
	stfidf = sorted(tfidf.items(), key=operator.itemgetter(1))
	for i in range(0,10):
		top.append(stfidf[i][0])
	print(time.time()-start)
	return top

def cosine_sim(listA,listB,n,es):
	u = []
	v = []
	valu = 0
	valv = 0
	bow = set()

	for i in range(0,n):
		res = es.get(index='data', doc_type='word', id=i)['_source'].get('words')
		for word in res:
			bow.add(word)
	for w in bow:
		valu = 0
		for t in listA:
			if t==w:
				valu=1
				break
		u.append(valu)
	for w in bow:
		valv = 0
		for t in listB:
			if t==w:
				valv=1
				break
		v.append(valv)
	dotpro = numpy.dot(u,v)

	return (dotpro) / (norm(u) * norm(v))

def top3_sim(id_,n,es):
	top = []
	cosList=[]
	start = time.time()
	listA = es.get(index='data', doc_type='word', id=id_)['_source'].get('words')
	for i in range(0,n):
		if (id_==i):
			cosList.append(-1.0)
		else:
			listB = es.get(index='data', doc_type='word', id=i)['_source'].get('words')
			cosList.append(cosine_sim(listA,listB,n,es))

	e = es.get(index='data', doc_type='word', id=id_)['_source']
	e['cos'] = cosList
	es.index(index='data', doc_type='word', id=id_, body=e)

	for i in range(0,3):
		largest=0
		for j in range(0,n):
			if (cosList[largest]<cosList[j]):
				largest=j
		top.append(largest)
		cosList[largest]=-1.0

	print(time.time()-start)

	return top


if __name__ == '__main__':
	app.run(debug=True)

