#!/usr/bin/python3
#-*- coding: utf-8 -*-

import sys, re, requests, math, nltk
nltk.download('stopwords')
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, session
from numpy import dot
from numpy.linalg import norm
import numpy
from math import log
from elasticsearch import Elasticsearch
from nltk.corpus import stopwords

es_host="127.0.0.1"
es_port="9200"

def crawling(url,id_):
	es = Elasticsearch([{'host':es_host, 'port':es_port}], timeout=30)
	words = []
	freq = []
	swlist = []
	n = 0

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

	e={ "num":n, "words":words, "frequencies":freq }
	res = es.index(index='data', doc_type='word', id=id_, body=e)
	print(res)

def compute_tf(list1, count):
	tf_d = []
	i = 0
	for wordcnt in list1:
		tf_d.append(wordcnt / count)
		i+=1

	return tf_d

def compute_idf(n):

	bow = set()
	idf_d = []
	
	es = Elasticsearch([{'host':es_host, 'port':es_port}],
timeout = 30)

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

def compute_tfidf(list1,count,n):
	tf = []
	idf = []
	res = []
	tf = compute_tf(list1,count)
	idf = compute_idf(n)

	print(tf)
	print(idf)
	for i in range(0,count):
		print(tf[i], idf[i], tf[i]*idf[i])
		res.append(tf[i]*idf[i])

	return res

def cosine_sim(listA,listB,n):
	es = Elasticsearch([{'host':es_host, 'port':es_port}], timeout=30)
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

	print(u)
	print(v)
	dotpro = numpy.dot(u,v)
	print(dotpro, (dotpro) / (norm(u) * norm(v)))
	return (dotpro) / (norm(u) * norm(v))


if __name__ == '__main__':
	es = Elasticsearch([{'host':es_host, 'port':es_port}], timeout=30)
	url = "http://airavata.apache.org/"
	id_ = 0
	crawling(url,id_)
	url = "http://buildr.apache.org/"
	id_ = 1
	crawling(url,id_)

	
	res1 = es.get(index='data', doc_type='word', id=0)['_source'].get('words')
	res2 = es.get(index='data', doc_type='word', id=1)['_source'].get('words')

	print(res1)
	print(res2)
	print(cosine_sim(res1, res1, 2))
	print(cosine_sim(res1, res2, 2))
