#!/usr/bin/python3
#-*- coding: utf-8 -*-

import re, requests, sys, nltk
from bs4 import BeautifulSoup
from flask import Flask, render_template
from elasticsearch import Elasticsearch
nltk.download('stopwords')
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

if __name__ == '__main__':
	url = "http://impala.apache.org/"
	id_ = 0
	crawling(url,id_)
