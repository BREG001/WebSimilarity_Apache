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


