#!/bin/bash

echo "create templates directory..."
mkdir ./templates
echo "copy files to temp directory..."
cp ./home.html ./templates
cp ./analysis.html ./templates
cp ./tfidf.html ./templates
cp ./cossim.html ./templates
echo "download modules..."
pip install nltk
pip install numpy
pip install beautifulsoup4
pip install konlpy
pip install requests
pip install elasticsearch

echo "run Flask..."
flask run
