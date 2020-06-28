#!/bin/bash

echo "create templates directory..."
mkdir ./templates
echo "copy files to temp directory..."
cp ./home.html ./templates
cp ./analysis.html ./templates
cp ./tfidf.html ./templates
cp ./cossim.html ./templates
echo "download modules..."


echo "run Flask..."
flask run
