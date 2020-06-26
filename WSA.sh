#!/bin/bash

echo "create templates directory..."
mkdir ./templates
echo "copy files to temp directory..."
cp ./home.html ./templates

echo "run Flask..."
flask run
