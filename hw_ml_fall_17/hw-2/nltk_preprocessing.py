import pandas as pd
import numpy as np
import math
import sys
import os
from nltk.stem.porter import *

stemmer = PorterStemmer()
path = ['/stemmed', '/stop_stem']

for folder in path:

    stopwords = []
    if folder == '/stop_stem':
        with open(os.getcwd()+'/stopwords.txt') as file:
            for line in file:
                stopwords.append(line.rstrip())


    path = os.getcwd()+'/train/ham'

    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r", encoding="Latin-1")
        with open(os.getcwd()+folder+'/train/ham/'+filename,'a+', encoding="Latin-1") as write_file:
            for line in file:
                for word in line.split():
                    word = stemmer.stem(word.lower())
                    if word not in stopwords:
                        write_file.write(word+'\n')


    path = os.getcwd()+'/train/spam'


    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r", encoding="Latin-1")
        with open(os.getcwd()+folder+'/train/spam/'+filename,'a+', encoding="Latin-1") as write_file:
            for line in file:
                for word in line.split():
                    word = stemmer.stem(word.lower())
                    if word not in stopwords:
                        write_file.write(word+'\n')

    path = os.getcwd()+'/test/ham'


    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r", encoding="Latin-1")
        with open(os.getcwd()+folder+'/test/ham/'+filename,'a+', encoding="Latin-1") as write_file:
            for line in file:
                for word in line.split():
                    word = stemmer.stem(word.lower())
                    if word not in stopwords:
                        write_file.write(word+'\n')

    path = os.getcwd()+'/test/spam'


    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r", encoding="Latin-1")
        with open(os.getcwd()+folder+'/test/spam/'+filename,'a+', encoding="Latin-1") as write_file:
            for line in file:
                for word in line.split():
                    word = stemmer.stem(word.lower())
                    if word not in stopwords:
                        write_file.write(word+'\n')
    
