import os
import re
import sys
import math
import pandas as pd
import numpy as np

def make_dictionary(path,features):
    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r")
        for word in file:
            word = word.strip()
            #if re.match(reg_exp,word):#taking just alphanumeric strings
            if word.isalnum():
                if word not in features:
                    features.append(word)
    return features

def insert_term_frequency_ham(path,df,row):
    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r")
        for word in file:
            word = word.strip()
            #if re.match(reg_exp,word):#taking just alphanumeric strings
            if word.isalnum():
                df.loc[row,word] = df.loc[row,word] + 1
        df.loc[row,'output'] = 0
        row = row + 1
    return df,row
    
def insert_term_frequency_spam(path,df,row):
    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r")
        for word in file:
            word = word.strip()
            #if re.match(reg_exp,word):#taking just alphanumeric strings
            if word.isalnum():
                df.loc[row,word] = df.loc[row,word] + 1
        df.loc[row,'output'] = 1
        row = row + 1
    return df,row

def calculated_value(row):
    global weight
    global matrix

    zee = np.dot(weight,matrix[row,:-1])#dot product of weight and current row

    if zee>0:
        return 1.0
    else:
        return 0.0

def train_algorithm(eta,features,iterations):
    global weight
    global matrix

    calculated_array = [0]*(row)#initializing array for storing predicted value for each datapoint
    
    for loop in range(iterations):#hard limit on iterations
        for doc in range(row):
            calculated_array[doc] = calculated_value(doc)#calculating predicted value for each datapoint
            
        for j in range(len(weight)):
            for i in range(row):
                if matrix[i,j]:
                    #updating all weights  
                    weight[j] += eta*matrix[i,j]*(matrix[i,-1] - calculated_array[i])

def insert_term_frequency_test_ham(path,df,row,features):
    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r")
        for word in file:
            word = word.strip()
            #if re.match(reg_exp,word):#taking just alphanumeric strings
            if word.isalnum():
                if word in features:
                    df.loc[row,word] = df.loc[row,word] + 1
        df.loc[row,'output'] = 0
        row = row + 1
    return df,row
    
def insert_term_frequency_test_spam(path,df,row,features):
    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r")
        for word in file:
            word = word.strip()
            #if re.match(reg_exp,word):#taking just alphanumeric strings
            if word.isalnum():
                if word in features:

                    df.loc[row,word] = df.loc[row,word] + 1
        df.loc[row,'output'] = 1
        row = row + 1
    return df,row

def test_algorithm(row):
    global matrix
    global weight
    
    correct = 0#count of correct prediction
    
    for i in range(row):
        #for all datapoints
        zee = np.dot(weight,matrix[i,:-1])#dot product of datapoint and weights
        if zee > 0:
            #if dot product is greater than 0 than predicted value is 1
            if matrix[i,-1] == 1.0:
                correct+=1
        else:
            #if dot product is less than 0 than predicted value is 0
            if matrix[i,-1] == 0.0: 
                correct+=1

    return correct       
    
reg_exp = '[a-z0-9]+'

#folder = "stemmed"
folder = "stop_stem"

print(folder)

path1 = os.getcwd() + "/" + folder + "/" +"train/ham"
path2 = os.getcwd() + "/" + folder + "/" +"train/spam"

features = []

features = make_dictionary(path1,features)
features = make_dictionary(path2,features)

features.append('output')

df = pd.DataFrame(columns=features)

for i in range(len(os.listdir(path1))+len(os.listdir(path2))):
    df.loc[i] = [0 for n in range(len(features))]

row = 0
df,row = insert_term_frequency_ham(path1,df,row)
df,row = insert_term_frequency_spam(path2,df,row)

one_col = [1]*(row)

df.insert(loc=0, column='threshold', value=one_col)

matrix = df.values


weight = [0]*(len(features))

iterations = 10
eta = 0.1

print('Iterations: ',iterations)
print('eta: ',eta)

train_algorithm(eta,features,iterations)

##############################################################################

path1 = os.getcwd() + "/" + folder + "/" +"test/ham"
path2 = os.getcwd() + "/" + folder + "/" +"test/spam"


df_test = pd.DataFrame(columns=features)

for i in range(len(os.listdir(path1))+len(os.listdir(path2))):
    df_test.loc[i] = [0 for n in range(len(features))]

row = 0
df_test,row = insert_term_frequency_test_ham(path1,df_test,row,features)
df_test,row = insert_term_frequency_test_spam(path2,df_test,row,features)

one_col = [1]*(row)

df_test.insert(loc=0, column='threshold', value=one_col)

matrix = df_test.values

correct = test_algorithm(row)

print(folder,correct/float(row))
