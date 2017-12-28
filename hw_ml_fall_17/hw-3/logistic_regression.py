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

    try:
        return float(math.exp(zee)/float(1.0 + math.exp(zee)))#sigmoid function
    except OverflowError:
        return 1.0


def train_algorithm(eta,lmbda,features,iterations):
    global weight
    global matrix

    calculated_array = [0]*(row)#initializing array for storing predicted value for each datapoint
    
    for loop in range(iterations):#hard limit on iterations

        for doc in range(row):
            calculated_array[doc] = calculated_value(doc)#calculating predicted value for each datapoint
            
        partial_deravitive = [0]*(len(features))#initializing partial derivative of all weights to zero

        for j in range(len(weight)):
            for i in range(row):
                #updating partial_deravitive for every document under each weight 
                partial_deravitive[j] = partial_deravitive[j] + matrix[i,j]*(matrix[i,-1] - calculated_array[i])

        for w in range(len(weight)):
            #updating all weights after updating all partial_deravitive 
            weight[w] = weight[w] + eta*((-lmbda*weight[w]) + partial_deravitive[w])
            


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

if sys.argv[1] == "stemmed":
    folder = "stemmed"
else:
    folder = "stop_stem"

print(folder)

path1 = os.getcwd() + "/" + folder + "/" +"train/ham"
#Getting ham from train dataset
path2 = os.getcwd() + "/" + folder + "/" +"train/spam"
#Getting spam from train dataset

features = [] #Toatal feature list

features = make_dictionary(path1,features)#Making features from ham dataset 
features = make_dictionary(path2,features)#Making features from spam dataset 

features.append('output')#adding output column that has actual values of ham or spam

df = pd.DataFrame(columns=features)#Making dataframe with features

#Initializing 0 for feature and its count in document
for i in range(len(os.listdir(path1))+len(os.listdir(path2))):
    df.loc[i] = [0 for n in range(len(features))]

row = 0#variable for keeping track of rows
df,row = insert_term_frequency_ham(path1,df,row)#calculating count of feature in each document of ham 
df,row = insert_term_frequency_spam(path2,df,row)#calculating count of feature in each document of spam 

one_col = [1]*(row)#adding column that has all values 1 for multiplying with w0

df.insert(loc=0, column='threshold', value=one_col)#actually adding column to dataframe

matrix = df.values#converting dataframe to matrix for fast calculation

weight = [0]*(len(features))#initializing weights to 0

lmbda = 1
iterations = 10
eta = .01

print('lmbda: ',lmbda)
print('Iterations: ',iterations)
print('eta: ',eta)

train_algorithm(eta,lmbda,features,iterations)

##############################################################################

path1 = os.getcwd() + "/" + folder + "/" +"test/ham"
#Getting ham from test dataset
path2 = os.getcwd() + "/" + folder + "/" +"test/spam"
#Getting spam from test dataset

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
