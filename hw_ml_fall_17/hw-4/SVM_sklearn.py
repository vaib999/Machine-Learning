import pandas as pd
import numpy as np
import math
import sys
import os
from sklearn import svm

def read_file(filename):
    df = pd.DataFrame()
    path = 'C:/Entertainment/My Subjects/Machine Learning/hw-4/dataset'
    file = open(path+'/'+filename, "r")

    row = 0
    for line in file:
        output = line[0]
        line = line[1:]

        col = 0
        for word in line.split():
           feature = word[-1]
           df.loc[row,col] = feature
           col+=1

        df.loc[row,col] = output
        row+=1
        
    return df

def predict(df_train,train_valid,ker):
    clf = svm.SVC(kernel=ker)
    clf.fit(df_train.iloc[:,:-1], df_train.iloc[:,-1])

    predicted_values = clf.predict(train_valid.iloc[:,:-1])

    correct = 0
    for i in range(len(predicted_values)):
        if predicted_values[i] == train_valid.iloc[i,-1]:
            correct+=1

    print(ker,correct/float(len(predicted_values)))


df_training = read_file('training.new')

df_validation = read_file('validation.new')


predict(df_training,df_validation,'rbf')

predict(df_training,df_validation,'linear')

predict(df_training,df_validation,'poly')
