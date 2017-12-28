import pandas as pd
import numpy as np
import math
import sys

class Node:
    def __init__(self,feature):
        self.data = feature
        self.left = None
        self.right = None
        self.majority = None
        self.end = 0

def accuracy(model,dataset):
    expected_values = dataset['Class']
    expected_values = list(expected_values)
    calculated_values = []
    start = model
    for i in range(len(dataset)):
        model = start
        while model.end == 0:
            path = dataset.loc[i,model.data]
            if path == 0:
                model = model.left
            else:
                model = model.right
        if model.majority  == 1:
            calculated_values.append(1)
        else:
            calculated_values.append(0)
    count = 0
    for i in range(len(expected_values)):
        if expected_values[i] == calculated_values[i]:
            count = count + 1

    return count/len(expected_values)

def draw(start,level):

    #print pipe according to levels
    for i in range(level):
        print("|",sep='',end=' ')
        
    print(start.data,"= 0 : ",end='')

    if start.left.end == 1:
        if start.left.majority == 0:
            print('0')
        elif start.left.majority == 1:
            print("1")        
    else:
        print()
        draw(start.left,level+1)


    for i in range(level):
        print("|",sep='',end=' ')
    
    print(start.data,"= 1 : ",end='')
    
    if start.right.end == 1:
        if start.right.majority == 0:
            print('0')
        elif start.right.majority == 1:
            print("1")        
    else:
        print()
        draw(start.right,level+1)

def prune(start,validation):
    initial_accuracy = accuracy(start,validation)
    subset_accuracy = initial_accuracy

    while initial_accuracy <= subset_accuracy:
        q = []
        
        if start.left.end == 0:
            q.append(start.left)

        if start.right.end == 0:
            q.append(start.right)
            
        d = {}
        while q and q[0].end == 0:
            if q[0].left.end == 0:
                q.append(q[0].left)

            if q[0].right.end == 0:
                q.append(q[0].right)

            q[0].end = 1

            acc = accuracy(start,validation)

            q[0].end = 0

            if acc > initial_accuracy:
                d = {}
                d[acc] = [q[0]]
                initial_accuracy = acc
            elif acc == initial_accuracy:
                if len(d):
                    d[acc].append(q[0])
                else:
                    d[acc] = [q[0]]
                    
            q.pop(0)

        if len(d):
            if list(d.keys())[0] >= initial_accuracy:
                for i in d[list(d.keys())[0]]:
                    i.end = 1
            subset_accuracy = list(d.keys())[0]
        else:
            subset_accuracy = acc
    
    return subset_accuracy
            

            
def ID3(example,target_attribute,attributes):

    child = Node(None)#Creating a new node with attribute value none and left, right values none

    temp = example.groupby(['Class']).groups#Getting index of class labels with 0 and 1 value

    if 0 in temp:
        #if there are some 0 values
        tot_0 = len(temp[(0)])#total 0 values
    else:
        tot_0 = 0#no 0 values
        
    if 1 in temp:
        #if there are some 1 values
        tot_1 = len(temp[(1)])#total 0 values
    else:
        tot_1 = 0#no 0 values

    class_num = [tot_0,tot_1]#list of number of 0 and 1
    total_values = tot_0 + tot_1#total number of 0 and 1

    if tot_0 > tot_1:
        child.majority = 0
    else:
        child.majority = 1
    
    if total_values == tot_0:
        #if all values are 0
        child.data = 0#leaf node is 0
        child.end = 1
        return child
    
    elif total_values == tot_1:
        #if all values are 1
        child.data = 1#leaf node is 1
        child.end = 1
        return child

    elif len(attributes) == 0:
        #if we run out of features
        if tot_0 > tot_1:
            #if num of 0 are more than num of 1
            child.data = 0
        else:
            #if num of 1 are more than num of 0
            child.data = 1
        child.end = 1    
        return child

    #entropy of super feature
    feature_entropy = 0
    for i in class_num:
        feature_entropy = feature_entropy + -1 * (i/total_values)* math.log(i/total_values,2)

    
    #Entropy of different feature taking different values
    info_gain = -2109281#this should be maximum
    for j in attributes:
        feature_value_class_value = example.loc[:,[j,'Class']]#taking feature and class column
        stat = feature_value_class_value.groupby([j,'Class']).groups#getting groups for all combination of feature and class

        if (0,0) in stat:
            #if (0,0) is there in dict
            tot_00 = len(stat[(0,0)])#class label 0 for 0 values in features
        else:
            tot_00 = 0#no (0,0) values
        
        if (0,1) in stat:
            #if (0,1) is there in dict
            tot_01 = len(stat[(0,1)])#class label 1 for 0 values in features
        else:
            tot_01 = 0
            
        if (1,0) in stat:
            #if (1,0) is there in dict
            tot_10 = len(stat[(1,0)])#class label 0 for 1 values in features
        else:
            tot_10 = 0
            
        if (1,1) in stat:
            #if (1,1) is there in dict
            tot_11 = len(stat[(1,1)])#class label 1 for 1 values in features
        else:
            tot_11 = 0
        
        class_feature_0 = [tot_00,tot_01]#list of 0,1 values for 0 values in features
        class_feature_1 = [tot_10,tot_11]#list of 0,1 values for 1 values in features

        feature_value = 0

        for k in class_feature_0:
            if k:
                feature_value = feature_value + -1 * (tot_0/(tot_0+tot_1))*(k/(tot_00+tot_01))* math.log(k/(tot_00+tot_01),2)
                
        for k in class_feature_1:
            if k:
                feature_value = feature_value + -1 * (tot_1/(tot_0+tot_1))*(k/(tot_10+tot_11))* math.log(k/(tot_10+tot_11),2)

    
        if feature_entropy - feature_value > info_gain:
            #information gain is bigger than previous
            info_gain = feature_entropy - feature_value 
            feature_selected = j

    feature_values = example.groupby([feature_selected])#getting values with 0 and 1 seperated
    features_list.remove(feature_selected)#remove selected feature from feature list

    feature_values = feature_values.groups
    feature_values[0] = list(feature_values[0])#index of 0 values in selected feature
    feature_values[1] = list(feature_values[1])#index of 1 values in selected feature
                        
    if len(feature_values[0]):
        #if there are non zero values for 0 value in features 
        child.data = feature_selected#setting feature name 
        child.left = ID3(example.loc[feature_values[0]],feature_selected,features_list) #getting left node either feature or leaf(0,1)
    else:
        #if there are no 0 values in selected feature
        if tot_0 > tot_1:
            #if class label 0 > class label 1
            child.data = 0
        else:
            #if class label 1 > class label 0
            child.data = 1
        child.end = 1
        return child
            
    if len(feature_values[1]):
        #if there are non zero values for 1 value in features 
        child.data = feature_selected    
        child.right = ID3(example.loc[feature_values[1]],feature_selected,features_list)#getting right node either feature or leaf(0,1)
    else:
        #if there are no 1 values in selected feature
        if tot_0 > tot_1:
            #if class label 0 > class label 1
            child.data = 0
        else:
            #if class label 1 > class label 0
            child.data = 1
        child.end = 1
        return child
            
    return child
    

def ID3_variance(example,target_attribute,attributes):

    child = Node(None)#Creating a new node with attribute value none and left, right values none

    temp = example.groupby(['Class']).groups#Getting index of class labels with 0 and 1 value

    if 0 in temp:
        #if there are some 0 values
        tot_0 = len(temp[(0)])#total 0 values
    else:
        tot_0 = 0#no 0 values
        
    if 1 in temp:
        #if there are some 1 values
        tot_1 = len(temp[(1)])#total 0 values
    else:
        tot_1 = 0#no 0 values

    class_num = [tot_0,tot_1]#list of number of 0 and 1
    total_values = tot_0 + tot_1#total number of 0 and 1

    if tot_0 > tot_1:
        child.majority = 0
    else:
        child.majority = 1
        
    if total_values == tot_0:
        #if all values are 0
        child.data = 0#leaf node is 0
        child.end = 1
        return child
    
    elif total_values == tot_1:
        #if all values are 1
        child.data = 1#leaf node is 1
        child.end = 1
        return child

    elif len(attributes) == 0:
        #if we run out of features
        if tot_0 > tot_1:
            #if num of 0 are more than num of 1
            child.data = 0
        else:
            #if num of 1 are more than num of 0
            child.data = 1
        child.end = 1
        return child

    #entropy of super feature
    feature_entropy = 0
    for i in class_num:
        feature_entropy = feature_entropy + (tot_0*tot_1)/(total_values*total_values)

    
    #Entropy of different feature taking different values
    info_gain = -2109281#this should be maximum
    for j in attributes:
        feature_value_class_value = example.loc[:,[j,'Class']]#taking feature and class column
        stat = feature_value_class_value.groupby([j,'Class']).groups#getting groups for all combination of feature and class

        if (0,0) in stat:
            #if (0,0) is there in dict
            tot_00 = len(stat[(0,0)])#class label 0 for 0 values in features
        else:
            tot_00 = 0#no (0,0) values
        
        if (0,1) in stat:
            #if (0,1) is there in dict
            tot_01 = len(stat[(0,1)])#class label 1 for 0 values in features
        else:
            tot_01 = 0
            
        if (1,0) in stat:
            #if (1,0) is there in dict
            tot_10 = len(stat[(1,0)])#class label 0 for 1 values in features
        else:
            tot_10 = 0
            
        if (1,1) in stat:
            #if (1,1) is there in dict
            tot_11 = len(stat[(1,1)])#class label 1 for 1 values in features
        else:
            tot_11 = 0
        
        class_feature_0 = [tot_00,tot_01]#list of 0,1 values for 0 values in features
        class_feature_1 = [tot_10,tot_11]#list of 0,1 values for 1 values in features

        feature_value = 0
        if (tot_0+tot_1) != 0 and (tot_00+tot_01) != 0:
            feature_value = feature_value + (tot_0/(tot_0+tot_1))*((tot_00*tot_01)/(tot_00+tot_01)*(tot_00+tot_01))                

        if (tot_0+tot_1) != 0 and (tot_10+tot_11) != 0:
            feature_value = feature_value + (tot_1/(tot_0+tot_1))*((tot_10*tot_11)/(tot_10+tot_11)*(tot_10+tot_11))

        if feature_entropy - feature_value >= info_gain:
            #information gain is bigger than previous
            info_gain = feature_entropy - feature_value 
            feature_selected = j

    feature_values = example.groupby([feature_selected])#getting values with 0 and 1 seperated
    features_list.remove(feature_selected)#remove selected feature from feature list

    feature_values = feature_values.groups
    feature_values[0] = list(feature_values[0])#index of 0 values in selected feature
    feature_values[1] = list(feature_values[1])#index of 1 values in selected feature
                        
    if len(feature_values[0]):
        #if there are non zero values for 0 value in features 
        child.data = feature_selected#setting feature name 
        child.left = ID3_variance(example.loc[feature_values[0]],feature_selected,features_list) #getting left node either feature or leaf(0,1)
    else:
        #if there are no 0 values in selected feature
        if tot_0 > tot_1:
            #if class label 0 > class label 1
            child.data = 0
        else:
            #if class label 1 > class label 0
            child.data = 1
        child.end = 1
        return child
            
    if len(feature_values[1]):
        #if there are non zero values for 1 value in features 
        child.data = feature_selected    
        child.right = ID3_variance(example.loc[feature_values[1]],feature_selected,features_list)#getting right node either feature or leaf(0,1)
    else:
        #if there are no 1 values in selected feature
        if tot_0 > tot_1:
            #if class label 0 > class label 1
            child.data = 0
        else:
            #if class label 1 > class label 0
            child.data = 1
        child.end = 1
        return child
            
    return child


df_training = pd.read_csv(sys.argv[1])#Training dataframe
df_test = pd.read_csv(sys.argv[3])#Test dataframe
df_validation = pd.read_csv(sys.argv[2])#validation Dataset

arg = sys.argv[4]
arg2 = sys.argv[5]

features = df_training.keys()#all features label including class
features_list = list(features[:-1])#all features label excluding class

result = ID3(df_training,None,features_list)#calling algorithm. returns pointer to first node

if arg2 == 'yes':
    prune(result,df_validation)
    

if arg == 'yes':
    draw(result,0)

print('Information Gain Test Accuracy:',accuracy(result,df_test))


df_training = pd.read_csv(sys.argv[1])#Training dataframe
df_test = pd.read_csv(sys.argv[3])#Test dataframe
df_validation = pd.read_csv(sys.argv[2])#validation Dataset

features = df_training.keys()#all features label including class
features_list = list(features[:-1])#all features label excluding class

result_variance = ID3_variance(df_training,None,features_list)

if arg2 == 'yes':
    prune(result_variance,df_validation)
    

if arg == 'yes':
    draw(result_variance,0)

print('Variance Test Accuracy:',accuracy(result_variance,df_test))
