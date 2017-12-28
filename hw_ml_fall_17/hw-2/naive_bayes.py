import os
import re
import sys
import math

reg_exp = '[a-z0-9]+'

path = ["stemmed", "stop_stem"]
'''
stop = sys.argv[1]
train = sys.argv[2]
test = sys.argv[3]
'''
for folder in path:
    
    #total distinct words in dictionary
    tot_dict_words = 0
    
    path = os.getcwd()+"/"+folder+'/train/ham'

    total_ham = len(os.listdir(path))#total ham document

    ham = {}#ham document dictionary

    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r")
        for word in file:
            if re.match(reg_exp,word):#taking just alphanumeric strings
                word = word.strip()
                if word in ham:
                    ham[word] = ham[word] + 1#adding count of particular string
                else:
                    ham[word] = 1#adding string for first time
                    tot_dict_words = tot_dict_words + 1#adding dictionary count


    #total words in ham
    tot_word_ham = 0 
    for i in ham:
        tot_word_ham = tot_word_ham + ham[i]
        

    path = os.getcwd()+"/"+folder+'/train/spam'

    total_spam = len(os.listdir(path))

    spam = {}
    for filename in os.listdir(path):
        #file = open(path+'/'+filename, "r", encoding="Latin-1")
        file = open(path+'/'+filename, "r")
        for word in file:
            if re.match(reg_exp,word):
                word = word.strip()
                if word in spam:
                    spam[word] = spam[word] + 1
                else:
                    spam[word] = 1
                    tot_dict_words = tot_dict_words + 1

    #total words in spam
    tot_word_spam = 0
    for i in spam:
        tot_word_spam = tot_word_spam + spam[i]


    ham_prior = float(total_ham)/(total_ham + total_spam)#prior of ham


    spam_prior = float(total_spam)/(total_ham + total_spam)#prior of spam



    path = os.getcwd()+"/"+folder+'/test/ham'

    predict_for_ham = []
    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r")
        
        prob_of_words_given_ham = math.log(ham_prior)
        prob_of_words_given_spam = math.log(spam_prior)
        
        for word in file:
            if re.match(reg_exp,word):
                word = word.strip()
                if word in ham:#adding conditional probabilities
                    prob_of_words_given_ham = prob_of_words_given_ham + math.log(float(ham[word]+1)/(tot_word_ham + tot_dict_words))
                elif word in spam:#if word not in ham
                    prob_of_words_given_ham = prob_of_words_given_ham + math.log(1/float(tot_word_ham + tot_dict_words))
                    
                if word in spam:
                    prob_of_words_given_spam = prob_of_words_given_spam + math.log(float(spam[word]+1)/(tot_word_spam + tot_dict_words))
                elif word in ham :
                    prob_of_words_given_spam = prob_of_words_given_spam + math.log(1/float(tot_word_spam + tot_dict_words))

        if prob_of_words_given_ham >= prob_of_words_given_spam:
            predict_for_ham.append('ham')
        else:
            predict_for_ham.append('spam')


    path = os.getcwd()+"/"+folder+'/test/spam'

    predict_for_spam = []
    for filename in os.listdir(path):
        file = open(path+'/'+filename, "r")
        
        prob_of_words_given_ham = math.log(ham_prior)
        prob_of_words_given_spam = math.log(spam_prior)

        for word in file:
            if re.match(reg_exp,word):
                word = word.strip()
                if word in ham :
                    prob_of_words_given_ham = prob_of_words_given_ham + math.log(float(ham[word]+1)/(tot_word_ham + tot_dict_words))
                elif word in spam:
                    prob_of_words_given_ham = prob_of_words_given_ham + math.log(1/float(tot_word_ham + tot_dict_words))
                    
                if word in spam:
                    prob_of_words_given_spam = prob_of_words_given_spam + math.log((spam[word]+1)/float(tot_word_spam + tot_dict_words))
                elif word in ham :
                    prob_of_words_given_spam = prob_of_words_given_spam + math.log(1/float(tot_word_spam + tot_dict_words))

        if prob_of_words_given_ham >= prob_of_words_given_spam:
            predict_for_spam.append('ham')
        else:
            predict_for_spam.append('spam')

    correct = 0
    incorrect = 0
    for i in predict_for_ham:
        if i == 'ham':
            correct = correct + 1
        else:
            incorrect = incorrect + 1
            
    for i in predict_for_spam:
        if i == 'spam':
            correct = correct + 1
        else:
            incorrect = incorrect + 1
    if folder == 'stemmed':
        print('Accuracy without stopwords:',correct/float(correct+incorrect))
    else:
        print('Accuracy with stopwords:',correct/float(correct+incorrect))
