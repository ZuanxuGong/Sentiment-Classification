# -*- coding: utf-8 -*-
import string
import numpy as np
import random
import nltk
import math
import time
import sys
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()

# encoding=utf8
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

file_names = ['amazon_cells_labelled.txt', 'yelp_labelled.txt', 'imdb_labelled.txt']

def idlist(n):
    L = []
    i = -1
    if n == 0:
        L = [0]
    else:
        while i < n-1:
            i = i + 1
            L.append(i)
    return L

def gender_features(line, wordstemp):
    wordsfeature = wordstemp.copy()
    for word in line.split():
        if word in wordsfeature:
            wordsfeature[word] = wordsfeature.get(word, 0) + 1 
    return wordsfeature

# FV1: delete words whose frequency is less than 5
def get_FV1(words):
    print "waiting... get FV1: delete words whose frequency is less than 5"
    i = 0
    wordsFV1 = {}
    for word in sorted(words.items(), key = lambda item:item[1], reverse = 1):
        if word[1] >= 5:
            wordsFV1[word[0]] = 0
        else:
            continue
        i = i + 1
    return wordsFV1

# FV2: remain words with top 256 frequency
def get_FV2(words):
    print "waiting... get FV2: remain words with top 256 frequency"
    i = 0
    wordsFV2 = {}
    for word in sorted(words.items(), key = lambda item:item[1], reverse = 1):
        if i < 256:
            wordsFV2[word[0]] = 0
        else:
            continue
        i = i + 1
    return wordsFV2

# FV3: remain words with top 256 TFIDF scores
def get_FV3(words, sentenceNum, wordsMaxFreq):
    print "waiting... get FV3: remain words with top 256 TFIDF scores"    
    IFIDF = {}
    wordsFV3 = {}
    i = 0
    for word in words.keys():
        IFIDF[word] = wordsMaxFreq[word] * math.log(3000.0 / float(sentenceNum[word]))
    for word in sorted(IFIDF.items(), key = lambda item:item[1], reverse = 1):
            if i < 256:
                wordsFV3[word[0]] = 0
            else:
                continue
            i = i + 1
    return wordsFV3

def getFV(wordsFV, trainingset, validationset, testingset):
    train_FV = [(gender_features(samples[i], wordsFV), labels[i]) for i in trainingset]
    validation_FV = [(gender_features(samples[i], wordsFV), labels[i]) for i in validationset]
    test_FV = [(gender_features(samples[i], wordsFV), labels[i]) for i in testingset]
    return train_FV, validation_FV, test_FV
            
def confusionMatrix(res, FV):
    TP = 0
    TN = 0
    FP = 0
    FN = 0   
    for i in range(len(res)):
        if(res[i] == '1'):
            if(FV[i][1] == '1'):
                TP += 1
            else:
                FP  += 1
        else:
            if(FV[i][1] == '0'):
                TN += 1
            else:
                FN += 1                           
    return TP, TN, FP, FN

def ClassifierNB(train_FV, validation_FV, test_FV, wordsFV):
    NBFV_s = time.time()
    classifierNBFV = nltk.classify.NaiveBayesClassifier.train(train_FV)
    NBFV_e = time.time()
    print("Time(training with 1800 samples): %f s" % (NBFV_e - NBFV_s))    
    resTrain = []
    resValidation = []
    resTest = []
    #Trainin set
    for i in range(1800):
        resTrain.append(classifierNBFV.classify(train_FV[i][0]))
    [TP, TN, FP, FN] = confusionMatrix(resTrain, train_FV)
    print("Training Accuarcy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/1800, TP/float(TP+FP), TP/float(TP+FN)))
    
    #Validation set
    for i in range(600):
        resValidation.append(classifierNBFV.classify(validation_FV[i][0]))
    [TP, TN, FP, FN] = confusionMatrix(resValidation, validation_FV)   
    print("Validation Accuarcy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/600, TP/float(TP+FP), TP/float(TP+FN)))
    
    #Testing set
    NBFV_st = time.time()
    for i in range(600):
        resTest.append(classifierNBFV.classify(test_FV[i][0]))
    NBFV_et = time.time()
    [TP, TN, FP, FN] = confusionMatrix(resTest, test_FV)
    print("Test Accuarcy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/600, TP/float(TP+FP), TP/float(TP+FN)))
    print("Time(testing for a new tuple): %f s" % ((NBFV_et - NBFV_st) / float(len(test_FV))))
    print("Example 1: %s predicted: %s" % (sentences[testingset[0]], classifierNBFV.classify(test_FV[0][0])))
    #classifierNBFV.show_most_informative_features(15)

def ClassifierDT(train_FV, validation_FV, test_FV, wordsFV):
    NBDT_s = time.time()
    classifierDTFV = nltk.classify.DecisionTreeClassifier.train(train_FV, depth_cutoff = 90)
    NBDT_e = time.time()
    print("Time(training with 1800 samples): %f s" % (NBDT_e - NBDT_s))
    resTrain = []
    resValidation = []
    resTest = []    
    
    #Training set
    for i in range(1800):
        resTrain.append(classifierDTFV.classify(train_FV[i][0]))
    [TP, TN, FP, FN] = confusionMatrix(resTrain, train_FV)
    print("Training Accuarcy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/1800, TP/float(TP+FP), TP/float(TP+FN)))
        
    #Validation set
    for i in range(600):
        resValidation.append(classifierDTFV.classify(validation_FV[i][0]))
    [TP, TN, FP, FN] = confusionMatrix(resValidation, validation_FV)   
    print("Validation Accuarcy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/600, TP/float(TP+FP), TP/float(TP+FN)))
        
    #Testing set
    NBDT_st = time.time()
    for i in range(600):
        resTest.append(classifierDTFV.classify(test_FV[i][0]))
    NBDT_et = time.time()
    [TP, TN, FP, FN] = confusionMatrix(resTest, test_FV)
    print("Test Accuarcy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/600, TP/float(TP+FP), TP/float(TP+FN)))
    print("Time(testing for a new tuple): %f s" % ((NBDT_et - NBDT_st) / float(len(test_FV))))
    print("Example 1: %s predicted: %s" % (sentences[testingset[0]], classifierDTFV.classify(test_FV[0][0])))
    #classifierDTFV.show_most_informative_features(15)

def kNN(test_data, train_data, istest, k):
    res = []
    id = 0
    KNN_st = time.time()
    for sample1 in test_data:
        i = 0
        distance = {}
        label = {}
        for sample2 in train_data:
            dist = 0
            for attri in sample1[0]:
                test_value = sample1[0][attri]
                train_value = sample2[0][attri]
                if(test_value > 0 or train_value > 0):
                    dist += (test_value - train_value) * (test_value - train_value)

            distance[i] = dist;
            label[i] = sample2[1];
            i = i + 1
        
        vote = 0
        num = 0
        for nearest in sorted(distance.items(), key = lambda item:item[1]):
            if num < k:
                vote += int(label[nearest[0]])             
            else:
                continue
            num = num + 1                    
        if vote > k / 2:
            res.append('1')
        else:
            res.append('0')
        id += 1
        sys.stdout.write(' ' * 10 + '\r')
        sys.stdout.flush()
        sys.stdout.write("Proceeding:" + str(id) + '/' + str(len(test_data)) +'\r')
        sys.stdout.flush()        
    KNN_et = time.time()
    if(istest):
        print("Time(testing for a new tuple): %f s" % ((KNN_et - KNN_st) / float(len(test_data))))                                
    
    [TP, TN, FP, FN] = confusionMatrix(res, test_data)
    return TP, TN, FP, FN

words = {}
sentenceNum = {}
wordsMaxFreq = {}
wordsFV1 = {}
strip = string.whitespace + string.punctuation + string.digits + "\"'"
#count words numbers(the length of words should >= 2)
print "waiting... count words numbers(the length of words should >= 2)"
sampleNum = 0
sentences = {}
samples = {}
labels = {}
for filename in file_names[0:]:
    for line in open(filename):
        samples[sampleNum] = ''
        wordExist = {}
        wordNum = 0
        for word in line.split():
            word = word.strip(strip)
            word = word.lower()
            word = lemm.lemmatize(word, 'n')
            word = lemm.lemmatize(word, 'v')
            word = lemm.lemmatize(word, 'a')              
            if len(word) >= 2:                
                words[word] = words.get(word, 0) + 1
                wordExist[word] = wordExist.get(word, 0) + 1
                wordNum += 1
            samples[sampleNum] = samples[sampleNum] + word + ' ';
        for word in wordExist.keys():
            sentenceNum[word] = sentenceNum.get(word, 0) + 1
            wordsMaxFreq[word] = max(wordsMaxFreq.get(word, 0), float(wordExist[word])/float(wordNum))
        sentences[sampleNum] = line
        labels[sampleNum] = line.split()[len(line.split()) - 1]        
        sampleNum = sampleNum + 1

#Dataset Split 60% for training, 20% for testing, 20% for validation
List = idlist(sampleNum)
random.shuffle(List)
trainingset = List[0 : 1800]
validationset = List[1800 : 2400]
testingset = List[2400 : 3000]

#Generate FV1 FV2 FV3
wordsFV1 = get_FV1(words)
wordsFV2 = get_FV2(words)
wordsFV3 = get_FV3(wordsFV1, sentenceNum, wordsMaxFreq)
[train_FV1, validation_FV1, test_FV1] = getFV(wordsFV1, trainingset, validationset, testingset)
[train_FV2, validation_FV2, test_FV2] = getFV(wordsFV2, trainingset, validationset, testingset)
[train_FV3, validation_FV3, test_FV3] = getFV(wordsFV3, trainingset, validationset, testingset)

# Naive Bayes classifier
print("Naive Bayes Classifier FV1")
ClassifierNB(train_FV1, validation_FV1, test_FV1, wordsFV1)
print("Naive Bayes Classifier FV2")
ClassifierNB(train_FV2, validation_FV2, test_FV2, wordsFV2)
print("Naive Bayes Classifier FV3")
ClassifierNB(train_FV3, validation_FV3, test_FV3, wordsFV3)

#Decision Tree Classifier
print("Deceision Tree Classifier FV1")
ClassifierDT(train_FV1, validation_FV1, test_FV1, wordsFV1)
print("Deceision Tree Classifier FV2")
ClassifierDT(train_FV2, validation_FV2, test_FV2, wordsFV2)
print("Deceision Tree Classifier FV3")
ClassifierDT(train_FV3, validation_FV3, test_FV3, wordsFV3)

# K-NN classifier
k = 5
print("K-NN Classifier  FV2 -- K = 5")
[TP, TN, FP, FN] = kNN(train_FV2, train_FV2, 0, k)
print("Training Accuracy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/1800, TP/float(TP+FP), TP/float(TP+FN)))
[TP, TN, FP, FN] = kNN(validation_FV2, train_FV2, 0, k)
print("Validation Accuracy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/600, TP/float(TP+FP), TP/float(TP+FN)))
[TP, TN, FP, FN] = kNN(test_FV2, train_FV2, 1, k)
print("Test Accuracy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/600, TP/float(TP+FP), TP/float(TP+FN)))

print("K-NN Classifier  FV3 -- K = 5")
[TP, TN, FP, FN] = kNN(train_FV3, train_FV3, 0, k)
print("Training Accuracy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/1800, TP/float(TP+FP), TP/float(TP+FN)))
[TP, TN, FP, FN] = kNN(validation_FV3, train_FV3, 0, k)
print("Validation Accuracy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/600, TP/float(TP+FP), TP/float(TP+FN)))
[TP, TN, FP, FN] = kNN(test_FV3, train_FV3, 1, k)
print("Test Accuracy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/600, TP/float(TP+FP), TP/float(TP+FN)))

print("K-NN Classifier  FV1 -- K = 5")
[TP, TN, FP, FN] = kNN(train_FV1, train_FV1, 0, k)
print("Training Accuracy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/1800, TP/float(TP+FP), TP/float(TP+FN)))
[TP, TN, FP, FN] = kNN(validation_FV1, train_FV1, 0, k)
print("Validation Accuracy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/600, TP/float(TP+FP), TP/float(TP+FN)))
[TP, TN, FP, FN] = kNN(test_FV1, train_FV1, 1, k)
print("Test Accuracy: %f, Presision: %f, Recall: %f" % (float(TP+TN)/600, TP/float(TP+FP), TP/float(TP+FN)))

