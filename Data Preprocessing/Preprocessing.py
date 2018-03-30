# -*- coding: utf-8 -*-
import string
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
# encoding=utf8
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

file_names = ['amazon_cells_labelled.txt', 'yelp_labelled.txt', 'imdb_labelled.txt']

words = {} 
strip = string.whitespace + string.punctuation + string.digits + "\"'"

#count words numbers(the length of words should >= 2)
print "waiting... count words numbers(the length of words should >= 2)"
sampleNum = 0
for filename in file_names[0:]:
    for line in open(filename):
        for word in line.split():
            word = word.strip(strip)
            word = word.lower()
            word = lemm.lemmatize(word, 'n')
            word = lemm.lemmatize(word, 'v')
            word = lemm.lemmatize(word, 'a')              
            if len(word) >= 2:                
                words[word] = words.get(word, 0) + 1
        sampleNum = sampleNum + 1

#give position to the words and delete some words whose frequency is less than 5 times
print "waiting... give position to the words and delete some words whose frequency is less than 5 times"
wordPosition = 0
position = {}
for word in sorted(words):
#    print("'{0}' occurs {1} times".format(word,words[word]))
    if words[word] >= 5:
        position[word] = wordPosition
        wordPosition = wordPosition + 1
    else:
        words.pop(word)
    
#construct feature matrix and label
print "waiting... construct feature matrix and label"
features = np.zeros([sampleNum, len(words)])
labels = np.zeros([sampleNum, 1])
sampleNum = 0

maxlinewordNum = 0;
for filename in file_names[0:]:
    for line in open(filename):
        linewords = {}        
        for word in line.split():
            word = word.strip(strip)
            word = word.lower()            
            word = lemm.lemmatize(word, 'n')            
            word = lemm.lemmatize(word, 'v')
            word = lemm.lemmatize(word, 'a')              
            if word in words:
                linewords[word] = linewords.get(word, 0) + 1                
                features[sampleNum, position[word]] = linewords[word]
                if maxlinewordNum < linewords[word]:
                    maxlinewordNum = linewords[word]
        labels[sampleNum] = line.split()[len(line.split()) - 1]
        sampleNum = sampleNum + 1

#feature normalization to [0,1]
features = features / maxlinewordNum        
        
print ("samples number： %d  words number: %d labels number： %d" %(sampleNum, len(words), len(labels)))
print ("the shape of feature matrix: %d %d" % (features.shape[0], features.shape[1]))

#show one example
print "example 1:"
for filename in file_names[0:]:
    for line in open(filename):
        print line
        exword = ''
        exwordafter = ''
        for word in line.split():
            word = word.strip(strip)
            word = word.lower()            
            exword = exword + word + ' '
            word = lemm.lemmatize(word, 'n')
            word = lemm.lemmatize(word, 'v')
            word = lemm.lemmatize(word, 'a')              
            if word in words:
                exwordafter = exwordafter + word + ' '
        print ("seperate words:\n  %s" %(exword))
        print ("words after preprocessing:\n  %s" %(exwordafter))        
        break
    break

print "feature vector for example 1: "
print features[0,:]