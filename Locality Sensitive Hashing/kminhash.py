# -*- coding: utf-8 -*-
import string
import numpy as np
import nltk
import random
import time
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
# encoding=utf8
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

def computeJaccardSim(shinglesAll, numPairs):
    JaccardSim = np.zeros((3000,3000))
    t0 = time.time()
    MSError = 0;
    for i in range(3000):
        shingle1 = shinglesAll[i]  
        JaccardSim[i, i] = 1
        for j in range(i + 1, 3000):
            shingle2 = shinglesAll[j]
            tempSim = (len(shingle1.intersection(shingle2)) / float(max(1,len(shingle1.union(shingle2)))))  
            JaccardSim[i, j] = tempSim
            JaccardSim[j, i] = tempSim
            if tempSim > 0:
                haha = 0
        sys.stdout.write(' ' * 10 + '\r')
        sys.stdout.flush()
        sys.stdout.write("Calculating Jaccard Similarities:" + str(i) + '/' + str(3000) +'\r')
        sys.stdout.flush()        
    elapsed = (time.time() - t0)    
    print ("Calculating Jaccard Similarities took %.2fsec" % elapsed)
    print ("Jaccard Similarity:")
    print JaccardSim
    return JaccardSim

def RandomCoeffs(KMinHash):
    Rands = []
    while KMinHash > 0:
        rand = random.randint(0, 3000) 
        while rand in Rands:
            rand = random.randint(0, 3000) 
        Rands.append(rand)
        KMinHash = KMinHash - 1
    return Rands

def generateSignatures(KMinHash, shinglesAll):
    t0 = time.time()
    coeffA = RandomCoeffs(KMinHash)
    coeffB = RandomCoeffs(KMinHash)    
    signatures = []
    for i in range(3000):
        signature = []
        for k in range(KMinHash):
            minHashCode = 3001 + 1
            for shingleID in shinglesAll[i]:
                hashCode = (coeffA[k] * shingleID + coeffB[k]) % 3001
                if hashCode < minHashCode:
                    minHashCode = hashCode
            signature.append(minHashCode)
        signatures.append(signature)
        sys.stdout.write(' ' * 10 + '\r')
        sys.stdout.flush()
        sys.stdout.write("generating Signatures:" + str(i) + '/' + str(3000) +'\r')
        sys.stdout.flush()          
    elapsed = (time.time() - t0)
    print ("Generating %d-MinHash signatures took %.2fsec" % (KMinHash, elapsed))   
    return signatures

def compareAllSignatures(KMinHash, signatures, JaccardSim, numPairs):
    MinHashSim = np.zeros((3000,3000))
    t0 = time.time()
    MSError = 0;
    for i in range(0, 3000):
        MinHashSim[i, i] = 1;
        signature1 = signatures[i]        
        for j in range(i + 1, 3000):
            signature2 = signatures[j]
            count = 0
            for k in range(0, KMinHash):
                count = count + (signature1[k] == signature2[k])
            GTSim = (JaccardSim[i, j])
            TempSim = count / float(KMinHash)
            MinHashSim[i, j] = TempSim
            MinHashSim[j, i] = TempSim
            MSError += ((TempSim - GTSim) * (TempSim - GTSim) / float(numPairs))    
        sys.stdout.write(' ' * 10 + '\r')
        sys.stdout.flush()
        sys.stdout.write("comparing all signatures:" + str(i) + '/' + str(3000) +'\r')
        sys.stdout.flush()         
    elapsed = (time.time() - t0)
    print ('Comparing all %d-minhash signatures took %.2f secs' % (KMinHash, elapsed))
    print ("MinHash Similarity:")
    print MinHashSim
    return MSError


file_names = ['amazon_cells_labelled.txt', 'yelp_labelled.txt', 'imdb_labelled.txt']

words = {} 
strip = string.whitespace + string.punctuation + string.digits + "\"'"


#----------------------------no shingle----------------------------------
#generating words(the length of words should >= 2) no shingle
print "waiting... generating words(the length of words should >= 2) no shingle"
sampleNum = 0
shinglesAll = {}
CountMap = {}
count = 0
for filename in file_names[0:]:
    for line in open(filename):
        shingleLine = set()
        for word in line.split():
            word = word.strip(strip)
            word = word.lower()
            word = lemm.lemmatize(word, 'n')
            word = lemm.lemmatize(word, 'v')
            word = lemm.lemmatize(word, 'a')
            if len(word) >= 2:                
                if word in CountMap:
                    shingleLine.add(CountMap[word])
                else:
                    CountMap[word] = count + 1
                    shingleLine.add(CountMap[word])
                    count += 1
        shinglesAll[sampleNum] = shingleLine
        sampleNum = sampleNum + 1

print ("samples number： %d features number:%d " %(sampleNum, count))

numPairs = int(3000 * (3000 - 1) / 2)

JaccardSim = computeJaccardSim(shinglesAll, numPairs)

signatures = generateSignatures(16, shinglesAll)
MSError = compareAllSignatures(16, signatures, JaccardSim, numPairs)
print("no shingle The mean squared error of 16-MinHash is %f \n" % (MSError))

signatures = generateSignatures(32, shinglesAll)
MSError = compareAllSignatures(32, signatures, JaccardSim, numPairs)
print("no shingle The mean squared error of 32-MinHash is %f \n" % (MSError)) 

signatures = generateSignatures(64, shinglesAll)
MSError = compareAllSignatures(64, signatures, JaccardSim, numPairs)
print("no shingle The mean squared error of 64-MinHash is %f \n" % (MSError)) 

signatures = generateSignatures(128, shinglesAll)
MSError = compareAllSignatures(128, signatures, JaccardSim, numPairs)
print("no shingle The mean squared error of 128-MinHash is %f \n" % (MSError)) 

signatures = generateSignatures(256, shinglesAll)
MSError = compareAllSignatures(256, signatures, JaccardSim, numPairs)
print("no shingle The mean squared error of 256-MinHash is %f \n" % (MSError)) 


#---------------------------- 3-shingle----------------------------------        
#generating shingles (the length of words should >= 2) 3-shingle
print "waiting... generating shingles(the length of words should >= 2) 3-shingle"
sampleNum = 0
shinglesAll = {}
CountMap = {}
count = 0
for filename in file_names[0:]:
    for line in open(filename):
        shingleLine = set()
        words = []
        for word in line.split():
            word = word.strip(strip)
            word = word.lower()
            word = lemm.lemmatize(word, 'n')
            word = lemm.lemmatize(word, 'v')
            word = lemm.lemmatize(word, 'a')
            if len(word) >= 2:                
                words.append(word)
        for index in range(0, len(words) - 2):
            shingle = words[index] + " " + words[index + 1] + " " + words[index + 2]                 
            if shingle in CountMap:
                shingleLine.add(CountMap[shingle])
            else:
                CountMap[shingle] = count + 1
                shingleLine.add(CountMap[shingle])
                count += 1
        shinglesAll[sampleNum] = shingleLine
        sampleNum = sampleNum + 1
        
print ("samples number： %d features number:%d " %(sampleNum, count))

JaccardSim = computeJaccardSim(shinglesAll, numPairs)

signatures = generateSignatures(16, shinglesAll)
MSError = compareAllSignatures(16, signatures, JaccardSim, numPairs)
print("3-shingle: The mean squared error of 16-MinHash is %f \n" % (MSError))

signatures = generateSignatures(32, shinglesAll)
MSError = compareAllSignatures(32, signatures, JaccardSim, numPairs)
print("3-shingle: The mean squared error of 32-MinHash is %f \n" % (MSError)) 

signatures = generateSignatures(64, shinglesAll)
MSError = compareAllSignatures(64, signatures, JaccardSim, numPairs)
print("3-shingle: The mean squared error of 64-MinHash is %f \n" % (MSError)) 

signatures = generateSignatures(128, shinglesAll)
MSError = compareAllSignatures(128, signatures, JaccardSim, numPairs)
print("3-shingle: The mean squared error of 128-MinHash is %f \n" % (MSError)) 

signatures = generateSignatures(256, shinglesAll)
MSError = compareAllSignatures(256, signatures, JaccardSim, numPairs)
print("3-shingle: The mean squared error of 256-MinHash is %f \n" % (MSError)) 
