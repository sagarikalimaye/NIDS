# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 00:34:48 2018

"""


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import csv
from collections import OrderedDict
import os.path
import time

def ACCSCORE(test_data):  
    test_label=np.asarray(test_data['Attack'])
    pred_label=np.asarray(test_data['prediction'])
    print(test_label)
    print(pred_label)
    TP=0
    TN=0
    FP=0
    FN=0
    Accuracy=0.00
    TP = np.sum(np.logical_and(pred_label == 1, test_label == 1))
    TN = np.sum(np.logical_and(pred_label == 0, test_label == 0))
    FP = np.sum(np.logical_and(pred_label == 0, test_label == 1))
    FN = np.sum(np.logical_and(pred_label == 1, test_label == 0))
    print('Number of True Positives:',TP)
    print('Number of True Negatives:',TN)
    print('Number of False Positives:',FP)
    print('Number of False Negatives:',FN)
    sum=TP+TN+FP+FN
    print("Total",sum)
    Accuracy=((TP+TN)/sum)*100
    sen=(TP/(TP+FN))*100
    spec=(FP/(FP+TN))*100
    print('Accuracy of the system is:',Accuracy)
    print('Sensitivity of the system is:',sen) #rate of correctly identified attack packets
    print('Specificity of the system is:',spec)

def TIMETEST(file):
    dataset = pd.read_csv(file)
    c1=dataset['tcp'].mean()
    print('mean:',c1)

    attack=[]
    for row in dataset['tcp']:
        if (row > 85):
            attack.append('1')
        else:
            attack.append('0')

        
    dataset['Attack']=attack
    print(dataset)
    dataset.to_csv('TESTCOUNT.csv')
    
    
def TIMETRAIN():
    dataset = pd.read_csv('TraindataCOUNT.csv')
    c1=dataset['tcp'].mean()
    print('mean:',c1)

    attack=[]
    for row in dataset['tcp']:
        if (row > 85):
            attack.append('1')
        else:
            attack.append('0')

        
    dataset['Attack']=attack
    print(dataset)
    dataset.to_csv('TRAINCOUNT.csv')
    
def NAIVEB():
    test = pd.read_csv('TESTCOUNT.csv')
    print(test)
#    for row in data['Time']:
#        row = row*100000
#        print(row)
    train=pd.read_csv('TRAINCOUNT.csv')
    
#    train, test = cross_validation.train_test_split(data,test_size=0.50)
    clf = SVC(kernel='linear')
    # Use all columns apart from the Attack column as feautures
    train_features = train.iloc[:,[2]]
  
    # Use the Attack column as the label
    train_label = train.iloc[:,[3]]
    
    test_features = test.iloc[:,[2]]
   
    test_label = test.iloc[:,[3]]
    
    # Train the naive bayes model
    clf.fit(train_features, train_label)

    # build a dataframe to show the expected vs predicted values
    test_data = pd.concat([test_features,test_label], axis=1)
    test_data["prediction"] = clf.predict(test_features)
    
    print(test_data)
    test['Prediction']=test_data['prediction']
    test.to_csv('ResNB.csv')
    ACCSCORE(test_data)
#    print('\nThe blaclisted IPs are:\n')
#    UpdateBlacklist(test)
    
    
def main():
    start = time.time()
    TIMETRAIN()
    TIMETEST('testDatacount1.csv')
    NAIVEB()
    print("\nThe execution time is:",time.time()-start)
    
    
if __name__ == "__main__":
    main()
