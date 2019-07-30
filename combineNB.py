# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 23:40:40 2018

@author: Devashree
"""


import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
from collections import OrderedDict
import os.path
import time





blacklist=[]


def PreprocessTrain():
    dataset = pd.read_csv('TrainDATA.csv')
    #P = dataset.iloc[:,[2]]
    #P2=dataset.iloc[:,[3]]
    
    df2 = pd.DataFrame(OrderedDict({
        'sourceIP': dataset.iloc[:,2],
        
        'destIP': dataset.iloc[:,3]
    
    	}))
    
    g1= df2.groupby(['sourceIP', 'destIP']).size().reset_index().rename(columns={0:'count'})
    
#    print(g1)
    
    #g1.add_suffix('_Count').reset_index()
    df2=g1
    c1=df2['count'].mean()
    print('mean:',c1)
    #df2.to_csv('Kmeans.csv')
    
    attack=[]
    for row in df2['count']:
        if (row > 85):
            attack.append('1')
        else:
            attack.append('0')
    
            
    df2['Attack']=attack
#    print(df)
    df2.to_csv('TRAIN.csv')
    
def CheckBlacklist(filename):
    if os.path.exists("E:\Dev@shree\Python\Project code\Final\FULLFINAL\LAST\BLACKLIST.csv"):
        print("Checking Blacklist!!!")
        
        BL = pd.read_csv('BLACKLIST.csv')
        dfc=pd.DataFrame({
        'Block':BL.iloc[:,0]
            })
        
 
        data = pd.read_csv(filename)
        df = pd.DataFrame(OrderedDict({
#                'N0':data.iloc[:,0],
                'Time': data.iloc[:,1],
                'source': data.iloc[:,2],
                'Destination' : data.iloc[:,3],
                'Protocol': data.iloc[:,4],
                'Length': data.iloc[:,5],
                'Info': data.iloc[:,6]
                }))
#        print(df)
     
        for i in dfc.iterrows():
            f=i[1]
            blk = f['Block']	
        
            df= df.loc[~df['source'].isin([blk])]
        print(df)
    
        df.to_csv('data1.csv')
    else: 
        print('File not there!!')


    
def PreprocessTest(file):
    dataset = pd.read_csv(file)
    print(dataset)
    #P = dataset.iloc[:,[2]]
    #P2=dataset.iloc[:,[3]]
    
    df1 = pd.DataFrame(({
        'sourceIP': dataset.iloc[:,2],
        
        'destIP': dataset.iloc[:,3]
    
    	
    }))
    print(df1)
    
    g1= df1.groupby(['sourceIP', 'destIP']).size().reset_index().rename(columns={0:'count'})
    
    print(g1)
#    
#    g1.add_suffix('_Count').reset_index()
    df1=g1
    df1.to_csv('KmeansNE12.csv')
##    c1=df1['count'].mean()
##    print('mean:',c1)
#    
#    
    print(df1)
    attack=[]
    for row in df1['count']:
        if (row > 85):
            attack.append('1')
        else:
            attack.append('0')
    
            
    df1['Attack']=attack
    print(df1)
    df1.to_csv('TEST.csv')
    
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
    print('Specificity of the system is:',spec) #rate of incorrectly identified attack packets
    
    
#    test_data.to_csv('ResultNB.csv')
    
def WriteBLACKLIST(blacklist):
#   

#    Assuming res is a flat list
    with open('BLACKLIST.csv', 'a') as output:
        writer = csv.writer(output, lineterminator='\n')
#        writer.write('Blacklist') 

        for val in blacklist:
            writer.writerow([val]) 

    
def UpdateBlacklist(test):
    src=[]
    
    
    for i in test.iterrows():
        f=i[1]
#        print(f)
        if(f['Prediction']==1 and f['sourceIP']!='192.168.1.102'):
            src=f['sourceIP']
#            print(src)
            blacklist.append(src)
#            print(blacklist)
            
    WriteBLACKLIST(blacklist)
            
           

def NAIVEB():
    test = pd.read_csv('TEST.csv')
    print(test)
#    for row in data['Time']:
#        row = row*100000
#        print(row)
    train=pd.read_csv('TRAIN.csv')
    
#    train, test = cross_validation.train_test_split(data,test_size=0.50)
    NaiveBayes = GaussianNB()
    # Use all columns apart from the Attack column as feautures
    train_features = train.iloc[:,[3]]
  
    # Use the Attack column as the label
    train_label = train.iloc[:,[4]]
    
    test_features = test.iloc[:,[3]]
   
    test_label = test.iloc[:,4]
    
    # Train the naive bayes model
    NaiveBayes.fit(train_features, train_label)

    # build a dataframe to show the expected vs predicted values
    test_data = pd.concat([test_features,test_label], axis=1)
    test_data["prediction"] = NaiveBayes.predict(test_features)
    
    print(test_data)
    test['Prediction']=test_data['prediction']
    test.to_csv('ResNB.csv')
    ACCSCORE(test_data)
#    print('\nThe blaclisted IPs are:\n')
    UpdateBlacklist(test)
    
    
def main():
    start = time.time()
    CheckBlacklist('data21.csv')
    PreprocessTrain()
    PreprocessTest('data21.csv')
    NAIVEB()
    print("\nThe execution time is:",time.time()-start)
    
    
if __name__ == "__main__":
    main()
    
    