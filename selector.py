# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""


import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import BAT as bat
import WOA as woa
import FFA as ffa
import csv
import numpy
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import fitnessFUNs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



def selector(algo,func_details,popSize,Iter,completeData):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
   
    
    DatasetSplitRatio=0.34   #Training 66%, Testing 34%
    
    DataFile="datasets/"+completeData
      
    data_set=numpy.loadtxt(open(DataFile,"rb"),delimiter=",",skiprows=0)
    numRowsData=numpy.shape(data_set)[0]    # number of instances in the  dataset
    numFeaturesData=numpy.shape(data_set)[1]-1 #number of features in the  dataset

    dataInput=data_set[0:numRowsData,0:-1]
    dataTarget=data_set[0:numRowsData,-1]  
    trainInput, testInput, trainOutput, testOutput = train_test_split(dataInput, dataTarget, test_size=DatasetSplitRatio, random_state=1) 
#
   
#    numRowsTrain=numpy.shape(trainInput)[0]    # number of instances in the train dataset
#    numFeaturesTrain=numpy.shape(trainInput)[1]-1 #number of features in the train dataset
#
#    numRowsTest=numpy.shape(testInput)[0]    # number of instances in the test dataset
#    numFeaturesTest=numpy.shape(testInput)[1]-1 #number of features in the test dataset
# 

    dim=numFeaturesData
    
    if(algo==0):
        x=pso.PSO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput)
    if(algo==1):
        x=mvo.MVO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput)
    if(algo==2):
        x=gwo.GWO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput)
    if(algo==3):
        x=mfo.MFO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput)
    if(algo==4):
        x=woa.WOA(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput)
    if(algo==5):
        x=ffa.FFA(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput)
    if(algo==6):
        x=bat.BAT(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput)

    # Evaluate MLP classification model based on the training set
#    trainClassification_results=evalNet.evaluateNetClassifier(x,trainInput,trainOutput,net)
 #   x.trainAcc=trainClassification_results[0]
  #  x.trainTP=trainClassification_results[1]
   # x.trainFN=trainClassification_results[2]
    #x.trainFP=trainClassification_results[3]
    #x.trainTN=trainClassification_results[4]
   
    # Evaluate MLP classification model based on the testing set   
    #testClassification_results=evalNet.evaluateNetClassifier(x,testInput,testOutput,net)
            
    reducedfeatures=[]
    for index in range(0,dim):
        if (x.bestIndividual[index]==1):
            reducedfeatures.append(index)
    reduced_data_train_global=trainInput[:,reducedfeatures]
    reduced_data_test_global=testInput[:,reducedfeatures]
               
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(reduced_data_train_global,trainOutput)
   

         # Compute the accuracy of the prediction
         
    target_pred_train = knn.predict(reduced_data_train_global)
    acc_train = float(accuracy_score(trainOutput, target_pred_train))
    x.trainAcc=acc_train
    
    target_pred_test = knn.predict(reduced_data_test_global)
    acc_test = float(accuracy_score(testOutput, target_pred_test))
    x.testAcc=acc_test
    
        #print('Test set accuracy: %.2f %%' % (acc * 100))

    #x.testTP=testClassification_results[1]
    #x.testFN=testClassification_results[2]
    #x.testFP=testClassification_results[3]
    #x.testTN=testClassification_results[4] 
    
    
    return x
    
#####################################################################    
