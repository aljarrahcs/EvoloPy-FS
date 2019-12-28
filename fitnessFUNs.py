# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer
from sklearn import metrics
          
#____________________________________________________________________________________       
def FN1(I,trainInput,trainOutput,dim):            
         data_train_internal, data_test_internal, target_train_internal, target_test_internal = train_test_split(trainInput, trainOutput, test_size=0.34, random_state=1)
         reducedfeatures=[]
         for index in range(0,dim):
           if (I[index]==1):
               reducedfeatures.append(index)

         reduced_data_train_internal=data_train_internal[:,reducedfeatures]
         reduced_data_test_internal=data_test_internal[:,reducedfeatures]
                  
         knn = KNeighborsClassifier(n_neighbors=5)
         knn.fit(reduced_data_train_internal, target_train_internal)
         target_pred_internal = knn.predict(reduced_data_test_internal)
         acc_train = float(accuracy_score(target_test_internal, target_pred_internal))
       
         fitness=0.99*(1-acc_train)+0.01*sum(I)/(dim)

         return fitness
#_____________________________________________________________________       
def getFunctionDetails(a):
    
    # [name, lb, ub, dim]
    param = {  0:["FN1",-1,1]

            }
    return param.get(a, "nothing")



