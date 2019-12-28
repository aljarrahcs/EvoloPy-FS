# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""


import math
import numpy
import random
import time
from solution import solution
from sklearn.preprocessing import Binarizer
import transfer_functions_benchmark
import fitnessFUNs

def BAT(objf,lb,ub,dim,N,Max_iteration,trainInput,trainOutput):
    
    n=N;      # Population size
    #lb=-50
    #ub=50
    N_gen=Max_iteration  # Number of generations
    
    A=0.5;      # Loudness  (constant or decreasing)
    r=0.5;      # Pulse rate (constant or decreasing)
    
    Qmin=0         # Frequency minimum
    Qmax=2         # Frequency maximum
    
    
    d=dim           # Number of dimensions 
    
    # Initializing arrays
    Q=numpy.zeros(n)  # Frequency
    v=numpy.zeros((n,d))  # Velocities
    Convergence_curve1=[];
    Convergence_curve2=[];

    # Initialize the population/solutions
    
   # Sol=numpy.random.rand(n,d)*(ub-lb)+lb      #generating continuous individuals
    
    Sol=numpy.random.randint(2, size=(n,d))     #generating binary individuals
     # the following statement insures that at least one feature is selected
        #(i.e the randomly generated individual has at least one value 1)
    
    for i in range(0,n):
      while numpy.sum(Sol[i,:])==0: 
          Sol[i,:]=numpy.random.randint(2, size=(1,d))
          
     
       
    S=numpy.zeros((n,d))
    S=numpy.copy(Sol)
    Fitness=numpy.zeros(n)
    
    # initialize solution for the final results   
    s=solution()
    print("BAT is optimizing  \""+objf.__name__+"\"")    
    
    # Initialize timer for the experiment
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    #Evaluate initial random solutions
    for i in range(0,n):    
      Fitness[i]=objf(S[i,:],trainInput,trainOutput,dim)
    
    # Find the initial best solution
    fmin = min(Fitness)
    I=numpy.argmin(Fitness)
    best=Sol[I,:]
       
    # Main loop
    for t in range (0,N_gen): 
        
        # Loop over all bats(solutions)
        for i in range (0,n):
            
         # for i in range(0,n):
         # while numpy.sum(S[i,:])==0: 
          # S[i,:]=numpy.random.randint(2, size=(1,d))   
          Q[i]=Qmin+(Qmin-Qmax)*random.random()
          v[i,:]=v[i,:]+(Sol[i,:]-best)*Q[i]
          S[i,:]=Sol[i,:]+v[i,:]


          
        # Check boundaries
        #  Sol=numpy.clip(Sol,lb,ub)
          
          
    
          # Pulse rate
          if random.random()>r:
              S[i,:]=best+0.001*numpy.random.randn(d) #update statement
       
         
          
          
          for f in range(0,dim):
              ss= transfer_functions_benchmark.s1(S[i,f])#transfer function
              if (random.random()<ss): 
                   S[i,f]=1;
              else:
                   S[i,f]=0;
          
             
        
          for i in range(0,n):
             while numpy.sum(S[i,:])==0: 
                S[i,:]=numpy.random.randint(2, size=(1,d))
        
        
          # Evaluate new solutions
          Fnew=objf(S[i,:],trainInput,trainOutput,dim)
          
          # Update if the solution improves
          if ((Fnew<=Fitness[i]) and (random.random()<A) ):
                Sol[i,:]=numpy.copy(S[i,:])

                Fitness[i]=Fnew;
           
    
          # Update the current best solution
          if Fnew<=fmin:
                best=S[i,:]
                fmin=Fnew
                
          featurecount=0
          for f in range(0,dim):
              if best[f]==1:
                featurecount=featurecount+1   
                
                              
        #update convergence curve
        Convergence_curve1.append(fmin)  
        Convergence_curve2.append(featurecount)  


        if (t%1==0):
          print(['At iteration'+ str(t+1)+' the best fitness on trainig is:'+ str(fmin)+', the best number of features: '+str(featurecount)]);
          
     
    
    
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=best
    s.convergence1=Convergence_curve1
    s.convergence2=Convergence_curve2

    s.optimizer="BAT"
    s.objfname=objf.__name__
    
    
    
    return s
