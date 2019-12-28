# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""


import random
import numpy
import math
from solution import solution
import time
import transfer_functions_benchmark
import fitnessFUNs

    

def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter,trainInput,trainOutput):
    
    #Max_iter=1000
    #lb=-100
    #ub=100
    #dim=30  
    #SearchAgents_no=5
    
    # initialize alpha, beta, and delta_pos
    Alpha_pos=numpy.zeros(dim)
    Alpha_score=float("inf")
    
    Beta_pos=numpy.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=numpy.zeros(dim)
    Delta_score=float("inf")
    
    #initialization stage of positions of the search agents(either continuous or discrete (binary) individual generation)
   # Positions=numpy.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb #generating continuous individuals
   
    Positions=numpy.random.randint(2, size=(SearchAgents_no,dim)) #generating binary individuals
    
    Convergence_curve1=numpy.zeros(Max_iter)
    Convergence_curve2=numpy.zeros(Max_iter)

    s=solution()

     # Loop counter
    print("GWO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            Positions[i,:]=numpy.clip(Positions[i,:], lb, ub)
            
            # the following statement insures that at least one feature is selected
            #(i.e the randomly generated individual has at least one value 1)       
            while numpy.sum(Positions[i,:])==0:   
                 Positions[i,:]=numpy.random.randint(2, size=(1,dim))

            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:],trainInput,trainOutput,dim)
            
            # Update Alpha, Beta, and Delta
            if fitness<Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness<Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score): 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
            
        
        
        
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
               # X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                temp=transfer_functions_benchmark.s1(A1*D_alpha)
                if temp<numpy.random.uniform(0,1):
                    temp=0
                else:
                    temp=1
                if (Alpha_pos[j]+temp)>=1:
                    X1=Alpha_pos[j]+temp
                
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
              #  X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2 
                temp=transfer_functions_benchmark.s1(A2*D_beta)
                
                if temp<numpy.random.uniform(0,1):
                    temp=0
                else:
                    temp=1
                    
                if (Beta_pos[j]+temp)>=1:
                    X2=Beta_pos[j]+temp
                
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
               # X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3   
                
                temp=transfer_functions_benchmark.s1(A3*D_delta)
                if temp<numpy.random.uniform(0,1):
                    temp=0
                else:
                    temp=1
                    
                if (Delta_pos[j]+temp)>=1:
                    X3=Delta_pos[j]+temp
                
            Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
            
            
        featurecount=0
        for f in range(0,dim):
            if Alpha_pos[f]==1:
                featurecount=featurecount+1    
            
                           
            
            
            
            
        
        Convergence_curve1[l]=Alpha_score;
        Convergence_curve2[l]=featurecount;
        if (l%1==0):
                print(['At iteration'+ str(l+1)+' the best fitness on trainig is:'+ str(Alpha_score)+', the best number of features: '+str(featurecount)]);
    
    
    
        
    
    
    
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=Alpha_pos
    s.convergence1=Convergence_curve1
    s.convergence2=Convergence_curve2

    s.optimizer="GWO"
    s.objfname=objf.__name__
    
    
    
    
    return s
    

