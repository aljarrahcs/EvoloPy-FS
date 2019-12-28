# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""

#% ======================================================== % 
#% Files of the Matlab programs included in the book:       %
#% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
#% Second Edition, Luniver Press, (2010).   www.luniver.com %
#% ======================================================== %    
#
#% -------------------------------------------------------- %
#% Firefly Algorithm for constrained optimization using     %
#% for the design of a spring (benchmark)                   % 
#% by Xin-She Yang (Cambridge University) Copyright @2009   %
#% -------------------------------------------------------- %

import numpy
import math
import time
from solution import solution
import transfer_functions_benchmark
import fitnessFUNs
import random


def alpha_new(alpha,NGen):
    #% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
    #% alpha_0=0.9
    delta=1-(10**(-4)/0.9)**(1/NGen);
    alpha=(1-delta)*alpha
    return alpha



def FFA(objf,lb,ub,dim,n,MaxGeneration,trainInput,trainOutput):

    #General parameters

    #n=50 #number of fireflies
    #dim=30 #dim  
    #lb=-50
    #ub=50
    #MaxGeneration=500
 
    #FFA parameters
    alpha=0.5  # Randomness 0--1 (highly random)
    betamin=0.20  # minimum value of beta
    gamma=1   # Absorption coefficient
    
    
    
    zn=numpy.ones(n)
    zn.fill(float("inf")) 
    
    
    #ns(i,:)=Lb+(Ub-Lb).*rand(1,d);
   # ns=numpy.random.uniform(0,1,(n,dim)) *(ub-lb)+lb #generating continuous individuals
    
    ns=numpy.random.randint(2, size=(n,dim))          #generating binary individuals
    
    
        
    Lightn=numpy.ones(n)
    Lightn.fill(float("inf")) 
    
    #[ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)
    
    convergence1=[]
    convergence2=[]

    s=solution()

     
    print("FFA is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main loop
    for k in range (0,MaxGeneration):     # start iterations
    
        #% This line of reducing alpha is optional
        alpha=alpha_new(alpha,MaxGeneration);
        
        #% Evaluate new solutions (for all n fireflies)
        for i in range(0,n):
     # the following statement insures that at least one feature is selected
    #(i.e the randomly generated individual has at least one value 1)  
             while numpy.sum(ns[i,:])==0:   
                ns[i,:]=numpy.random.randint(2, size=(1,dim))
            
             zn[i]=objf(ns[i,:],trainInput,trainOutput,dim);
             Lightn[i]=zn[i]
        
        
                
        
        # Ranking fireflies by their light intensity/objectives
    
        
        Lightn=numpy.sort(zn)
        Index=numpy.argsort(zn)
        ns=ns[Index,:]
        
        
        #Find the current best
        nso=ns
        Lighto=Lightn
        nbest=ns[0,:] 
        Lightbest=Lightn[0]
        
        #% For output only
        fbest=Lightbest;
        
        
        BestQuality=fbest

        featurecount=0
        for f in range(0,dim):
            if nbest[f]==1:
                featurecount=featurecount+1
        
        convergence1.append(BestQuality)
        convergence2.append(featurecount)
        	
        
        if (k%1==0):
               print(['At iteration '+ str(k)+ ' the best fitness on trainig is '+ str(BestQuality)+', the best number of features: '+str(featurecount)]);
        
        
        
        
            
      
        
        
          
                
        
        #% Move all fireflies to the better locations
    #    [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,...
    #          Lightbest,alpha,betamin,gamma,Lb,Ub);
        scale=numpy.ones(dim)*abs(ub-lb)
        for i in range (0,n):
            # The attractiveness parameter beta=exp(-gamma*r)
            for j in range(0,n):
                r=numpy.sqrt(numpy.sum((ns[i,:]-ns[j,:])**2));
                #r=1
                # Update moves
                if Lightn[i]>Lighto[j]: # Brighter and more attractive
                   beta0=1
                   beta=(beta0-betamin)*math.exp(-gamma*r**2)+betamin
                   tmpf=alpha*(numpy.random.rand(dim)-0.5)*scale
                   ns[i,:]=ns[i,:]*(1-beta)+nso[j,:]*beta+tmpf #update statement
                   for j in range (0,dim):
                       ss= transfer_functions_benchmark.s1(ns[i,j])
                   
                     
                       if (random.random()<ss): 
                         ns[i,j]=1;
                       else:
                         ns[i,j]=0;
        
        #ns=numpy.clip(ns, lb, ub)
        
       
    #    
       ####################### End main loop
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=nbest
    s.convergence1=convergence1
    s.convergence2=convergence2

    s.optimizer="FFA"
    s.objfname=objf.__name__
    
    return s
    
    
    
    
    
