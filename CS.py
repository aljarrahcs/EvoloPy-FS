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
import transfer_functions_benchmark
import fitnessFUNs
    
def get_cuckoos(nest,best,lb,ub,n,dim):
    
    # perform Levy flights
    tempnest=numpy.zeros((n,dim))
    tempnest=numpy.array(nest)
    beta=3/2;
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta);

    s=numpy.zeros(dim)
    for j in range (0,n):
        s=nest[j,:]
        u=numpy.random.randn(len(s))*sigma
        v=numpy.random.randn(len(s))
        step=u/abs(v)**(1/beta)
 
        stepsize=0.01*(step*(s-best))

        s=s+stepsize*numpy.random.randn(len(s))
        tempnest[j,:]=transfer_functions_benchmark.s1(s)
        for i in range (0,dim):
            ss= transfer_functions_benchmark.s1(tempnest[j,i])
            if (random.random()<ss): 
               tempnest[j,i]=1;
            else:
               tempnest[j,i]=0;
        
        
        while numpy.sum(tempnest[j,:])==0: 
         tempnest[j,:]=numpy.random.randint(2, size=(1,dim))
        #tempnest[j,:]=numpy.clip(s, lb, ub)
    return tempnest

def get_best_nest(nest,newnest,fitness,n,dim,objf,trainInput,trainOutput):
# Evaluating all new solutions
    tempnest=numpy.zeros((n,dim))
    tempnest=numpy.copy(nest)
    for j in range(0,n):
    #for j=1:size(nest,1),
        fnew=objf(newnest[j,:],trainInput,trainOutput,dim);
        
        if fnew<=fitness[j]:
           fitness[j]=fnew
           tempnest[j,:]=newnest[j,:]
        
    # Find the current best

    fmin = min(fitness)
    K=numpy.argmin(fitness)
    bestlocal=tempnest[K,:]

    return fmin,bestlocal,tempnest,fitness

# Replace some nests by constructing new solutions/nests
def empty_nests(nest,pa,n,dim):
    # Discovered or not 
    tempnest=numpy.zeros((n,dim))

    K=numpy.random.uniform(0,1,(n,dim))>pa
   # K=numpy.random.randint(2, size=(n,dim))>pa 
    stepsize=random.random()*(nest[numpy.random.permutation(n),:]-nest[numpy.random.permutation(n),:])

    
    tempnest=nest+stepsize*K
    for i in range(0,n):
        for j in range(0,dim):
          if tempnest[i,j] >=.5:
              tempnest[i,j]=1
          else:
              tempnest[i,j]=0
    
    for i in range(0,n):
      while numpy.sum(tempnest[i,:])==0: 
          tempnest[i,:]=numpy.random.randint(2, size=(1,dim))
    
   #      print(tempnest[j,:])
    
    return tempnest
##########################################################################


def CS(objf,lb,ub,dim,n,N_IterTotal,trainInput,trainOutput):


    #lb=-1
    #ub=1
    #n=50
    #N_IterTotal=1000
    #dim=30
    
    # Discovery rate of alien eggs/solutions
    pa=0.25
    
    
    nd=dim
    
    
#    Lb=[lb]*nd
#    Ub=[ub]*nd
    convergence1=[]
    convergence2=[]

    # RInitialize nests randomely
    #nest=numpy.random.rand(n,dim)*(ub-lb)+lb
    nest=numpy.random.randint(2, size=(n,dim))  

    for i in range(0,n):
      while numpy.sum(nest[i,:])==0: 
          nest[i,:]=numpy.random.randint(2, size=(1,dim))

    
    new_nest=numpy.zeros((n,dim))
    new_nest=numpy.copy(nest)
    
    bestnest=[0]*dim;
     
    fitness=numpy.zeros(n) 
    fitness.fill(float("inf"))
    

    s=solution()

     
    print("CS is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    fmin,bestnest,nest,fitness =get_best_nest(nest,new_nest,fitness,n,dim,objf,trainInput,trainOutput)
    # Main loop counter
    for iter in range (0,N_IterTotal):
        # Generate new solutions (but keep the current best)
     
         new_nest=get_cuckoos(nest,bestnest,lb,ub,n,dim)
         # Evaluate new solutions and find best
         fnew,best,nest,fitness=get_best_nest(nest,new_nest,fitness,n,dim,objf,trainInput,trainOutput)
        
         new_nest=empty_nests(new_nest,pa,n,dim) ;

       
        
        
        # Evaluate new solutions and find best
         fnew,best,nest,fitness=get_best_nest(nest,new_nest,fitness,n,dim,objf,trainInput,trainOutput)
    
         if fnew<fmin:
            fmin=fnew
            bestnest=best
            
            
         featurecount=0
         for f in range(0,dim):
            if best[f]==1:
                featurecount=featurecount+1
         convergence1.append(fmin)
         convergence2.append(featurecount)

  
            
                       
    
         if (iter%10==0):
            print(['At iteration '+ str(iter)+ ' the best fitness on trainig is '+ str(fmin)+ ',the best number of features: '+str(featurecount) ]);

    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=best
    s.convergence1=convergence1
    s.convergence2=convergence2

    s.optimizer="CS"
    s.objfname=objf.__name__
    
     
    
    return s
    
