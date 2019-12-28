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



def WOA(objf,lb,ub,dim,SearchAgents_no,Max_iter,trainInput,trainOutput):


    #dim=30
    #SearchAgents_no=50
    #lb=-100
    #ub=100
    #Max_iter=500
        
    
    # initialize position vector and score for the leader
    Leader_pos=numpy.zeros(dim)
    Leader_score=float("inf")  #change this to -inf for maximization problems
    
    
    #Initialize the positions of search agents
   # Positions=numpy.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb #generating continuous individuals
    Positions=numpy.random.randint(2, size=(SearchAgents_no,dim))#generating binary individuals
    #Initialize convergence
    convergence_curve1=numpy.zeros(Max_iter)
    convergence_curve2=numpy.zeros(Max_iter)

    
    ############################
    s=solution()

    print("WOA is optimizing  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################
    
    t=0  # Loop counter
    
    # Main loop
    while t<Max_iter:
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            
            #Positions[i,:]=checkBounds(Positions[i,:],lb,ub)          
           # Positions[i,:]=numpy.clip(Positions[i,:], lb, ub)
            
            
             # the following statement insures that at least one feature is selected
             #(i.e the randomly generated individual has at least one value 1)
            while numpy.sum(Positions[i,:])==0:   
                 Positions[i,:]=numpy.random.randint(2, size=(1,dim))
            
            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:],trainInput,trainOutput,dim);
            
            # Update the leader
            if fitness<Leader_score: # Change this to > for maximization problem
                Leader_score=fitness; # Update alpha
                Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position
            
            
            featurecount=0
            for f in range(0,dim):
              if Leader_pos[f]==1:
                 featurecount=featurecount+1
            
            
            convergence_curve1[t]=Leader_score
            convergence_curve2[t]=featurecount
            if (t%1==0):
               print(['At iteration '+ str(t)+ ' the best fitness on trainig is: '+ str(Leader_score)+'the best number of features: '+str(featurecount)]);
        
        
        
                
        a=2-t*((2)/Max_iter); # a decreases linearly fron 2 to 0 in Eq. (2.3)
        
        # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a2=-1+t*((-1)/Max_iter);
        
        # Update the Position of search agents 
        for i in range(0,SearchAgents_no):
            r1=random.random() # r1 is a random number in [0,1]
            r2=random.random() # r2 is a random number in [0,1]
            
            A=2*a*r1-a  # Eq. (2.3) in the paper
            C=2*r2      # Eq. (2.4) in the paper
            
            
            b=1;               #  parameters in Eq. (2.5)
            l=(a2-1)*random.random()+1   #  parameters in Eq. (2.5)
            
            p = random.random()        # p in Eq. (2.6)
            
            for j in range(0,dim):
                
                if p<0.5:
                    if abs(A)>=1:
                        rand_leader_index = math.floor(SearchAgents_no*random.random());
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand=abs(C*X_rand[j]-Positions[i,j]) 
                        Positions[i,j]=X_rand[j]-A*D_X_rand   #update statement
                        Positions[i,j]= transfer_functions_benchmark.v1(Positions[i,j])
                        
                    elif abs(A)<1:
                        D_Leader=abs(C*Leader_pos[j]-Positions[i,j]) 
                        Positions[i,j]=Leader_pos[j]-A*D_Leader    #update statement  
                        
                        ss= transfer_functions_benchmark.s1(Positions[i,j])
                    
                        if (random.random()<ss): 
                            Positions[i,j]=1;
                        else:
                            Positions[i,j]=0;

                    
                elif p>=0.5:
                  
                    distance2Leader=abs(Leader_pos[j]-Positions[i,j])
                    # Eq. (2.5)
                    Positions[i,j]=distance2Leader*math.exp(b*l)*math.cos(l*2*math.pi)+Leader_pos[j]
                    Positions[i,j]= transfer_functions_benchmark.v1(Positions[i,j])
                    
                    ss= transfer_functions_benchmark.s1(Positions[i,j])
                    
                    if (random.random()<ss): 
                        Positions[i,j]=1;
                    else:
                        Positions[i,j]=0;
                    

      
        
       
        t=t+1
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=Leader_pos
    s.convergence1=convergence_curve1
    s.convergence2=convergence_curve2

    s.optimizer="WOA"
    s.objfname=objf.__name__
    
    return s


