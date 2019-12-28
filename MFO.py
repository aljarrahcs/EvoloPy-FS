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

  
def MFO(objf,lb,ub,dim,N,Max_iteration,trainInput,trainOutput):

    
    
    
    
    #Initialize the positions of moths
   # Moth_pos=numpy.random.uniform(0,1,(N,dim)) *(ub-lb)+lb #generating continuous individuals
    Moth_pos=numpy.random.randint(2, size=(N,dim))          #generating binary individuals
    
    Moth_fitness=numpy.full(N,float("inf"))
    #Moth_fitness=numpy.fell(float("inf"))
    
    Convergence_curve1=numpy.zeros(Max_iteration)
    Convergence_curve2=numpy.zeros(Max_iteration)

    
    sorted_population=numpy.copy(Moth_pos)
    fitness_sorted=numpy.zeros(N)
    #####################
    best_flames=numpy.copy(Moth_pos)
    best_flame_fitness=numpy.zeros(N)
    ####################
    double_population=numpy.zeros((2*N,dim))
    double_fitness=numpy.zeros(2*N)
    
    double_sorted_population=numpy.zeros((2*N,dim))
    double_fitness_sorted=numpy.zeros(2*N)
    #########################
    previous_population=numpy.zeros((N,dim));
    previous_fitness=numpy.zeros(N)


    s=solution()

    print("MFO is optimizing  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    Iteration=1;    
    
    # Main loop
    while (Iteration<Max_iteration+1):
        
        # Number of flames Eq. (3.14) in the paper
        Flame_no=round(N-Iteration*((N-1)/Max_iteration));
        
        for i in range(0,N):
            
            # Check if moths go out of the search spaceand bring it back
           # Moth_pos[i,:]=numpy.clip(Moth_pos[i,:], lb, ub) 
    
    # the following statement insures that at least one feature is selected
   #(i.e the randomly generated individual has at least one value 1)       
            while numpy.sum(Moth_pos[i,:])==0:   
                 Moth_pos[i,:]=numpy.random.randint(2, size=(1,dim))

            # evaluate moths
            Moth_fitness[i]=objf(Moth_pos[i,:],trainInput,trainOutput,dim)
            
        
           
        if Iteration==1:
            # Sort the first population of moths
            fitness_sorted=numpy.sort(Moth_fitness)
            I=numpy.argsort(Moth_fitness)
            
            sorted_population=Moth_pos[I,:]
               
            
            #Update the flames
            best_flames=sorted_population;
            best_flame_fitness=fitness_sorted;
        else:
    #        
    #        # Sort the moths
            double_population=numpy.concatenate((previous_population,best_flames),axis=0)
            double_fitness=numpy.concatenate((previous_fitness, best_flame_fitness),axis=0);
    #        
            double_fitness_sorted =numpy.sort(double_fitness);
            I2 =numpy.argsort(double_fitness);
    #        
    #        
            for newindex in range(0,2*N):
                double_sorted_population[newindex,:]=numpy.array(double_population[I2[newindex],:])           
            
            fitness_sorted=double_fitness_sorted[0:N]
            sorted_population=double_sorted_population[0:N,:]
    #        
    #        # Update the flames
            best_flames=sorted_population;
            best_flame_fitness=fitness_sorted;
    
    #    
    #   # Update the position best flame obtained so far
        Best_flame_score=fitness_sorted[0]
        Best_flame_pos=sorted_population[0,:]
    #      
        previous_population=Moth_pos;
        previous_fitness=Moth_fitness;
        
        
        
        
        
        
        
        featurecount=0
        for f in range(0,dim):
            if Best_flame_pos[f]==1:
                featurecount=featurecount+1
        
        
#        print(Best_flame_pos)
#        print(Best_flame_score)
#        
#   Convergence_curve[Iteration-1]=(Best_flame_score)
        Convergence_curve1[Iteration-1]=Best_flame_score# store the best number of features
        Convergence_curve2[Iteration-1]=featurecount#store the best fitness on testing returened from F11


      #Display best fitness along the iteration
        if (Iteration%1==0):
            print(['At iteration'+ str(Iteration+1)+' the best fitness on trainig is:'+ str(Best_flame_score)+', the best number of features: '+str(featurecount)]);
         
        
        
        # a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a=-1+Iteration*((-1)/Max_iteration);
        

        
        # Loop counter
        for i in range(0,N):
    #        
            for j in range(0,dim):
                if (i<=Flame_no): #Update the position of the moth with respect to its corresponsing flame
    #                
                    # D in Eq. (3.13)
                    distance_to_flame=abs(sorted_population[i,j]-Moth_pos[i,j])
                    b=1
                    t=(a-1)*random.random()+1;
    #                
    #                % Eq. (3.12)
                    Moth_pos[i,j]=distance_to_flame*math.exp(b*t)*math.cos(t*2*math.pi)+sorted_population[i,j]#update statement
                    ss= transfer_functions_benchmark.s1(Moth_pos[i,j])
                    
                    if (random.random()<ss): 
                       Moth_pos[i,j]=1;
                    else:
                       Moth_pos[i,j]=0;

    #            end
    #            
                if i>Flame_no: # Upaate the position of the moth with respct to one flame
    #                
    #                % Eq. (3.13)
                    distance_to_flame=abs(sorted_population[i,j]-Moth_pos[i,j]);
                    b=1;
                    t=(a-1)*random.random()+1;
    #                
    #                % Eq. (3.12)
                    Moth_pos[i,j]=distance_to_flame*math.exp(b*t)*math.cos(t*2*math.pi)+sorted_population[Flame_no,j]#update statement
                    ss= transfer_functions_benchmark.s1(Moth_pos[i,j])
                    
                    if (random.random()<ss): 
                       Moth_pos[i,j]=1;
                    else:
                       Moth_pos[i,j]=0;
        #Display best fitness along the iteration
#        if (Iteration%1==0):
#            print(['At iteration '+ str(Iteration)+ ' the best fitness is '+ str(Best_flame_score)]);
#            Convergence_curve[Iteration-1]=(Best_flame_score)

    

    
        Iteration=Iteration+1; 

    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=Best_flame_pos
    s.convergence1=Convergence_curve1
    s.convergence2=Convergence_curve2

    s.optimizer="MFO"
    s.objfname=objf.__name__
    
    
    
    return s
    




