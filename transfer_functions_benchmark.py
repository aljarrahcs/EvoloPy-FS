"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""


#import skfuzzy
import numpy as np
from math import pi
from scipy.special import erf
import matplotlib.pylab as plt

#________________________V-shaped transfer functions______________________
def v1(x):
   v1=abs(erf((np.sqrt(pi)/2)*x))
   return v1

  
def v2(x):
   v2=abs(np.tanh(x))
   return v2
   
   
def v3(x):
   v3= abs(x/np.sqrt(1+np.square(x)))
   return v3  
   
   
def v4(x):
   v4= abs((2/pi)*np.arctan((pi/2)*x))
   return v4  
##______________________S-shaped transfer functions_______________________

def s1(x):
    
    s1=1 / (1 + np.exp(-2*x))
    
    return s1

def s2(x):
    s2 = 1 / (1 + np.exp(-x))  
    return s2
# s2 is called logistic function and can be imported using scipy.special.expit(x) library

def s3(x):
    s3=1 / (1 + np.exp(-x/3))
    return s3


def s4(x):
    s4=1 / (1 + np.exp(-x/2))
    return s4


##________________________the sigmoid functions_________________________

# A customized function for SIGMOID 

def sigmf1(x,b,c):
    b=10
    c=.5
    y = 1 / (1. + np.exp(- c * (x - b)))
   
    return y



## Built-in function for SIGMOID using skfuzzy.membership library

def sigmf2(x,b,c):
    b=10
    c=.5
    y=skfuzzy.membership.sigmf(x,b,c)

    return y





x = np.arange(-8, 8, 0.1) 
# x is used inside this script and will be replaced by a binary individual (1-d binary vector)
#when the transfer function is called or imported in the optimizers scripts


#__________________Calling for transfer functions inside this script____________________

#T1=v1(x)
#T2=v2(x)
#T3=v3(x)
#T4=v4(x)
#
#T5=s1(x)
#T6=s2(x)
#T7=s3(x)
#T8=s4(x)
#
#T9=sigmf1(x,.5,10)
#T10=sigmf2(x,.5,10)
#
##_______________Plotting of transfer functions inside this script________________________
#plt.figure(1)
#plt.subplot(211)
#
#plt.plot(x, T1) 
#plt.plot(x, T2) 
#plt.plot(x, T3) 
#plt.plot(x, T4)
#plt.xlabel('x')
#plt.ylabel('f(x)')
##plt.show()
##
##plt.figure(2)
##
#plt.subplot(212)
#plt.plot(x, T5) 
#plt.plot(x, T6) 
#plt.plot(x, T7) 
#plt.plot(x, T8) 
#
#
#plt.xlabel('x')
#plt.ylabel('f(x)')
##plt.show()
##
##plt.figure(3)
##plt.subplot(222)
##plt.plot(x, T9) 
#
##plt.xlabel('x')
##plt.ylabel('f(x)')
##plt.show()
##
###plt.subplot(212)
##plt.plot(x, T10) 
##
##
##plt.xlabel('x')
##plt.ylabel('f(x)')
#plt.show()
##
##
#
#
#
#
#



























  
  
    
    
