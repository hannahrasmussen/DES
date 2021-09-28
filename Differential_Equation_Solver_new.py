#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


# In[2]:


a2 = 1/5
a3 = 3/10
a4 = 3/5
a5 = 1
a6 = 7/8
b21 = 1/5
b31 = 3/40
b32 = 9/40
b41 = 3/10
b42 = -9/10
b43 = 6/5
b51 = -11/54
b52 = 5/2
b53 = -70/27
b54 = 35/27
b61 = 1631/55296
b62 = 175/512
b63 = 575/13824
b64 = 44275/110592
b65 = 253/4096
c1 = 37/378
c2 = 0
c3 = 250/621
c4 = 125/594
c5 = 0
c6 = 512/1771
c10 = 2825/27648
c20 = 0
c30 = 18575/48384
c40 = 13525/55296
c50 = 277/14336
c60 = 1/4
eps = 10**-8                       #allows us to find ideal delta at each step, which is this fraction of y
S = 0.9                            #Safety value


# In[3]:


def Cash_Karp (x,y,der,dx):
    k1 = dx*der(x,y)                  
    k2 = dx*der(x+(a2*dx),y+(b21*k1))    
    k3 = dx*der(x+(a3*dx),y+(b31*k1)+(b32*k2))
    k4 = dx*der(x+(a4*dx),y+(b41*k1)+(b42*k2)+(b43*k3))
    k5 = dx*der(x+(a5*dx),y+(b51*k1)+(b52*k2)+(b53*k3)+(b54*k4))
    k6 = dx*der(x+(a6*dx),y+(b61*k1)+(b62*k2)+(b63*k3)+(b64*k4)+(b65*k5))
    y5th = y+(c1*k1)+(c2*k2)+(c3*k3)+(c4*k4)+(c5*k5)+(c6*k6) 
    y4th = y+(c10*k1)+(c20*k2)+(c30*k3)+(c40*k4)+(c50*k5)+(c60*k6)
    return y5th,y4th,dx #y5th and y4th are arrays because y is an array


# In[4]:


def evaluate_delta(y,y5,y4,der,dx):
    delta_array = abs(y5-y4)
    delta = max(delta_array)
    delta_ideal = eps*(abs(y)+0.5*abs(y5-y)+0.5*abs(y4-y)) #OK so the problem is that delta is not 0 but delta_ideal is
    if min(delta_ideal) == 0:
        delta_ideal = eps*(abs(y)+0.5*abs(y5-y)+0.5*abs(y4-y) + 1e-15)
    if delta==0:
        return "good", delta
    nz = np.nonzero(delta_array)
    d_rat = delta_ideal[nz]/delta_array[nz]
    d = min(d_rat)
    if d>1: #checks to make sure EVERY delta is less than the maximum delta
        return "good", d 
    else:
        return "bad", d


# In[5]:


def driver(x,y,der,dx):
    for i in range(10): #don't want it to go through this loop more than 10 times
        y5,y4,dx = Cash_Karp(x,y,der,dx) #dx doesn't change here because of the way Cash_Karp is written
        gb, d = evaluate_delta(y,y5,y4,der,dx) #d is delta if delta=0, but d is delta_ratio otherwise
        if gb=="good":
            if d==0:
                dx_new = dx*5
            else: 
                dx_new = min(S*dx*abs(d)**0.2, 5*dx) #will avoid taking too big of a step
            break #otherwise, the loop should keep running until either 
                  #a good delta is found or it goes through 10 times
        else: #to get here, gb must have been bad, so we just need to set a new dx for it to try again
            dx = S*dx*abs(d)**0.25
            dx_new = dx #just in case it's the 10th time, we need to have a dx_new to return
    x = x+dx #the most recent dx used in Cash_Karp as opposed to the new dx
    return x,y5,dx_new  


# In[6]:

def destination_x_dx(der, y_in, N, dN, xi, xf, dx_init): #dN is number of steps between steps saved, which is N
    dx = dx_init
    if len(y_in)!=len(der(xi,y_in)):
        print("ERROR: Number of y initial values given is not the same as the number of derivatives given.")
#    dx = .05*min(abs(y_in/der(xi,y_in)).any(),.001) #hopefully the derivative isn't 0 at first?
    y5th = np.copy(y_in)
    x_values = np.zeros(N+1) #making length of N even if we don't need to use the whole array
    y = []
    for i in range (len(y_in)):
        y.append(np.zeros(len(x_values)))
    x_values[0] = xi #sets initial value of x
    for m in range (len(y_in)):
        y[m][0] = y5th[m] #sets all the initial values of the functions; do I really need this part or can it be incorporated into my next for loop?
        
    for i in range(1,len(x_values)):
        for j in range(dN):
            xi,y5th,dx = driver(xi,y5th,der,dx) #resaves x in x_values, current y values in y5th, and dx every time
        x_values[i] = xi
#        print(i, xi, dx)
        for m in range (len(y_in)):
            y[m][i] = y5th[m]
        if xi>xf:
            break
        if i==N:
#            print("ERROR: Maximum number of steps used without reaching final x value.")
            return x_values, y, dx
    
    result = []
    for k in range (len(y_in)):
        result.append(y[k][:i+1]) #the colon must tell the code to append everything THROUGH i+1
     #is there a way to truncate all the zeros at the end of x?
    return x_values[:(i+1)], result, dx




def destination_x(der, y_in, N, dN, xi, xf): #dN is number of steps between steps saved, which is N
    dx = .01
    if len(y_in)!=len(der(xi,y_in)):
        print("ERROR: Number of y initial values given is not the same as the number of derivatives given.")
    dx = .05*min(abs(y_in/der(xi,y_in)).any(),.001) #hopefully the derivative isn't 0 at first?
    y5th = np.copy(y_in)
    x_values = np.zeros(N) #making length of N even if we don't need to use the whole array
    y = []
    for i in range (len(y_in)):
        y.append(np.zeros(len(x_values)))
    x_values[0] = xi #sets initial value of x
    for m in range (len(y_in)):
        y[m][0] = y5th[m] #sets all the initial values of the functions; do I really need this part or can it be incorporated into my next for loop?
        
    for i in range(1,len(x_values)):
        for j in range(dN):
            xi,y5th,dx = driver(xi,y5th,der,dx) #resaves x in x_values, current y values in y5th, and dx every time
        x_values[i] = xi
        print(i, xi, dx)
        for m in range (len(y_in)):
            y[m][i] = y5th[m]
        if xi>xf:
            break
        if i==N-1:
            print("ERROR: Maximum number of steps used without reaching final x value.")
            return x_values, y
    
    result = []
    for k in range (len(y_in)):
        result.append(y[k][:i+1]) #the colon must tell the code to append everything THROUGH i+1
     #is there a way to truncate all the zeros at the end of x?
    return x_values[:(i+1)], result


