#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


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
eps = 10**-8       
S = 0.9   


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
    return y5th,y4th


# In[4]:


def evaluate_delta(y,y5,y4):
    delta_array = abs(y5-y4)
    delta = max(delta_array)
    delta_ideal = eps*(abs(y)+0.5*abs(y5-y)+0.5*abs(y4-y)) 
    if min(delta_ideal) == 0:
        delta_ideal = eps*(abs(y)+0.5*abs(y5-y)+0.5*abs(y4-y) + 1e-15)
    if delta==0:
        return "good", delta
    nz = np.nonzero(delta_array)
    d_rat = delta_ideal[nz]/delta_array[nz]
    d = min(d_rat)
    if d>1: 
        return "good", d 
    else:
        return "bad", d


# In[5]:


def driver(x,y,der,dx):
    for i in range(10): 
        y5,y4 = Cash_Karp(x,y,der,dx)
        gb, d = evaluate_delta(y,y5,y4)
        if gb=="good":
            if d==0:
                dx_new = dx*5
            else: 
                dx_new = min(S*dx*abs(d)**0.2, 5*dx)
            break 
        else:
            if (i<9):
                dx = S*dx*abs(d)**0.25
            else: #last for loop iteration
                dx_new = S*dx*abs(d)**0.25
    x = x+dx 
    return x,y5,dx_new  


# In[6]:

def destination_x_dx(der, y_in, N, dN, xi, xf, dx_init):
    dx = dx_init
    if len(y_in)!=len(der(xi,y_in)):
        print("ERROR: Number of y initial values given is not the same as the number of derivatives given.")
    y5th = np.copy(y_in)
    x_values = np.zeros(N+1)
    y = []
    for i in range (len(y_in)):
        y.append(np.zeros(len(x_values)))
    x_values[0] = xi 
    for m in range (len(y_in)):
        y[m][0] = y5th[m] 
        
    for i in range(1,len(x_values)):
        for j in range(dN):
            xi,y5th,dx = driver(xi,y5th,der,dx)
        x_values[i] = xi
        for m in range (len(y_in)):
            y[m][i] = y5th[m]
        if xi>xf:
            break
        if i==N:
            return x_values, y, dx
    
    result = []
    for k in range (len(y_in)):
        result.append(y[k][:i+1])
    return x_values[:(i+1)], result, dx
