# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 20:07:56 2018

@author: wojzw
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

fn="seq.txt"



start_index=0
end_index=1000000+start_index

def read_asc_1d_npArray(filename):
    with open(filename) as file:
        file.seek(start_index)
        data = file.read(end_index - start_index).replace(",",".")
        parser=re.compile(r'-?\d+\.\d+')
        array1d=[float(x) for x in parser.findall(data)]
        array1d=np.array(array1d, dtype=np.float)
    return array1d
x=read_asc_1d_npArray(fn)


def unit_exp(length, period):
    func= np.zeros(length,dtype=float)
    for i in range(length):
        func[i]=(1/period)*math.exp(-i/period)
    return func


def step_exp(x,x0,offset,amp,t0):

    res = amp*np.exp(-(x-x0)/t0)+offset
    res[:x0] = offset                 
    return res

def fit_decay(data):
    l = len(data)
    x = np.linspace(0, l-1, l)
    t0 = 500
    c = data.min()
    a = data.max()-data.min()
    x0 = np.where(data==data.max())[0][0]
    popt, pcov = curve_fit(step_exp, x, data, p0=(x0,c,a,t0))
    
    return popt


def square(length):
    func= np.zeros(length,dtype=float)
    for i in range(length):
        func[i]=(1/length)
    return func

def conv(T1,T2):
    conv= np.zeros(len(T1),dtype=float)
    for i in range(len(T1)):
        for j in range(len(T2)):
            if(i+j<len(T1)):
                conv[i]+=T2[j]*T1[i+j]
    return conv

def find_mean(array):
    s=0
    for x in np.nditer(array):
        s += x
    return s/np.size(array)

def cor_square(T,length):
    i=0
    Tlen=len(T)
    Clen=Tlen-length+1
    C=np.zeros(Clen,dtype=float)
    while i<length:
        C[0]=C[0]+T[i]
        i += 1
    i=1
    while i<Clen:
        C[i]=C[i-1]-T[i-1]+T[i+length-1]
        i+=1
    return C/length

def cor_exp(T,dec_time,length):
    Tlen=len(T)
    Clen=Tlen-length+1
    C=np.zeros(Clen,dtype=float)
    q=np.exp(-1/dec_time)
    i=0
    while i<length:
        C[0]=C[0]+T[i]*q**(i)
        i += 1
    i=1
    ql=q**(length-1)
    while i<Clen:
        C[i]=(C[i-1]-T[i-1])/q+T[i+length-1]*ql
        i+=1
    return C/(dec_time*(1-np.exp(-length/dec_time)))

#y=conv(x,square(500))
#plt.plot(y)

mean=np.average(x[0:200])
x2=mean-x
#y1=conv(x2,unit_exp(2000,500))
y2=cor_square(x2,1000)
y3=cor_exp(x2,550,1500)
plt.figure(figsize=(30, 20))
#plt.plot(x2)
#plt.plot(y1)
plt.plot(y2)
plt.plot(y3)

th=1000
expT=500
def region_finder(T):
    CC=cor_square(x2,2*expT)
    region_tables=[]
    i=0
    while i < len(CC):
        if CC[i]>th:
            mi=np.argmax(T[i:i+3*expT])
            reg=T[i+mi-expT:i+mi+4*expT]
            region_tables.append((i+mi,reg))
            i=i+mi+4*expT
        else:
            i=i+1
    return region_tables
            
reg=region_finder(x2)
