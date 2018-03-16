# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 20:07:56 2018

@author: wojzw
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import math

fn="peak2.txt"



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

#y=conv(x,square(500))
#plt.plot(y)

y=conv(x,unit_exp(1000,500))
plt.plot(y)

def find_mean(array):
    s=0
    for x in np.nditer(array):
        s += x
    return s/np.size(array)

def step_exp(x,y,x0,y0,amp,sig):
    return amp*math.exp(-(x-x0))



array_mean=0

pht_size=5
th=1.5
def photon_finder(array):
    mean=find_mean(array)
    sh=np.shape(array)
    chkd=np.full(sh,False,dtype=bool)
    photons_tables=[]
    for (x,y),value in np.ndenumerate(array):
        if(chkd[x][y]):
            continue
        if(value<th*mean):
            chkd[x][y]=True
            continue
        t=array[max(0,x-pht_size):min(sh[0],x+pht_size),max(0,y-pht_size):min(sh[0],y+pht_size)]
        xm,ym=np.unravel_index(t.argmax(), t.shape)
        xm+=max(0,x-pht_size)
        ym+=max(0,y-pht_size)
        chkd[max(0,xm-pht_size):min(sh[0],xm+pht_size),max(0,ym-pht_size):min(sh[0],ym+pht_size)]=True
        t=array[max(0,xm-pht_size):min(sh[0],xm+pht_size),max(0,ym-pht_size):min(sh[0],ym+pht_size)]
        xp=max(0,xm-pht_size)
        yp=max(0,ym-pht_size)
        photons_tables.append(((xp,yp),t))
    return photons_tables


pht_width=2
pht_amp=array_mean
def fit_gauss(t,initial_guess=(array_mean,pht_size,pht_size,pht_width,pht_width,0,array_mean)):
    side_x = np.arange(0,t.shape[0])
    side_y = np.arange(0,t.shape[0])
    X1, X2 = np.meshgrid(side_x, side_y)
    size = X1.shape
    x1_1d = X1.reshape((1, np.prod(size)))
    x2_1d = X2.reshape((1, np.prod(size)))
    xdata = np.vstack((x1_1d, x2_1d))
    
    t_fit=t.reshape((np.prod(t.shape)))
    
    return opt.curve_fit(twoD_Gaussian, xdata, t_fit, p0=initial_guess)
    
#fit=fit_gauss(t[0][1])
def photon_fitter(t_photons):
    photons=[]
    for (xp,yp),ph_table in t_photons:
        fit=fit_gauss(ph_table)
        photons.append((fit[0][0],fit[0][1]+xp,fit[0][2]+yp))
    return photons