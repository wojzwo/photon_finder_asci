import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import seaborn as sns
import scipy.optimize as opt
from timeit import default_timer as timer

fn="ixon0015.asc"


    
def read_asc_2d_npArray(filename):
    with open(filename) as file:
        array2d = [[float(digit) for digit in line.split()] for line in file]
        array2d=np.array(array2d, dtype=np.int32)
        sh=np.shape(array2d)
        array2d=array2d[::,1:sh[1]]
    return array2d
    

def find_mean(array):
    s=0
    for x in np.nditer(array):
        s += x
    return s/np.size(array)

    
array=read_asc_2d_npArray(fn)
array_mean=find_mean(array)

start = timer() 
asize=100
a=np.full((asize,asize), 1, dtype=np.float64)

#def dist(x1,x2,y1,y2):
#    return np.sqrt((x1-x2)**2+(y1-y2)**2)
#
#def gauss(x,y,x0,y0,amp,sig):
#    return amp*math.exp(-dist(x,x0,y,y0)**2/(2*sig**2))
#
#def add_gauss(array,x0,y0,amp,sig):
#    for (x,y),value in np.ndenumerate(array):
#        array[x][y] += gauss(x,y,x0,y0,sig,amp)
#    return array
        
#sns.heatmap(array)

def twoD_Gaussian(r, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x=r[0]
    y=r[1]
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


pht_size=5
th=10
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



t=photon_finder(array)
#plt.figure(figsize=(20, 20))
#sns.heatmap(array,cmap="inferno")
end = timer()


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

photons=photon_fitter(t)




for photon in photons:
    print(photon)
print(end - start)