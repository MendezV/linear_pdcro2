import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import Lattice
import StructureFactor
import Dispersion
import time
import sys
from scipy import integrate
import concurrent.futures
import functools
from datetime import datetime
import pandas as pd
import seaborn as sns
from scipy.special import spence as dilog2
import matplotlib as mpl


import csv
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

c=18.087*(1e-10) #in meters  #from sunkos thesis
a=2.93*(1e-10) #in meters  #from sunkos thesis
dbz=2*np.pi/(c/3)
Abz=8*np.pi*np.pi*np.sqrt(3)/(3*a*a) #in 1/m^2 

Vbz=Abz*dbz
fracvol=Vbz/2
spin_deg=2 #from sum over spins
n_cr=spin_deg*fracvol/((2*np.pi)**3) #m-3 #extra factor of 1/(2pi)^3 from the momentum space integral 
me=9.1093837*1e-31 #kg
m_cr=1.5*me 
e=1.6021*1e-19 #c
hbar=1.05457*1e-34 # m2 kg / s
kb=1.3806*1e-23 #m2 kg s-2 K-1

J=6.25 #in mev
U=4000
g=107
JK=4*g*g/U
mevtoK=11.6045250061598
Tval=J*mevtoK
print(Tval, 'validity above this temp in K')
print(JK, 'Kondo coupling in mev')

S=3.0/2.0
kappa=S*(S+1)
alpha=2

#electron Parameters
tp1=568 #in units of Js\
tp2=-tp1*108/568 #/tpp1
tperp=tp1/150
fill=0.5

import Dispersion

#random parameters and integration grid size
size_E=2**8+1
size_Z=2**8+1
save=True
Machine='FMAC'


JKrightUnits=JK*mevtoK*kb

def Li2(x):
    return dilog2(1-x)

def Li3(x):
    a=[]
    
    for l in x:
        t=np.linspace(1e-7,l,100)
        dt=t[1]-t[0]
        
        a.append(np.trapz(dilog2(1-t)/t)*dt)
        
    return np.array(a)

def Jc(y,z):

    eps=1j*1e-17
    exp_sum=np.exp(-z-y)
    exp_dif=np.exp(z-y)+eps
    exp_zero=np.exp(-y)
    
    exp_zero2=np.exp(z)
    exp_zero3=np.exp(y)
    exp_sum2=np.exp(z+y)

    den=1/(2*y*z)
    
    # print(Li2(exp_sum),Li2(exp_dif))
    
    Li3part=2*Li3(exp_sum)-2*Li3(exp_dif)
    Li2Dif=(2*y+z)*Li2(exp_sum)+(-2*y+z)*Li2(exp_dif)
    Li2_zero=-2*z*Li2(exp_zero)
    
    Lipart=(Li3part+Li2Dif+Li2_zero)*den
    
    polypart=z**2/(12*y) + (y+z)/2 + np.pi*np.pi/(3*y)
    
    
    Logpart=(y/(2*z))*np.log( (np.sinh((y-z)/2)/np.sinh((y+z)/2) )+eps) +np.log(exp_zero3-1+eps)-0.5*np.log((exp_zero3-exp_zero2)*(exp_sum2-1)+eps) 
    
    return np.real(Lipart+polypart+Logpart)


def Ic(y):

    exp_zero=np.exp(-y)
    exp_zero2=np.exp(y)
    
    return -4*Li2(exp_zero)/y -2*y/(exp_zero2-1) - 4*y + 2*np.pi*np.pi/(3*y) +4*np.log(exp_zero2-1)

def K_tau(T,lambda_K):
    return (2*np.pi/hbar)*lambda_K*JKrightUnits*Ic(alpha*np.pi*J*mevtoK/(T))

def E_tau(T,omega_0, lambda_0,mevtoK):
    return (2*np.pi/hbar)*lambda_0*kb*T*((0.5*omega_0*mevtoK/T)/np.sinh((0.5*omega_0*mevtoK/T)))**2

def EME_tau(T, omega0,lambda_eme):
    return (2*np.pi/hbar)*lambda_eme*kb*T*((omega0*mevtoK/T)**2/(np.cosh( omega0*mevtoK/T)- 1 ))*Jc(alpha*np.pi*J*mevtoK/T,omega0*mevtoK/T) 


def A_tau(T,omega_0_D, lambda_0_a,mevtoK):

    size_x=2**8+1
    inti=[]
    for Ts in T:
        ratio=0.5*omega_0_D*mevtoK/Ts
        x = np.linspace(1e-10, np.pi/2-1e-10, size_x)
        dx=x[1]-x[0]
        inti.append(  (1/np.pi)*dx*integrate.romb(  np.sin(x)**4 / (np.sinh(np.sin(x)*ratio)**2) )   )
    ratio=0.5*omega_0_D*mevtoK/T 
    inti_arr=np.array(inti)*(ratio**2)
    
    return (8*np.pi/hbar)*lambda_0_a*kb*T*inti_arr



def EA_tau(T,omega_0_D, lambda_0_a,omega_0, lambda_0,mevtoK):
    return A_tau(T,omega_0_D, lambda_0_a,mevtoK)+E_tau(T,omega_0, lambda_0,mevtoK)

def main() -> int:

    

    #random parameters and integration grid size
    ed=Dispersion.Dispersion_TB_single_band([tp1,tp2],fill,size_E,Machine)
    [dens2,bins,valt,rho ]=[ed.dens2,ed.bins,ed.valt,ed.f2 ]
    [nn,earr,Dos]=[ed.nn,ed.earr,ed.Dos]
    mu=ed.mu

    #dimensionless
    lambda_K=2*alpha*kappa*JK*rho(mu)
    
    poptE=[36.9553148 ,  0.05531515]  #freq in ev and dimensionless coupling
    popt=[3.99286313e+01, 2.55040778e-02]  #freq in ev and  dimensionless coupling 
    poptEA=[2.86942388e+01, 4.28936187e-02, 1.20288925e+02, 2.38349684e-02] #freq in ev and  dimensionless coupling Ac #freq in ev and  dimensionless coupling opt
    
    T1=J*mevtoK*1.1
    T2=475
    T=np.linspace(T1,T2,400)
                   
    tauinvList=[]
    tauinvList_2=[]
    tauinvList_3=[]
    
    fac=m_cr*1e+8/(e*e*n_cr)
    
    tauinv=K_tau(T,lambda_K)*fac
    tauinv_2=EME_tau(T, popt[0], popt[1])*fac
    tauinv_3=EA_tau(T,poptEA[0],poptEA[1],poptEA[2],poptEA[3],mevtoK)*fac
    
    
    
    ####plotting starts
    fig, ax = plt.subplots(1, 1, figsize=[5, 4])

    LW=1.5
    # Create inset of width 30% and height 40% of the parent axes' bounding box
    # at the upper left corner (loc=3)  Einstein
    axins1 = inset_axes(ax, width="40%", height="65%", loc=2,borderpad=0.0)
    axins1.plot(T,tauinv_3, lw=LW,  label=r'$\rho_{\rm{el-ph}}$', c='k')
    axins1.plot(T,tauinv_2, lw=LW, label=r'$\rho_{\rm{EME}}$', c='teal')
    axins1.plot(T,tauinv, lw=LW,  label=r'$\rho_{\rm{K}}$', c='darkorange')
    axins1.set_ylim([0,np.max(tauinv_3)*1.5])
    axins1.yaxis.set_label_position("right")
    axins1.yaxis.tick_right()
    axins1.tick_params(axis='x', labelsize=15 ,direction="in")
    axins1.tick_params(axis='y', labelsize=15 ,direction="in")
    axins1.legend( prop={'size':14.25},borderpad=0.2,borderaxespad=0.1,framealpha=0.7,frameon=False)
    axins1.set_ylabel(r'$\rho_{ab}$ [$\mu \Omega $cm] ', size=20, rotation=270, labelpad=25)
    axins1.set_xlabel(r'$T$ [K]', size=15)
    
    ax.plot(T,tauinv_3+tauinv+tauinv_2, lw=LW, c='b',label=r'$\rho_{\rm{K}}+\rho_{\rm{el-ph}} $'+'\n'+r'$+\rho_{\rm{EME}}$')
    ax.plot(T,tauinv_3, lw=LW,c='r', label=r'$\rho_{\rm{el-ph}}$')
    
    
    # # Turn ticklabels of insets off
    # for axi in [axins1, axins2, axins3]:
    #     axi.tick_params(left=False, labelleft=False, labelbottom=False)
    
    Tpdcro2=[]
    rhopdcro2=[]
    with open('pdcro2.csv', newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in filereader:
            Tpdcro2.append(float(row[0]))
            rhopdcro2.append(float(row[1]))

    Tpdcro2=np.array(Tpdcro2)
    rhopdcro2=np.array(rhopdcro2)
        
    Tpdcoo2=[]
    rhopdcoo2=[]
    with open('pdcoo2.csv', newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in filereader:
            Tpdcoo2.append(float(row[0]))
            rhopdcoo2.append(float(row[1]))
            
    Tpdcoo2=np.array(Tpdcoo2)  
    rhopdcoo2=np.array(rhopdcoo2)  
    ax.plot(Tpdcoo2,rhopdcoo2, lw=LW, c='r',label=r'PdCoO$_2$', ls=':')
    ax.plot(Tpdcro2,rhopdcro2, lw=LW,c='b', label=r'PdCrO$_2$', ls=':')
    ax.legend( prop={'size':13.6}, loc='upper right',borderpad=0.2,borderaxespad=0.1,framealpha=0.7,frameon=False)
    
    dis=np.max(tauinv_3+tauinv+tauinv_2)*1.75
    ax.set_ylim([0,dis])
    # ax.text(T2*0.6,dis*0.25,r"$\tau^{-1}_{K} + \tau^{-1}_{EME} + \tau^{-1}_{E}$", c='k', fontsize=15)
    ax.set_ylabel(r'$\rho_{ab}$ [$\mu \Omega $cm] ', size=20)
    ax.set_xlabel(r'$T$ [K]', size=20)
    ax.tick_params(axis='x', labelsize=20,direction="in" )
    ax.tick_params(axis='y', labelsize=20 )
    # plt.tight_layout()

    plt.savefig("in-plane.png", dpi=500, bbox_inches='tight')
    
   
if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
