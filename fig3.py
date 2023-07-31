import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import Lattice
import StructureFactor
import Dispersion
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
e2=(1.6021766*1e-19)**2
hbar=1.05457*1e-34 # m2 kg / s
kb=1.3806*1e-23 #m2 kg s-2 K-1

J=6.25 #in mev
U=4000
g=110
JK=4*g*g/U
print("Kondo coupling of" , JK)
mevtoK=11.6045250061598
mevtoJ=e*0.001 #conversion
Tval=J*mevtoK
print(Tval, 'validity above this temp')


S=3.0/2.0
kappa=S*(S+1)
alpha=2

#electron Parameters
tp1=568 #in units of Js\
tp2=-tp1*108/568 #/tpp1
tperp=(0.002/2)*(3*tp1+9*tp2)*((6*a/c)**2)
print(tp1/tperp, tperp/tp1,'tperp ratio',tperp)
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



def nf( e, T):
    rat=np.max(np.abs(e/T))
    if rat<700:
        return 1/(1+np.exp( e/T ))
    else:
        return np.heaviside(-e, 0.5)
    
def min_dnf( e, T):
    return nf(e, T)*( 1 - nf(e,T) )/T


def nb( e, T):
    rat=np.max(np.abs(e/T))
    if rat<700:
        return 1/(np.exp( e/T )-1)
    else:
        return -np.heaviside(-e,0.5)
    
def be_nf( e, T):
    rat=np.max(np.abs(e/T))
    if rat<700:
        x=e/T
        return x/(np.exp(x)+1)
    else:
        return np.heaviside(-e,0.5)

def be_nb( e, T):
    rat=np.max(np.abs(e/T))
    if rat<700:
        x=e/T
        expr=np.real(x/(np.exp(x)-1+1j*1e-17))
        problems=np.where(np.isnan(expr))[0]
        expr[problems]=1
        return expr
    else:
        return -(e/T)*np.heaviside(-e,0.5)
    
def be_csch( e, T):
    rat=np.max(np.abs(e/T))
    if rat<700:
        x=e/T
        expr=x/np.sinh(x)
        problems=np.where(np.isnan(expr))[0]
        expr[problems]=1
        return expr
    else:
        return -(e/T)*np.heaviside(-e,0.5)

def GG(T,e,z,S,rho):
    return rho(e)*min_dnf( e, T)*rho(e-z)*S(z)*(be_nb( z, T) +nf(z - e, T)*z/T )/(np.pi**2)
    

def FF(T,omega0,e,z,S,rho):
    return np.cosh( e/(2*T) )*rho(e)*min_dnf( e, T)*S(z)*(rho(e+z+omega0)/np.cosh( (e+z+omega0)/(2*T) ) +rho(e+z-omega0)/np.cosh( (e+z-omega0)/(2*T) ) )*be_csch( e, 2*T)/(np.pi**2)


def sigm11_f(T,Kcou_11_2,e_1,z_1,S,rho, de, dz,mevtoK):
    sigm11List=[]
    for Tt in T:
        
        In=Kcou_11_2*GG(Tt/mevtoK,e_1,z_1,S,rho)*de*dz
        Inte=integrate.romb(integrate.romb(In))
        sigm11List.append(Inte)
    
    sigm11=np.array(sigm11List)
    return sigm11

def sigm22_f(T,Kcou_22_2,omega0tilde,e_1,z_1,S, rho, de, dz,mevtoK):
    sigm22List=[]
    for Tt in T:
        IIn=( Kcou_22_2/np.sinh(omega0tilde*mevtoK/(2*Tt)) ) *FF(Tt/mevtoK,omega0tilde,e_1,z_1,S,rho)*de*dz
        Inte=integrate.romb(integrate.romb(IIn))
        sigm22List.append(Inte)
    
    sigm22=np.array(sigm22List)
    return sigm22


def main() -> int:

    

    #random parameters and integration grid size
    ed=Dispersion.Dispersion_TB_single_band([tp1,tp2],fill,size_E,Machine)
    [dens2,bins,valt,rho ]=[ed.dens2,ed.bins,ed.valt,ed.f2 ]
    [nn,earr,Dos]=[ed.nn,ed.earr,ed.Dos]
    mu=ed.mu
    
    
    ####estimation of e-p-s coupling
    axy=2.93*1e-10
    az=c/3
    dlay=az/2 #distance from Pd to Cr
    dnn=axy
    ell=np.log(tp1/g)*(dlay-dnn)
    lam=1/ell #1/meter
    print("decay constant for eps couping estimate...",ell, lam)

    #dimensionless
    lambda_K=2*alpha*kappa*JK*rho(mu)
    
    poptE=[36.9553148 ,  0.05531515]  #freq in ev and dimensionless coupling
    popt=[4.00633908e+01, 2.55520073e-02]  #freq in ev and  dimensionless coupling 
    poptEA=[2.86942388e+01, 4.28936187e-02, 1.20288925e+02, 2.38349684e-02] #freq in ev and  dimensionless coupling Ac #freq in ev and  dimensionless coupling opt
    
    omega0=poptEA[2]
    omegaD=poptEA[0]
    lambda_A=poptEA[1]
    lambda_E=poptEA[3]
    
    omega0tilde=popt[0]
    lambda_EME=popt[1]
    
    
                   

    
    fac_ab=m_cr*1e+8/(e2*n_cr)
    nu0=rho(mu)/mevtoJ
    teff=tp1*0.042*mevtoJ
    fac_enm_c= 1e+8/(e*e*2*nu0*((teff/hbar)**2)*(az/(axy**2))) #8.940033980000002e-11
    
    
    

    # ####plotting starts
    fig, ax = plt.subplots(1, 1, figsize=[5, 4])

    LW=1
    
    eps_arr=np.linspace(-0.1,0.,6)
    nnn=np.size(eps_arr)
    colors = plt.cm.coolwarm(np.linspace(0,1,nnn))
    
    

    
    
    
    ####out of plane
    #struct fac params
    
    T1=J*mevtoK
    T2=1300# 100*J*mevtoK
    T=np.linspace(T1,T2,40)
    
    
    SF=StructureFactor.StructureFac_HT(1, kappa, alpha*np.pi,  size_Z, J)
    S=SF.omega_Dynamical
    
    W1=np.max(earr)
    W2=np.min(earr)
    
    
    e_pre=np.linspace(W2,W1,size_E)
    z_pre=np.linspace(-alpha*np.pi*J,alpha*np.pi*J,size_Z)
    
    de=e_pre[1]-e_pre[0]
    dz=z_pre[1]-z_pre[0]
    
    
    e_1,z_1=np.meshgrid(e_pre,z_pre)


    for i,eps in enumerate(eps_arr):
        aufac=(np.exp(-eps))**4
        
        fac_geom=(  1e+8 )*hbar*axy*axy/(dlay*e2) #\mu\Omega cm
        print(fac_geom, 'fac_geom',aufac, 'aufac')
        
        Kcou_11_2=(aufac*(2*JK)**2 )/fac_geom
        Kcou_22_2=((lambda_EME*aufac/(np.pi*kappa))*(omega0tilde/rho(0)))/fac_geom
        
        
        
        sigm11=sigm11_f(T,Kcou_11_2,e_1,z_1,S,rho, de, dz,mevtoK)
        sigm22=sigm22_f(T,Kcou_22_2,omega0tilde,e_1,z_1,S,rho, de, dz,mevtoK)
        
        
        tauinv=K_tau(T,lambda_K*aufac)*fac_enm_c/aufac
        tauinv_2=EME_tau(T, omega0tilde, lambda_EME*aufac)*fac_enm_c/aufac
        tauinv_3=EA_tau(T,omegaD, lambda_A, omega0, lambda_E, mevtoK)*fac_enm_c/aufac

        
        res=1/( (1/(tauinv_3 +tauinv+tauinv_2 ))+sigm11+sigm22)

        ax.plot(T,res/1e+3, lw=LW,color=colors[i])
        # ax.plot(T,tauinv_3, lw=LW,label=r'K',color=colors[i])
        # ax.plot(T,1/(sigm22), lw=LW,label=r'$EME$',color=colors[i], ls=':')
    
    # creating ScalarMappable for colorbar
    cmap = plt.get_cmap('coolwarm', nnn)
    norm = mpl.colors.Normalize(vmin=eps_arr[0], vmax=eps_arr[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, ticks=np.linspace(eps_arr[0],eps_arr[-1], 2),ax=ax)
    cbar.ax.tick_params(labelsize=20)
    # cbar.formatter.set_powerlimits((0, 0))
    # cbar.ax.yaxis.offsetText.set_fontsize(15)
    ax.annotate(r"$\frac{a_z}{\xi}   \varepsilon_{zz} $",xy=(0.85,0.55),xycoords='figure fraction',fontsize=22) #note that the coordinates are given as a fraction of the figure, not necessarily of the area enclosed by the axes 

    
    # ax.legend()
    
    
    # ax.set_xlim([0,1400])
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylabel(r'$\rho_{c}$ [m$\Omega $cm] ', size=22)#,rotation=270, labelpad=18)
    ax.set_xlabel(r'$T$[K]', size=22)
    ax.tick_params(axis='x', labelsize=20,direction="in" )
    ax.tick_params(axis='y', labelsize=20,direction="in" )
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    
    
    axins1 = inset_axes(ax, width="40%", height="40%", loc=4,borderpad=0.0)
    
    for i,eps in enumerate(eps_arr):
        
        aufac=(np.exp(-eps))**4
        tauinv=K_tau(T,lambda_K*aufac)*fac_ab
        tauinv_2=EME_tau(T, omega0tilde, lambda_EME*aufac)*fac_ab
        tauinv_3=EA_tau(T,omegaD, lambda_A,omega0, lambda_E,mevtoK)*fac_ab
        
        axins1.plot(T,tauinv_3+tauinv+tauinv_2, lw=LW,label=r'$\rho_{K}+\rho_{ep} $'+'\n'+r'$+\rho_{EME}$',color=colors[i])



    dis=np.max(tauinv_3+tauinv+tauinv_2)*1.2
    axins1.set_ylim([0,45])

    axins1.set_ylabel(r'$\rho_{ab}$ [$\mu \Omega $cm] ', size=18)
    axins1.set_xlabel(r'$T$ [K]', size=18, labelpad=10)
    yt=[r"10",r"40"]
    ytpos=np.array([10,40])
    axins1.set_yticks(ytpos, pad=0.001)
    axins1.set_yticklabels(yt,size=18)
    axins1.tick_params(axis='x', labelsize=18,direction="in" )
    axins1.tick_params(axis='y', labelsize=18,direction="in" )
    axins1.xaxis.set_label_position("top")
    axins1.xaxis.tick_top()
    


    plt.savefig("out_of_plane.png", dpi=500, bbox_inches='tight')
    
if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit

##leftover comments from trying to fit the hopping 
# e2nmz= 6*e2*(az/(axy*axy)) * (rho(mu)/mevtoJ) * ( (tperp*mevtoJ)/hbar )**2  # #density of states already has a factor of 2 for spin sum
        # meff=(hbar**2)/(2*tperp*mevtoJ*dlay*dlay)
        # meff3=(hbar**2)/(3*tp1*mevtoJ*axy*axy+9*tp2*mevtoJ*axy*axy)
        
        # fac2=2.38623858e-13
        # print(meff,meff3, m_cr,me )
        # print(fac*(1e-8*e2*n_cr) ,fac2*(1e-8*e2*n_cr),fac2/fac)