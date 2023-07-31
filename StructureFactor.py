import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator # You may have some better interpolation methods
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.optimize import curve_fit
import sys
import Lattice
from scipy.interpolate import interp1d

class StructureFac_fit:

    #initializes temperature and parameters for the fits
    def __init__(self, T, KX, KY ):

        self.T=T
        self.KX=KX
        self.KY=KY
        ##fit parameters for different temperatures:

        #T=1.0
        if(T==1.0):
            self.alph=np.array([ 0.7097908959336873,  -0.0043594581070084135,  -0.004495974146928671, -0.024777430963518057,   0.0030982360905670333,   0.0004539363283678258])
            self.et=np.array([0.23331490064983912,  0.06490355420597822,    -0.03601601298488789,   -0.04655841264762831,    -0.010189892955121571, -0.006643162950435294])
            self.lam=4.178642027077301

        #T=2.0
        elif(T==2.0):
            self.alph=np.array([0.6222520758430777,   -0.009461521513823186,     -0.006053452180584596,   -0.010702516424885714,   0.0020003919364024714,     -1.0205060481952911e-05])
            self.et=np.array([ 0.09658650764378539,  0.013384215891118253,    -0.010516833605162713,      -0.01090143816114151,      0.0004144707853819521,      0.0001204480954048534])   
            self.lam= 3.370944783098885

        #T=3.0
        elif(T==3.0):
            self.alph=np.array([ 0.6071415409901372, -0.007643725101933083,    -0.004102812828401734,    -0.0064882051217971795,  0.001523532730774404, 2.9287972860276336e-05 ])
            self.et=np.array([0.09131678420721018,    0.005815174776661578,  -0.00670989716658747,   -0.006410702279227802,   0.0011528049552485798,  0.0003122379970753175])
            self.lam=3.1806350971738353 

        #T=4.0
        elif(T==4.0):
            self.alph=np.array([0.6039735698700098,  -0.006047485346815116,  -0.002472531200563213,  -0.004275139567596024,  0.0013215101055664485,  7.175902842573632e-05])
            self.et=np.array([0.09553529940513966,    0.0018186539475888166, -0.004739620696629819,   -0.004494332347087183,  0.0013142362882892138,  0.000291236774009811])
            self.lam=3.106684811722399

        #T=5.0
        elif(T==5.0):
            self.alph=np.array([0.6040810534137876,  -0.004732865622040766,  -0.001127349144342822,  -0.0027706232289313806,     0.0013410007192414624,      0.0002091028903354993])
            self.et=np.array([0.1011836355370694,    -0.0009584775768396963, -0.003315265600935579  ,    -0.0033082929540142105,  0.0013589600008588089,  0.00027718680067235644])
            self.lam=3.0703759314285626 

        #T=6.0
        elif(T==6.0):
            self.alph=np.array([0.6048825771130464,  0.008438891265335053,   -0.00020510810262157957,    -0.0018087562251271861 ,    0.0012503605738861706, 0.00019497167542968042]) 
            self.et=np.array([0.106654110761412, -0.0032025150693937156, -0.0021998569308273493, -0.0024967677601908135, 0.0012877521033920923,  0.00021783472445787648])
            self.lam=3.0498886949148036

        #T=7.0
        elif(T==7.0):
            self.alph=np.array([0.6060957381074707,  -0.0030658995822706505, 0.0006420851435837167,  -0.0009717074866372171,  0.0013131665577559325, 0.0003018262351056688]) 
            self.et=np.array([0.11175745410773015,   -0.005039700509756049,  -0.0012315951887033312, -0.001841829723904746,  0.0012274174484950366,   0.00019865688157838827])
            self.lam=3.037203270341933

        #T=8.0
        elif(T==8.0):
            self.alph=np.array([0.6070551862962202,  -0.002697101301333131,  0.00107489031858635,    -0.0005590867861957349,     0.0011475930927934638,   0.00018205256220494518])
            self.et=np.array([ 0.11616954883061426,  -0.003988873565023515,   -0.0004657571027108079,    -0.0013839453692115198, 0.0011044737627527907, 0.00012737532156294055]) 
            self.lam=3.028807008086399

        
        #T=9.0
        elif(T==9.0):
            self.alph=np.array([0.6083718387945705,  0.010508584347927811,   0.0018022331604978998,  0.00012473785812128,    0.0013244365310333586,  0.00037723431092535686])
            self.et=np.array([0.12033706994827463,    -0.007946874704261314, 0.00030420836462192086,  -0.00090404839139119,  0.001061248299460572,   0.00014865459778067692]) 
            self.lam= 3.0229631820378184


        #T=10.0
        elif(T==10.0):
            self.alph=np.array([0.6092518069471177,   -0.0017454331191290237,    0.0021259053889015845,   0.0004188012953199125, 0.0012489555790225417,  0.0003255774536971311])
            self.et=np.array([0.12385676180579733,   -0.009155564378675983,   0.0008941115202702899,      -0.0005938474219710233,    0.0019469008555008608,      0.0001013876862340809])
            self.lam=3.018732903302169

        #T=50.0
        elif(T==100.0):
            self.alph=np.array([ 0.6201774944069754, 0.0012530224630861754,   0.005833622305428591,  0.003605079260905222,   0.0014324738571124083,   0.0006316328735677271])
            self.et=np.array([0.16281297212996357,   -0.021355929675790707,   0.007802291525953633,  0.002565178078007373,   -0.0004250484192336371, -3.285858634623746e-05])
            self.lam=3.000789439969265
        #T=100.0
        else:
            print("T may not be fitted, setting it to T=1000")
            self.alph=np.array([0.6220438193343075,  0.0016537316919072811,  0.006387742935248672,   0.004060505526695932,    0.0014967727700990639,  0.000700872036530507])
            self.et=np.array([0.1697667873959355,    -0.023474171445420244,   0.009095251231202181,  0.0030821033954326386,  -0.0007082689712385551, -2.655211696552507e-05])
            self.lam=3.00019867333284


        self.params=self.params_fit()
        self.SF_stat=self.Static_SF()
        self.name="fit_SF_arr"
                    
    def __repr__(self):
        return "Structure factorat T={T}".format(T=self.T)

    def Static_SF(self):
        KX=self.KX
        KY=self.KY
        gg2=2*np.cos(KX)+4*np.cos(KX/2)*np.cos(np.sqrt(3)*KY/2)
        return 3/(self.lam+(1/self.T)*gg2)
    
    def Int_Static_SF(self, KX, KY):
        curlyN=np.size(KX)
        return np.sum(self.Static_SF(KX,KY))/curlyN -1

    def params_fit(self):
        KX=self.KX
        KY=self.KY
        ##nearest neighbour expansion
        gamma0=1
        gamma1=(1/3.0)*(np.cos(KX)+2*np.cos(KX/2)*np.cos(np.sqrt(3)*KY/2))
        gamma2=(1/3.0)*(2*np.cos(3*KX/2)*np.cos(np.sqrt(3)*KY/2)+np.cos(np.sqrt(3)*KY))
        gamma3=(1/3.0)*(np.cos(2*KX)+2*np.cos(2*KX/2)*np.cos(2*np.sqrt(3)*KY/2))
        gamma4=(1/3.0)*( np.cos(5*KX/2)*np.cos(np.sqrt(3)*KY/2) +np.cos(2*KX)*np.cos(np.sqrt(3)*KY) +np.cos(KX/2)*np.cos(3*np.sqrt(3)*KY/2) )
        gamma5=(1/3.0)*(np.cos(3*KX)+2*np.cos(3*KX/2)*np.cos(3*np.sqrt(3)*KY/2))

 
        sum_et_gam=self.et[0]*gamma0+self.et[1]*gamma1+self.et[2]*gamma2+self.et[3]*gamma3+self.et[4]*gamma4+self.et[5]*gamma5
        et_q=sum_et_gam*((6-6*gamma1)**2)
        alpha_q=self.alph[0]*gamma0+self.alph[1]*gamma1+self.alph[2]*gamma2+self.alph[3]*gamma3+self.alph[4]*gamma4+self.alph[5]*gamma5
        #additional 2 pi for the correct normalization of the frequency integral
        NN=2*np.pi*np.abs( alpha_q*np.sqrt( et_q*( et_q-1 +1j*1e-17) )/np.arcsinh( np.sqrt( (et_q-1+1j*1e-17) ) ) )

        return [sum_et_gam,et_q,alpha_q,NN]


    def Dynamical_SF( self,kx,ky, f):

        [sum_et_gam,et_q,alpha_q,NN]=self.params
        a1=np.abs(alpha_q*f)
        x=np.heaviside(300-a1,1.0)
        a2=a1*x
        sinhal=np.sinh(a2)
        fac=x*NN/(sinhal*sinhal+et_q)
        return self.SF_stat*fac # this has to be called in the reverse order for some reason.
    
 
class StructureFac_fit_F:

    #initializes temperature and parameters for the fits
    def __init__(self, T ):

        self.T=T
        self.name="fit_SF_func"

        ##fit parameters for different temperatures:

        #T=1.0
        if(T==1.0):
            self.alph=np.array([ 0.7097908959336873,  -0.0043594581070084135,  -0.004495974146928671, -0.024777430963518057,   0.0030982360905670333,   0.0004539363283678258])
            self.et=np.array([0.23331490064983912,  0.06490355420597822,    -0.03601601298488789,   -0.04655841264762831,    -0.010189892955121571, -0.006643162950435294])
            self.lam=4.178642027077301

        #T=2.0
        elif(T==2.0):
            self.alph=np.array([0.6222520758430777,   -0.009461521513823186,     -0.006053452180584596,   -0.010702516424885714,   0.0020003919364024714,     -1.0205060481952911e-05])
            self.et=np.array([ 0.09658650764378539,  0.013384215891118253,    -0.010516833605162713,      -0.01090143816114151,      0.0004144707853819521,      0.0001204480954048534])   
            self.lam= 3.370944783098885

        #T=3.0
        elif(T==3.0):
            self.alph=np.array([ 0.6071415409901372, -0.007643725101933083,    -0.004102812828401734,    -0.0064882051217971795,  0.001523532730774404, 2.9287972860276336e-05 ])
            self.et=np.array([0.09131678420721018,    0.005815174776661578,  -0.00670989716658747,   -0.006410702279227802,   0.0011528049552485798,  0.0003122379970753175])
            self.lam=3.1806350971738353 

        #T=4.0
        elif(T==4.0):
            self.alph=np.array([0.6039735698700098,  -0.006047485346815116,  -0.002472531200563213,  -0.004275139567596024,  0.0013215101055664485,  7.175902842573632e-05])
            self.et=np.array([0.09553529940513966,    0.0018186539475888166, -0.004739620696629819,   -0.004494332347087183,  0.0013142362882892138,  0.000291236774009811])
            self.lam=3.106684811722399

        #T=5.0
        elif(T==5.0):
            self.alph=np.array([0.6040810534137876,  -0.004732865622040766,  -0.001127349144342822,  -0.0027706232289313806,     0.0013410007192414624,      0.0002091028903354993])
            self.et=np.array([0.1011836355370694,    -0.0009584775768396963, -0.003315265600935579  ,    -0.0033082929540142105,  0.0013589600008588089,  0.00027718680067235644])
            self.lam=3.0703759314285626 

        #T=6.0
        elif(T==6.0):
            self.alph=np.array([0.6048825771130464,  0.008438891265335053,   -0.00020510810262157957,    -0.0018087562251271861 ,    0.0012503605738861706, 0.00019497167542968042]) 
            self.et=np.array([0.106654110761412, -0.0032025150693937156, -0.0021998569308273493, -0.0024967677601908135, 0.0012877521033920923,  0.00021783472445787648])
            self.lam=3.0498886949148036

        #T=7.0
        elif(T==7.0):
            self.alph=np.array([0.6060957381074707,  -0.0030658995822706505, 0.0006420851435837167,  -0.0009717074866372171,  0.0013131665577559325, 0.0003018262351056688]) 
            self.et=np.array([0.11175745410773015,   -0.005039700509756049,  -0.0012315951887033312, -0.001841829723904746,  0.0012274174484950366,   0.00019865688157838827])
            self.lam=3.037203270341933

        #T=8.0
        elif(T==8.0):
            self.alph=np.array([0.6070551862962202,  -0.002697101301333131,  0.00107489031858635,    -0.0005590867861957349,     0.0011475930927934638,   0.00018205256220494518])
            self.et=np.array([ 0.11616954883061426,  -0.003988873565023515,   -0.0004657571027108079,    -0.0013839453692115198, 0.0011044737627527907, 0.00012737532156294055]) 
            self.lam=3.028807008086399

        
        #T=9.0
        elif(T==9.0):
            self.alph=np.array([0.6083718387945705,  0.010508584347927811,   0.0018022331604978998,  0.00012473785812128,    0.0013244365310333586,  0.00037723431092535686])
            self.et=np.array([0.12033706994827463,    -0.007946874704261314, 0.00030420836462192086,  -0.00090404839139119,  0.001061248299460572,   0.00014865459778067692]) 
            self.lam= 3.0229631820378184


        #T=10.0
        elif(T==10.0):
            self.alph=np.array([0.6092518069471177,   -0.0017454331191290237,    0.0021259053889015845,   0.0004188012953199125, 0.0012489555790225417,  0.0003255774536971311])
            self.et=np.array([0.12385676180579733,   -0.009155564378675983,   0.0008941115202702899,      -0.0005938474219710233,    0.0019469008555008608,      0.0001013876862340809])
            self.lam=3.018732903302169

        #T=50.0
        elif(T==100.0):
            self.alph=np.array([ 0.6201774944069754, 0.0012530224630861754,   0.005833622305428591,  0.003605079260905222,   0.0014324738571124083,   0.0006316328735677271])
            self.et=np.array([0.16281297212996357,   -0.021355929675790707,   0.007802291525953633,  0.002565178078007373,   -0.0004250484192336371, -3.285858634623746e-05])
            self.lam=3.000789439969265
        #T=100.0
        else:
            print("T may not be fitted, setting it to T=1000")
            self.alph=np.array([0.6220438193343075,  0.0016537316919072811,  0.006387742935248672,   0.004060505526695932,    0.0014967727700990639,  0.000700872036530507])
            self.et=np.array([0.1697667873959355,    -0.023474171445420244,   0.009095251231202181,  0.0030821033954326386,  -0.0007082689712385551, -2.655211696552507e-05])
            self.lam=3.00019867333284
                    
    def __repr__(self):
        return "Structure factor at T={T}".format(T=self.T)

    def Static_SF(self, KX, KY):
        gg2=2*np.cos(KX)+4*np.cos(KX/2)*np.cos(np.sqrt(3)*KY/2)
        return 3/(self.lam+(1/self.T)*gg2)
    
    def Int_Static_SF(self, KX, KY):
        curlyN=np.size(KX)
        return np.sum(self.Static_SF(KX,KY))/curlyN -1

    def Dynamical_SF( self, kx, ky, f):

        gamma0=1
        gamma1=(1/3.0)*(np.cos(kx)+2*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2))
        gamma2=(1/3.0)*(2*np.cos(3*kx/2)*np.cos(np.sqrt(3)*ky/2)+np.cos(np.sqrt(3)*ky))
        gamma3=(1/3.0)*(np.cos(2*kx)+2*np.cos(2*kx/2)*np.cos(2*np.sqrt(3)*ky/2))
        gamma4=(1/3.0)*( np.cos(5*kx/2)*np.cos(np.sqrt(3)*ky/2) +np.cos(2*kx)*np.cos(np.sqrt(3)*ky) +np.cos(kx/2)*np.cos(3*np.sqrt(3)*ky/2) )
        gamma5=(1/3.0)*(np.cos(3*kx)+2*np.cos(3*kx/2)*np.cos(3*np.sqrt(3)*ky/2))

 
        sum_et_gam=self.et[0]*gamma0+self.et[1]*gamma1+self.et[2]*gamma2+self.et[3]*gamma3+self.et[4]*gamma4+self.et[5]*gamma5
        et_q=sum_et_gam*((6-6*gamma1)**2)
        alpha_q=self.alph[0]*gamma0+self.alph[1]*gamma1+self.alph[2]*gamma2+self.alph[3]*gamma3+self.alph[4]*gamma4+self.alph[5]*gamma5
        #additional 2 pi for the correct normalization of the frequency integral
        NN=2*np.pi*np.abs( alpha_q*np.sqrt( et_q*( et_q-1 +1j*1e-17) )/np.arcsinh( np.sqrt( (et_q-1+1j*1e-17) ) ) )

    

        ##preventing overflow in sinh and in multiply
        a1=np.abs(alpha_q*f)
        x=np.heaviside(300-a1,1.0)
        a2=a1*x
        ###
        sinhal=np.sinh(a2)
        fac=x*NN/(sinhal*sinhal+et_q)

        SF_stat=3.0/(self.lam+(1/self.T)*6.0*gamma1)
        return SF_stat*fac # this has to be called in the reverse order for some reason.
    
    def momentum_cut_high_symmetry_path(self, latt, Nomegs,Nt_points ):
        omeg_max=6
        kpath=latt.High_symmetry_path(Nt_points)
        ##geneerating arrays for imshow of momentum cut
        omegas=np.linspace(0.0001,omeg_max ,Nomegs)
        t=np.arange(0,len(kpath),1)
        t_m,omegas_m=np.meshgrid(t,omegas)
        SSSfw=self.Dynamical_SF(kpath[t_m,0],kpath[t_m,1],omegas_m)
        im=plt.imshow(SSSfw, vmax=6 ,origin='lower', aspect='auto')
        Nty=4
        Ntx=3
        Npl2=np.linspace(0,Nomegs,Nty)
        Npl=np.linspace(0,len(kpath),Ntx)
        om=np.round(np.linspace(0,omeg_max,Nty),3)
        t=np.round(np.linspace(0,1,Ntx),3)
        cbar = plt.colorbar(im)
        tick_font_size = 20
        cbar.ax.tick_params(labelsize=tick_font_size)
        # pyplot.locator_params(axis='y', nbins=4)
        # pyplot.locator_params(axis='x', nbins=3)
        plt.xticks(Npl,t, size=20)
        plt.yticks(Npl2,om, size=20)
        plt.xlabel(r"$q$", size=20)
        plt.ylabel(r"$\omega$", size=20)
        plt.tight_layout()
        plt.savefig(self.name+ ".png")
        plt.close()

        return SSSfw
    

class StructureFac_fit_F_w:

    #initializes temperature and parameters for the fits
    def __init__(self, T, Nomega, Machine):

        self.T=T
        self.name="fit_SF_func_w"
        self.Npoi_ints=2**7
        self.latt_int=Lattice.TriangLattice(self.Npoi_ints, False, Machine) #temp grid for integrating and getting filling

        [self.KX,self.KY]=self.latt_int.Generate_lattice()
        Vol_rec=self.latt_int.Vol_BZ()
        self.ds=1/np.size(self.KX)
        
        self.ome=np.linspace(0.000001, 2*np.pi, Nomega)
        self.dome=self.ome[1]-self.ome[0]

        ##fit parameters for different temperatures:

        #T=1.0
        if(T==1.0):
            self.alph=np.array([ 0.7097908959336873,  -0.0043594581070084135,  -0.004495974146928671, -0.024777430963518057,   0.0030982360905670333,   0.0004539363283678258])
            self.et=np.array([0.23331490064983912,  0.06490355420597822,    -0.03601601298488789,   -0.04655841264762831,    -0.010189892955121571, -0.006643162950435294])
            self.lam=4.178642027077301

        #T=2.0
        elif(T==2.0):
            self.alph=np.array([0.6222520758430777,   -0.009461521513823186,     -0.006053452180584596,   -0.010702516424885714,   0.0020003919364024714,     -1.0205060481952911e-05])
            self.et=np.array([ 0.09658650764378539,  0.013384215891118253,    -0.010516833605162713,      -0.01090143816114151,      0.0004144707853819521,      0.0001204480954048534])   
            self.lam= 3.370944783098885

        #T=3.0
        elif(T==3.0):
            self.alph=np.array([ 0.6071415409901372, -0.007643725101933083,    -0.004102812828401734,    -0.0064882051217971795,  0.001523532730774404, 2.9287972860276336e-05 ])
            self.et=np.array([0.09131678420721018,    0.005815174776661578,  -0.00670989716658747,   -0.006410702279227802,   0.0011528049552485798,  0.0003122379970753175])
            self.lam=3.1806350971738353 

        #T=4.0
        elif(T==4.0):
            self.alph=np.array([0.6039735698700098,  -0.006047485346815116,  -0.002472531200563213,  -0.004275139567596024,  0.0013215101055664485,  7.175902842573632e-05])
            self.et=np.array([0.09553529940513966,    0.0018186539475888166, -0.004739620696629819,   -0.004494332347087183,  0.0013142362882892138,  0.000291236774009811])
            self.lam=3.106684811722399

        #T=5.0
        elif(T==5.0):
            self.alph=np.array([0.6040810534137876,  -0.004732865622040766,  -0.001127349144342822,  -0.0027706232289313806,     0.0013410007192414624,      0.0002091028903354993])
            self.et=np.array([0.1011836355370694,    -0.0009584775768396963, -0.003315265600935579  ,    -0.0033082929540142105,  0.0013589600008588089,  0.00027718680067235644])
            self.lam=3.0703759314285626 

        #T=6.0
        elif(T==6.0):
            self.alph=np.array([0.6048825771130464,  0.008438891265335053,   -0.00020510810262157957,    -0.0018087562251271861 ,    0.0012503605738861706, 0.00019497167542968042]) 
            self.et=np.array([0.106654110761412, -0.0032025150693937156, -0.0021998569308273493, -0.0024967677601908135, 0.0012877521033920923,  0.00021783472445787648])
            self.lam=3.0498886949148036

        #T=7.0
        elif(T==7.0):
            self.alph=np.array([0.6060957381074707,  -0.0030658995822706505, 0.0006420851435837167,  -0.0009717074866372171,  0.0013131665577559325, 0.0003018262351056688]) 
            self.et=np.array([0.11175745410773015,   -0.005039700509756049,  -0.0012315951887033312, -0.001841829723904746,  0.0012274174484950366,   0.00019865688157838827])
            self.lam=3.037203270341933

        #T=8.0
        elif(T==8.0):
            self.alph=np.array([0.6070551862962202,  -0.002697101301333131,  0.00107489031858635,    -0.0005590867861957349,     0.0011475930927934638,   0.00018205256220494518])
            self.et=np.array([ 0.11616954883061426,  -0.003988873565023515,   -0.0004657571027108079,    -0.0013839453692115198, 0.0011044737627527907, 0.00012737532156294055]) 
            self.lam=3.028807008086399

        
        #T=9.0
        elif(T==9.0):
            self.alph=np.array([0.6083718387945705,  0.010508584347927811,   0.0018022331604978998,  0.00012473785812128,    0.0013244365310333586,  0.00037723431092535686])
            self.et=np.array([0.12033706994827463,    -0.007946874704261314, 0.00030420836462192086,  -0.00090404839139119,  0.001061248299460572,   0.00014865459778067692]) 
            self.lam= 3.0229631820378184


        #T=10.0
        elif(T==10.0):
            self.alph=np.array([0.6092518069471177,   -0.0017454331191290237,    0.0021259053889015845,   0.0004188012953199125, 0.0012489555790225417,  0.0003255774536971311])
            self.et=np.array([0.12385676180579733,   -0.009155564378675983,   0.0008941115202702899,      -0.0005938474219710233,    0.0019469008555008608,      0.0001013876862340809])
            self.lam=3.018732903302169

        #T=50.0
        elif(T==100.0):
            self.alph=np.array([ 0.6201774944069754, 0.0012530224630861754,   0.005833622305428591,  0.003605079260905222,   0.0014324738571124083,   0.0006316328735677271])
            self.et=np.array([0.16281297212996357,   -0.021355929675790707,   0.007802291525953633,  0.002565178078007373,   -0.0004250484192336371, -3.285858634623746e-05])
            self.lam=3.000789439969265
        #T=100.0
        else:
            print("T may not be fitted, setting it to T=1000")
            self.alph=np.array([0.6220438193343075,  0.0016537316919072811,  0.006387742935248672,   0.004060505526695932,    0.0014967727700990639,  0.000700872036530507])
            self.et=np.array([0.1697667873959355,    -0.023474171445420244,   0.009095251231202181,  0.0030821033954326386,  -0.0007082689712385551, -2.655211696552507e-05])
            self.lam=3.00019867333284
    
        self.f2 = interp1d(self.ome,self.omega_DynamicalPre(self.ome), kind='cubic',bounds_error=False, fill_value=0) 
                       
    def __repr__(self):
        return "Structure factor at T={T}".format(T=self.T)

    def Static_SF(self, KX, KY):
        gg2=2*np.cos(KX)+4*np.cos(KX/2)*np.cos(np.sqrt(3)*KY/2)
        return 3/(self.lam+(1/self.T)*gg2)
    
    def Int_Static_SF(self, KX, KY):
        curlyN=np.size(KX)
        return np.sum(self.Static_SF(KX,KY))/curlyN -1

    def Dynamical_SF( self, kx, ky, f):

        gamma0=1
        gamma1=(1/3.0)*(np.cos(kx)+2*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2))
        gamma2=(1/3.0)*(2*np.cos(3*kx/2)*np.cos(np.sqrt(3)*ky/2)+np.cos(np.sqrt(3)*ky))
        gamma3=(1/3.0)*(np.cos(2*kx)+2*np.cos(2*kx/2)*np.cos(2*np.sqrt(3)*ky/2))
        gamma4=(1/3.0)*( np.cos(5*kx/2)*np.cos(np.sqrt(3)*ky/2) +np.cos(2*kx)*np.cos(np.sqrt(3)*ky) +np.cos(kx/2)*np.cos(3*np.sqrt(3)*ky/2) )
        gamma5=(1/3.0)*(np.cos(3*kx)+2*np.cos(3*kx/2)*np.cos(3*np.sqrt(3)*ky/2))

 
        sum_et_gam=self.et[0]*gamma0+self.et[1]*gamma1+self.et[2]*gamma2+self.et[3]*gamma3+self.et[4]*gamma4+self.et[5]*gamma5
        et_q=sum_et_gam*((6-6*gamma1)**2)
        alpha_q=self.alph[0]*gamma0+self.alph[1]*gamma1+self.alph[2]*gamma2+self.alph[3]*gamma3+self.alph[4]*gamma4+self.alph[5]*gamma5
        #additional 2 pi for the correct normalization of the frequency integral
        NN=2*np.pi*np.abs( alpha_q*np.sqrt( et_q*( et_q-1 +1j*1e-17) )/np.arcsinh( np.sqrt( (et_q-1+1j*1e-17) ) ) )

    

        ##preventing overflow in sinh and in multiply
        a1=np.abs(alpha_q*f)
        x=np.heaviside(300-a1,1.0)
        a2=a1*x
        ###
        sinhal=np.sinh(a2)
        fac=x*NN/(sinhal*sinhal+et_q)

        SF_stat=3.0/(self.lam+(1/self.T)*6.0*gamma1)
        return SF_stat*fac # this has to be called in the reverse order for some reason.
    
    def Dynamical_SF_w( self, f):

        return np.sum(self.Dynamical_SF(self.KX,self.KY, f))*self.ds
    
    def momentum_cut_high_symmetry_path(self, latt, Nomegs,Nt_points ):
        
        omeg_max=6
        kpath=latt.High_symmetry_path(Nt_points)
        ##geneerating arrays for imshow of momentum cut
        omegas=np.linspace(0.0001,omeg_max ,Nomegs)
        t=np.arange(0,len(kpath),1)
        t_m,omegas_m=np.meshgrid(t,omegas)
        SSSfw=self.Dynamical_SF(kpath[t_m,0],kpath[t_m,1],omegas_m)
        im=plt.imshow(SSSfw, vmax=6 ,origin='lower', aspect='auto')
        Nty=4
        Ntx=3
        Npl2=np.linspace(0,Nomegs,Nty)
        Npl=np.linspace(0,len(kpath),Ntx)
        om=np.round(np.linspace(0,omeg_max,Nty),3)
        t=np.round(np.linspace(0,1,Ntx),3)
        cbar = plt.colorbar(im)
        tick_font_size = 20
        cbar.ax.tick_params(labelsize=tick_font_size)
        # pyplot.locator_params(axis='y', nbins=4)
        # pyplot.locator_params(axis='x', nbins=3)
        plt.xticks(Npl,t, size=20)
        plt.yticks(Npl2,om, size=20)
        plt.xlabel(r"$q$", size=20)
        plt.ylabel(r"$\omega$", size=20)
        plt.tight_layout()
        plt.savefig(self.name+ ".png")
        plt.close()

        return SSSfw
    
    def omega_DynamicalPre(self,omegas):
        S=[]
        
        for omega in omegas:
            Siav= self.Dynamical_SF_w(omega)
            S.append(Siav)
            
        return np.array(S)
    
    
    def omega_Dynamical(self,omegas):
        return self.f2(np.abs(omegas))


class StructureFac_HT:

    #initializes temperature and parameters for the fits
    def __init__(self, T, kappa, alpha, Nomega, J=None):

        self.T=T
        self.name="HT_SF_func"
        
        self.ome=np.linspace(0.0001, 2*np.pi, Nomega)
        self.dome=self.ome[1]-self.ome[0]
        self.kappa=kappa
        self.alpha=alpha
        
        if J>0:
            self.J=J
            print('the heisenberg coupling is',self.J,'the limit should be',self.alpha*self.J)
        else:
            self.J=1
                    
    def __repr__(self):
        return "Structure factor at T={T}".format(T=self.T)

    def omega_Dynamical(self,omegas):
        
        return self.kappa*np.heaviside(self.alpha*self.J-np.abs(omegas),0)/self.J


class MD_SF:

    def __init__(self, T ):

        ############################################################
        # Reading the structure factor
        ############################################################
        self.name="MD_data_SF"
        self.L = 120
        self.n_freqs = 4097
        # Momentum and frequency ranges (with some built in buffers)
        K1 = np.arange(-4*self.L//3, 4*self.L//3)
        K2 = np.arange(-4*self.L//3, 4*self.L//3)
        nX,nY=np.meshgrid(K1,K2)
        F = np.arange(0, self.n_freqs)
        Ta=str(T)
        T=float(Ta)



        print("loading data for the structure factor at T="+Ta)
        s=time.time()
        # Load the data files
        #MAC
        dsf_data = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/dsf_TLHAF_L=120_tf=4096_T='+Ta+'.npy')
        ##DebMac
        #dsf_data = np.load('/Users/Felipe/Documents/Struc_dat/dsf_TLHAF_L=120_tf=4096_T='+Ta+'.npy')
        #linux
        # dsf_data = np.load('/home/juan/Documents/Projects/Delafossites/SF_data/dsf_TLHAF_L=120_tf=4096_T=0.5.npy')
        e=time.time()
        print("time for loading", e-s)


        def dsf_func(k1, k2, w):
            return dsf_data[k1%self.L, k2%self.L, w]


        print("reshaping the original array....")
        s=time.time()
        #dsf_func_data = np.array([[[dsf_func(k1, k2, w) for k1 in K1] for k2 in K2] for w in F])
        dsf_func_data = np.array([ dsf_func( nX,nY, w) for w in F])
        # print(np.shape(dsf_func_data ))
        e=time.time()
        print("time for recasting", e-s)



        ############################################################
        # Interpolating 
        ############################################################
        s=time.time()
        self.dsf_interp = RegularGridInterpolator((F, K1, K2), dsf_func_data, method='linear')
        e=time.time()
        print("time for interpolation", e-s)

    def Dynamical_SF(self, qx, qy, f):
        thres=1.0
        k1 = self.L*qx/(2*np.pi)
        k2 = self.L*( qx/2 + np.sqrt(3)*qy/2 )/ (2*np.pi)
        fp=f*np.piecewise(f, [f <= thres, f >thres], [1, .0]) +thres*np.heaviside(f-thres,0) #treatment to avoid exceeding the range of the interpolation
        w = self.n_freqs*fp/(2*np.pi)
        return self.dsf_interp((w, k2, k1)) # this has to be called in the reverse order for some reason.

    def momentum_cut_high_symmetry_path(self, latt, Nomegs,Nt_points ):
        omeg_max=1.0
        kpath=latt.High_symmetry_path(Nt_points)
        ##geneerating arrays for imshow of momentum cut
        omegas=np.linspace(0.0001,omeg_max ,Nomegs)
        t=np.arange(0,len(kpath),1)
        t_m,omegas_m=np.meshgrid(t,omegas)
        SSSfw=self.Dynamical_SF(kpath[t_m,0],kpath[t_m,1],omegas_m)
        plt.imshow(SSSfw ,vmax=65 ,origin='lower', aspect='auto')
        Npl2=np.linspace(0,Nomegs,6)
        Npl=np.linspace(0,len(kpath),6)
        om=np.round(np.linspace(0,omeg_max,6),3)
        t=np.round(np.linspace(0,1,6),3)
        plt.colorbar()
        plt.xticks(Npl,t)
        plt.yticks(Npl2,om)
        plt.xlabel(r"$q$")
        plt.ylabel(r"$\omega$")
        plt.tight_layout()
        plt.savefig(self.name+ ".png")
        plt.close()

        return SSSfw


class Langevin_SF:


     #initializes temperature and parameters
    def __init__(self, T , KX,KY):
        self.name="Langevin_SF"

        self.T=T

        self.alphl=[0.0054342689, 0.00645511652936,0.0085441664872,0.008896935]

        self.alph=self.alphfunc(T)
        self.lam=self.bisection(self.f,3/T,40,170,KX,KY)


    def gamma2(self,kx,ky):
        return 2*np.cos(kx)+4*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2)


    def Sf(self,kx,ky,lam):
        return 3/(lam+(1/self.T)*self.gamma2(kx,ky))

    def f(self,lam,kx,ky):
        curlyN=np.size(kx)
        return np.sum(self.Sf(kx,ky,lam))/curlyN -1

    ##bisection method to solve the large n self consistency equation
    def bisection(self,f,a,b,N,KX,KY):
        '''Approximate solution of f(x)=0 on interval [a,b] by bisection method.

        Parameters
        ----------
        f : function
            The function for which we are trying to approximate a solution f(x)=0.
        a,b : numbers
            The interval in which to search for a solution. The function returns
            None if f(a)*f(b) >= 0 since a solution is not guaranteed.
        N : (positive) integer
            The number of iterations to implement.

        Returns
        -------
        x_N : number
            The midpoint of the Nth interval computed by the bisection method. The
            initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
            midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
            If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
            iteration, the bisection method fails and return None.

        Examples
        --------
        >>> f = lambda x: x**2 - x - 1
        >>> bisection(f,1,2,25)
        1.618033990263939
        >>> f = lambda x: (2*x - 1)*(x - 3)
        >>> bisection(f,0,1,10)
        0.5
        '''
        if f(a,KX,KY)*f(b,KX,KY) >= 0:
            print("Bisection method fails.")
            return None
        a_n = a
        b_n = b
        for n in range(1,N+1):
            m_n = (a_n + b_n)/2
            f_m_n = f(m_n,KX,KY)
            if f(a_n,KX,KY)*f_m_n < 0:
                a_n = a_n
                b_n = m_n
            elif f(b_n,KX,KY)*f_m_n < 0:
                a_n = m_n
                b_n = b_n
            elif f_m_n == 0:
                print("Found exact solution.")
                return m_n
            else:
                print("Bisection method fails.")
                return None
        return (a_n + b_n)/2

    

    ################################
    ################################
    ################################
    ################################

    ###Dynamical structure factor
    #temperature dependent fit parameters for the langevin dynamics
    
    def alphfunc(self,T):
        return np.piecewise(T, [T <= 0.5, (T <= 1.0) & (T>0.5), (T <= 10.0) & (T>1.0), T>10.0], self.alphl)


    
    #dynamic structure fac
    def Dynamical_SF(self, kx,ky,f):
        gam=2*np.cos(kx)+4*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2)
        SP=3/(self.lam+(1/self.T)*gam)
        fq=(self.gamma2(kx,ky)**2)/self.T +self.gamma2(kx,ky)*(self.lam-6/self.T)- 6*self.lam
        fq=self.alph*fq
        return -2*SP*(fq/(f**2+fq**2))

    def momentum_cut_high_symmetry_path(self, latt, Nomegs,Nt_points ):
        omeg_max=1.0
        kpath=latt.High_symmetry_path(Nt_points)
        ##geneerating arrays for imshow of momentum cut
        omegas=np.linspace(0.0001,omeg_max ,Nomegs)
        t=np.arange(0,len(kpath),1)
        t_m,omegas_m=np.meshgrid(t,omegas)
        SSSfw=self.Dynamical_SF(kpath[t_m,0],kpath[t_m,1],omegas_m)
        plt.imshow(SSSfw ,vmax=65 ,origin='lower', aspect='auto')
        Npl2=np.linspace(0,Nomegs,6)
        Npl=np.linspace(0,len(kpath),6)
        om=np.round(np.linspace(0,omeg_max,6),3)
        t=np.round(np.linspace(0,1,6),3)
        plt.colorbar()
        plt.xticks(Npl,t)
        plt.yticks(Npl2,om)
        plt.xlabel(r"$q$")
        plt.ylabel(r"$\omega$")
        plt.tight_layout()
        plt.savefig(self.name+ ".png")
        plt.close()

        return SSSfw
 
        
def main() -> int:
    
    ##########################
    ##########################
    # parameters
    ##########################
    ##########################

    # # #electronic parameters
    J=2*5.17 #in mev
    tp1=568/J #in units of Js\
    tp2=-tp1*108/568 #/tpp1
    ##coupling 
    U=4000/J
    g=100/J
    Kcou=g*g/U
    # fill=0.67 #van hove
    fill=0.5
    
    Machine='FMAC'
    
    
    
    Npoints_int_pre=100 
    save=False

    T=1
    SS=StructureFac_fit_F_w(T , 400, Machine)
    
    kappa=2.0
    alpha=np.pi
    SS2=StructureFac_HT(T, kappa, alpha, 400)
    
    sw=SS.omega_Dynamical(SS.ome)
    sw2=SS2.omega_Dynamical(SS.ome)
    
    plt.plot(SS.ome,sw*SS.ome/T)
    plt.plot(SS2.ome,sw2*SS2.ome/T)
    plt.show()
    plt.plot(SS.ome, sw )
    plt.plot(SS2.ome, sw2 )
    plt.show()
    
    ome2=np.linspace(-10,10,200)
    
    sw=SS.omega_Dynamical(ome2)
    sw2=SS2.omega_Dynamical(ome2)
    
    plt.plot(ome2,sw*ome2/T)
    plt.plot(ome2,sw2*ome2/T)
    plt.show()
    plt.plot(ome2, sw )
    plt.plot(ome2, sw2 )
    plt.show()


    
if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
