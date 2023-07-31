import numpy as np
import time
import Lattice
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys 


#Hack for now, search for: choose dispersion wherever we want to change dispersion
class Dispersion_TB_single_band:

    def __init__(self, hop, fill,size_E,Machine=None):

        self.hop=hop
        if Machine is None:
            self.Machine = ""
        else:
            self.Machine = Machine 
        

        #GRIDS AND INTEGRATION MEASURES
        print("started calculating filling for chemical potential and dispersion parameters TB_single_band..")

        self.Npoi_ints=1000 # 1200 for accurate calculation, 400 for quick
        self.latt_int=Lattice.TriangLattice(self.Npoi_ints, False, Machine) #temp grid for integrating and getting filling
        
        # [KX,KY]=self.latt_int.Generate_lattice()
        [self.KX,self.KY]=self.latt_int.read_lattice()
        Vol_rec=self.latt_int.Vol_BZ()
        self.ds=Vol_rec/np.size(self.KX)
        self.ds2=1/np.size(self.KX)

        self.energy_k = self.Disp(self.KX,self.KY)
        
      
        Wbdw=np.max(self.energy_k)-np.min(self.energy_k)

        #DISPERSION PARAMS
        self.bandmin=np.min(self.energy_k)
        self.bandmax=np.max(self.energy_k)
        self.bandwidth=Wbdw

        #getting chempot for filling
        [self.nn,self.earr,Dos]=self.DOS(size_E)
        self.Dos=Dos/Vol_rec
        indemin=np.argmin((self.nn-fill)**2)
        mu=self.earr[indemin]
        # mu=24
        self.mu=mu
        self.name="lattice_disp"


        #validating actual filling
        self.EF= self.mu-self.bandmin #fermi energy from the bottom of the band
        self.energy_k_mu = self.Disp_mu(self.KX,self.KY)
        nu_fill=np.sum(np.heaviside(-self.energy_k_mu,1)*self.ds)/Vol_rec
        print("finished calculating filling for chemical potential")
        print("Filling: {f} .... chemical potential: {m}".format(f=nu_fill,m=mu))
        
        self.filling=nu_fill
        self.target_fill=fill
        
        [self.dens2,self.bins,self.valt,self.f2 ]=self.DOS_2()

    
    def Disp(self,kx,ky):
        [tp1,tp2]=self.hop
        ed=-tp1*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
        ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky))+4*np.cos(3*(kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
        return ed

    def Disp_mu(self,kx,ky):

        [tp1,tp2]=self.hop
        ed=-tp1*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
        ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky))+4*np.cos(3*(kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
        return ed-self.mu


    def Fermi_Vel(self,kx,ky):

        [tp1,tp2]=self.hop
        sq3y2=np.sqrt(3)*ky/2
        sq3y=np.sqrt(3)*ky
        vx=-tp1*(-2*np.cos(sq3y2)*np.sin(kx/2)-2*np.sin(kx)) +6*tp2*np.cos(sq3y2)*np.sin(3*kx/2)
        vy=2*np.sqrt(3)*tp1*np.cos(kx/2)*np.sin(sq3y2)-2*np.sqrt(3)*tp2*(-np.cos(3*kx/2)*np.sin(sq3y2)-np.sin(sq3y))
        return [vx,vy]
  


    #if used in the middle of plotting will close the plot
    def FS_contour(self, Np):
        s=time.time()
        print('starting contour.....')
        y = np.linspace(-4,4, 4603)
        x = np.linspace(-4.1,4.1, 4603)
        X, Y = np.meshgrid(x, y)
        Z = self.Disp(X,Y)  #choose dispersion
        c= plt.contour(X, Y, Z, levels=[self.mu],linewidths=3, cmap='summer');
        plt.close()
        #plt.show()
        numcont=np.shape(c.collections[0].get_paths())[0]
        
        if numcont==1:
            v = c.collections[0].get_paths()[0].vertices
        else:
            contourchoose=0
            v = c.collections[0].get_paths()[0].vertices
            sizecontour_prev=np.prod(np.shape(v))
            for ind in range(1,numcont):
                v = c.collections[0].get_paths()[ind].vertices
                sizecontour=np.prod(np.shape(v))
                if sizecontour>sizecontour_prev:
                    contourchoose=ind
            v = c.collections[0].get_paths()[contourchoose].vertices
        NFSpoints=Np
        xFS_dense = v[::int(np.size(v[:,1])/NFSpoints),0]
        yFS_dense = v[::int(np.size(v[:,1])/NFSpoints),1]
        e=time.time()
        print('finished contour.....', e-s)
        return [xFS_dense,yFS_dense]
    
    def FS_contour_HT(self, Np):
        s=time.time()
        print('starting high res contour.....')
        sizegrid=int(Np/3)
        y = np.linspace(-3,3, sizegrid) #3 is able to capture half filling FS
        x = np.linspace(-3,3, sizegrid) #3 is able to capture half filling FS
        X, Y = np.meshgrid(x, y)
        Z = self.Disp(X,Y)  #choose dispersion
        c= plt.contour(X, Y, Z, levels=[self.mu],linewidths=3, cmap='summer');
        plt.close()
        #plt.show()
        numcont=np.shape(c.collections[0].get_paths())[0]
        print('number of sheets.....',numcont)
        if numcont==1:
            v = c.collections[0].get_paths()[0].vertices
        else:
            contourchoose=0
            v = c.collections[0].get_paths()[0].vertices
            sizecontour_prev=np.prod(np.shape(v))
            for ind in range(1,numcont):
                v = c.collections[0].get_paths()[ind].vertices
                sizecontour=np.prod(np.shape(v))
                if sizecontour>sizecontour_prev:
                    contourchoose=ind
            v = c.collections[0].get_paths()[contourchoose].vertices
        NFSpoints=Np
        print('contour size and intended span.....',np.size(v[:,1]),NFSpoints,int(np.size(v[:,1])/NFSpoints))
        xFS_dense = v[:,0]
        yFS_dense = v[:,1]
        e=time.time()
        print('finished high res contour.....', e-s)
        return [xFS_dense,yFS_dense]
    
    def FS_contour_HT2(self,Nangles):
        s=time.time()
        print('starting high res contour.....')
        Np=(2**9+1)*40
        print("attempting countour of size",Np)
        sizegrid=int(Np)
        y = np.linspace(-3,3, sizegrid) #3 is able to capture half filling FS
        x = np.linspace(-3,3, sizegrid) #3 is able to capture half filling FS
        X, Y = np.meshgrid(x, y)
        Z = self.Disp(X,Y)  #choose dispersion
        c= plt.contour(X, Y, Z, levels=[self.mu],linewidths=3, cmap='summer');
        plt.close()
        #plt.show()
        numcont=np.shape(c.collections[0].get_paths())[0]
        print('number of sheets.....',numcont)
        if numcont==1:
            v = c.collections[0].get_paths()[0].vertices
        else:
            contourchoose=0
            v = c.collections[0].get_paths()[0].vertices
            sizecontour_prev=np.prod(np.shape(v))
            for ind in range(1,numcont):
                v = c.collections[0].get_paths()[ind].vertices
                sizecontour=np.prod(np.shape(v))
                if sizecontour>sizecontour_prev:
                    contourchoose=ind
            v = c.collections[0].get_paths()[contourchoose].vertices

        print('contour size before interpolation.....',np.size(v[:,1]))
        xFS_dense = v[:,0]
        yFS_dense = v[:,1]
        e=time.time()
        print('finished high res contour.....', e-s)
        
        
        angdens=np.arctan2(yFS_dense,xFS_dense)
        #sorting the arrays traversing them from -pi to pi
        list_ang, listx = zip(*sorted(zip(list(angdens), list(xFS_dense) )))
        list_ang, listy = zip(*sorted(zip(list(angdens), list(yFS_dense) )))
        angdens=np.array(list_ang)
        xFS_dense=np.array(listx)
        yFS_dense=np.array(listy)
        
        
        fx = interp1d(angdens, xFS_dense)
        fy = interp1d(angdens, yFS_dense)
        ang=np.linspace(np.min(angdens), np.max(angdens),Nangles)
        print("range of angles from marching squares",np.min(angdens), np.max(angdens))
        
        return [fx(ang),fy(ang)]
    
    def deltad(self,x, epsil):
        return (1/(np.pi*epsil))/(1+(x/epsil)**2)

    def DOS(self,size_E):

        #DOMAIN OF THE DOS
        minE=self.bandmin-0.001*self.bandwidth
        maxE=self.bandmax+0.001*self.bandwidth
        earr=np.linspace(minE,maxE,size_E)

        
        Vol_rec=self.latt_int.Vol_BZ()


        
        #parameter for delta func approximation
        epsil=0.02*self.bandwidth

        ##DOS 
        Dos=[]
        for i in earr:
            dosi=np.sum(self.deltad(self.energy_k-i,epsil))*self.ds
            Dos.append(dosi)
            
        de=earr[1]-earr[0]
        Dos=np.array(Dos)
        print("norm of Dos,", np.sum(Dos)*de, self.latt_int.VolBZ)
        
        #FILLING FOR EACH CHEMICAL POTENTIAL
        ndens=[]
        for mu_ind in range(size_E):
            
            N=np.trapz(Dos[0:mu_ind])*de
            ndens.append(N)
        nn=np.array(ndens)
        nn=nn/nn[-1]
        
        print("sum of the hist, normed?", np.sum(Dos)*de)

        return [nn,earr,Dos]
    
    def DOS_2(self):
        Ene_BZ=self.energy_k
        
        eps_l=[]
        
        eps_l.append(np.mean( np.abs( np.diff( Ene_BZ.flatten() )  ) )/2)
        eps_a=np.array(eps_l)
        eps=np.min(eps_a)*30
        
        mmin=np.min(Ene_BZ)
        mmax=np.max(Ene_BZ)
        NN=int((mmax-mmin)/eps)+int((int((mmax-mmin)/eps)+1)%2) #making sure there is a bin at zero energy
        binn=np.linspace(mmin,mmax,NN+1)
        valt=np.zeros(NN)

        val_p,bins_p=np.histogram(Ene_BZ.flatten(), bins=binn,density=True)
        valt=valt+val_p

       
        bins=(binn[:-1]+binn[1:])/2
        
        #taking into account spin
        valt=valt*2
        
        f2 = interp1d(binn[:-1],valt, kind='cubic',bounds_error=False, fill_value=0)
        de=(bins[1]-bins[0])
        print("sum of the hist, normed?", np.sum(valt)*de)
        
        
        
        #FILLING FOR EACH CHEMICAL POTENTIAL
        ndens=[]
        for mu_ind in range(NN):
            N=np.trapz(valt[0:mu_ind])*de
            ndens.append(N)
        nn=np.array(ndens)
        dens2=nn/nn[-1]
        
        

        return [dens2,bins,valt,f2 ]

    #######random functions
    def nf(self, e, T):
        rat=np.max(np.abs(e/T))
        if rat<700:
            return 1/(1+np.exp( e/T ))
        else:
            return np.heaviside(-e,0.5)

    def nb(self, e, T):
        rat=np.max(np.abs(e/T))
        if rat<700:
            return 1/(np.exp( e/T )-1)
        else:
            return -np.heaviside(-e,0.5)
        
    def be_nf(self, e, T):
        rat=np.max(np.abs(e/T))
        if rat<700:
            x=e/T
            return x/(np.exp(x)+1)
        else:
            return np.heaviside(-e,0.5)

    def be_nb(self, e, T):
        rat=np.max(np.abs(e/T))
        if rat<700:
            x=e/T
            expr=x/(np.exp(x)-1)
            problems=np.where(np.isnan(expr))[0]
            expr[problems]=1
            return expr
        else:
            return -(e/T)*np.heaviside(-e,0.5)
    
    def PlotFS(self, lat):
        l=Lattice.TriangLattice(100,False )
        Npoi=1000
        Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
        VV=np.array(Vertices_list+[Vertices_list[0]])
        [KxFS,KyFS]=self.FS_contour(Npoi)
        plt.plot(VV[:,0], VV[:,1],c='k')
        plt.scatter(KxFS, KyFS, s=1, c='r')
        plt.show()


class Dispersion_circ:

    def __init__(self, hop, fill, Machine=None):

        self.hop=hop
        
        if Machine is None:
            self.Machine = ""
        else:
            self.Machine = Machine 

        #GRIDS AND INTEGRATION MEASURES
        print("started calculating filling for chemical potential and dispersion parameters _circ.. ")

        self.Npoi_ints=1200
        self.latt_int=Lattice.TriangLattice(self.Npoi_ints, True,Machine) #temp grid for integrating and getting filling
        
        # [KX,KY]=l.Generate_lattice()
        [KX,KY]=self.latt_int.read_lattice()
        Vol_rec=self.latt_int.Vol_BZ()
        ds=Vol_rec/np.size(KX)

        energy_k = self.Disp(KX,KY)
        
      
        Wbdw=np.max(energy_k)-np.min(energy_k)

        #DISPERSION PARAMS
        self.bandmin=np.min(energy_k)
        self.bandmax=np.max(energy_k)
        self.bandwidth=Wbdw

        #getting chempot for filling
        [nn,earr,Dos]=self.DOS(size_E=500, Npoi_ints=1200)
        indemin=np.argmin((nn-fill)**2)
        mu=earr[indemin]
        self.mu=mu
        self.name="parabolic_disp"


        #validating actual filling
        self.EF= self.mu-self.bandmin #fermi energy from the bottom of the band
        energy_k_mu = self.Disp_mu(KX,KY)
        nu_fill=np.sum(np.heaviside(-energy_k_mu,1)*ds)/Vol_rec
        print("finished calculating filling for chemical potential")
        print("Filling: {f} .... chemical potential: {m}".format(f=nu_fill,m=mu))
        self.filling=nu_fill

    
    def Disp(self,kx,ky):
        
        [tp1,tp2]=self.hop
        DD2=0.5*(3*tp1+9*tp2) #multiplied by length squared
        ed=0.5*DD2*(kx**2+ky**2)
        return ed

    def Disp_mu(self,kx,ky):

        [tp1,tp2]=self.hop
        DD2=0.5*(3*tp1+9*tp2) #multiplied by length squared
        ed=0.5*DD2*(kx**2+ky**2)
        return ed-self.mu


    def Fermi_Vel(self,kx,ky):

        [tp1,tp2]=self.hop
        DD2=0.5*(3*tp1+9*tp2) #multiplied by length squared
        vx=DD2*kx
        vy=DD2*ky

        return [vx,vy]
  
    #if used in the middle of plotting will close the plot
    def FS_contour2(self, Np):
        theta = np.linspace(-np.pi,np.pi, Np)
        [tp1,tp2]=self.hop
        m=2/(3*tp1+9*tp2)
        
        kf=np.sqrt(2*self.EF*m)

        xFS_dense=kf*np.cos(theta)
        yFS_dense=kf*np.sin(theta)
        return [xFS_dense,yFS_dense]
    
    def FS_contour(self, Np):
        y = np.linspace(-4,4, 10603)
        x = np.linspace(-4,4, 10603)
        X, Y = np.meshgrid(x, y)
        Z = self.Disp_mu(X,Y)  
        c= plt.contour(X, Y, Z, levels=[0]);
        plt.close()
        #plt.show()
        numcont=np.shape(c.collections[0].get_paths())[0]
        v = c.collections[0].get_paths()[0].vertices
        
        # if numcont==1:
        #     v = c.collections[0].get_paths()[0].vertices
        # else:
        #     contourchoose=0
        #     v = c.collections[0].get_paths()[0].vertices
        #     sizecontour_prev=np.prod(np.shape(v))
        #     for ind in range(1,numcont):
        #         v = c.collections[0].get_paths()[ind].vertices
        #         sizecontour=np.prod(np.shape(v))
        #         if sizecontour>sizecontour_prev:
        #             contourchoose=ind
        #     v = c.collections[0].get_paths()[contourchoose].vertices
        NFSpoints=Np
        chunksize=int(np.size(v[:,0])/NFSpoints)


        xFS_dense = v[::chunksize,0]
        yFS_dense = v[::chunksize,1]
        
        return [xFS_dense,yFS_dense]

    def deltad(self,x, epsil):
        return (1/(np.pi*epsil))/(1+(x/epsil)**2)

    def DOS(self,size_E, Npoi_ints):

        #DOMAIN OF THE DOS
        minE=self.bandmin-0.001*self.bandwidth
        maxE=self.bandmax+0.001*self.bandwidth
        earr=np.linspace(minE,maxE,size_E)

        #INTEGRATION LATTICE
        latt_int=Lattice.TriangLattice(Npoi_ints, False, self.Machine) #temp grid for integrating and getting filling
        
        # [KX,KY]=l.Generate_lattice()
        [KX,KY]=latt_int.read_lattice()
        Vol_rec=latt_int.Vol_BZ()
        ds=Vol_rec/np.size(KX)

        #DISPERSION FOR INTEGRAL
        energy_k = self.Disp(KX,KY)
        #parameter for delta func approximation
        epsil=0.002*self.bandwidth

        ##DOS 
        Dos=[]
        for i in earr:
            dosi=np.sum(self.deltad(energy_k-i,epsil))*ds
            Dos.append(dosi)

        #FILLING FOR EACH CHEMICAL POTENTIAL
        ndens=[]
        for mu_ind in range(size_E):
            de=earr[1]-earr[0]
            N=np.trapz(Dos[0:mu_ind])*de
            ndens.append(N)
        nn=np.array(ndens)
        nn=nn/nn[-1]

        return [nn,earr,Dos]

    #######random functions
    def nf(self, e, T):
        rat=np.abs(np.max(e/T))
        if rat<700:
            return 1/(1+np.exp( e/T ))
        else:
            return np.heaviside(-e,0.5)

    def nb(self, e, T):
        rat=np.abs(np.max(e/T))
        if rat<700:
            return 1/(np.exp( e/T )-1)
        else:
            return -np.heaviside(-e,0.5)

    def PlotFS(self, lat):
        l=Lattice.TriangLattice(100,False )
        Npoi=1000
        Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
        VV=np.array(Vertices_list+[Vertices_list[0]])
        [KxFS,KyFS]=self.FS_contour(Npoi)
        plt.plot(VV[:,0], VV[:,1],c='k')
        plt.scatter(KxFS, KyFS, s=1, c='r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
        
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
    size_E=4000
    

    ##########################
    ##########################
    # Geometry/Lattice
    ##########################
    ##########################

    save=True
    Machine='FMAC'
    
    # ##########################
    # ##########################
    # # Fermi surface and structure factor
    # ##########################
    # ##########################

    ed=Dispersion_TB_single_band([tp1,tp2],fill,size_E,Machine)
    [dens2,bins,valt,f2 ]=[ed.dens2,ed.bins,ed.valt,ed.f2 ]
    [nn,earr,Dos]=[ed.nn,ed.earr,ed.Dos]
    mu=ed.mu
    plt.plot(earr,Dos)
    plt.show()
    
    
if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
