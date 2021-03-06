import pyfits
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
# import pylab as pb

class fitsimage:
    """A class for the manipulation of .fits images - in particular for implementing deconvolution."""

    # Opens the original .fits images and stores their contents in appropriate variables for later use. Also
    # initialises some variables which are used throughout the class, in particular the central index of the PSF.
    def __init__(self,imagename,psfname):
        self.imagename = imagename
        self.psfname = psfname
        self.imgHDUlist = pyfits.open("{}".format(self.imagename))
        self.psfHDUlist = pyfits.open("{}".format(self.psfname))
        self.dirtyoriginal = self.imgHDUlist[0].data[0,0,:,:]
        self.dirtyimg = self.imgHDUlist[0].data[0,0,:,:]
        self.psf = self.psfHDUlist[0].data[0,0,:,:]
        self.imgHDUlist.close()
        self.psfHDUlist.close()

        self.cleanmap = np.zeros(self.dirtyimg.shape)
        self.m,self.n = np.unravel_index(self.psf.argmax(), self.psf.shape)

    def CLEAN(self,maxiter=1000,threshold=0.1,gain=0.1):
        """
        A method of the fitsimage class, this implements the Hogbom CLEAN algorithm for deconvolution. This method is
        powerful for the extraction of point sources, but does not perform well on extended sources. Multiscale CLEAN
        is a better approach for such sources. The algorithm only works on the central quadrant of the image. This is
        a result of the manipulation of the dirty beam.
        """
        self.maxiter = maxiter
        self.threshold = threshold
        self.gain = gain

        # The following determines the location of the current maximum pixel and then moves the center of the PSF to
        # that location by rolling the PSF and then zeroing entries which rolled beyond the edges of the array. The
        # rolled PSF is multiplied by a gain value and then subtracted from the dirty image. This process is repeated
        # iteratively until such time as a maximum iteration is reached, or the specified threshold value is reached.
        for iter in range(self.maxiter):
            i,j = np.unravel_index(self.dirtyimg[self.n/2:self.m/2+self.m,self.n/2:self.n/2+self.n].argmax(),
                                   self.dirtyimg[self.m/2:self.m/2+self.m,self.n/2:self.n/2+self.n].shape)
            i,j = i+self.m/2,j+self.n/2

            if self.dirtyimg[i,j]<self.threshold:
                break

            psfmoved = np.roll(np.roll(self.psf,j-self.n,axis=1),i-self.m,axis=0)

            if j-self.n >=0:
                psfmoved[:,0:abs(j-self.n)]=0
            else:
                psfmoved[:,(psfmoved.shape[1]-abs(j-self.n)):]=0
            if i-self.m <=0:
                psfmoved[(psfmoved.shape[0]-abs(i-self.m)):,:]=0
            else:
                psfmoved[:abs(i-self.m),:]=0

            ## OMS: self.dirtyimg[i,j] and self.dirtyimg.max() is the same thing, isn't it?
            ## better use just self.dirtyimg[i,j] then -- also somewhat quicker as
            ## you don't need to recompute the max()
            self.cleanmap[i,j] += self.gain*self.dirtyimg[i,j]
            self.dirtyimg = self.dirtyimg - (self.gain*psfmoved*self.dirtyimg.max())

        if iter+1 == self.maxiter:
            print "Maximum iteration reached."
        else:
            print "Threshold reached. {} iterations performed.".format(iter)

        # The following determines the region which the main lobe of the PSF occupies by determining the indices of
        # the elements inside the second zero of the PSF. CAUTION: this works in this scenario, but kk may need to be
        # reduced in certain cases to ensure the gaussian can be fit.

        ii,jj = 0,0
        psfpos = (self.psf>0)

        while True:
            if psfpos[self.m+ii,self.n]==1:
                ii += 1
            else:
                if psfpos[self.m+ii+1,self.n]!=1:
                    ii += 1
                else:
                    break

        while True:
            if psfpos[self.m,self.n+jj]==1:
                jj += 1
            else:
                if psfpos[self.m,self.n+jj+1]!=1:
                    jj += 1
                else:
                    break

        kk = int(np.sqrt(jj**2+ii**2))

        # Following creates row and column values of interest for the central PSF lobe and then selects those values
        # from the PSF using np.where. Additionally, the inputs for the fitting are created by reshaping the x,y,
        # and z data into columns.
        y = np.arange(self.m-kk,self.m+kk,1)
        x = np.arange(self.n-kk,self.n+kk,1)

        z = np.where(self.psf[self.m-kk:self.m+kk,self.n-kk:self.n+kk]>0,1,0)*self.psf[self.m-kk:self.m+kk,
                                                                                 self.n-kk:self.n+kk]

        gridx, gridy = np.meshgrid(x-self.m,y-self.n)
        xyz = np.column_stack((gridx.reshape(-1,1),gridy.reshape(-1,1),z.reshape(-1,1,order="C")))

        # Elliptical gaussian which can be fit to the central lobe of the PSF. xy must be an Nx2 array consisting of
        # pairs of row and column values for the region of interest. This is obtained from kk above.
        def ellipgauss(xy,A,xshift,xsigma,yshift,ysigma):
            return A*np.exp(-1*(((xy[:,1]-xshift)**2)/(2*(xsigma**2))+((xy[:,0]-yshift)**2)/(2*(ysigma**2))))

        # This command from scipy performs the fitting of the 2D gaussian, and returns the optimal coefficients in opt.
        opt = curve_fit(ellipgauss, xyz[:,0:2],xyz[:,2],(1,0,1,0,1))[0]

        # Following create the data for the new images. The cleanbeam has to be reshaped to reclaim it in 2D.
        self.cleanbeam = ellipgauss(xyz[:,0:2],opt[0],opt[1],opt[2],opt[3],opt[4]).reshape(gridx.shape,order="C")
        self.cleancomp = fftconvolve(self.cleanmap,self.cleanbeam,mode='same')
        self.cleanimg = self.cleancomp + self.dirtyimg
        self.residual = self.dirtyoriginal - self.cleancomp

        # TEST CODE - CHECK CONTOUR MAPS
        # pb.contour(gridx,gridy,self.cleanbeam)
        # pb.show()
        # pb.contour(gridx,gridy,ellipgauss(xyz[:,0:2],1,0,20,0,20).reshape(gridx.shape))
        # pb.show()
        # pb.contour(xyz[:,0].reshape(gridx.shape),xyz[:,1].reshape(gridx.shape),xyz[:,2].reshape(gridx.shape,order="C"))
        # pb.show()

    def saveimage(self):
        """
        This method simply saves the CLEAN components, residual and restored image.
        """
        data = np.zeros([1,1,self.dirtyimg.shape[0],self.dirtyimg.shape[1]])
        data[0,0,:,:] += self.residual
        newfile = pyfits.PrimaryHDU(data,self.imgHDUlist[0].header)
        newfile.writeto("residual_"+"{}".format(self.imagename))

        data = np.zeros([1,1,self.dirtyimg.shape[0],self.dirtyimg.shape[1]])
        data[0,0,512:(512+1024),512:(512+1024)] += self.cleanimg[512:(512+1024),512:(512+1024)]
        newfile = pyfits.PrimaryHDU(data[0],self.imgHDUlist[0].header)
        newfile.writeto("restored_"+"{}".format(self.imagename))

        data = np.zeros([1,1,self.dirtyimg.shape[0],self.dirtyimg.shape[1]])
        data[0,0,:,:] += self.cleanmap
        newfile = pyfits.PrimaryHDU(data,self.imgHDUlist[0].header)
        newfile.writeto("cleancomp_"+"{}".format(self.imagename))

img = fitsimage("KAT7_1445_1x16_12h.ms.CORRECTED_DATA.channel.1ch_nobright.fits","KAT7_1445_1x16_12h.ms.psf.channel"
                                                                                 ".1ch.fits")
img.CLEAN(1000,0.01,0.1)
img.saveimage()



