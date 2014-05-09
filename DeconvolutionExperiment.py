import pyfits
import numpy as np
import time
import pylab as pb
from scipy import ndimage
from scipy.signal import convolve
from scipy.signal import fftconvolve
import math
import multiprocessing as mp
from scipy.optimize import curve_fit

def linearconv(C0, wavelet, sliceind, queue, rorc="row"):
    if rorc == "row":
        for i in range(C0.shape[0]):
            C0[i,:] = ndimage.convolve1d(C0[i,:],wavelet)
                # linearconvker(C0[i,:], wavelet, scale)

    elif rorc == "column":
        for i in range(C0.shape[1]):
            C0[:,i] = ndimage.convolve1d(C0[:,i],wavelet)
                # linearconvker(C0[:,i], wavelet, scale)

    queue.put([sliceind,C0])

# def linearconvker(C0,wavelet,scale):
#
#     paddingwidth = ((2**scale)-1)
#     paddingblock = paddingwidth + 1
#
#     waveletwidth = (wavelet.shape[0]-1)*paddingwidth+wavelet.shape[0]
#     wavelethalfwidth = (waveletwidth-1)//2
#
#     extendedC0 = np.empty([C0.shape[0]+2*wavelethalfwidth])
#
#     extendedC0[0:wavelethalfwidth] = C0[wavelethalfwidth-1::-1]
#     extendedC0[wavelethalfwidth:-wavelethalfwidth] = C0
#     extendedC0[-wavelethalfwidth:] = C0[-1:-wavelethalfwidth-1:-1]
#
#     out1 = np.empty(C0.shape)
#
#     for i in range(out1.shape[0]):
#         out1[i] = np.sum(wavelet*extendedC0[i:i+waveletwidth:paddingblock])
#
#     return out1

class FitsImage:
    """A class for the manipulation of .fits images - in particular for implementing deconvolution."""

    def __init__(self, image_name, psf_name):
        """
        Opens the original .fits images specified by image_name and psf_name.
        """
        self.image_name = image_name
        self.psf_name = psf_name

        self.img_HDU_list = pyfits.open("{}".format(self.image_name))
        self.psf_HDU_list = pyfits.open("{}".format(self.psf_name))

        self.dirty_data = self.img_HDU_list[0].data[0,0,:,:]
        self.psf = self.psf_HDU_list[0].data[0,0,:,:]

        self.img_HDU_list.close()
        self.psf_HDU_list.close()

    def altatrous(self, C0, scale, corecount):
        processes = []
        queue = mp.Queue()

        # wavelet = (1./16)*np.array([1,4,6,4,1])

        spline = (1./16)*np.array([1,4,6,4,1])
        wavelet = np.zeros(4*((2**scale)-1)+spline.shape[0])
        wavelet[0::((2**scale))] = spline

        for i in range(corecount):
            process = mp.Process(target=linearconv,args=(C0[i*C0.shape[0]//corecount:(i+1)*C0.shape[
                0]//corecount,:],wavelet,i,queue,'row',))
            process.start()
            processes.append(process)

        for i in range(corecount):
            slicenum, C0[slicenum*C0.shape[0]//corecount:(slicenum+1)*C0.shape[0]//corecount,:] = queue.get()

        for i in range(corecount):
            process = mp.Process(target=linearconv,args=(C0[:,i*C0.shape[0]//corecount:(i+1)*C0.shape[
                0]//corecount],wavelet,i,queue,'column',))
            process.start()
            processes.append(process)

        for i in range(corecount):
            slicenum, C0[:,slicenum*C0.shape[0]//corecount:(slicenum+1)*C0.shape[0]//corecount] = queue.get()

        return C0

    def IUWT(self, in1, scale_count):
        """
        This function replicates the functionality of stardec_g2.m in Arwa's original code. In particular it makes use of
        the a trous algorithm for the decomposition of the image into wavelets. The output of this function is a matrix
        containing the wavelet coefficients. This is the isotropic undecimated wavelet transform. Accepts the following
        parameters:

        in1                 (no default): Image, on which the decomposition is to be performed.
        scale_count         (no default): Maximum scale to be considered.

        """
        filter = (1./16)*np.array([1,4,6,4,1])      # Filter for use in the a trous algorithm.
        filteredimgsize = in1.shape         # Stores the size of the filtered image.

        C0 = in1                            # Sets the initial value to be the image.
        coeffs = np.zeros([scale_count+1,filteredimgsize[0],filteredimgsize[1]]) # Initialises an zero array to store the
                                                                                 # coefficients.

        # The following loop calculates the wavelet coefficients based on values of a trous algorithm for C0 and C. C0 is
        # resassigned the value of C on each loop.

        for i in range(scale_count):
            C = self.atrous(filter,C0,i)                 # Approximation coefficients.
            C1 = self.atrous(filter,C,i)
            detailcoeffs = C0-C1                    # Detail coefficients.
            coeffs[i,:,:] = detailcoeffs            # Store the detail coefficients.
            C0 = C

        coeffs[scale_count,:,:] = C      # The coeffs at value scale_count are assigned to the last value of C.
        return coeffs

    def atrous(self, filter, C0, scale):
        """
        The following is an implementation of the a trous algorithm for wavelet decomposition. Accepts the following
        parameters:

        filter      (no default): The filter which is applied to the components of the transform.
        C0          (no default): The current array being for decomposition.
        scale       (no default): The scale for which the decomposition is being carried out.

        """
        C0size = C0.shape

        v = np.empty(C0size)
        C1 = np.empty(C0size)

        v = filter[2]*C0

        v[(2**(scale+1)):,:] += filter[0]*C0[:-(2**(scale+1)),:]
        v[:(2**(scale+1)),:] += filter[0]*C0[(2**(scale+1))-1::-1,:]

        v[(2**scale):,:] += filter[1]*C0[:-(2**scale),:]
        v[:(2**scale),:] += filter[1]*C0[(2**scale)-1::-1,:]

        v[:-(2**scale),:] += filter[3]*C0[(2**scale):,:]
        v[-(2**scale):,:] += filter[3]*C0[:-(2**scale)-1:-1,:]

        v[:-(2**(scale+1)),:] += filter[4]*C0[(2**(scale+1)):,:]
        v[-(2**(scale+1)):,:] += filter[4]*C0[:-(2**(scale+1))-1:-1,:]

        C1 = filter[2]*v

        C1[:,(2**(scale+1)):] += filter[0]*v[:,:-(2**(scale+1))]
        C1[:,:(2**(scale+1))] += filter[0]*v[:,(2**(scale+1))-1::-1]

        C1[:,(2**scale):] += filter[1]*v[:,:-(2**scale)]
        C1[:,:(2**scale)] += filter[1]*v[:,(2**scale)-1::-1]

        C1[:,:-(2**scale)] += filter[3]*v[:,(2**scale):]
        C1[:,-(2**scale):] += filter[3]*v[:,:-(2**scale)-1:-1]

        C1[:,:-(2**(scale+1))] += filter[4]*v[:,(2**(scale+1)):]
        C1[:,-(2**(scale+1)):] += filter[4]*v[:,:-(2**(scale+1))-1:-1]

        return C1

    def saveImage(self, fits_header, fits_data, output_file_name):
        """
        This method simply saves a .fits file.
        """
        save_data = np.zeros([1,1,fits_data.shape[0],fits_data.shape[1]])
        save_data[0,0,:,:] = fits_data
        save_file = pyfits.PrimaryHDU(save_data,fits_header)
        save_file.writeto("{}.fits".format(output_file_name),clobber=True)

if __name__ == "__main__":
    test_image = FitsImage("3C147.fits","3C147_PSF.fits")
    t1 = time.time()
    # coeffs = test_image.atrous((1./16)*np.array([1,4,6,4,1]),test_image.dirty_data,2)
    t2 = time.time()
    t = t2-t1
    # print coeffs
    # print t

    t1 = time.time()
    # coeffs = test_image.altatrous(test_image.dirty_data,2,4)
    t2 = time.time()
    t = t2-t1
    # print coeffs
    # print t

def beamfit(psf,psfheader):
    """
    The following contructs a restoring beam from the psf. This is accoplished by fitting a Gaussian to the central
    lobe. Accpets the following paramters:

    psf     (no default):   Array containing the psf for the image in question.
    """
    psf_max_location = np.unravel_index(np.argmax(psf), psf.shape)

    threshold_psf = np.where(psf>0.01,psf,0)

    labelled_psf, labels = ndimage.label(threshold_psf)

    # primary_beam = np.where(labelled_psf==labelled_psf[psf_max_location],1,0)*threshold_psf
    labelled_primary_beam = np.where(labelled_psf==labelled_psf[psf_max_location],1,0)

    extracted_primary_beam = threshold_psf[ndimage.find_objects(labelled_primary_beam)[0]]
    # extracted_primary_beam = extracted_primary_beam/np.max(extracted_primary_beam)

    extracted_primary_beam_shape = extracted_primary_beam.shape
    extracted_primary_beam_max_location = np.unravel_index(np.argmax(extracted_primary_beam),extracted_primary_beam_shape)

    # Following creates row and column values of interest for the central PSF lobe and then selects those values
    # from the PSF using np.where. Additionally, the inputs for the fitting are created by reshaping the x,y,
    # and z data into columns.

    y = np.arange(-extracted_primary_beam_max_location[0],-extracted_primary_beam_max_location[
        0]+extracted_primary_beam_shape[0],1)
    x = np.arange(-extracted_primary_beam_max_location[1],-extracted_primary_beam_max_location[
        1]+extracted_primary_beam_shape[1],1)
    z = extracted_primary_beam

    pb.imshow(z)
    pb.show()

    gridx, gridy = np.meshgrid(x,-y)

    xyz = np.column_stack((gridx.reshape(-1,1),gridy.reshape(-1,1),z.reshape(-1,1,order="C")))

    # Elliptical gaussian which can be fit to the central lobe of the PSF. xy must be an Nx2 array consisting of
    # pairs of row and column values for the region of interest. This is obtained from kk above.

    def ellipgauss(xy,A,xsigma,ysigma,theta):
        # (xy,A,a,b,c)
        # return A*np.exp(-1*(a*(xy[:,0]**2)+b*xy[:,1]*xy[:,0]+c*(xy[:,1]**2)))
        # (xy,A,xshift,xsigma,yshift,ysigma)
        return A*np.exp(-1*(((xy[:,1]*np.cos(theta)-xy[:,0]*np.sin(theta))**2)/(2*(xsigma**2))+(xy[:,1]*np.sin(theta)+(
            xy[:,0]*np.cos(theta))**2)/(2*(ysigma**2))))

    # This command from scipy performs the fitting of the 2D gaussian, and returns the optimal coefficients in opt.

    opt = curve_fit(ellipgauss, xyz[:,0:2],xyz[:,2],(1,1,1,0))[0]

    # Following create the data for the new images. The cleanbeam has to be reshaped to reclaim it in 2D.

    cleanbeam = ellipgauss(xyz[:,0:2],opt[0],opt[1],opt[2],opt[3]).reshape(gridx.shape,order="C")

    rotationangle = math.degrees(opt[3])
    FWHMmajor = 2*np.sqrt(2*np.log(2))*max(opt[1],opt[2])*psfheader[0].header['CDELT1']
    FWHMminor = 2*np.sqrt(2*np.log(2))*min(opt[1],opt[2])*psfheader[0].header['CDELT2']

    beamparams = [rotationangle,abs(FWHMmajor),abs(FWHMminor)]

    maja = 1

    return cleanbeam, beamparams

ans, params = beamfit(test_image.psf,test_image.psf_HDU_list)
print params
pb.imshow(ans,aspect="equal",interpolation="bicubic")
pb.show()

    # spline = (1./16)*np.array([1,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1])

    # test_image.saveImage(test_image.img_HDU_list[0].header, test_image.dirty_data, "DIRTY_COPY")

    # result = test_image.dirty_data
    #
    # result = test_image.dirty_data

    # WEIRD behaviour here - for equivalent scale = 1 decomposition, filter has one zero padding. For scale 2,
    # filter has 3 zero padding. For scale 3 filter has 7 zero padding - behaviour appears to be ((2**scale) - 1).
    #
    # t1 = time.time()
    # for i in range(result.shape[0]):
    #     result[i,:] = ndimage.convolve1d(result[i,:],spline,mode="reflect")
    # for i in range(result.shape[0]):
    #     result[:,i] = ndimage.convolve1d(result[:,i],spline,mode="reflect")
    # t2 = time.time()
    # print result
    # print t2-t1






    #
    # # pb.imshow(result)
    # # pb.show()
    # t2 = time.time()
    # t = t2-t1
    # print result
    # print t

#     t1 = time.time()
#     for j in range(1):
#         for i in range(result.shape[0]):
#             result[i,:] = np.convolve(result[i,:],(1./16)*np.array([1,0,4,0,6,0,4,0,1]),mode='same')
#         for i in range(result.shape[0]):
#             result[:,i] = np.convolve(result[:,i],(1./16)*np.array([1,0,4,0,6,0,4,0,1]),mode='same')
#
#     # pb.imshow(result)
#     # pb.show()
#     t2 = time.time()
#     t = t2-t1
#     print result
#     print t
#
#     t1 = time.time()
#     for j in range(1):
#         for i in range(result.shape[0]):
#             result[i,:] = ndimage.convolve1d(result[i,:],(1./16)*np.array([1,4,6,4,1]))
#         for i in range(result.shape[0]):
#             result[:,i] = ndimage.convolve1d(result[:,i],(1./16)*np.array([1,4,6,4,1]))
#
#     # pb.imshow(result)
#     # pb.show()
#     t2 = time.time()
#     t = t2-t1
#     print result
#     print t
#
# def myconv(in1,in2):
#     out1 = []
#
#     outer_limit = in1.shape[0]//2
#     inner_limit = in2.shape[0]//2
#
#     for i in range(in2.shape[0]//2 - 1, in1.shape[0] - in2.shape[0]//2):
#         out1.append(np.sum(in1[i - inner_limit + 1:i + inner_limit + 1]*in2))
#
#     return out1
#
# a = np.array([1,2,3,4,5,6,7,8])
# b = np.array([1,1,1,1])
#
# print 'hello'
# print myconv(a,b)
# print ndimage.convolve1d(a,b)

    # t1 = time.time()

    # fftfilter = np.fft.fft((1./16)*np.array([1,0,4,0,6,0,4,0,1]))
    # for j in range(1):
    #     for i in range(result.shape[0]):
    #         result[i,:] = np.fft.fftshift(np.fft.ifft((np.fft.fft(result[i,:])*fftfilter)))
    #     for i in range(result.shape[0]):
    #         result[:,i] = np.fft.fftshift(np.fft.ifft((np.fft.fft(result[:,i])*fftfilter)))
    #
    # # pb.imshow(result)
    # # pb.show()
    # t2 = time.time()
    # t = t2-t1
    # print result
    # print t


    # t1 = time.time()
    # ans = fftconvolve(test_image.dirty_data.astype(np.float32),(1./256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,
    #                                                                                                          24,6],[4,16,
    #                                                                                                             24,16,
    #                                                                                                       4],[1,4,
    #                                                                                                             6,4,
    #                                                                                                        1]],
    #                                                           dtype=np.float32))
    # t2 = time.time()
    # t = t2-t1
    # print ans
    # print t

      # def cleanProximate(self):
    #     dirty_data_shape = self.dirty_data.shape
    #
    #     psf_max_location = np.unravel_index(np.argmax(self.psf), self.psf.shape)
    #
    #     threshold_psf = self.psf*np.where(self.psf>0,1,0)
    #
    #     labelled_psf, labels = ndimage.label(threshold_psf)
    #     # primary_beam = np.where(labelled_psf==labelled_psf[psf_max_location],1,0)*threshold_psf
    #     labelled_primary_beam = np.where(labelled_psf==labelled_psf[psf_max_location],1,0)
    #     extracted_primary_beam = threshold_psf[ndimage.find_objects(labelled_primary_beam)[0]]
    #
    #     extracted_primary_beam = extracted_primary_beam/np.max(extracted_primary_beam)
    #
    #     extracted_primary_beam_shape = extracted_primary_beam.shape
    #     extracted_primary_beam_max_location = np.unravel_index(np.argmax(extracted_primary_beam),extracted_primary_beam_shape)
    #
    #     print extracted_primary_beam_max_location
    #
    #     max_iter = 1
    #
    #     for i in range(0,max_iter):
    #         dirty_data_max_location = np.unravel_index(np.argmax(self.dirty_data), dirty_data_shape)
    #         dirty_data_max = self.dirty_data[dirty_data_max_location]
    #         normalised_dirty_data = self.dirty_data/dirty_data_max
    #
    #
    #         dirty_data_corner_row = dirty_data_max_location[0]-extracted_primary_beam_max_location[0]
    #         dirty_data_corner_column = dirty_data_max_location[1]-extracted_primary_beam_max_location[1]
    #
    #         dirty_data_bounds = \
    #             tuple([slice(dirty_data_corner_row,dirty_data_corner_row+extracted_primary_beam_shape[0],None),
    #                   slice(dirty_data_corner_column,dirty_data_corner_column+extracted_primary_beam_shape[1],None)])
    #
    #         print normalised_dirty_data[dirty_data_bounds] - extracted_primary_beam/np.max(extracted_primary_beam)
    #
    #         pb.imshow(np.where(abs(normalised_dirty_data[dirty_data_bounds] - extracted_primary_beam/np.max(
    #             extracted_primary_beam))<0.1,1,0))
    #         pb.show()

            #
            # dirty_data_bounds = tuple(slice(,
            #                                 dirty_data_max_location[0]+))

        # sumx = np.empty([self.dirty_data.shape[0],1])
        # sumy = np.empty([self.dirty_data.shape[0],1])
        #
        # for i in range(self.dirty_data.shape[0]):
        #     sumx[i,0] = np.sum(self.dirty_data[i,:])
        #     sumy[i,0] = np.sum(self.dirty_data[:,i])
        #
        # pb.plot(sumx)
        # pb.plot(sumy)
        # pb.show()
        #
        # print dirty_data_max_location
        # print psf_max_location
        #
        # print self.dirty_data[dirty_data_max_location[0]-1:dirty_data_max_location[0]+2,dirty_data_max_location[
        #                                                                              1]-1:dirty_data_max_location[1]+2]
        # print self.psf[psf_max_location[0]-1:psf_max_location[0]+2,psf_max_location[1]-1:psf_max_location[1]+2]