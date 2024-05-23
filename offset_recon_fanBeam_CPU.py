# -*- coding: utf-8 -*-
"""
Created on Thu Apr 4 16:50:47 2024

@author:
Harry Allan, harry.allan.21@ucl.ac.uk

AXIm (Advanced X-ray Imaging Group),
Department of Medical Physics and Biomedical Engineering, 
University College London, UK

Code to suport the publication: "Extended Field-of-View X-Ray Attenuation, Phase, and Darkfield Microtomography 
using an Offset Cone-Beam Geometry", Harry Allan et al. 

Code is adapted to demonstrate the offset geometry reconstruction using a CPU compatible fan-beam geometry, based on the
same principles as the cone-beam geometry. Data is kept small to remain compatible on a range of systems. 
Should be easily adaptable for real user data.

"""
#%%

import numpy as np
from matplotlib import pyplot as plt
import astra
from skimage.data import shepp_logan_phantom
import matplotlib as mpl
mpl.rcParams['figure.dpi']=400


def apply_weighting(sino,offset,mag,detector_pixel_size,distance_source_origin):
    """
    
    """

    num_columns = sino.shape[1]    

    weight_lim = int(num_columns -(offset*2*mag)) # int(num_columns/2 - offset) #

    #weight_lim = int(num_columns/2 - offset)

    phi = weight_lim*detector_pixel_size
    t = np.linspace(-phi,phi,weight_lim)
    R = distance_source_origin

    top = np.arctan(t/R)
    bottom = np.arctan(phi/R)

    y = 0.5*( np.sin(np.pi/2 * top/bottom) + 1 )

    sino[:,:weight_lim] = sino[:,:weight_lim] * y
    sino = sino*2

    return sino

def apply_filter(sino,filter_name='ramp'):
    """
    Applying variation of ramp filter to sinogram.

    Input:
    sino: sinogram, assumed to be 2D array of shape (num_projections, num_columns)
    filter_name: name of filter, available filters: ramp, shepp-logan, cosine, hamming, hann

    Output:
    sino: filtered sinogram

    """

    # padding sino to power of 2
    columns = sino.shape[1]
    pad_columns = 2**int(np.ceil(np.log2(columns)))

    # if pad_columns is less than 10% of the original columns, increase to the next power of 2 tp reduce truncation artefacts
    if pad_columns < 1.1*columns:
        pad_columns *= 2

    # applying edge padding
    pad_left = (pad_columns - columns)//2
    pad_right = pad_columns - columns - pad_left
    sino = np.pad(sino, ((0,0),(pad_left,pad_right)), 'edge')

    # create filter from fft of sampled real space filter
    n = np.concatenate(
        (
            np.arange(1, pad_columns / 2 + 1, 2, dtype=int),
            np.arange(pad_columns / 2 - 1, 0, -2, dtype=int),
        )
    )   

    f = np.zeros(pad_columns)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    fourier_filter = 2 * np.real(np.fft.fft(f))  # ramp filter
    fourier_filter = np.fft.fftshift(fourier_filter)

    # optionally apply other filters
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # not super sure on this one, need to check
        omega = np.pi * np.fft.fftfreq(pad_columns)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, pad_columns, endpoint=False)
        cosine_filter = np.sin(freq)
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= np.hamming(pad_columns)
    elif filter_name == "hann":
        fourier_filter *= np.hanning(pad_columns)
    elif filter_name is None:
        fourier_filter[:] = 1
        
    # apply filter
    sino = np.fft.fft(sino, axis=1)
    sino = np.fft.fftshift(sino, axes=1)
    sino = sino*fourier_filter
    sino = np.fft.ifftshift(sino, axes=1)
    sino = np.fft.ifft(sino, axis=1)

    # remove padding
    sino = sino[:,pad_left:-pad_left]

    # remove imaginary component
    sino = np.real(sino)

    return sino

#%% Offset scan simulation and reconstruction using a CPU compatible fan-beam geometry

# generating a phantom of default size 400 x 400
phantom = shepp_logan_phantom()

# setting some arbitrary geometry parameters
distance_source_origin = 800 # [mm]
distance_origin_detector = 200 # [mm]
detector_pixel_size =  20**(-3) # [mm]
no_projections = 721
angular_range = 360
num_rows = 1 
num_columns = 256

mag = (distance_source_origin+distance_origin_detector)/distance_source_origin

eff_pix_size = detector_pixel_size/mag

# offset is defined as the distance between the perpendicular line from the source to the centre of the detector,
# and the centre of rotation
offset = 80 

# array containing angles, flipped as convention to fit the acquisition software in our system
angles = np.flip((np.linspace(0, angular_range*2*np.pi/360, num=no_projections, endpoint=False)))

# setting deltaU as the offset, not this step may be used to adjust the vectors depending on how the offset is defined
deltaU = offset

# creating input vectors for the fan-beam geometry following the convention from https://astra-toolbox.com/docs/geom2d.html
S_vecs=np.zeros((no_projections, 2))
D_vecs=np.zeros((no_projections, 2))
U_vecs=np.zeros((no_projections, 2))


for i in range(len(angles)):
    
    # source
    S_vecs[i,:] = np.array([np.sin(angles[i]), -np.cos(angles[i])]) * (distance_source_origin/eff_pix_size)

    # center of detector
    D_vecs[i,:] = np.array([- np.sin(angles[i]), np.cos(angles[i])]) * (distance_origin_detector/eff_pix_size)

    # vector from detector pixel (0,0) to (0,1)
    U_vecs[i,:]=np.array([np.cos(angles[i]), np.sin(angles[i])]) * (detector_pixel_size/eff_pix_size)
    
    # vector transformations
    S_vecs[i,:] = S_vecs[i,:] + deltaU * U_vecs[i,:] 
    D_vecs[i,:] = D_vecs[i,:] + deltaU * U_vecs[i,:]
    U_vecs[i,:] = U_vecs[i,:]
    
# vectors matrix
vecs = np.zeros((no_projections, 6))

vecs[:, 0:2] = S_vecs
vecs[:, 2:4] = D_vecs
vecs[:, 4:] = U_vecs

# simulating an offset fan-beam scan and showing the sinogram

vol_geom = astra.creators.create_vol_geom(400, 400)

proj_geom = astra.create_proj_geom('fanflat_vec', num_columns, vecs)

proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

sinogram_id, sinogram = astra.create_sino(phantom[:,:], proj_id, returnData=True)

plt.imshow(sinogram,cmap='gray')

#%% carrying out post-convolution weighted FBP

dat = np.copy(sinogram)
dat = dat.astype(float)

# applying some variation of the ramp filter to dampen low-frequency components
dat = apply_filter(dat,filter_name='ramp')

# getting reconstruction now as an intermediate result to show naive reconstruction without weighting
[idd, volume_noWeight] = astra.create_backprojection(dat, proj_id)

# normalising the reconstruction
volume_noWeight = volume_noWeight * np.pi / (2 * no_projections)

# apply offset geometry weighting POST ramp filter
dat = apply_weighting(dat,offset,mag,detector_pixel_size,distance_source_origin)

[idd, volume_post] = astra.create_backprojection(dat, proj_id)

# normalising the reconstruction
volume_post = volume_post * np.pi / (2 * no_projections)

plt.subplot(1,3,1)
plt.imshow(phantom,cmap='gray',vmin=0,vmax=0.8)
plt.axis('off')
plt.title('Phantom')

plt.subplot(1,3,2)
plt.imshow(volume_noWeight,cmap='gray',vmin=0,vmax=0.8)
plt.axis('off')
plt.title('Recon: no weight')

plt.subplot(1,3,3)
plt.imshow(volume_post,cmap='gray',vmin=0,vmax=0.8)
plt.axis('off')
plt.title('Recon: weighted')

