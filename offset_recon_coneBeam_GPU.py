# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:10:00 2024

@author:
Harry Allan, harry.allan.21@ucl.ac.uk

AXIm (Advanced X-ray Imaging Group),
Department of Medical Physics and Biomedical Engineering, 
University College London, UK

Code includes functions for weighting, filtering, and 
reconstruction, which may be applied to data acquired in a range of conventional and phase-sensitive CT systems.

Code demonstrates offset cone-beam geometry reconstruction by simulating a central-slice. Setting this up as a cone-beam
rather than fan-beam geometry makes this code easily adapted for real user data with cone-beam systems. Note that the 
cone-beam geometry requires a GPU, to see a simpler CPU-compatible demonstration with a fan-beam geometry please see
an alternative script "offset_recon_fanBeam_CPU.py".

"""
#%%

import numpy as np
import math
from matplotlib import pyplot as plt
import astra
from skimage.data import shepp_logan_phantom
import matplotlib as mpl
mpl.rcParams['figure.dpi']=400


def apply_weighting_offsetDet(sino,offset,mag,detector_pixel_size,distance_source_origin):
    """
    A weighting function using offset detector weights, for which the weighting function is spatially symmetric about the 
    centre of rotation. This has been tested with offset rotation axis data, for which this weighting function is no longer 
    correct (though it may stil provide reasonable results for small cone angles). Function is provided for potential 
    usefullness but not used in this script or yet tested with offset detector data.
    
    Input:
    sino: sinogram, assumed to be 2D array of shape (num_projections, num_columns)
    offset: offset of the detector from the centre of rotation
    mag: magnification factor
    detector_pixel_size: size of detector pixel
    distance_source_origin: distance from source to centre of rotation

    Output:
    sino: weighted sinogram of same shape as input

    """

    num_columns = sino.shape[1]    

    weight_lim = int(num_columns -(offset*2*mag)) 

    phi = weight_lim*detector_pixel_size
    t = np.linspace(-phi,phi,weight_lim)
    R = distance_source_origin

    top = np.arctan(t/R)
    bottom = np.arctan(phi/R)

    y = 0.5*( np.sin(np.pi/2 * top/bottom) + 1 )

    sino[:,:weight_lim] = sino[:,:weight_lim] * y
    sino = sino*2

    return sino

def apply_weighting_offsetCOR(sino,offset,detector_pixel_size,distance_source_origin,distance_origin_detector):
    """
    A weighting function using offset centre-of-rotation weights, for which the weighting function is angularly symmetric 
    about the centre of rotation. This function is used in this script for the offset fan-beam geometry.

    Weighting functions based on the work of G. Belotti, G. Fattori, G. Baroni, and S. Rit, “Extension of the cone-beam ct field-of-view
    using two complementary short scans,” Medical Physics, 2023.

    Input:
    sino: sinogram, assumed to be 2D array of shape (num_projections, num_columns)
    offset: offset of the centre of rotation from the centre of the detector
    detector_pixel_size: size of detector pixel
    distance_source_origin: distance from source to centre of rotation
    distance_origin_detector: distance from centre of rotation to detector

    Output:
    sino: weighted sinogram of same shape as input
    y: weighting function applied to sinogram

    """

    num_columns = sino.shape[1]    

    t = np.linspace(-num_columns/2,num_columns/2,num_columns)

    # normalised source to detector distance
    SDD = (distance_source_origin + distance_origin_detector)/detector_pixel_size

    # angle between optical axis and COR
    tau = np.arctan(offset/(distance_source_origin/detector_pixel_size))

    alpha_u = np.arctan(t/SDD) + tau
    alpha = np.arctan((num_columns/2)/SDD)
    
    weight_end = int(np.abs(num_columns/2 + SDD*np.tan(alpha-2*np.abs(tau))))

    top = alpha_u
    bottom = alpha-np.abs(tau) 

    y = 0.5*( math.copysign(1,tau) * np.sin(np.pi/2 * top/bottom) + 1 )

    if tau > 0:
        y[weight_end:] = 1
    elif tau < 0:
        y[:-weight_end] = 1
    
    y = y*2

    sino = sino * y

    return sino, y

def apply_filter(sino,filter_name='ramp'):
    """
    Applying variation of ramp filter to sinogram.

    Input:
    sino: sinogram, assumed to be 2D array of shape (num_projections, num_columns)
    filter_name: name of filter, available filters: ramp, shepp-logan, cosine, hamming, hann

    Output:
    sino: filtered sinogram of same shape as input

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

if astra.use_cuda():
    print('Found GPU, continuing.')
    
    
    # generating a phantom of default size 400 x 400
    phantom = shepp_logan_phantom()
    phantom = np.expand_dims(phantom,0)
    
    # setting and calculating some arbitrary geometry parameters
    distance_source_origin = 800 # [mm]
    distance_origin_detector = 200 # [mm]
    detector_pixel_size =  20**(-3) # [mm]
    no_projections = 721
    angular_range = 360
    num_rows = 1 
    num_columns = 256
    
    mag = (distance_source_origin+distance_origin_detector)/distance_source_origin
    
    eff_pix_size = detector_pixel_size/mag
    
    # array containing angles, flipped as convention to fit the acquisition software in our system
    angles = np.flip((np.linspace(0, angular_range*2*np.pi/360, num=no_projections, endpoint=False)))
    
    # offset is defined as the distance between the perpendicular line from the source to the centre of the detector,
    # and the centre of rotation
    offset = 80 
    
    # setting deltaU as the offset, not this step may be used to adjust the vectors depending on how the offset is defined
    deltaU = offset
    # deltaV is unused, can be optionally included to account for imperfect geometry alignment
    deltaV = 0
    
    # these are left unused, assumed well aligned geometry
    eta=np.radians(0) # In plane rotation(skew)
    theta=np.radians(0) # Out-of-plane rotation (tilt)
    phi=np.radians(0) # Out-of-plane rotation (slant)
    
    # creating input vectors for the cone-beam geometry following the convention from https://astra-toolbox.com/docs/geom3d.html
    S_vecs=np.zeros((no_projections, 3))
    D_vecs=np.zeros((no_projections, 3))
    U_vecs=np.zeros((no_projections, 3))
    V_vecs=np.zeros((no_projections, 3))
    
    M_rotx = np.float32([
        [1, 0, 0],
     	[0, np.cos(eta), np.sin(eta)],
     	[0, -np.sin(eta), np.cos(eta)],    
    ])
    
    M_roty = np.float32([
     	[np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
     	[np.sin(theta), 0, np.cos(theta)],    
    ])
    
    M_rotz = np.float32([
     	[np.cos(phi), np.sin(phi), 0], 
     	[-np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    
    M_rot = np.dot(np.dot(M_rotz, M_roty), M_rotx)
    
    for i in range(len(angles)):
        
        # source
        S_vecs[i,:] = np.array([np.sin(angles[i]), -np.cos(angles[i]), 0]) * (distance_source_origin/eff_pix_size)
        # center of detector
        D_vecs[i,:] = np.array([- np.sin(angles[i]), np.cos(angles[i]), 0]) * (distance_origin_detector/eff_pix_size)
        # vector from detector pixel (0,0) to (0,1)
        U_vecs[i,:]=np.array([np.cos(angles[i]), np.sin(angles[i]), 0]) * (detector_pixel_size/eff_pix_size)
        # vector from detector pixel (0,0) to (1,0) 
        V_vecs[i,:]=np.array([0, 0,  detector_pixel_size/eff_pix_size])
        
        # Vector Transformations
        S_vecs[i,:] = S_vecs[i,:] + deltaU * U_vecs[i,:] + deltaV * V_vecs[i,:]
        D_vecs[i,:] = D_vecs[i,:] + deltaU * U_vecs[i,:] + deltaV * V_vecs[i,:]
        U_vecs[i,:] = np.dot(M_rot, U_vecs[i,:])
        V_vecs[i,:] = np.dot(M_rot, V_vecs[i,:])
        
    # vectors matrix
    vecs = np.zeros((no_projections, 12))
    
    vecs[:, :3] = S_vecs
    vecs[:, 3:6] = D_vecs
    vecs[:, 6:9] = U_vecs
    vecs[:,9:] = V_vecs
    
    # simulating an offset fan-beam scan and showing the sinogram
    vol_geom = astra.creators.create_vol_geom(400, 400, 1)
    
    proj_geom = astra.create_proj_geom('cone_vec', num_rows, num_columns, vecs)
    
    sinogram_id, sinogram = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom, returnData=True)
    
    plt.imshow(sinogram[0,:,:],cmap='gray')

else:
    print('No GPU found, running this script will crash kernel. Please use "offset_recon_fanBeam_CPU.py" instead for a CPU-compatible version.')

#%% carrying out post-convolution weighted FBP

dat = np.copy(sinogram[0,:,:])
dat = dat.astype(float)

# applying some variation of the ramp filter to dampen low-frequency components
dat = apply_filter(dat,filter_name='ramp')

# must expand dims as this is a 3D geometry
dat = np.expand_dims(dat,0)

# getting reconstruction now as an intermediate result to show naive reconstruction without weighting
[idd, volume_noWeight] = astra.create_backprojection3d_gpu(dat, proj_geom, vol_geom, returnData=True)

# normalising the reconstruction
volume_noWeight = volume_noWeight * np.pi / (2 * no_projections)

# apply offset geometry weighting POST ramp filter
dat,_ = apply_weighting_offsetCOR(dat[0,:,:],offset,detector_pixel_size,distance_source_origin,distance_origin_detector)

dat = np.expand_dims(dat,0)

[idd, volume_post] = astra.create_backprojection3d_gpu(dat, proj_geom, vol_geom, returnData=True)

# normalising the reconstruction
volume_post = volume_post * np.pi / (2 * no_projections)

plt.subplot(1,3,1)
plt.imshow(phantom[0,:,:],cmap='gray',vmin=0,vmax=0.8)
plt.axis('off')
plt.title('Phantom')

plt.subplot(1,3,2)
plt.imshow(volume_noWeight[0,:,:],cmap='gray',vmin=0,vmax=0.8)
plt.axis('off')
plt.title('Recon: no weight')

plt.subplot(1,3,3)
plt.imshow(volume_post[0,:,:],cmap='gray',vmin=0,vmax=0.8)
plt.axis('off')
plt.title('Recon: weighted')














