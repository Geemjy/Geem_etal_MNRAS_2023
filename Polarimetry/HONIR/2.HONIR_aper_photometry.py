
#==============================================
# BEFORE RUNNING
#==============================================

'''
This is the code for making the aperture photometry of the asteroids taken by the Hiroshima optical and near-infrared camera 
(HONIR; Akitaya et al. 2014) on the 1.5-m Kanata Telescope at the Higashi–Hiroshima observatory.


1. 
 - Input file:  
   '*.fits'                     Preprocessed FITS file
   'Phaethon_coord.csv'         The file containing the center of 
                                the target's o-ray and e-ray produced by '1.HONIR_masking.py'.
   'mask_*.fits'                Masking image produced by '1.HONIR_masking.py'.  
 
 
 - Outout file:
   Phot_{DATE}_{Object_name}.csv         Photometric result of each images 
             

2. What you need to run this code. The following packages must be installed.
  - astropy (https://www.astropy.org/)
  

3. Directory should contain the complete sets consist of 4 images (taken at HWP=0+90*n, 22.5+90*n, 45+90*n, 67.5+90n deg where n=0,1,2,3).
If the number of images in the directory is not a multiple of 4, an error occurs.
'''


#==============================================
# INPUT VALUE FOR THE APERTURE PHOTOMETRY
#==============================================

subpath  = 'The directory path where fits & Phaethon_coord.csv * mask_*.fits files are saved.'
Observatory = {'lon': 132.77,
               'lat': 34.27,
               'elevation': 0.502} #Higashihiroshima Observatory
Target_name = 3200



####################################
# Photometry Parameter
####################################
#Values below are examples. Different values were used for each data.
Aperture_scale = 1.8    # Aperture radius = Aperture_scale * FWHM 
ANN_scale = 4         # Annulus radius = ANN_scale * FWHM
Dan = 20          # [pix] #Dannulus size


fig_plot = 'yes' #Will you plot the image? or 'No'






#==============================================
# IMPORT PACKAGES AND DEFINE THE FUNCTION
#==============================================
import glob 
import os
import astropy
import photutils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
from astropy.time import Time
from photutils import CircularAperture,CircularAnnulus,aperture_photometry
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter

import sep

def signal_to_noise(source_eps, sky_eps, rd, npix,
                            gain):
    signal = source_eps 
    noise = np.sqrt((source_eps  + npix *
                         (sky_eps * gain )) + npix * rd ** 2)
    return signal / noise   


def skyvalue(data, y0, x0, r_in, r_out,masking=None):
    if masking is not None:
        masking = masking.astype(bool)
    else:
        masking = np.zeros(np.shape(data))

    # Determine sky and std
    y_in = int(y0-r_out)
    y_out = int(y0+r_out)
    x_in = int(x0-r_out)
    x_out = int(x0+r_out)
    if y_in < 0:
        y_in = 0
    if y_out > len(data) :
        y_out = len(data)
    if x_in < 0:
        x_in = 0
    if x_out > len(data[0]):
        x_out =  len(data[0])
        
    sky_deriving_area = data[y_in:y_out, x_in:x_out]
    masking = masking[y_in:y_out, x_in:x_out]
    
    new_mask = np.zeros(np.shape(sky_deriving_area))+1
    for yi in range(len(sky_deriving_area)):
        for xi in range(len(sky_deriving_area[0])):
            position = (xi - r_out)**2 + (yi-r_out)**2
            if position < (r_out)**2 and position > r_in**2:
                new_mask[yi, xi] = 0
    new_mask = new_mask.astype(bool)
    mask = new_mask + masking
    
    Sky_region = np.ma.masked_array(sky_deriving_area, mask)
    std = np.ma.std(Sky_region)
    sky = np.ma.median(Sky_region)
    npix = np.shape(sky_deriving_area)[0]*np.shape(sky_deriving_area)[1] - np.sum(mask)
    
    return(sky, std, npix)

def circle(x,y,r):
    theta = np.linspace(0, 2*np.pi, 100)
    x1 = r*np.cos(theta)+y
    x2 = r*np.sin(theta)+x
    return(x2.tolist(),x1.tolist())

#==============================================
# BRING THE TARGET IMAGE
#==============================================

file = glob.glob(os.path.join(subpath,'Cr*.fits'))
file = sorted(file)

log = pd.DataFrame({})
for fi in file:
    hdul = fits.open(fi)
    header = hdul[0].header
    data = hdul[0].data
    log = log.append({'FILENAME':os.path.split(fi)[-1],
                      'OBJECT':header['OBJECT'],
                      'DATE':header['DATE-OBS'],
                      'HWPANG':header['HWPANGLE'],
                      'EXPTIME':header['EXPTIME']},
                      ignore_index=True)

    
    
#======================================#
#             Photometry               #
#======================================#

Photo_Log = pd.DataFrame({})
order = np.arange(0,len(file),4)
for z in order:
    SET = [file[z],file[z+1], file[z+2], file[z+3]]
    kappa = []
    err_kappa = []
    for ang in range(0,4):
        RET = SET[ang]
        hdul = fits.open(RET)
        header = hdul[0].header 
        data = hdul[0].data
        epoch = header['DATE-OBS']+'T'+header['UT']
        epoch_jd = Time(epoch, format='isot', scale='utc').jd 
        gain = header['GAIN']
        exp = header['EXPTIME']
        RN = header['RDNOISE']
        OBJECT = header['OBJECT']
        


        #Bring the masking image==================================================
        hdul_mask = fits.open(subpath+'/mask_'+RET.split('/')[-1])[0]
        masking = hdul_mask.data  
        
        
        #Bring the center==========================================================
        target_center = os.path.join(subpath,'Phaethon_coord.csv')
        target_center = pd.read_csv(target_center)
        target_center_i = target_center[target_center['file']==RET]

        xo = target_center_i['Xo'].values[0]
        yo = target_center_i['Yo'].values[0]
        xe = target_center_i['Xe'].values[0]
        ye = target_center_i['Ye'].values[0]


        #####INTERPOLATE###########################
        mask_ = np.zeros(np.shape(masking))
        mask_[masking==2]=1
        mask_ = mask_.astype(bool)

        mask_use_ = np.zeros(np.shape(masking))
        mask_use_[masking==1]=1
        mask_use_ = mask_use_.astype(bool)

        data = data.byteswap().newbyteorder()      
        bkg_ = sep.Background(data, mask=mask_.astype(bool), bw=5, bh=5, fw=5, fh=5)
        bkg_image_ = bkg_.back()      
        old_data_ = np.copy(data)
        data[mask_] = bkg_image_[mask_]




        #Determine FWHM =======================================================
        crop_index = 10
        ## Ordinary ===========================================================
        y_1, y_2 = int(yo-crop_index), int(yo+crop_index)
        x_1, x_2 = int(xo-crop_index), int(xo+crop_index)
        crop_o = data[y_1:y_2,x_1:x_2]

        sky_tem,std_tem,sky_area_tem = skyvalue(data,yo,xo,30,40,mask_use_)
        crop_o = crop_o - sky_tem
        y, x = np.mgrid[:len(crop_o), :len(crop_o[0])]
        g_init = Gaussian2D(x_mean = crop_index,y_mean=crop_index,
                            theta=0,
                            amplitude=crop_o[crop_index,crop_index],
                            bounds={'x_mean':(crop_index-5,crop_index+5),
                                    'y_mean':(crop_index-5,crop_index+5)})

        fitter = LevMarLSQFitter()
        fitted = fitter(g_init, x,y, crop_o)
        center_x = fitted.x_mean.value
        center_y = fitted.y_mean.value
        fwhm_o = max(fitted.x_fwhm,fitted.y_fwhm)


        ## Extra-Ordinary
        y_1, y_2 = int(ye-crop_index), int(ye+crop_index)
        x_1, x_2 = int(xe-crop_index), int(xe+crop_index)
        crop_e = data[y_1:y_2,x_1:x_2]
        sky_tem,std_tem,sky_area_tem = skyvalue(data,ye,xe,30,40,mask_use_)
        crop_e = crop_e - sky_tem
        y, x = np.mgrid[:len(crop_e), :len(crop_e[0])]
        g_init = Gaussian2D(x_mean = 20,y_mean=20,
                            theta=0,
                            amplitude=crop_e[crop_index,crop_index],
                            bounds={'x_mean':(crop_index-5,crop_index+5),
                                    'y_mean':(crop_index-5,crop_index+5)})
        fitter = LevMarLSQFitter()
        fitted = fitter(g_init, x,y, crop_e)
        center_x = fitted.x_mean.value
        center_y = fitted.y_mean.value
        fwhm_e = max(fitted.x_fwhm,fitted.y_fwhm)    

        FWHM_sel = max(fwhm_o,fwhm_e)


        #Aperture photometry=======================================================

        #### Set aperture size
        Aperture_radius = Aperture_scale*FWHM_sel/2
        Ann = ANN_scale*FWHM_sel/2
        Ann_out = Ann+Dan


        ##Determine sky value by aperture   
        Aper_o = CircularAperture([xo,yo],Aperture_radius) #Set aperture
        sky_o,std_o,area_o = skyvalue(data,yo,xo,Ann,Ann_out,mask_use_) # Set area determinung Sk

        Aper_e = CircularAperture([xe,ye],Aperture_radius) #Set aperture
        sky_e,std_e,area_e = skyvalue(data,ye,xe,Ann,Ann_out,mask_use_) # Set area determinung Sk


        Flux_o = aperture_photometry(data - sky_o,Aper_o,mask_use_)['aperture_sum'][0]*gain
        ERR_o = np.sqrt(Flux_o + 3.14*Aperture_radius**2*(sky_o*gain + (std_o*gain)**2 +(RN*gain)**2))
        Snr_o = signal_to_noise(Flux_o,sky_o,RN,Aperture_radius**2*3.14,gain)

        Flux_e = aperture_photometry(data - sky_e, Aper_e,mask_use_)['aperture_sum'][0]*gain
        ERR_e = np.sqrt(Flux_e + 3.14*Aperture_radius**2*(sky_e*gain + (std_e*gain)**2 + (RN*gain)**2))
        Snr_e = signal_to_noise(Flux_e,sky_e,RN,Aperture_radius**2*3.14,gain)
        
        sky_o, std_o = sky_o*gain, std_o*gain
        sky_e, std_e = sky_e*gain, std_e*gain
        
        Photo_Log = Photo_Log.append({'Object': OBJECT,
                                  'Filename':RET.split('/')[-1],
                                  'HWPANG':header['HWPANGLE'],
                                  'Filter':header['FILTER02'],
                                  'TIME':epoch,
                                  'DATE':epoch.split('T')[0],
                                  'JD':epoch_jd,
                                  'Aper [pix]':Aperture_radius,
                                  'EXP [s]':exp,
                                  'Ann':Ann,
                                  'Ann_out':Ann_out,
                                  'Flux_o':Flux_o,
                                  'eFlux_o':ERR_o,
                                  'Flux_e':Flux_e,
                                  'eFlux_e':ERR_e,
                                  'SNR_o':Snr_o,
                                  'SNR_e':Snr_e,
                                  'Sky_o':sky_o,
                                  'eSky_o':std_o,
                                  'Sky_e':sky_e,
                                  'eSky_e':std_e,
                                  'Airmass':header['AIRMASS']}, ignore_index=True)     
        
        
        if fig_plot =='yes':
            lim = 200
            fig,ax = plt.subplots(1,2,figsize=(20,10))
            plot_data = np.ma.masked_array(data,mask_use_)
            figsize=100
            im = ax[0].imshow(plot_data - sky_o/gain,vmin=-lim,vmax=lim,cmap='seismic')
            xi,yi = circle(xo,yo,Aperture_radius)
            ax[0].plot(xi,yi,color='y',lw=4)
            xi,yi = circle(xo,yo,Ann)
            ax[0].plot(xi,yi ,color='c',lw=4)
            xi,yi = circle(xo,yo,Ann+Dan)
            ax[0].plot(xi,yi ,color='c',lw=4)
            ax[0].plot(xo,yo,marker='+',ls='',color='b')
            ax[0].set_xlim(xo-figsize,xo+figsize)
            ax[0].set_ylim(yo-figsize,yo+figsize)
            ax[0].set_title('Ordinary'+RET.split('/')[-1],fontsize=18)
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im,cax=cax) 

            im = ax[1].imshow(plot_data - sky_e/gain,vmin=-lim,vmax=lim,cmap='seismic')
            xi,yi = circle(xe,ye,Aperture_radius)
            ax[1].plot(xi,yi,color='y',lw=4)
            xi,yi = circle(xe,ye,Ann)
            ax[1].plot(xi,yi ,color='c',lw=4)
            xi,yi = circle(xe,ye,Ann+Dan)
            ax[1].plot(xi,yi ,color='c',lw=4)
            ax[1].plot(xe,ye,marker='+',ls='',color='b')
            ax[1].set_xlim(xe-figsize,xe+figsize)
            ax[1].set_ylim(ye-figsize,ye+figsize)
            ax[1].set_title('ExtaOrdinary'+RET.split('/')[-1],fontsize=18)
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im,cax=cax) 
            plt.show()

        
new_index = ['Filename','Object','HWPANG','Filter',
             'TIME','DATE','JD','EXP [s]',
             'Ann','Ann_out','Flux_o','eFlux_o',
             'Flux_e','eFlux_e','SNR_o','SNR_e',
             'Sky_o','eSky_o','Sky_e','eSky_e',
             'Aper [pix]','Airmass']
Photo_Log = Photo_Log.reindex(columns = new_index)  
Photo_Log = Photo_Log.round({'Aper [pix]':1,'EXP [s]':1,
             'Ann':1,'Ann_out':1,'Flux_o':1,'eFlux_o':1,
             'Flux_e':1,'eFlux_e':1,'SNR_o':1,'SNR_e':1,
             'Sky_o':1,'eSky_o':1,'Sky_e':1,'eSky_e':1,
             'Airmass':1})
Photo_Log.to_csv(os.path.join(subpath,'Phot_{0}_{1}.csv'.format(epoch.split('T')[0].replace('-','_'),OBJECT))) 

