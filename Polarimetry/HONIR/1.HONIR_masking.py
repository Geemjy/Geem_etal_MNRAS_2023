
#==============================================
# BEFORE RUNNING
#==============================================
'''

This is the code for making the "Masking image" of the FITS file for images taken by the Hiroshima optical and near-infrared camera 
(HONIR; Akitaya et al. 2014) on the 1.5-m Kanata Telescope at the Higashi–Hiroshima observatory.
The "Masking image" masks the 1) nearby stars and 2) Cosmic rays.


1. 
 - Input file:  
   '*.fits'         Preprocessed FITS file
   '*.mag.1'        IRAF Phot file containing center of target's o-ray.
   '*.mag.2'        Since WCS is not applied to the HONIR data, 
                    the position of the star was manually entered. 
                    IRAF Phot file containing the start and end points of passing stars (elongated trajectories).
 
 - Output file:
   'Phaethon_coord.csv'   CSV file containing the center of the target (both o-ray and e-ray)
   'mask_*.fits'          Masking image in FITS format
   
   

2. What do you need to run this code? The following packages must be installed.
  - astropy (https://www.astropy.org/)
  - Astro-SCRAPPY (https://github.com/astropy/astroscrappy)
  - "*.mag.1" file from IRAF's Phot package that contains the center of the target's o-ray component.
  - "*.mag.2" file from IRAF's Phot package that contains the start and end points of passing stars 
              (Required only when there are nearby stars)
              

3. 
In this code, the center of the target is found by using the phot of IRAF. So, we need the ".mag" file to bring the coordinate of the target's center.
There is no problem if you find the target's center by other methods. 
All you need to do is modify the part that brings the central coordinate of the target.

  
4. Directory should contain the complete sets consist of 4 images (taken at HWP=0+90*n, 22.5+90*n, 45+90*n, 67.5+90n deg where n=0,1,2,3).
If the number of images in the directory is not a multiple of 4, an error occurs.
'''




#==============================
# INPUT
#==============================
subpath = 'The directory path where fits & mag.* files are saved.'
Target_name = 3200


#==============================================
# IMPORT PACKAGES AND DEFINE THE FUNCTION
#==============================================
from astropy.io import ascii, fits
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
import astroscrappy
import warnings
from astropy.io.fits.verify import VerifyWarning
from tqdm import tqdm
warnings.simplefilter('ignore', category=VerifyWarning)
plt.rcParams['figure.max_open_warning'] = 0



def pill_masking(image,x1,x2,y1,y2,height,
                 target_xo,target_yo,target_xe,target_ye):
    x_star_str = x1
    x_star_end = x2
    y_star_str = y1
    y_star_end = y2
    
    Masking_image = np.zeros(np.shape(image))
    for yi in range(len(image)):
        for xi in range(len(image)):
            for star in range(len(x_star_end)):
                star = star
                slope = (y_star_end[star] - y_star_str[star])/(x_star_end[star]-x_star_str[star])
                y_up = slope *xi + y_star_str[star] + height - slope *x_star_str[star]
                y_low = slope *xi + y_star_str[star] - height - slope *x_star_str[star]
                x_str = min(x_star_str[star],x_star_end[star])
                x_end = max(x_star_str[star],x_star_end[star])
                
                if (xi - x_star_str[star])**2 + (yi-y_star_str[star])**2 < (height)**2:
                    Masking_image[yi,xi] = 1
                if (xi - x_star_end[star])**2 + (yi-y_star_end[star])**2 < (height)**2:
                    Masking_image[yi,xi] = 1    
                if yi >= y_low and  y_up >= yi and xi > x_str and x_end > xi:
                    Masking_image[yi,xi] = 1      
    return Masking_image    

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
    if y_out < 0:
        y_out = 0
    if x_in < 0:
        x_in = 0
    if x_out < 0:
        x_out = 0
        
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





FILE = glob.glob(os.path.join(subpath,'Cr*HN*.fits'))
FILE = sorted(FILE)

##==============================================
##  FIND TARGET CENTER
##==============================================
df = pd.DataFrame({})

for fi in tqdm(FILE):
    print(fi)
    hdul = fits.open(fi)[0]
    data = hdul.data
    header = hdul.header
    OBJECT = header['OBJECT']
    

    Magfile = ascii.read(fi+'.mag.1')
    xo,yo = Magfile['XCENTER'][0], Magfile['YCENTER'][0]
    xe, ye = xo + 160, yo
    
    #Determine FWHM=======================================================
    crop_index = 50
    #Ordinray========================================
    
    lim = 5
    y_1, y_2 = int(yo-crop_index), int(yo+crop_index)
    x_1, x_2 = int(xo-crop_index), int(xo+crop_index)
    crop_o = data[y_1:y_2,x_1:x_2]
    sky_tem_o,std_tem,sky_area_tem = skyvalue(data,yo,xo,45,49)
    crop_o = crop_o - sky_tem_o
    y, x = np.mgrid[:len(crop_o), :len(crop_o[0])]
    g_init = Gaussian2D(x_mean = crop_index,y_mean=crop_index,
                        theta=0,
                        amplitude=crop_o[crop_index,crop_index],
                        bounds={'x_mean':(crop_index-lim,crop_index+lim),
                                'y_mean':(crop_index-lim,crop_index+lim)})

    fitter = LevMarLSQFitter()
    fitted = fitter(g_init, x,y, crop_o)
    center_xo = fitted.x_mean.value
    center_yo = fitted.y_mean.value
    

    
    re_g_init2 = Gaussian2D(x_mean = center_xo,y_mean=center_yo,
                    theta=0,
                    amplitude=crop_o[crop_index,crop_index],
                    bounds={'x_mean':(center_xo-lim,center_xo+lim),
                            'y_mean':(center_yo-lim,center_yo+lim)})
    
    fitter = LevMarLSQFitter()
    fitted = fitter(re_g_init2, x,y, crop_o)
    center_xo = fitted.x_mean.value  + x_1
    center_yo = fitted.y_mean.value  + y_1
    
    
    #Extra========================================
    y_1, y_2 = int(ye-crop_index), int(ye+crop_index)
    x_1, x_2 = int(xe-crop_index), int(xe+crop_index)
    crop_e = data[y_1:y_2,x_1:x_2]
    sky_tem,std_tem,sky_area_tem = skyvalue(data,ye,xe,45,49)
    crop_e = crop_e - sky_tem
    y, x = np.mgrid[:len(crop_e), :len(crop_e[0])]
    g_init = Gaussian2D(x_mean = crop_index,y_mean=crop_index,
                        theta=0,
                        amplitude=crop_e[crop_index,crop_index],
                        bounds={'x_mean':(crop_index-lim,crop_index+lim),
                                'y_mean':(crop_index-lim,crop_index+lim)})

    fitter = LevMarLSQFitter()
    fitted = fitter(g_init, x,y, crop_e)
    center_xe = fitted.x_mean.value
    center_ye = fitted.y_mean.value    
    
    re_g_init2 = Gaussian2D(x_mean = center_xe,y_mean=center_ye,
                    theta=0,
                    amplitude=crop_e[crop_index,crop_index],
                    bounds={'x_mean':(center_xe-lim,center_xe+lim),
                            'y_mean':(center_ye-lim,center_ye+lim)})
    
    fitter = LevMarLSQFitter()
    fitted = fitter(re_g_init2, x,y, crop_e)
    center_xe = fitted.x_mean.value  + x_1
    center_ye = fitted.y_mean.value  + y_1
    
    
    df = df.append({'file':fi,
                   'Xo':center_xo,
                   'Yo':center_yo,
                   'Xe':center_xe,
                   'Ye':center_ye,
                   'Xo_init':xo,
                   'Xe_init':xe,
                   'Yo_init':yo,
                   'Ye_init':ye},
                  ignore_index=True)
    
    
    fig,ax = plt.subplots(1,2,figsize=(8,5))
    figsize=30
    im = ax[0].imshow(data - sky_tem_o,vmin=-100,vmax=100,cmap='seismic')
    ax[0].plot(center_xo,center_yo,marker='+',ls='',color='c',ms=10)
    ax[0].set_xlim(center_xo-figsize,center_xo+figsize)
    ax[0].set_ylim(center_yo-figsize,center_yo+figsize)
    ax[0].set_title('Ordi '+fi.split('/')[-1],fontsize=10)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im,cax=cax)
    
    im = ax[1].imshow(data - sky_tem,vmin=-100,vmax=100,cmap='seismic')
    ax[1].plot(center_xe,center_ye,marker='+',ls='',color='c',ms=10)
    ax[1].set_xlim(center_xe-figsize,center_xe+figsize)
    ax[1].set_ylim(center_ye-figsize,center_ye+figsize)
    ax[1].set_title('Extra '+fi.split('/')[-1],fontsize=10)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im,cax=cax)
    plt.show()
df.to_csv(os.path.join(subpath,'{0}_coord.csv'.format(OBJECT)))







##==============================================
##  STAR MASKING
##==============================================
order = np.arange(0,len(FILE),4)


for z in tqdm(order):
    SET = [FILE[z],FILE[z+1], FILE[z+2], FILE[z+3]]
    for i in range(0,4):
        RET = SET[i]  #Bring the fits file
        print(RET)
    
        hdul = fits.open(RET)[0]
        header = hdul.header 
        image = hdul.data

        #MAKE THE MASKED IMAGE
        Mask_image = np.zeros(np.shape(image))
        
              
        #MASKING THE BACKGROUND STARS
        #Bring the observer quantities from JPL Horizons
        #Querying the RA, DEC of target based on JD at exposure start
        X_str = []
        Y_str = []   
        try:
            Magfile = ascii.read(RET+'.mag.2')
        except FileNotFoundError:
            Masking_image_str = np.zeros(np.shape(image))
        else:
            Magfile = ascii.read(RET+'.mag.2')
            
            x_list = Magfile['XINIT']
            y_list = Magfile['YINIT']
            
            #For ordinary component
            X_str = []
            Y_str = []  
            X_end = []
            Y_end = [] 
            for t in range(len(x_list)):
                if t%2 == 0:
                    X_str.append(x_list[t])
                    Y_str.append(y_list[t])
                    X_str.append(x_list[t]+162)
                    Y_str.append(y_list[t])
                elif t%2 == 1:
                    X_end.append(x_list[t])
                    Y_end.append(y_list[t])
                    X_end.append(x_list[t]+162)
                    Y_end.append(y_list[t])
                    
            print(RET)
            target_center_ = df[df['file']==RET]
            xo = target_center_['Xo'].values[0]
            xe = target_center_['Xe'].values[0]
            yo = target_center_['Yo'].values[0]
            ye = target_center_['Ye'].values[0]
            
            Masking_image_str = pill_masking(image,X_str,X_end,Y_str,Y_end,16,
                                            xo,yo,xe,ye)       
        MASK = Mask_image  + Masking_image_str     
        #MASK THE COSMIC-RAY
        gain = header['GAIN']
        m_LA,cor_image = astroscrappy.detect_cosmics(image,
                                                      gain = gain,
                                                      readnoise = 4.9,
                                                       sigclip=4,
                                                     sigfrac =1,
                                                    objlim=1.5)
        tmLA = m_LA.astype(int)
        MASK[tmLA == 1 ] = 2
        objpath = os.path.join(subpath,'mask_'+RET.split('/')[-1])
        fits.writeto(objpath,data = MASK,header = header,overwrite=True)
        ccccrop = image[int(xo-50):int(xo+50),int(yo-50):int(yo+50)]
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(image-np.median(ccccrop), vmin=-30,vmax=30,cmap='seismic')
        ax[1].imshow(MASK)
        ax[0].set_xlim(xo-50,xo+50)
        ax[1].set_xlim(xo-50,xo+50)
        ax[0].set_ylim(yo-50,yo+50)
        ax[1].set_ylim(yo-50,yo+50)
        plt.show()
        print(os.path.join(subpath,'mask_'+RET.split('/')[-1] +' is created.'))


