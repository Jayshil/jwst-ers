import numpy as np
from jwst import datamodels
import os
from jwst.pipeline import calwebb_detector1
from jwst.pipeline import calwebb_spec2
from glob import glob
import time
import utils

# ---------------------------------
# For calibrated stellar spectra
# ---------------------------------

seg = '009'
#segs = segs[-2:]
p2 = os.getcwd() + '/RateInts/StelSpec'    # To store corrected files

# And correcting the data
fname_cal = glob(p2 + '/*' + seg + '_mirimage_calints.fits')[0]
calints = datamodels.open(fname_cal)

## Saving the DQ array
darkdq = calints.dq
np.save(p2 + '/bmap_seg' + seg + '.npy', darkdq)
##

print('>>>> --- Correcting errorbars (for zeros and NaNs)...')
## Correct errorbars
med_err = np.nanmedian(calints.err.flatten())
## Changing Nan's and zeros in error array with median error
corr_err1 = np.copy(calints.err)
corr_err2 = np.where(calints.err != 0., corr_err1, med_err)                     # Replacing error == 0 with median error
corrected_errs = np.where(np.isnan(calints.err) != True, corr_err2, med_err)    # Replacing error == Nan with median error
print('>>>> --- Done!!')

print('>>>> --- Creating a bad-pixel map...')
## Making a bad-pixel map
mask_bp1 = np.ones(calints.data.shape)
mask_bp2 = np.where(calints.err != 0., mask_bp1, 0.)                 # This will place 0 in mask where errorbar == 0
mask_bp3 = np.where(np.isnan(calints.err) != True, mask_bp2, 0.)     # This will place 0 in mask where errorbar is Nan
mask_badpix = np.where(darkdq == 0., mask_bp3, 0.)                               # This will place 0 in mask where darkdq != 0
## Mask with cosmic rays
### Essentially this mask will have 0s in the places of bad pixels...
mask_bcr = utils.identify_crays(calints.data, mask_badpix)
print('>>>> --- Done!!')

print('>>>> --- Correcting data...')
corrected_data = np.copy(calints.data)

print('>>> --- Saving results...')
np.save(p2 + '/Corrected_data_seg' + seg + '.npy', corrected_data)
np.save(p2 + '/Corrected_errors_seg' + seg + '.npy', corrected_errs)
np.save(p2 + '/Mask_bcr_seg' + seg + '.npy', mask_bcr)
print('>>>> --- Done (Stage 2 processing)!!')