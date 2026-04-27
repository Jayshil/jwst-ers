import numpy as np
from jwst import datamodels
from jwst.pipeline import Spec2Pipeline
from scipy.interpolate import interp1d
import os

# This file is to estimate the wavelength solution for NIRSpec.
# The output would be xpos vs wavelength.

# -----------------------------------------------------
#   Data path and file name
# -----------------------------------------------------
pin = os.getcwd() + '/NRSPR/RateInts'
pout = os.getcwd() + '/NRSPR/Outputs'

fname = 'jw01366004001_04101_00001-seg001_nrs1_gainscalestep.fits'

# -----------------------------------------------------
#    Running stage 2 of the jwst pipeline
# -----------------------------------------------------
data = datamodels.open(pin + '/' + fname)
det2 = Spec2Pipeline.call(data, save_results=False)

# -----------------------------------------------------
#   Extracting the wavelength solution
# -----------------------------------------------------
### ---- First, let's load the trace positions
traces = np.load(pout + '/Trace_seg001.npz')
xpos, med_trace = traces['xpos'], traces['traces']

### Generating an array, full of np.nan, where we will store the wavelength solution
wav_on_orignal_data_shape = np.zeros( data.data[0,:,:].shape )
wav_on_orignal_data_shape[wav_on_orignal_data_shape==0] = np.nan

## Now, we will fill the array with the _available_ wavelength solution.
row_start, row_end = det2[0].ystart-1, det2[0].ystart+det2[0].ysize-1
col_start, col_end = det2[0].xstart-1, det2[0].xstart+det2[0].xsize-1
wav_on_orignal_data_shape[row_start:row_end, col_start:col_end] = np.copy( det2[0].wavelength )

## 1D wavlength solution along the trace
wav_along_trace = np.empty( len(xpos) )
for i in range(len(xpos)):
    wav_along_trace[i] = wav_on_orignal_data_shape[ int(med_trace[i]), int(xpos[i]) ]

### There would be several NaN in the wav_along_trace, because the wavelength solution is not available for all the pixels. 
### We will use interpolation to fill the NaN values.
valid_indices = ~np.isnan(wav_along_trace)                  # Get the indices of the valid (non-NaN) values
interp_func = interp1d(xpos[valid_indices], wav_along_trace[valid_indices], kind='linear', fill_value='extrapolate')      # Create an interpolation function based on the valid values
wav_along_trace_filled = interp_func(xpos)                  # Use the interpolation function to fill the NaN values

# -----------------------------------------------------
#   Save the wavelength solution
# -----------------------------------------------------
np.savez(pout + '/Wavelength_solution.npz', xpos=xpos, wav_along_trace=wav_along_trace_filled)