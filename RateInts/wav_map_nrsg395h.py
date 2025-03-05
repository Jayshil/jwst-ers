import numpy as np
import matplotlib.pyplot as plt
from jwst import datamodels, assign_wcs
from jwst.assign_wcs import AssignWcsStep
from gwcs import wcstools
import os

detector = 'nrs1'
fname = os.getcwd() + '/RateInts/Ramp_NRSG395H/jw01366003001_04101_00001-seg001_' + detector + '_rateints.fits'
ramp = datamodels.open(fname)

awcs = AssignWcsStep.call(ramp, save_results=False)
wcs_out = assign_wcs.nrs_wcs_set_input(awcs, awcs.meta.instrument.fixed_slit)

wcs_out.bounding_box = ( (-0.5, awcs.data.shape[2]-0.5), (-0.5, awcs.data.shape[1]-0.5) )
bb_cols, bb_rows = wcstools.grid_from_bounding_box( wcs_out.bounding_box )

_, _, bb_wav_map = wcs_out(bb_cols, bb_rows)

plt.figure(figsize=(15,5))
plt.imshow(bb_wav_map, interpolation='none', aspect='auto')
plt.show()

np.save(os.getcwd() + '/Data/wav_map_nrsg395h_' + detector + '.npy', bb_wav_map)