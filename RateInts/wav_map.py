import numpy as np
import matplotlib.pyplot as plt
from jwst import datamodels
from jwst.assign_wcs import AssignWcsStep
import os
from tqdm import tqdm

#fname = os.getcwd() + '/Tests/Data/jw01952002001_04103_00001-seg009_mirimage_rateints.fits'
fname = os.getcwd() + '/RateInts/Ramp_NIRSpec/jw01366004001_04101_00001-seg004_nrs1_1_rampfitstep.fits'

rateints = datamodels.open(fname)
awcs = AssignWcsStep.call(rateints, save_results=False)

print(rateints.data.shape)

col_idx = np.arange(0, rateints.data.shape[2], 1)
row_idx = np.arange(0, rateints.data.shape[1], 1)

ab = awcs.meta.wcs(30, 30, 1)   # This takes the arguments column index, row index and order no
print(ab)
# Output would be, (RA, DEC, Wavelength, Order)

"""lams = np.zeros(rateints.data[0,:,:].shape)
#print(lams.shape)

for i in tqdm(range(len(col_idx))):
    for j in range(len(row_idx)):
        _, _, lams[j,i], _ = awcs.meta.wcs(col_idx[i], row_idx[j], 1)

np.save(os.getcwd() + '/Data/wave_map_NIRCam.npy', lams)

plt.imshow(lams, origin='lower', aspect='equal', interpolation='None')
plt.colorbar()
plt.show()"""