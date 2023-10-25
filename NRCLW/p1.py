import numpy as np
import matplotlib.pyplot as plt
import pickle
from transitspectroscopy import spectroscopy as tspec
from poetss import poetss
import os
import time

# This file is to compute FWHM for all integrations for all visits

visit = 'NRCLW'
p1 = os.getcwd() + '/RateInts/Corr_' + visit
pout = os.getcwd() + '/NRCLW/Outputs/' + visit

## Segment!!!
segs = []
for i in range(4):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

corrected_data_all = np.load(p1 + '/Corrected_data_seg' + segs[0] + '.npy')
for seg in range(len(segs)-1):
    # Loading the data
    print('>>>> --- Loading the dataset...')
    corrected_data = np.load(p1 + '/Corrected_data_seg' + segs[seg+1] + '.npy')
    corrected_data_all = np.vstack((corrected_data_all, corrected_data))
    print('>>>> --- Done!!')


# Loading the trace positions
print('>>>> --- Finding trace positions...')
xstart, xend = 25, 1600
cent_cub1 = poetss.find_trace_cof(clean_cube=corrected_data_all[:,4:,xstart:xend], margin=5)
median_trace, dx1 = poetss.fit_multi_trace(cent_mat=cent_cub1, deg=2, clip=3)
xpos = np.arange(xstart, xend, 1)
print('>>>> --- Done!!')

# And, computing the FWHM
print('>>>> --- Finding the FWHM...')
fwhm, super_fwhm = tspec.trace_fwhm(tso=corrected_data_all[:,4:,:], x=xpos, y=median_trace, distance_from_trace=10)
print('>>>> --- Done!!')


np.save(pout + '/fwhm_' + visit + '.npy', fwhm)
np.save(pout + '/super_fwhm_' + visit + '.npy', super_fwhm)