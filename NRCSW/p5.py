import numpy as np
import matplotlib.pyplot as plt
import os

visit = 'NRCSW'

pin = os.getcwd() + '/RateInts/Corr_' + visit
pout = os.getcwd() + '/SW/Outputs/' + visit

segs = []
for i in range(4):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

data = np.load(pin + '/Corrected_data_seg' + segs[0] + '.npy')
errs = np.load(pin + '/Corrected_errors_seg' + segs[0] + '.npy')
psf_data = data[:,75:,462:632]
psf_errs = errs[:,75:,462:632]

for i in range(len(segs) - 1):
    seg = segs[i+1]
    # Loading the data
    print('>>>> --- Working on Segment ' + str(seg))
    # Loading the data
    corrected_data = np.load(pin + '/Corrected_data_seg' + seg + '.npy')
    corrected_errs = np.load(pin + '/Corrected_errors_seg' + seg + '.npy')
    psf_data = np.vstack((psf_data, corrected_data[:,75:,462:632]))
    psf_errs = np.vstack((psf_errs, corrected_errs[:,75:,462:632]))

photometry = np.average(psf_data, axis=(1,2), weights=psf_errs**2)
errors = np.sqrt(np.nansum(psf_errs**6, axis=(1,2))) / np.nansum(psf_errs**2, axis=(1,2))

plt.errorbar(np.arange(len(photometry)), photometry/np.nanmedian(photometry), yerr=errors/np.nanmedian(photometry), fmt='.')
plt.show()