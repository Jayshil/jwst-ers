import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import utils

# This file is to visualise how FWHM varies with time

visit = 'NRCLW'
pin = os.getcwd() + '/NRCLW/Outputs/' + visit
pin2 = os.getcwd() + '/RateInts/Corr_' + visit

## Segment!!!
segs = []
for i in range(4):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

fwhm = np.load(pin + '/fwhm_' + visit + '.npy')
super_fwhm = np.load(pin + '/super_fwhm_' + visit + '.npy')
times = np.load(pin2 + '/Times_bjd_seg001.npy')

for i in range(len(segs)-1):
    time_seg = np.load(pin2 + '/Times_bjd_seg' + segs[i+1] + '.npy')
    times = np.hstack((times, time_seg))

med_fwhm = np.nanmedian(fwhm, axis=1)
times = times + 2400000.5# - 2459702.

# For FWHM
tbin, fbin, febin, _ = utils.lcbin(time=times, flux=med_fwhm, binwidth=0.005)
tbin2, fbin2, febin2, _ = utils.lcbin(time=times, flux=med_fwhm, binwidth=0.05)

# For super FWHM
tbin_sf, fbin_sf, flebin_sf, _ = utils.lcbin(time=times, flux=super_fwhm, binwidth=0.005)
tbin_sf2, fbin_sf2, flebin_sf2, _ = utils.lcbin(time=times, flux=super_fwhm, binwidth=0.05)

# For Full model
fig, ax = plt.subplots(figsize=(16/1.5,9/1.5))

# Top panel
ax.errorbar(times, med_fwhm, fmt='.', c='cornflowerblue', alpha=0.5)
ax.errorbar(tbin, fbin, yerr=febin, fmt='.', c='gray', alpha=0.7, zorder=50)
ax.errorbar(tbin2, fbin2, yerr=febin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax.set_ylabel('FWHM', fontsize=14)
ax.set_xlabel('Time (BJD)', fontsize=14)
ax.set_xlim(np.min(times), np.max(times))
#ax.set_ylim(0.9990,1.0015)
plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
ax.set_title('FWHM for Visit ' + str(visit), fontsize=15)
#plt.show()
plt.savefig(pin + '/Figures/Full_fwhm_' + visit + '.png', dpi=500)

# For Full model
fig, ax = plt.subplots(figsize=(16/1.5,9/1.5))

# Top panel
ax.errorbar(times, super_fwhm, fmt='.', c='cornflowerblue', alpha=0.5)
ax.errorbar(tbin_sf, fbin_sf, yerr=flebin_sf, fmt='.', c='gray', alpha=0.7, zorder=50)
ax.errorbar(tbin_sf2, fbin_sf2, yerr=flebin_sf2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax.set_ylabel('FWHM', fontsize=14)
ax.set_xlabel('Time (BJD)', fontsize=14)
ax.set_xlim(np.min(times), np.max(times))
#ax.set_ylim(0.9990,1.0015)
plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
ax.set_title('Super FWHM for Visit ' + str(visit), fontsize=15)
#plt.show()
plt.savefig(pin + '/Figures/Full_super_fwhm_' + visit + '.png', dpi=500)

"""f1 = open(pin2 + '/fwhm_full_' + visit + '.dat', 'w')
f1.write('# Median FWHM\t Super FWHM\n')
for i in range(len(med_fwhm)):
    f1.write(str(med_fwhm[i]) + '\t' + str(super_fwhm[i]) + '\n')
f1.close()"""