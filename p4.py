import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import juliet
import h5py
from glob import glob
from utils import natural_keys

pout = os.getcwd() + '/WASP-39_NC/Analysis/Spectra_ind_100'
lst = os.listdir(pout)
#lst.remove('.DS_Store')
lst.sort(key = natural_keys)

# Wavelengths
f1 = h5py.File(os.getcwd() + '/WASP-39_NC/Stage4/S4_2022-11-14_wasp39_run3/ap8_bg20/S4_wasp39_ap8_bg20_LCData.h5')
wav = np.asarray(f1['wave_mid'])
wav_h, wav_l = np.asarray(f1['wave_hi']), np.asarray(f1['wave_low'])
wav_err = (wav_h - wav_l)/2

#wav, wav_err = wav[0:len(lst)], wav_err[0:len(lst)]

# Retrieving (Rp/R*)^2 from each analysis
dep, dep_uerr, dep_derr = np.ones(len(lst)), np.ones(len(lst)), np.ones(len(lst))
for i in range(len(lst)):
    f2 = glob(pout + '/' + lst[i] + '/*.pkl')[0]
    post = pickle.load(open(f2, 'rb'), encoding='latin1')
    post1 = post['posterior_samples']
    for j in post1.keys():
        if j[0:4] == 'p_p1':
            rprs1 = j
    rprs2 = post1[rprs1]
    dep1 = (rprs2**2)*1e6
    qua_dep = juliet.utils.get_quantiles(dep1)
    dep[i], dep_uerr[i], dep_derr[i] = qua_dep[0], qua_dep[1]-qua_dep[0], qua_dep[0]-qua_dep[2]

#wav, wav_err, dep, dep_derr, dep_uerr = wav[:-5], wav_err[:-5], dep[:-5], dep_derr[:-5], dep_uerr[:-5]
dep_avg_err = (dep_derr + dep_uerr)/2
dep_err_med = np.median(dep_avg_err)
dep_err_std = np.std(dep_avg_err)

msk = np.where(dep_avg_err < (dep_err_med + (2*dep_err_std)))[0]
wav, wav_err, dep, dep_derr, dep_uerr = wav[msk], wav_err[msk], dep[msk], dep_derr[msk], dep_uerr[msk]

plt.figure(figsize=(16/1.5, 9/1.5))
plt.errorbar(wav, dep, xerr=wav_err, yerr=[dep_derr, dep_uerr], fmt='o', c='orangered', mfc='white')
plt.xlabel(r'Wavelength (in $\mu m$)')
plt.ylabel(r'$(R_p/R_\star)^2$ (in ppm)')
plt.grid()
plt.show()
#plt.savefig(pout + '/Spectrum.png', dpi=100)