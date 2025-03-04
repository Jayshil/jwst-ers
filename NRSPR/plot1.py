import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from glob import glob
from exotoolbox.utils import get_quantiles
from utils import make_low_res_spec

pin = os.getcwd() + '/NRSPR/Analysis/SpectralLC7'
ch_nos = 150

# Fake wavelength bins
wav, wav_bin = np.loadtxt(pin + '/wavelength_ch.dat', usecols=(0,1), unpack=True)
dep, dep_up_err, dep_lo_err = np.zeros(len(wav)), np.zeros(len(wav)), np.zeros(len(wav))

std_err = np.zeros(len(wav))
for i in range(len(wav)):
    fname = glob(pin + '/CH' + str(i) + '/*.pkl')[0]
    post = pickle.load(open(fname, 'rb'))
    post1 = post['posterior_samples']
    rprs2 = post1['p_p1_CH' + str(i)]**2
    qua = get_quantiles(rprs2*1e2)
    dep[i], dep_up_err[i], dep_lo_err[i] = qua[0], qua[1]-qua[0], qua[0]-qua[2]
    std_err[i] = np.nanstd(rprs2*1e2)

bin_wav, _, bin_dep, bin_deperr_lo, bin_deperr_up = make_low_res_spec(native_wav=wav, native_spec=dep/1e6, native_spec_err=std_err/1e6, ch_nos=ch_nos)

fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
#axs.errorbar(wav, dep, yerr=[dep_lo_err, dep_up_err], color='k', fmt='o', mfc='white', elinewidth=2, capthick=2, capsize=3)
axs.errorbar(bin_wav, bin_dep, yerr=[bin_deperr_lo, bin_deperr_up], color='k', fmt='o', mfc='white', elinewidth=2, capthick=2, capsize=3)

axs.set_xlabel(r'Fake wavelength [$\mu$m]', fontsize=14, fontfamily='serif')
axs.set_ylabel(r'Transit depth [ppm]', fontsize=14, fontfamily='serif')

axs.tick_params(labelfontfamily='serif')
plt.setp(axs.get_yticklabels(), fontsize=12)
plt.setp(axs.get_xticklabels(), fontsize=12)

plt.show()