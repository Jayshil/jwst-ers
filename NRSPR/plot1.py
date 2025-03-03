import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from glob import glob
from exotoolbox.utils import get_quantiles

pin = os.getcwd() + '/NRSPR/Analysis/SpectralLC5'

# Fake wavelength bins
wav, wav_bin = np.loadtxt(pin + '/wavelength_ch.dat', usecols=(0,1), unpack=True)
dep, dep_up_err, dep_lo_err = np.zeros(len(wav)), np.zeros(len(wav)), np.zeros(len(wav))

for i in range(len(wav)):
    fname = glob(pin + '/CH' + str(i) + '/*.pkl')[0]
    post = pickle.load(open(fname, 'rb'))
    post1 = post['posterior_samples']
    rprs2 = post1['p_p1_CH' + str(i)]**2
    qua = get_quantiles(rprs2*1e6)
    dep[i], dep_up_err[i], dep_lo_err[i] = qua[0], qua[1]-qua[0], qua[0]-qua[2]

fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
axs.errorbar(wav, dep, yerr=[dep_lo_err, dep_up_err], color='k', fmt='o', mfc='white', elinewidth=2, capthick=2, capsize=3)

axs.set_xlabel(r'Fake wavelength [$\mu$m]', fontsize=14, fontfamily='serif')
axs.set_ylabel(r'Transit depth [ppm]', fontsize=14, fontfamily='serif')

axs.tick_params(labelfontfamily='serif')
plt.setp(axs.get_yticklabels(), fontsize=12)
plt.setp(axs.get_xticklabels(), fontsize=12)

plt.show()