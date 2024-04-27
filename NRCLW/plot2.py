import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
import os
import juliet

# This file is to visualize eclipse spectrum

# Visits
visit = 'NRCLW'
ch_nos = 50
pin = os.getcwd() + '/NRCLW/Analysis/'
pout = os.getcwd() + '/NRCLW/Analysis/Figures'
analysis1 = 'SpectralLC5'

# For all channels
chs = []
for i in range(ch_nos):
    chs.append('CH' + str(i))

# Loading white-light analysis
f12_wht = glob(pin + '/White/*.pkl')[0]
post_wht = pickle.load(open(f12_wht, 'rb'))
post_all = post_wht['posterior_samples']
rprs_wht_qua = juliet.utils.get_quantiles(post_all['p_p1_' + visit])

# Extracting the spectroscopic data
wav_all, rprs_med_all, rprs_do_all, rprs_up_all = [], [], [], []
rprs_med, rprs_uerr, rprs_derr = np.zeros(len(chs)), np.zeros(len(chs)), np.zeros(len(chs))

for i in range(len(chs)):
    f12 = glob(pin + '/' + analysis1 + '/' + chs[i] + '/*.pkl')[0]
    post = pickle.load(open(f12, 'rb'))
    rp1 = post['posterior_samples']['p_p1_' + chs[i]]
    rp1_qua = juliet.utils.get_quantiles(rp1)
    rprs_med[i], rprs_uerr[i], rprs_derr[i] = rp1_qua[0], rp1_qua[1]-rp1_qua[0], rp1_qua[0]-rp1_qua[2]

wav, wav_bin = np.loadtxt(pin + '/' + analysis1 + '/wavelength_ch.dat', usecols=(0,1), unpack=True)

fig, ax = plt.subplots(figsize=(16/1.5,9/1.5))
ax.errorbar(wav, rprs_med, yerr=[rprs_derr, rprs_uerr], fmt='o', c='orangered', mfc='white', zorder=10, elinewidth=2, capthick=2, capsize=3)
#ax.plot(wav, fp_med, color='darkgreen', lw=1.5, zorder=5, alpha=0.7)
ax.fill_between(np.linspace(np.min(wav)-0.1,np.max(wav)+0.1,100), rprs_wht_qua[2], rprs_wht_qua[1], color='cornflowerblue', alpha=0.2, zorder=5)
ax.set_xlim([np.min(wav)-0.02, np.max(wav)+0.02])
ax.set_xlabel('Wavelength (in micron)', fontsize=14)
ax.set_ylabel('Rp/R*', fontsize=14)
ax.set_title('Transmission spectrum: Visit ' + str(visit), fontsize=14)
plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.show()
#plt.savefig(pout + '/Spec_' + visits[vis] + '.png', dpi=500)