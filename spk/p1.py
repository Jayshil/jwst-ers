import numpy as np
import matplotlib.pyplot as plt
import utils
import spelunker
import os
import pickle
from path import Path

visit = 'NRCSW'
pout = os.getcwd() + '/spk/Outputs/' + visit + '/'

# Photometric/SW data points
tim_sw, fl_sw1, fle_sw1 = np.loadtxt(os.getcwd() + '/NRCSW/Outputs/' + visit + '/Photometry_' + visit + '_photutils.dat', usecols=(0,1,2), unpack=True)
tim_sw = tim_sw + 2400000.5
fl_sw, fle_sw = fl_sw1/np.nanmedian(fl_sw1), fle_sw1/np.nanmedian(fl_sw1)
## Binning them (for plotting purposes)
sw_tbin, sw_flbin, sw_flebin, _ = utils.lcbin(time=tim_sw, flux=fl_sw, binwidth=0.05)
sw_tbin2, sw_flbin2, sw_flebin2, _ = utils.lcbin(time=tim_sw, flux=fl_sw, binwidth=0.5)

print('>>> --- Exposure time of Science data: {:.10f} sec'.format((tim_sw[1]-tim_sw[0])*24*60*60))

f1 = Path(pout + '/Guide_star_unbinned_' + visit + '.dat')
f2 = Path(pout + '/Gaussian_fit_unbinned_' + visit + '.pkl')

if f1.exists() and f2.exists():
    print('>>> --- Guide-star photometry already exists. Loading them...')
    # Loading the guide-star photometry
    fg_time, fg_flx_nonnormal = np.loadtxt(pout + '/Guide_star_unbinned_' + visit + '.dat', usecols=(0,1), unpack=True)
    fg_flx = fg_flx_nonnormal / np.nanmedian(fg_flx_nonnormal)
    print('>>> --- PSF properties of the guide-star already exists. Loading them...')
    # Loading the PSF properties
    results = pickle.load(open(pout + '/Gaussian_fit_unbinned_' + visit + '.pkl', 'rb'))
else:
    # Downloading the guide star data
    spk = spelunker.load(pid=1366, obs_num='2', visit='1', token='token', dir=pout, save=True)
    spk.optimize_photometry()
    fg_time = spk.fg_time + 2400000.5
    fg_flx_nonnormal = spk.fg_flux 
    fg_flx = fg_flx_nonnormal / np.nanmedian(spk.fg_flux)
    ## Saving the guide-star data at original exposure
    fname1 = open(pout + '/Guide_star_unbinned_' + visit + '.dat', 'w')
    for i in range(len(fg_time)):
        fname1.write(str(fg_time[i]) + '\t' + str(fg_flx_nonnormal[i]) + '\n')
    fname1.close()

    # And PSF properties of the guidestar
    spk.gauss2d_fit(ncpus=8)
    results = {}
    for key in list(spk.gaussfit_results.keys()):
        results[key] = spk.gaussfit_results[key].value
    pickle.dump(results, open(pout + '/Gaussian_fit_unbinned_' + visit + '.pkl','wb'))

print('>>> --- Exposure time of Guide Star data: {:.10f} sec'.format((fg_time[101]-fg_time[100])*24*60*60))

# Plotting unbinned guide star data
## Computing binning data (only for plotting purposes)
fg_tbin, fg_flbin, fg_flebin, _ = utils.lcbin(time=fg_time, flux=fg_flx, binwidth=0.0005)
fg_tbin2, fg_flbin2, fg_flebin2, _ = utils.lcbin(time=fg_time, flux=fg_flx, binwidth=0.005)

# For Full model
fig, ax = plt.subplots(figsize=(16/1.5,9/1.5))
# Top panel
ax.errorbar(fg_time, fg_flx, fmt='.', c='cornflowerblue', alpha=0.25)
ax.errorbar(fg_tbin, fg_flbin, yerr=fg_flebin, fmt='.', c='gray', alpha=0.7, zorder=50)
ax.errorbar(fg_tbin2, fg_flbin2, yerr=fg_flebin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax.set_ylabel('Relative Flux', fontsize=14)
ax.set_xlabel('Time (BJD)', fontsize=14)
ax.set_xlim(np.min(fg_time), np.max(fg_time))
plt.ylim([0.95,1.06])
plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
ax.set_title('Guide star (unbinned) data for Visit ' + str(visit), fontsize=15)
#plt.show()
plt.savefig(os.getcwd() + '/spk/Figures/GS_unbinned_' + visit + '.png', dpi=500)


# Binning the guide-star data to the exposure of the science data
f3 = Path(pout + '/Guide_star_binned_' + visit + '.dat')

if f3.exists():
    print('>>> --- Binned guide-star photometry already exists. Loading them...')
    fbin, fbinerr = np.loadtxt(pout + '/Guide_star_binned_' + visit + '.dat', usecols=(1,2), unpack=True)
else:
    fbin, fbinerr = utils.bin_fgs_to_science(tim_sw, fg_time, fg_flx)
    fname2 = open(pout + '/Guide_star_binned_' + visit + '.dat', 'w')
    for i in range(len(fbin)):
        fname2.write(str(tim_sw[i]) + '\t' + str(fbin[i]) + '\t' + str(fbinerr[i]) + '\n')
    fname2.close()


## Binning them (for plotting purposes)
fg_tbin3, fg_flbin3, fg_flebin3, _ = utils.lcbin(time=tim_sw, flux=fbin, binwidth=0.05)
fg_tbin4, fg_flbin4, fg_flebin4, _ = utils.lcbin(time=tim_sw, flux=fbin, binwidth=0.5)

# For Full model
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15/1.5,10/1.5), sharex=True)

ax[0].errorbar(tim_sw, fl_sw, fmt='.', c='cornflowerblue', alpha=0.25)
ax[0].errorbar(sw_tbin, sw_flbin, yerr=sw_flebin, fmt='.', c='gray', alpha=0.7, zorder=50)
ax[0].errorbar(sw_tbin2, sw_flbin2, yerr=sw_flebin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax[0].set_ylabel('Relative Flux', fontsize=14)
ax[0].set_title('SW photometry data for Visit ' + str(visit), fontsize=15)

# Top panel
ax[1].errorbar(tim_sw, fbin, fmt='.', c='cornflowerblue', alpha=0.25)
ax[1].errorbar(fg_tbin3, fg_flbin3, yerr=fg_flebin3, fmt='.', c='gray', alpha=0.7, zorder=50)
ax[1].errorbar(fg_tbin4, fg_flbin4, yerr=fg_flebin4, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax[1].set_ylabel('Relative Flux', fontsize=14)
ax[1].set_xlabel('Time (BJD)', fontsize=14)
ax[1].set_xlim(np.min(tim_sw), np.max(tim_sw))
#ax[2].set_ylim([0.98,1.02])

plt.setp(ax[0].get_xticklabels(), fontsize=12)
plt.setp(ax[0].get_yticklabels(), fontsize=12)
plt.setp(ax[1].get_xticklabels(), fontsize=12)
plt.setp(ax[1].get_yticklabels(), fontsize=12)

ax[1].set_title('Guide star (binned) data for Visit ' + str(visit), fontsize=15)

#plt.show()
plt.savefig(os.getcwd() + '/spk/Figures/GS_binned_' + visit + '.png', dpi=500)

# Guide-star PSF properties
f4 = Path(pout + '/Gaussian_fit_binned_' + visit + '.pkl')
if f4.exists():
    print('>>> --- PSF properties of the guide-star already exists. Loading them...')
    results_bin = pickle.load(open(pout + '/Gaussian_fit_binned_' + visit + '.pkl', 'rb'))
else:
    results_bin = {}
    for key in list(results.keys()):
        results_bin[key], _ = utils.bin_fgs_to_science(tim_sw, 
                                                    fg_time, 
                                                    results[key])
    pickle.dump(results_bin, open(pout + '/Gaussian_fit_binned_' + visit + '.pkl','wb'))


for key in list(results_bin.keys()):

    median = np.nanmedian(results_bin[key])
    std = np.nanmedian(np.abs(results_bin[key] - median)) * 1.4826

    ## For binning
    tbin5, res_bin5, rese_bin5, _ = utils.lcbin(time=tim_sw, flux=results_bin[key], binwidth=0.05)
    tbin6, res_bin6, rese_bin6, _ = utils.lcbin(time=tim_sw, flux=results_bin[key], binwidth=0.5)
    
    # For Full model
    fig, ax = plt.subplots(figsize=(15,5))
    # Top panel
    ax.errorbar(tim_sw, results_bin[key], fmt='.', c='cornflowerblue', alpha=0.25)
    ax.errorbar(tbin5, res_bin5, yerr=rese_bin5, fmt='.', c='gray', alpha=0.7, zorder=50)
    ax.errorbar(tbin6, res_bin6, yerr=rese_bin6, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
    ax.set_ylabel('Relative Flux', fontsize=14)
    ax.set_xlabel('Time (BJD)', fontsize=14)
    ax.set_xlim(np.min(tim_sw), np.max(tim_sw))
    ax.set_ylim(median-3*std,median+3*std)

    ax.set_title(key+' for FGS 2D Gaussian Fit', fontsize = 18)
    ax.set_xlabel('Time (BJD)', fontsize = 18)
    ax.set_ylabel(key, fontsize = 18)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    #plt.show()
    plt.savefig(os.getcwd() + '/spk/Figures/GS_PSF_' + key + '_' + visit + '.png', dpi=500)