import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter
from matplotlib.patches import Wedge
from astropy.stats import mad_std
from tqdm import tqdm
import utils as utl
from pathlib import Path

# This file is to extract aperture photometry from NIRCam SW channel

visit = 'NRCSW'
method = 'photutils'
pin = os.getcwd() + '/RateInts/Corr_' + visit
pout = os.getcwd() + '/NRCSW/Outputs/' + visit
if not Path(pout + '/Figures').exists():
    os.mkdir(pout + '/Figures')

segs = []
for i in range(4):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

aprad = 85.
skrad1, skrad2 = None, None

# Arrays to store the data products
tim_all, fl_all, fle_all = np.array([]), np.array([]), np.array([])
cenr_all, cenc_all, bkg_all = np.array([]), np.array([]), np.array([])

for i in range(len(segs)):
    fphoto = Path(pout + '/Photometry_' + visit + '_' + method + '.dat')
    if fphoto.exists():
        print('>>>> --- Looks like the data already exisits...')
        continue
    else:
        pass
    seg = segs[i]
    print('>>>> --- Working on Segment ' + str(seg))
    # Loading the data
    corrected_data = np.load(pin + '/Corrected_data_seg' + seg + '.npy')
    corrected_errs = np.load(pin + '/Corrected_errors_seg' + seg + '.npy')
    times_bjd = np.load(pin + '/Times_bjd_seg' + seg + '.npy')

    # Saving times
    tim_all = np.hstack((tim_all, times_bjd))

    # Making a figure to display the data along with aperture and sky radii
    nint = np.random.randint(0, corrected_data.shape[0])
    cen1, cen2 = utl.find_center(image=corrected_data[nint,:,:], rmin=75, rmax=255, cmin=458, cmax=638)

    fig, ax = plt.subplots(figsize=(15,5))

    im = ax.imshow(corrected_data[nint,:,:])#, cmap='plasma')
    ax.errorbar(cen2, cen1, c='orangered', mfc='white', fmt='o')
    ax.axvline(458, color='yellow', ls='--')
    ax.axvline(638, color='yellow', ls='--')
    ax.axhline(75, color='yellow', ls='--')
    ax.axhline(255, color='yellow', ls='--')
    Aperture = Wedge((cen2, cen1), aprad, 0, 360, width=0.5, color='coral')
    ax.add_patch(Aperture)
    if (skrad1 != None) & (skrad2 != None):
        Skuannulus1 = Wedge((cen2, cen1), skrad1, 0, 360, width=0.5, color='cyan')
        Skuannulus2 = Wedge((cen2, cen1), skrad2, 0, 360, width=0.5, color='cyan')
        ax.add_patch(Skuannulus1)
        ax.add_patch(Skuannulus2)

    ax.set_ylim([0, 255])

    plt.title('Data and Apertures for an arbitrary integration (Int No: ' + str(nint) + ')')
    plt.tight_layout()
    plt.savefig(pout + '/Figures/Data_and_ApSky_seg' + seg + '.png', dpi=400)
    plt.close(fig)

    # For a quick show
    for integrations in tqdm(range(corrected_data.shape[0])):
        # Finding the centroids
        cen_r, cen_c = utl.find_center(image=corrected_data[integrations,:,:], rmin=75, rmax=255, cmin=458, cmax=638)
        # And now the aperture photometry
        fl1, fle1, bkg1 = utl.aperture_photometry(image=corrected_data[integrations,:,:],\
                                                  err=corrected_errs[integrations,:,:],\
                                                  cen_r=cen_r, cen_c=cen_c,
                                                  rad=aprad, sky_rad1=skrad1, sky_rad2=skrad2,\
                                                  method=method)
        # Saving it
        fl_all, fle_all = np.hstack((fl_all, fl1)), np.hstack((fle_all, fle1))
        cenr_all, cenc_all = np.hstack((cenr_all, cen_r)), np.hstack((cenc_all, cen_c))
        bkg_all = np.hstack((bkg_all, bkg1))

# Saving the whole dataset (or, loading it, if it already exists)
fphoto = Path(pout + '/Photometry_' + visit + '_' + method + '.dat')
if fphoto.exists():
    tim_all, fl_all, fle_all = np.loadtxt(pout + '/Photometry_' + visit + '_' + method + '.dat', usecols=(0,1,2), unpack=True)
    cenr_all, cenc_all, bkg_all = np.loadtxt(pout + '/Photometry_' + visit + '_' + method + '.dat', usecols=(3,4,5), unpack=True)
else:
    f12 = open(pout + '/Photometry_' + visit + '_' + method + '.dat', 'w')
    f12.write('# Time (MJD)\tFlux\tFlux_err\tCen row\tCen col\tBkg\n')
    for i in range(len(tim_all)):
        f12.write(str(tim_all[i]) + '\t' + str(fl_all[i]) + '\t' + str(fle_all[i]) + '\t' + \
                str(cenr_all[i]) + '\t' + str(cenc_all[i]) + '\t' + str(bkg_all[i]) + '\n')
    f12.close()

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 7.5), sharex=True)

axs[0].errorbar(tim_all, fl_all/np.median(fl_all), yerr=fle_all/np.median(fl_all), fmt='.', c='k')
axs[0].plot(tim_all, median_filter(fl_all/np.median(fl_all), size=100), 'r-', lw=2., zorder=100)
axs[0].set_ylabel('Relative Flux')

axs[1].errorbar(tim_all, cenr_all, fmt='.', c='k')
axs[1].plot(tim_all, median_filter(cenr_all, size=100), 'r-', lw=2., zorder=100)
axs[1].set_ylabel('Row Center')

axs[2].errorbar(tim_all, cenc_all, fmt='.', c='k')
axs[2].plot(tim_all, median_filter(cenc_all, size=100), 'r-', lw=2., zorder=100)
axs[2].set_ylabel('Column Center')

axs[3].errorbar(tim_all, bkg_all/np.median(fl_all), fmt='.', c='k')
axs[3].plot(tim_all, median_filter(bkg_all/np.median(fl_all), size=100), 'r-', lw=2., zorder=100)
axs[3].set_ylabel('Background')

axs[3].set_xlabel('Time (MJD)')

axs[0].set_title('Aperture Photometry (MAD = {:.4f} ppm) and Centroids'.format(mad_std(fl_all/np.median(fl_all)) * 1e6))

plt.tight_layout()
#plt.show()
plt.savefig(pout + '/Figures/Photometry_Centroids_' + visit + '_' + method + '.png', dpi=500)