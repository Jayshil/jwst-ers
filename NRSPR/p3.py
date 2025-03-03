import numpy as np
import matplotlib.pyplot as plt
from utils import outlier_removal, computeRMS
import matplotlib.gridspec as gd
from poetss import poetss
import juliet
import pickle
import os

import multiprocessing
multiprocessing.set_start_method('fork')

pin = os.getcwd() + '/NRSPR/Outputs/'
pout = os.getcwd() + '/NRSPR/Analysis/White-light'
catwoman = False

def pipe_mad(data):
    return np.nanmedian(np.abs(np.diff(data, axis=0)), axis=0)

## Segment!!!
segs = []
for i in range(3):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

# Data files
instrument = 'NRSPR'
tim, fl, fle = {}, {}, {}

# --------------------------------------------------------------------------
#
#                       Loading the data
#
# --------------------------------------------------------------------------
lc_all, lc_err_all = [], []
times = np.array([])
for i in range(len(segs)):
    dataset1 = pickle.load(open(pin + '/Spectrum_cube_NRSPR_' + str(i+1) + '.pkl', 'rb'))
    lc_all.append(dataset1['spectra'])
    lc_err_all.append(dataset1['variance'])
    times = np.hstack((times, dataset1['times']))

## A giant cube with all lightcurves and their errors
lc1, lc_err1 = np.vstack(lc_all), np.vstack(lc_err_all)
lc_err1 = np.sqrt(lc_err1)

## White-light lightcurve
wht_light_lc, wht_light_err = poetss.white_light(lc1, lc_err1)

# And the final lightcurve
tim9, fl9, fle9 = times, wht_light_lc, wht_light_err

# Removing Nan values
tim7, fl7, fle7 = tim9[~np.isnan(fl9)], fl9[~np.isnan(fl9)], fle9[~np.isnan(fl9)]

# Outlier removal
msk2 = outlier_removal(tim7, fl7, fle7, clip=5, msk1=False)
tim7, fl7, fle7 = tim7[msk2], fl7[msk2], fle7[msk2]

# Saving them!
tim[instrument], fl[instrument], fle[instrument] = tim7, fl7/np.median(fl7), fle7/np.median(fl7)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
#
#                    Planetary parameters and priors
#
# --------------------------------------------------------------------------

# ------- Defining the priors
## Some planetary parameters
per, per_err = 4.05527892, 0.00000086                       # Ivshina & Winn 2022
bjd0, bjd0_err = 2456401.39763, 0.00013                     # Ivshina & Winn 2022
ar, ar_err = 11.55, 0.13                                    # Fischer et al. 2016
inc, inc_err = 87.32, 0.17                                  # Mancini et al. 2018
bb, bb_err = 0.447, (0.041+0.055)/2                         # Maciejewski et al. 2016
ecc, omega = 0., 90.                                        # Faedi et al. 2011

cycle = round((tim[instrument][0]-bjd0)/per)
tc1 = np.random.normal(bjd0, bjd0_err, 100000) + (cycle*np.random.normal(per, per_err, 100000))

par_P = ['P_p1', 't0_p1', 'b_p1', 'q1_' + instrument, 'q2_' + instrument, 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['fixed', 'normal', 'truncatednormal', 'uniform', 'uniform', 'fixed', 'fixed', 'normal']
hyper_P = [per, [np.median(tc1), np.std(tc1)], [bb, 3*bb_err, 0., 1.], [0., 1.], [0., 1.], ecc, omega, [ar, 3*ar_err]]
if not catwoman:
    par_P = par_P + ['p_p1_' + instrument]
    dist_P = dist_P + ['uniform']
    hyper_P = hyper_P + [[0., 1.]]
else:
    par_P = par_P + ['p1_p1_' + instrument, 'p2_p1_' + instrument, 'phi_p1']
    dist_P = dist_P + ['uniform', 'uniform', 'fixed']
    hyper_P = hyper_P + [[0., 1.], [0., 1.], 90.]

### Instrumental parameters
par_ins = ['mdilution_' + instrument, 'mflux_' + instrument, 'sigma_w_' + instrument]
dist_ins = ['fixed', 'normal', 'loguniform']
hyper_ins = [1.0, [0., 0.1], [0.1, 10000.]]

### GP parameters
par_gp = ['GP_sigma_' + instrument, 'GP_timescale_' + instrument, 'GP_rho_' + instrument]
dist_gp = ['loguniform', 'loguniform', 'loguniform']
hyper_gp = [[1e-5, 10000.], [1e-3, 1e2], [1e-3, 1e2]]

## Total priors
par_tot = par_P + par_ins + par_gp
dist_tot = dist_P + dist_ins + dist_gp
hyper_tot = hyper_P + hyper_ins + hyper_gp

priors = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
#
#                             Juliet fitting
#
# --------------------------------------------------------------------------


# ------- And, fitting
dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, GP_regressors_lc=tim,\
                      out_folder=pout)
res = dataset.fit(sampler = 'dynesty', nthreads=8)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
#
#                             Plots
#
# --------------------------------------------------------------------------

# Full model and errors on it
model, model_uerr, model_derr, comps = res.lc.evaluate(instrument, return_err=True, return_components=True, all_samples=True)
gp_model = res.lc.model[instrument]['GP']

# Out-of-transit baseline flux
mflx = np.median(res.posteriors['posterior_samples']['mflux_' + instrument])

# Detrended flux
fl9 = (fl[instrument] - gp_model) * (1 + mflx)

# Detrended error
errs1 = np.median(res.posteriors['posterior_samples']['sigma_w_' + instrument] * 1e-6)
fle9 = np.sqrt((errs1**2) + (fle[instrument]**2))

# Residuals
resid = (fl[instrument]-model)*1e6

# Transit model
tmodel = (model - gp_model) * (1 + mflx)


## --------------------------
##       Full model
## --------------------------
# For Full model
fig = plt.figure(figsize=(16/1.5,9/1.5))
gs = gd.GridSpec(2,1, height_ratios=[2,1])

# Top panel
ax1 = plt.subplot(gs[0])
ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.', c='cornflowerblue')
ax1.plot(tim[instrument], model, c='k', zorder=100)
ax1.set_ylabel('Relative Flux', fontsize=14)
ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
plt.setp(ax1.get_xticklabels(), fontsize=12)
plt.setp(ax1.get_yticklabels(), fontsize=12)
ax1.xaxis.set_major_formatter(plt.NullFormatter())
ax1.set_title('Full model', fontsize=15)

# Bottom panel
ax2 = plt.subplot(gs[1])
ax2.errorbar(tim[instrument], resid, yerr=fle9*1e6, c='cornflowerblue', fmt='.')
ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
ax2.set_ylabel('Residuals (ppm)', fontsize=14)
ax2.set_xlabel('Time (BJD)', fontsize=14)
ax2.set_ylim(-10000,10000)
ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
plt.setp(ax2.get_xticklabels(), fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=12)
plt.savefig(pout + '/Full_model.png')

## --------------------------
##      Detrended model
## --------------------------

# For Detrended
fig = plt.figure(figsize=(16/1.5,9/1.5))
gs = gd.GridSpec(2,1, height_ratios=[2,1])

# Top panel
ax1 = plt.subplot(gs[0])
ax1.errorbar(tim[instrument], fl9, yerr=fle9, fmt='.', c='cornflowerblue')
ax1.plot(tim[instrument], tmodel, c='k', zorder=100)
ax1.set_ylabel('Relative Flux', fontsize=14)
ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
plt.setp(ax1.get_xticklabels(), fontsize=12)
plt.setp(ax1.get_yticklabels(), fontsize=12)
ax1.xaxis.set_major_formatter(plt.NullFormatter())
ax1.set_title('Detrended transit model', fontsize=15)

# Bottom panel
ax2 = plt.subplot(gs[1])
ax2.errorbar(tim[instrument], resid, yerr=fle9*1e6, c='cornflowerblue', fmt='.')
ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
ax2.set_ylabel('Residuals (ppm)', fontsize=14)
ax2.set_xlabel('Time (BJD)', fontsize=14)
ax2.set_ylim(-10000,10000)
ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
plt.setp(ax2.get_xticklabels(), fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=12)
plt.savefig(pout + '/Detrended_model.png')


## --------------------------
##       Allan deviation
## --------------------------
rms, stderr, binsz = computeRMS(resid*1e-6, binstep=1)
normfactor = 1e-6

mad_sigs = pipe_mad(resid)
rms_resids = np.sqrt(np.mean(resid**2))
print('Precision of the lightcurve: {:.4f} ppm'.format(mad_sigs))
print('RMS of the residuals: {:.4f} ppm'.format(rms_resids))
print('Median errorbar after the fitting: {:.4f} ppm'.format(np.nanmedian(fle9)*1e6))

plt.figure(figsize=(8,6))
plt.plot(binsz, rms / normfactor, color='black', lw=1.5,
                label='Fit RMS', zorder=3)
plt.plot(binsz, stderr / normfactor, color='red', ls='-', lw=2,
                label=r'Std. Err. ($1/\sqrt{N}$)', zorder=1)
plt.xlim(0.95, binsz[-1] * 2)
plt.ylim(stderr[-1] / normfactor / 2., stderr[0] / normfactor * 2.)
plt.xlabel("Bin Size (N frames)", fontsize=14)
plt.ylabel("RMS (ppm)", fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig(pout + '/Allan_deviation.png')