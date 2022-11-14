import numpy as np
import matplotlib.pyplot as plt
import h5py
import juliet
import os
import utils as utl
import matplotlib.gridspec as gd
from gpcheops.utils import corner_plot

# This file is to analyse white light curve

pin = os.getcwd() + '/WASP-39_4/Stage4/S4_2022-11-07_wasp39_run2/ap7_bg9'
pout = os.getcwd() + '/WASP-39_4/Analysis/White-Light1'

f1 = h5py.File(pin + '/S4_wasp39_ap7_bg9_LCData.h5')
tim9, fl9, fle9, ycen9 = np.asarray(f1['time']), np.asarray(f1['flux_white']), np.asarray(f1['err_white']), np.asarray(f1['centroid_y'])
tim9 = tim9 + 2400000.5# + 0.0008578943306929432
# Removing Nan values
tim7, fl7, fle7, ycen7 = tim9[~np.isnan(fl9)], fl9[~np.isnan(fl9)], fle9[~np.isnan(fl9)], ycen9[~np.isnan(fl9)]

# Outlier removal
## For ycen
msk1 = utl.outlier_removal_ycen(ycen7)
tim7, fl7, fle7, ycen7 = tim7[msk1], fl7[msk1], fle7[msk1], ycen7[msk1]
msk2 = utl.outlier_removal(tim7, fl7, fle7, clip=10)
tim7, fl7, fle7 = tim7[msk2], fl7[msk2], fle7[msk2]

# Normalizing the lightcurve
tim7, fl7, fle7 = tim7, fl7/np.median(fl7), fle7/np.median(fl7)

# Some planetary parameters
per, per_err = 4.05527892, 0.00000086          # Ivshina & Winn 2022
bjd0, bjd0_err = 2456401.39763, 0.00013        # Ivshina & Winn 2022
ar, ar_err = 11.55, 0.13                       # Fischer et al. 2016
bb, bb_err = 0.447, (0.041+0.055)/2            # Maciejewski et al. 2016
cycle = round((tim7[0]-bjd0)/per)
tc1 = np.random.normal(bjd0, bjd0_err, 100000) + (cycle*np.random.normal(per, per_err, 100000))

# Fitting using juliet
## Instrument
instrument = 'JWST-WHT'
## Dataset
tim, fl, fle = {}, {}, {}
gp_pars, lin_pars = {}, {}
tim[instrument], fl[instrument], fle[instrument] = tim7, fl7, fle7
gp_pars[instrument] = tim7
lins = np.vstack([tim7-tim7[0], (tim7-tim7[0])**2])
lin_pars[instrument] = np.transpose(lins)

## Priors
### Planetary parameters
par_P = ['P_p1', 't0_p1', 'p_p1_' + instrument, 'b_p1', 'q1_' + instrument, 'q2_' + instrument, 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['fixed', 'normal', 'uniform', 'normal', 'uniform', 'uniform', 'fixed', 'fixed', 'normal']
hyper_P = [per, [np.median(tc1), np.std(tc1)], [0., 1.], [bb, bb_err], [0., 1.], [0., 1.], 0., 90., [ar, ar_err]]
### Instrumental parameters
par_ins = ['mflux_' + instrument, 'mdilution_' + instrument, 'sigma_w_' + instrument]
dist_ins = ['normal', 'fixed', 'loguniform']
hyper_ins = [[0., 0.5], 1., [0.1, 10000.]]
### Linear parameters
par_lin = ['theta0_' + instrument, 'theta1_' + instrument]
dist_lin = ['uniform', 'uniform']
hyper_lin = [[-3., 3.], [-3., 3.]]

### Total
par_tot = par_P + par_ins + par_lin
dist_tot = dist_P + dist_ins + dist_lin
hyper_tot = hyper_P + hyper_ins + hyper_lin

priors = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)

## And fitting
dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, linear_regressors_lc=lin_pars, out_folder=pout)
res = dataset.fit(sampler = 'dynesty', nthreads=4)

# Some plots
model = res.lc.evaluate(instrument)

# Let's make sure that it works:
fig = plt.figure(figsize=(16,9))
gs = gd.GridSpec(2,1, height_ratios=[2,1])

# Top panel
ax1 = plt.subplot(gs[0])
ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.', alpha=0.3)
ax1.plot(tim[instrument], model, c='k', zorder=100)
ax1.set_ylabel('Relative Flux')
ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
ax1.xaxis.set_major_formatter(plt.NullFormatter())

# Bottom panel
ax2 = plt.subplot(gs[1])
ax2.errorbar(tim[instrument], (fl[instrument]-model)*1e6, yerr=fle[instrument]*1e6, fmt='.', alpha=0.3)
ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
ax2.set_ylabel('Residuals (ppm)')
ax2.set_xlabel('Time (BJD)')
ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))

plt.savefig(pout + '/full_model.png')

corner_plot(pout, False)