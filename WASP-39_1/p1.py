import numpy as np
import matplotlib.pyplot as plt
import h5py
import juliet
import os
import utils as utl
import matplotlib.gridspec as gd

f1 = h5py.File(os.getcwd() + '/Stage4/S4_2022-10-20_wasp39_run1/ap8_bg9/S4_wasp39_ap8_bg9_LCData.h5')
tim7, fl7, fle7 = np.asarray(f1['time']), np.asarray(f1['flux_white']), np.asarray(f1['err_white'])
tim7 = tim7 + 2400000.5
# Removing Nan values
tim7, fl7, fle7 = tim7[~np.isnan(fl7)], fl7[~np.isnan(fl7)], fle7[~np.isnan(fl7)]

# Outlier removal
msk1 = utl.outlier_removal(tim7, fl7, fle7, clip=10)
tim7, fl7, fle7 = tim7[msk1], fl7[msk1], fle7[msk1]

# Normalizing the lightcurve
tim7, fl7, fle7 = tim7, fl7/np.median(fl7), fle7/np.median(fl7)

# Some planetary parameters
per, bjd0 = 4.0552941, 2455342.96913
ar, ar_err = 11.55, 0.13
cycle = round((tim7[0]-bjd0)/per)
tc1 = bjd0 + (cycle*per)

# Fitting using juliet
## Instrument
instrument = 'JWST-WHT'
## Dataset
tim, fl, fle = {}, {}, {}
tim[instrument], fl[instrument], fle[instrument] = tim7, fl7, fle7

## Priors
### Planetary parameters
par_P = ['P_p1', 't0_p1', 'p_p1_' + instrument, 'b_p1', 'q1_' + instrument, 'q2_' + instrument, 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['fixed', 'normal', 'uniform', 'uniform', 'uniform', 'uniform', 'fixed', 'fixed', 'normal']
hyper_P = [per, [tc1, 0.1], [0., 1.], [0., 1.], [0., 1.], [0., 1.], 0., 90., [ar, ar_err]]
### Instrumental parameters
par_ins = ['mflux_' + instrument, 'mdilution_' + instrument, 'sigma_w_' + instrument]
dist_ins = ['normal', 'fixed', 'loguniform']
hyper_ins = [[0., 0.1], 1., [0.1, 100000.]]
### Total
par_tot = par_P + par_ins
dist_tot = dist_P + dist_ins
hyper_tot = hyper_P + hyper_ins

priors = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)

## And fitting
dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, out_folder=os.getcwd() + '/Analysis/White_light1')
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

plt.savefig(os.getcwd() + '/Analysis/White_light/full_model.png')