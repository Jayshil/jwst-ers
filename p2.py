import numpy as np
import matplotlib.pyplot as plt
import h5py
import juliet
import os
import utils as utl
import matplotlib.gridspec as gd
from pathlib import Path

pin = os.getcwd() + '/WASP-39_4/Stage4/S4_2022-11-07_wasp39_run1/ap7_bg9'
pout = os.getcwd() + '/WASP-39_4/Analysis/Spectra_all'

f1 = h5py.File(pin + '/S4_wasp39_ap7_bg9_LCData.h5')
instruments = np.array([])
for i in range(len(f1['wave_mid'])):
    instruments = np.hstack((instruments,'CH' + str(i)))
instruments = np.hstack((instruments, 'WHITE'))

# Making the dataset:
tim, fl, fle = {}, {}, {}
gp_pars, lin_pars = {}, {}
for i in range(len(instruments)):
    if instruments[i] != 'WHITE':
        tim7, fl7, fle7 = np.asarray(f1['time']), np.asarray(f1['data'][i]), np.asarray(f1['err'][i])
        tim7 = tim7 + 2400000.5# + 0.0008578943306929432
        # Removing nans
        tim7, fl7, fle7 = tim7[~np.isnan(fl7)], fl7[~np.isnan(fl7)], fle7[~np.isnan(fl7)]
    else:
        tim7, fl7, fle7 = np.asarray(f1['time']), np.asarray(f1['flux_white']), np.asarray(f1['err_white'])
        tim7 = tim7 + 2400000.5# + 0.0008578943306929432
        # Removing Nan values
        tim7, fl7, fle7 = tim7[~np.isnan(fl7)], fl7[~np.isnan(fl7)], fle7[~np.isnan(fl7)]
    # Outlier removal
    msk1 = utl.outlier_removal(tim7, fl7, fle7, clip=10)
    tim7, fl7, fle7 = tim7[msk1], fl7[msk1], fle7[msk1]
    # Normalizing the lightcurve
    tim7, fl7, fle7 = tim7, fl7/np.median(fl7), fle7/np.median(fl7)
    tim[instruments[i]], fl[instruments[i]], fle[instruments[i]] = tim7, fl7, fle7
    gp_pars[instruments[i]] = tim7
    # Linear parameters
    lins = np.vstack([tim7])
    lin_pars[instruments[i]] = np.transpose(lins)

# Some planetary parameters
per, per_err = 4.05527892, 0.00000086          # Ivshina & Winn 2022
bjd0, bjd0_err = 2456401.39763, 0.00013        # Ivshina & Winn 2022
ar, ar_err = 11.55, 0.13                       # Fischer et al. 2016
bb, bb_err = 0.447, (0.041+0.055)/2            # Maciejewski et al. 2016
cycle = round((tim['WHITE'][0]-bjd0)/per)
tc1 = np.random.normal(bjd0, bjd0_err, 100000) + (cycle*np.random.normal(per, per_err, 100000))

## Priors
### Planetary parameters
par_P = ['P_p1', 't0_p1', 'b_p1', 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['fixed', 'normal', 'normal', 'fixed', 'fixed', 'normal']
hyper_P = [per, [np.median(tc1), np.std(tc1)], [bb, bb_err], 0., 90., [ar, ar_err]]
#### For p_p1 and LDCs
par_ps_ldcs, dist_ps_ldcs, hyper_ps_ldcs = [], [], []
for i in range(len(instruments)):
    par_ps_ldcs = par_ps_ldcs + ['p_p1_' + instruments[i], 'q1_' + instruments[i], 'q2_' + instruments[i]]
    dist_ps_ldcs = dist_ps_ldcs + ['uniform', 'uniform', 'uniform']
    hyper_ps_ldcs = hyper_ps_ldcs + [[0., 1.], [0., 1.], [0., 1.]]
### Instrumental parameters
par_ins, dist_ins, hyper_ins = [], [], []
for i in range(len(instruments)):
    par_ins = par_ins + ['mflux_' + instruments[i], 'mdilution_' + instruments[i], 'sigma_w_' + instruments[i]]
    dist_ins = dist_ins + ['normal', 'fixed', 'loguniform']
    hyper_ins = hyper_ins + [[0., 0.1], 1., [0.1, 100000.]]
### GP parameters
par_gp, dist_gp, hyper_gp = [], [], []
for i in range(len(instruments)):
    par_gp = par_gp + ['GP_sigma_' + instruments[i], 'GP_timescale_' + instruments[i], 'GP_rho_' + instruments[i]]
    dist_gp = dist_gp + ['loguniform', 'loguniform', 'loguniform']
    hyper_gp = hyper_gp + [[1e-5, 10000.], [1e-3,1e2], [1e-3,1e2]]
### Linear parameters
par_lin, dist_lin, hyper_lin = [], [], []
for i in range(len(instruments)):
    par_lin.append('theta0_' + instruments[i])
    dist_lin.append('uniform')
    hyper_lin.append([-1., 1.])
### Total
par_tot = par_P + par_ps_ldcs + par_ins# + par_gp + par_lin
dist_tot = dist_P + dist_ps_ldcs + dist_ins# + dist_gp + dist_lin
hyper_tot = hyper_P + hyper_ps_ldcs + hyper_ins# + hyper_gp + hyper_lin

priors = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)

## And fitting
#dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, GP_regressors_lc=gp_pars,\
#    linear_regressors_lc=lin_pars, out_folder=os.getcwd() + '/Analysis/Spectra')
dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, out_folder=pout)
res = dataset.fit(sampler = 'dynesty', nthreads=8)

path_plots = Path(pout + '/Plots')
if not path_plots.exists():
    os.mkdir(pout + '/Plots')

for i in range(len(instruments)):
    instrument = instruments[i]
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

    plt.savefig(pout + '/Plots/full_model_' + instrument + '.png')