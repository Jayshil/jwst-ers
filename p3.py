import numpy as np
import matplotlib.pyplot as plt
import h5py
import juliet
import os
import utils as utl
import matplotlib.gridspec as gd
from gpcheops.utils import corner_plot

pin = os.getcwd() + '/WASP-39_4/Stage4/S4_2022-11-07_wasp39_run2/ap7_bg9'
pout = os.getcwd() + '/WASP-39_4/Analysis/Spectra_ind'

f1 = h5py.File(pin + '/S4_wasp39_ap7_bg9_LCData.h5')
instruments = np.array([])
for i in range(len(f1['wave_mid'])):
    instruments = np.hstack((instruments,'CH' + str(i)))

# Planetary parameters
per = 4.05527892                               # Ivshina & Winn 2022
ar = 11.3535042768                             # Fixed from white-light analysis
bb = 0.4528981002                              # Fixed from white-light analysis
tc1 = 2459771.3355443380                       # Fixed from white-light analysis
q1, q2 = 0.0691207538, 0.1229216390            # Fixed from white-light analysis

for i in range(len(instruments)):
    # Making data
    tim7, fl7, fle7, ycen7 = np.asarray(f1['time']), np.asarray(f1['data'][i]), np.asarray(f1['err'][i]), np.asarray(f1['centroid_y'])
    tim7 = tim7 + 2400000.5# + 0.0008578943306929432
    # Removing nans
    tim7, fl7, fle7, ycen7 = tim7[~np.isnan(fl7)], fl7[~np.isnan(fl7)], fle7[~np.isnan(fl7)], ycen7[~np.isnan(fl7)]
    ## Outlier removal: ycen
    msk1 = utl.outlier_removal_ycen(ycen7)
    tim7, fl7, fle7, ycen7 = tim7[msk1], fl7[msk1], fle7[msk1], ycen7[msk1]
    ## Outlier removal: time
    msk2 = utl.outlier_removal(tim7, fl7, fle7, clip=10)
    tim7, fl7, fle7 = tim7[msk2], fl7[msk2], fle7[msk2]
    # Normalizing the lightcurve
    tim7, fl7, fle7 = tim7, fl7/np.median(fl7), fle7/np.median(fl7)
    # Data that juliet understand
    tim, fl, fle, lin_pars = {}, {}, {}, {}
    tim[instruments[i]], fl[instruments[i]], fle[instruments[i]] = tim7, fl7, fle7
    lins = np.vstack([tim7-tim7[0], (tim7-tim7[0])**2])
    # Priors
    ## Planetary priros
    par_P = ['P_p1', 't0_p1', 'p_p1_' + instruments[i], 'b_p1', 'q1_' + instruments[i], 'q2_' + instruments[i], 'ecc_p1', 'omega_p1', 'a_p1']
    dist_P = ['fixed', 'fixed', 'uniform', 'fixed', 'truncatednormal', 'truncatednormal', 'fixed', 'fixed', 'fixed']
    hyper_P = [per, tc1, [0., 1.], bb, [q1, 0.05, 0., 1.], [q2, 0.05, 0., 1.], 0., 90., ar]
    ## Instrumental priors
    par_ins = ['mflux_' + instruments[i], 'mdilution_' + instruments[i], 'sigma_w_' + instruments[i]]
    dist_ins = ['normal', 'fixed', 'loguniform']
    hyper_ins = [[0., 0.5], 1., [0.1, 10000.]]
    ## Linear priors
    par_lin = ['theta0_' + instruments[i], 'theta1_' + instruments[i]]
    dist_lin = ['uniform', 'uniform']
    hyper_lin = [[-3., 3.], [-3., 3.]]
    ## Total priros
    par_tot = par_P + par_ins + par_lin
    dist_tot = dist_P + dist_ins + dist_lin
    hyper_tot = hyper_P + hyper_ins + hyper_lin
    priors = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)
    # And fitting
    dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, linear_regressors_lc=lin_pars, out_folder=pout + '/' + instruments[i])
    res = dataset.fit(sampler = 'dynesty', nthreads=4)

    # Some plots
    instrument = instruments[i]
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

    plt.savefig(pout + '/' + instruments[i] + '/full_model.png')

    corner_plot(pout + '/' + instruments[i], False)