import numpy as np
import matplotlib.pyplot as plt
from krithika import SpectroscopicLC, julietPlots
from krithika import plotstyles
from astropy.stats import mad_std
from krithika import pipe_mad, clip_outliers
import juliet
import os

import multiprocessing
multiprocessing.set_start_method('fork')

# This file is to calculate and fit white-light light curves.

# -----------------------------------------------------
#   Data path and file name
# -----------------------------------------------------
pin = os.getcwd() + '/NRSPR/Outputs'
pout = os.getcwd() + '/NRSPR/Analysis'
pout_juliet = os.getcwd() + '/NRSPR/Analysis/White_light'
instrument = 'NRSPR'

# -----------------------------------------------------
#  List of segments
# -----------------------------------------------------
segments = ['seg' + str(i).zfill(3) for i in range(1, 5)]

# -----------------------------------------------------
# Looping through segments to load the data
# -----------------------------------------------------
spectral_timeseries, spectral_err_timeseries = [], []
times_all = []
trace_loc_all = []
for seg in segments:
    print('Processing ' + seg + '...')
    # ------ Loading the data
    spectral_data = np.load(pin + '/Extracted_spectra_' + seg + '.npz')
    spec1d, var1d = spectral_data['spec1d'], spectral_data['var1d']
    times = spectral_data['times_bjd'] + 2400000.5
    # ------ Appending the data to the lists
    spectral_timeseries.append(spec1d)
    spectral_err_timeseries.append(np.sqrt(var1d))
    times_all.append(times)

    plt.axvline(times[-1], color='k', ls='--', zorder=-10)

    # ------ Loading the trace locations
    trace_loc = np.load(pin + '/Trace_' + seg + '.npz')
    trace_loc_all.append(trace_loc['dx1'])

spectral_timeseries = np.vstack(spectral_timeseries)
spectral_err_timeseries = np.vstack(spectral_err_timeseries)
times_all = np.hstack(times_all)
wavelengths = np.linspace(0.6, 5.3, spectral_timeseries.shape[1])
trace_loc_all = np.hstack(trace_loc_all)

# -----------------------------------------------------
#.  Use SpectroscopicLC to calculate the white-light light curve
# -----------------------------------------------------
SpLC = SpectroscopicLC(times=times_all, lc=spectral_timeseries, lc_errs=spectral_err_timeseries, wavelengths=wavelengths)
white_lc, white_lc_errs = SpLC.white_light_lc()

print( 'MAD of the white-light light curve: ' + str(pipe_mad(white_lc / np.median(white_lc)) * 1e6 ) )

# ------ Clipping outliers
### In flux
idx = clip_outliers(flux=white_lc, clip=5, width=51)
times_all = times_all[idx]
white_lc, white_lc_errs = white_lc[idx], white_lc_errs[idx]
trace_loc_all = trace_loc_all[idx]
### In trace location
idx = clip_outliers(flux=trace_loc_all, clip=5, width=51)
times_all = times_all[idx]
white_lc, white_lc_errs = white_lc[idx], white_lc_errs[idx]
trace_loc_all = trace_loc_all[idx]

# -----------------------------------------------------
#.   Fitting the light curve
# -----------------------------------------------------
# ------ A function to standardise the data
def standard( x ):
    return ( x - np.nanmedian(x) ) / mad_std(x)

## Saving the white-light light curve the way juliet expects it
tim, fl, fle = {}, {}, {}
tim[instrument], fl[instrument], fle[instrument] = times_all, white_lc / np.nanmedian(white_lc), white_lc_errs / np.nanmedian(white_lc)

for ins in tim.keys():
    plt.errorbar(tim[ins], fl[ins], yerr=fle[ins], fmt='.', label=ins)
    plt.show()

"""lin_pars = {}
lin_pars[instrument] = np.transpose( np.vstack( [ standard(times_all), standard(trace_loc_all) ] ) )

# Priors
## Some planetary parameters
per, per_err = 4.05527892, 0.00000086                       # Ivshina & Winn 2022
bjd0, bjd0_err = 2456401.39763, 0.00013                     # Ivshina & Winn 2022
ar, ar_err = 11.55, 0.13                                    # Fischer et al. 2016
inc, inc_err = 87.32, 0.17                                  # Mancini et al. 2018
b, b_err = 0.447, np.sqrt(0.041**2+0.055**2)                # Maciejewski et al. 2016
ecc, omega = 0., 90.                                        # Faedi et al. 2011

cycle = round((tim[instrument][0]-bjd0)/per)
t0, t0_err = bjd0 + cycle * per, np.sqrt(bjd0_err**2 + (cycle * per_err)**2)

## Planetary priors
par_P = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_' + instrument, 'q2_' + instrument, 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['normal', 'normal', 'uniform', 'normal', 'uniform', 'uniform', 'fixed', 'fixed', 'normal']
hyper_P = [[per, per_err], [t0, t0_err], [0., 1.], [b, b_err], [0., 1.], [0., 1.], 0, 90, [ar, ar_err]]

## Instrumental parameters
par_ins = ['mdilution_' + instrument, 'mflux_' + instrument, 'sigma_w_' + instrument]
dist_ins = ['fixed', 'normal', 'loguniform']
hyper_ins = [1., [0., 0.1], [0.1, 1e4]]

## Linear parameters
par_lin, dist_lin, hyper_lin = [], [], []
for i in range(lin_pars[instrument].shape[1]):
    par_lin = par_lin + ['theta' + str(i) + '_' + instrument]
    dist_lin = dist_lin + ['uniform']
    hyper_lin = hyper_lin + [[-1., 1.]]

## Total priors
par_tot = par_P + par_ins + par_lin
dist_tot = dist_P + dist_ins + dist_lin
hyper_tot = hyper_P + hyper_ins + hyper_lin

priors = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)

# -------- Fitting the light curve with juliet
dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, linear_regressors_lc=lin_pars, out_folder=pout_juliet)
res = dataset.fit(sampler='dynamic_dynesty', nthreads=14, nlive=500)

# ------------------------------------------
#   Plotting the results
# ------------------------------------------
jp = julietPlots(input_folder=pout_juliet, sampler='dynamic_dynesty', light_travel_delay=True, stellar_radius=0.843)

# Let's plot some cool results!
## Full model
fig, axs1, axs2 = jp.full_model_lc(instruments=[instrument], save=True, quantile_models=False)

## Fake allan deviation
fig, axs, _, _, _ = jp.plot_fake_allan_deviation(instruments=[instrument])
plt.savefig(pout_juliet + '/Fake_allan_deviation_' + instrument + '.png', dpi=300)

jp.phase_folded_lc(highres=True, quantile_models=False)
plt.savefig(pout_juliet + '/Phase_folded_lc_' + instrument + '.png', dpi=300)

jp.plot_corner()"""