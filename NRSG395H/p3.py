import numpy as np
import matplotlib.pyplot as plt
import pickle
import juliet
import os
import utils as utl
import matplotlib.gridspec as gd
from astropy.stats import mad_std
from glob import glob
from pathlib import Path
from tqdm import tqdm
import multiprocessing
multiprocessing.set_start_method('fork')

# This file is to analyse spectroscopic lightcurves: for stark
# However, I am using multiprocesing, so multiple lightcurves analysed simultaneously

visit = 'NRSG395H'
analysis1 = 'SpectralLC7_NRS1'
chanel_nos = 1390
nthreads = 8
detector = 'nrs1'

# For `stark` data
pin = os.getcwd() + '/NRSG395H/Outputs/'

# Corresponding white-light lightcurve analysis
f12_joint_wht = glob(os.getcwd() + '/NRSG395H/Analysis/White-light-' + detector + '/*.pkl')[0]
post_joint_wht = pickle.load(open(f12_joint_wht, 'rb'))
post_joint_wht1 = post_joint_wht['posterior_samples']

# Wavelength map
wav_map = np.load(os.getcwd() + '/Data/wav_map_nrsg395h_' + detector + '.npy')
xpos = np.load(os.getcwd() + '/NRSG395H/Outputs/xpos_' + detector + '.npy')
# Loading median trace position
median_trace = np.load(os.getcwd() + '/NRSG395H/Outputs/median_trace_' + detector + '.npy')
wav_soln = np.zeros(len(xpos))
for i in range(len(xpos)):
    wav_soln[i] = wav_map[int(median_trace[i]), xpos[i]]

#-------------------------------------
#  Loading the data
#-------------------------------------
dataset1 = pickle.load(open(pin + '/Spectrum_cube_NRSG395H_' + detector + '.pkl', 'rb'))
times = dataset1['times']

## A giant cube with all lightcurves and their errors
all_lc, all_lc_var = np.vstack(dataset1['spectra']), np.vstack(dataset1['variance'])
all_lc_err = np.sqrt(all_lc_var)


## Spectroscopic lightcurve analysis
f16 = Path(os.getcwd() + '/NRSG395H/Analysis/' + analysis1 + '/wavelength_ch.dat')
f17 = Path(os.getcwd() + '/NRSG395H/Analysis/' + analysis1 + '/spectroscopic_lc_ch_' + str(chanel_nos) + '.pkl')
if f16.exists() and f17.exists():
    print('>>>> --- The spectroscopic lightcurves already exists...')
    print('         Loading them...')

    spectral_lcs = pickle.load(open(os.getcwd() + '/NRSG395H/Analysis/' + analysis1 + '/spectroscopic_lc_ch_' + str(chanel_nos) + '.pkl', 'rb'))
    spec_lc1, spec_lc_err1 = spectral_lcs['lc'], spectral_lcs['err']
else:
    spec_lc1, spec_lc_err1, wavs, wav_bins = utl.spectral_lc(all_lc, all_lc_err, wav_soln, chanel_nos)
    
    # Saving them!
    spectral_lcs = {}
    spectral_lcs['lc'] = spec_lc1
    spectral_lcs['err'] = spec_lc_err1

    pickle.dump(spectral_lcs, open(os.getcwd() + '/NRSG395H/Analysis/' + analysis1 + '/spectroscopic_lc_ch_' + str(chanel_nos) + '.pkl','wb'))

    ## Saving wavelengths and wavelength bins
    f12 = open(os.getcwd() + '/NRSG395H/Analysis/' + analysis1 + '/wavelength_ch.dat', 'w')
    f12.write('# Mid wavelength \t Bin size\n')
    for i in range(len(wavs)):
        f12.write(str(wavs[i]) + '\t' + str(wav_bins[i]) + '\n')
    f12.close()

# Storing all lightcurves in a big dictionary
all_lightcurve_data = {}
for i in range(spec_lc1.shape[1]):
    # Storing all lightcurves in a big dictionary
    all_lightcurve_data['CH' + str(i)] = {}
    ## And now storing the actual lightcurve data
    all_lightcurve_data['CH' + str(i)]['times'] = times
    all_lightcurve_data['CH' + str(i)]['lc'] = spec_lc1[:,i]
    all_lightcurve_data['CH' + str(i)]['err'] = spec_lc_err1[:,i]

# Planetary parameters
per, per_err = 4.05527892, 0.00000086                       # Ivshina & Winn 2022
## All parameters below are from White-light light curve analysis
bjd0, bjd0_err = np.nanmedian(post_joint_wht1['t0_p1']), np.nanstd(post_joint_wht1['t0_p1'])
ar, ar_err = np.nanmedian(post_joint_wht1['a_p1']), np.nanstd(post_joint_wht1['a_p1'])
bb, bb_err = np.nanmedian(post_joint_wht1['b_p1']), np.nanstd(post_joint_wht1['b_p1'])

## For generating priors
def generate_priors(instrument):
    ### Planetary parameters
    par_P = ['P_p1', 't0_p1', 'p_p1_' + instrument, 'b_p1', 'ecc_p1', 'omega_p1', 'a_p1', 'q1_' + instrument, 'q2_' + instrument]
    dist_P = ['fixed', 'fixed', 'uniform', 'fixed', 'fixed', 'fixed', 'fixed', 'uniform', 'uniform']
    hyper_P = [per, bjd0, [0., 1.], bb, 0., 90., ar, [0., 1.], [0., 1.]]
    ### Instrumental parameters
    par_ins = ['mflux_' + instrument, 'mdilution_' + instrument, 'sigma_w_' + instrument]
    dist_ins = ['normal', 'fixed', 'loguniform']
    hyper_ins = [[0., 0.1], 1., [1.e-3, 10000.]]
    ### Linear parameters
    par_lin = ['theta0_' + instrument, 'theta1_' + instrument, 'theta2_' + instrument]
    dist_lin = ['uniform', 'uniform', 'uniform']#, 'uniform', 'uniform']
    hyper_lin = [[-1., 1.], [-1., 1.], [-1., 1.]]#, [-1., 1.], [-1., 1.]]

    ### Total
    par_tot = par_P + par_ins + par_lin
    dist_tot = dist_P + dist_ins + dist_lin
    hyper_tot = hyper_P + hyper_ins + hyper_lin

    priors = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)

    return priors


# MultiProcessing helper function (does the actual fitting)
def fit_lc(lightcurves, ch_name):
    print('---------------------------------')
    print('Working on Channel: ' + ch_name)
    print('')
    # Output folder
    pout = os.getcwd() + '/NRSG395H/Analysis/' + analysis1 + '/' + ch_name
    f15 = Path(pout + '/_dynesty_NS_posteriors.pkl')
    f16 = Path(pout + '/model_resids.dat')
    f17 = Path(pout + '/posteriors.dat')
    if f15.exists() and f16.exists() and f17.exists():
        print('>>>> --- The result files already exists...')
        print('         Continuing to the next channel...')
        res = np.zeros(10)
    else:
        # Extracting the data
        tim9, fl9, fle9 = lightcurves['times'], lightcurves['lc'], lightcurves['err']
        
        # Removing Nan values
        tim7, fl7, fle7 = tim9[~np.isnan(fl9)], fl9[~np.isnan(fl9)], fle9[~np.isnan(fl9)]

        # Normalizing the lightcurve
        tim7, fl7, fle7 = tim7, fl7/np.median(fl7), fle7/np.median(fl7)

        step1, step2 = np.ones(len(tim7)), np.ones(len(tim7))
        step1[0:261], step2[263:] = 0., 0.

        # Making data such that juliet can understand
        tim, fl, fle = {}, {}, {}
        lin_pars = {}
        tim[ch_name], fl[ch_name], fle[ch_name] = tim7, fl7, fle7
        lins = np.vstack([ (tim7-np.median(tim7)) / mad_std(tim7), step1, step2 ])

        lin_pars[ch_name] = np.transpose(lins)

        # Priors
        priors = generate_priors(ch_name)

        # Fitting
        dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, linear_regressors_lc=lin_pars, out_folder=pout)
        res = dataset.fit(sampler = 'dynesty')#, nthreads=8)

        # Some plots
        model = res.lc.evaluate(ch_name)
        residuals = fl[ch_name]-model

        data12 = np.vstack((model, residuals))
        np.savetxt(pout + '/model_resids.dat', np.transpose(data12))

        print('>>>> --- Done!!')
    return res

# Function that does the multiprocessing
def multi_fit_lcs(lightcurves, nthreads=4):
    input_data = [(lightcurves[lc], lc) for lc in lightcurves]
        
    with multiprocessing.Pool(nthreads) as p:
        result_list = p.starmap(fit_lc, input_data)
                
    return np.array(result_list)

# And calling the function
res_all = multi_fit_lcs(all_lightcurve_data, nthreads=nthreads)

# To make figures
res_all = {}
for i in tqdm(range(spec_lc1.shape[1])):
    pout = os.getcwd() + '/NRSG395H/Analysis/' + analysis1 + '/CH' + str(i)
    model, residuals = np.loadtxt(pout + '/model_resids.dat', usecols=(0,1), unpack=True)
    res_all[i] = residuals


fig = plt.figure(figsize=(8,6))
for i in tqdm(range(spec_lc1.shape[1])):
    # Alan deviation plot
    res_standard = (res_all[i] - np.nanmedian(res_all[i])) / mad_std(res_all[i])
    rms, stderr, binsz = utl.computeRMS(res_standard, binstep=1)
    normfactor = 1e-6

    plt.plot(binsz, rms / normfactor, color='black', lw=1.5,
                    label='Fit RMS', zorder=3, alpha=0.5)
    plt.plot(binsz, stderr / normfactor, color='red', ls='-', lw=2,
                    label=r'Std. Err. ($1/\sqrt{N}$)', zorder=1)
    #plt.xlim(0.95, binsz[-1] * 2)
    #plt.ylim(stderr[-1] / normfactor / 2., stderr[0] / normfactor * 2.)
    if i == 0:
        plt.legend(loc='best')

plt.xlabel("Bin Size (N frames)", fontsize=14)
plt.ylabel("RMS (ppm)", fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.getcwd() + '/NRSG395H/Analysis/' + analysis1 + '/alan_deviation_' + detector + '.png')
plt.close(fig)