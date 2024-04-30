import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from glob import glob
from luas import GP
from luas import kernels
from luas import LuasKernel
from luas.exoplanet import ld_from_kipping
import jax
import jax.numpy as jnp
from jaxoplanet.orbits import KeplerianOrbit
from jaxoplanet.light_curves import QuadLightCurve


visit = 'NRCLW'
analysis1 = 'SpectralLC7'
chanel_nos = 50
nthreads = 8
ch_nos_plt = 10

# For `stark` data
pin = os.getcwd() + '/NRCLW/Outputs/' + visit# + '_v2'
pout = os.getcwd() + '/NRCLW/Analysis/' + analysis1

# Corresponding white-light lightcurve analysis
f12_joint_wht = glob(os.getcwd() + '/NRCLW/Analysis/White/*.pkl')[0]
post_joint_wht = pickle.load(open(f12_joint_wht, 'rb'))
post_joint_wht1 = post_joint_wht['posterior_samples']
## Planetary parameters
per, per_err = 4.05527892, 0.00000086                       # Ivshina & Winn 2022
## All parameters below are from White-light light curve analysis
bjd0, bjd0_err = np.nanmedian(post_joint_wht1['t0_p1']), np.nanstd(post_joint_wht1['t0_p1'])
ar, ar_err = np.nanmedian(post_joint_wht1['a_p1']), np.nanstd(post_joint_wht1['a_p1'])
bb, bb_err = np.nanmedian(post_joint_wht1['b_p1']), np.nanstd(post_joint_wht1['b_p1'])
rprs_wht = np.nanmedian(post_joint_wht1['p_p1_NRCLW'])

# Wavelength map
wav_map = np.load(os.getcwd() + '/Data/wavelength_map_NRCLW.npy')
xpos = np.arange(25, 1600, 1)
wav_soln = np.zeros(len(xpos))
for i in range(len(xpos)):
    wav_soln[i] = wav_map[128, xpos[i]]

#-------------------------------------
#  Loading the data
#-------------------------------------
dataset1 = pickle.load(open(pin + '/Spectrum_cube_NRCLW.pkl', 'rb'))
times = dataset1['times'] - bjd0

## Spectroscopic lightcurve analysis
spectral_lcs = pickle.load(open(os.getcwd() + '/NRCLW/Analysis/' + analysis1 + '/spectroscopic_lc_ch_' + str(chanel_nos) + '.pkl', 'rb'))
spec_lc1_unnorm, spec_lc_err1_unnorm = spectral_lcs['lc'], spectral_lcs['err']
spec_lc1_unnorm, spec_lc_err1_unnorm = np.transpose(spec_lc1_unnorm), np.transpose(spec_lc_err1_unnorm)
wav = np.loadtxt(os.getcwd() + '/NRCLW/Analysis/' + analysis1 + '/wavelength_ch.dat', usecols=0, unpack=True)
N_l = len(wav)

## Normalising spectroscopic lightcurves
spec_lc1, spec_lc_err1 = spec_lc1_unnorm / np.nanmedian(spec_lc1_unnorm, axis=1)[:, None], spec_lc_err1_unnorm / np.nanmedian(spec_lc1_unnorm, axis=1)[:, None]

# --------------------------------------------------------------------------
#               Best-fit values
# --------------------------------------------------------------------------

# Loading the results
res_num = pickle.load(open(pout + '/res_numpyro.pkl', 'rb'))
posteriors = res_num.posterior
def best_post(key):
    post_key = np.asarray(posteriors[key])
    post_key = post_key.reshape((post_key.shape[0]*post_key.shape[1], post_key.shape[2]))
    return np.nanmedian(post_key, axis=0)

p_best_fit = {
    # Mean function parameters
    'rprs':best_post('rprs'),       # Radius ratio rho aka Rp/R* for each wavelength
    'q1':best_post('q1'),           # First quadratic limb darkening coefficient for each wavelength
    'q2':best_post('q2'),           # Second quadratic limb darkening coefficient for each wavelength
    'base':best_post('base'),       # Baseline flux out of transit for each wavelength
    'Tgrad':best_post('Tgrad'),     # Gradient in baseline flux for each wavelength (days^-1)
    
    # Hyperparameters
    'log_h':np.nanmedian(np.asarray(posteriors['log_h']).flatten()),         # log height scale
    'log_l_l':np.nanmedian(np.asarray(posteriors['log_l_l']).flatten()),     # log length scale in wavelength
    'log_l_t':np.nanmedian(np.asarray(posteriors['log_l_t']).flatten()),     # log length scale in time
    'log_sigma':best_post('log_sigma'),                              # log white noise amplitude for each wavelength
}

# --------------------------------------------------------------------------
#               Functions defining 1D and 2D light curves
# --------------------------------------------------------------------------
# First defining transit light curve for 1 light curve
def transit_light_curve(par, t):
    """Function from luas documentation
    Uses the package `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_ to calculate
    transit light curves using JAX assuming quadratic limb darkening and a simple circular orbit.
    
    This particular function will only compute a single transit light curve but JAX's vmap function
    can be used to calculate the transit light curve of multiple wavelength bands at once.
    
    Args:
        par (PyTree): The transit parameters stored in a PyTree/dictionary (see example above).
        t (JAXArray): Array of times to calculate the light curve at.           
    Returns:
        JAXArray: Array of flux values for each time input.
    """
    light_curve = QuadLightCurve.init(u1=par['u1'], u2=par['u2'])
    orbit = KeplerianOrbit.init(
        time_transit=0.,
        period=per,
        semimajor=ar,
        impact_param=bb,
        radius=par['rprs'],
    )
    
    flux = (par['base'] + par['Tgrad'] * (t-0.) ) * ( 1 + light_curve.light_curve(orbit, t)[0] )
    
    return flux


# JAX vmap function to map 2d variable
transit_light_curve_vmap = jax.vmap(
    transit_light_curve, 
    in_axes=(# Other parameters
        {'rprs':0, 'u1':0, 'u2':0, 'base':0, 'Tgrad':0
        },
        # Time parameter
         None,  
    ),
    # Specify the output dimension to expand along, this will default to 0 anyway
    # Will output extra flux values for each light curve as additional rows
    out_axes = 0,
)

# A funtion to make 2d light curves (given parameters)
def transit_light_curve_2D(p, x_l, x_t):
    # vmap requires that we only input the parameters which have been explicitly defined how they vectorise
    transit_params = ['rprs', 'base', 'Tgrad']
    mfp = {k:p[k] for k in transit_params}
    # Calculate limb darkening coefficients from the Kipping (2013) parameterisation.
    mfp['u1'], mfp['u2'] = ld_from_kipping(p['q1'], p['q2'])
    # Use the vmap of transit_light_curve to calculate a 2D array of shape (M, N) of flux values
    # For M wavelengths and N time points.
    return transit_light_curve_vmap(mfp, x_t)


# --------------------------------------------------------------------------
#               Gaussian process definition
# --------------------------------------------------------------------------

# The wavelength kernel functions take the wavelength regression variable(s) x_l as input (of shape (N_l) or (d_l, N_l))
def Kl_fn(hp, wav1, wav2, wn = True):
    Kl = jnp.exp(2*hp['log_h'])*kernels.squared_exp(wav1, wav2, jnp.exp(hp['log_l_l']))
    return Kl

# The time kernel functions take the time regression variable(s) x_t as input (of shape (N_t) or (d_t, N_t))
def Kt_fn(hp, tim1, tim2, wn = True):
    return kernels.squared_exp(tim1, tim2, jnp.exp(hp['log_l_t']))

def Sl_fn(hp, x_l1, x_l2, wn = True):
    Sl = jnp.zeros((x_l1.shape[-1], x_l2.shape[-1]))   
    if wn:
        Sl += jnp.diag(jnp.exp(2*hp["log_sigma"]))
    return Sl
Sl_fn.decomp = "diag" # Sl is a diagonal matrix

def St_fn(p, x_t1, x_t2, wn = True):
    return jnp.eye(x_t1.shape[-1])
St_fn.decomp = "diag" # St is a diagonal matrix

# Build a LuasKernel object using these component kernel functions
# The full covariance matrix applied to the data will be K = Kl KRON Kt + Sl KRON St
kernel = LuasKernel(Kl = Kl_fn, Kt = Kt_fn, Sl = Sl_fn, St = St_fn,
                    use_stored_values = True,)


# --------------------------------------------------------------------------
#               Initialising GPo= object
# --------------------------------------------------------------------------

# Initialise our GP object
# Make sure to include the mean function and log prior function if you're using them
gp = GP(kernel,  # Kernel object to use
        wav,     # Regression variable(s) along wavelength/vertical dimension
        times,     # Regression variable(s) along time/horizontal dimension
        mf = transit_light_curve_2D,  # (optional) mean function to use, defaults to zeros
       )

best_gp_model, _, best_tra_model_gp = gp.predict(p=p_best_fit, Y=spec_lc1, x_l_pred=wav, x_t_pred=times)
best_tra_model = transit_light_curve_2D(p=p_best_fit, x_l=wav, x_t=times)


for i in range(50):
    ch_nos_plt = i
    plt.errorbar(times, spec_lc1[ch_nos_plt,:], fmt='.')
    plt.plot(times, best_gp_model[ch_nos_plt,:], 'k-', zorder=10)
    plt.plot(times, best_tra_model_gp[ch_nos_plt,:], 'b-', zorder=10)
    plt.title(ch_nos_plt)
    plt.show()