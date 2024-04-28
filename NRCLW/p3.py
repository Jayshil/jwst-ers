import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import corner
from glob import glob
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.initialization import init_to_value
from jax import random
from luas.numpyro_ext import LuasNumPyro
from luas import GP
from luas import kernels
from luas import LuasKernel
from luas.exoplanet import ld_from_kipping
import jax
from jax.random import split, PRNGKey
import jax.numpy as jnp
from jaxoplanet.orbits import KeplerianOrbit
from jaxoplanet.light_curves import QuadLightCurve
import arviz as az
from copy import deepcopy
jax.config.update("jax_enable_x64", True)

cpu_cores = 2
numpyro.set_host_device_count(cpu_cores)

# This file is to analyse spectroscopic lightcurves: for stark
# However, I am using multiprocesing, so multiple lightcurves analysed simultaneously

visit = 'NRCLW'
analysis1 = 'SpectralLC7'
chanel_nos = 50
nthreads = 8

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
#               Initial values
# --------------------------------------------------------------------------

p_initial = {
    # Mean function parameters
    'rprs':rprs_wht*jnp.ones(len(wav)),       # Radius ratio rho aka Rp/R* for each wavelength
    'q1':jnp.linspace(0.8, 0.2, len(wav)),    # First quadratic limb darkening coefficient for each wavelength
    'q2':0.2*jnp.ones(len(wav)),              # Second quadratic limb darkening coefficient for each wavelength
    'base':1.*jnp.ones(len(wav)),             # Baseline flux out of transit for each wavelength
    'Tgrad':0.*jnp.ones(len(wav)),            # Gradient in baseline flux for each wavelength (days^-1)
    
    # Hyperparameters
    'log_h':jnp.log(5e-4)*jnp.ones(1),             # log height scale
    'log_l_l':jnp.log(0.1)*jnp.ones(1),            # log length scale in wavelength
    'log_l_t':jnp.log(0.011)*jnp.ones(1),          # log length scale in time
    'log_sigma':jnp.log(5e-4)*jnp.ones(len(wav)),  # log white noise amplitude for each wavelength
}

# --------------------------------------------------------------------------
#               Log priors
# --------------------------------------------------------------------------

def logPrior(p):
    u1_mean, u2_mean = jnp.linspace(0.18, 0.35, len(wav)), jnp.linspace(0.27, 0.54, len(wav))
    u1, u2 = ld_from_kipping(p['q1'], p['q2'])
    u1_priors = -0.5*((u1 - u1_mean)/0.01)**2
    u2_priors = -0.5*((u2 - u2_mean)/0.01)**2
    
    logPrior = u1_priors.sum() + u2_priors.sum()

    return logPrior.sum()


# --------------------------------------------------------------------------
#               Initialising GPo= object
# --------------------------------------------------------------------------

# Initialise our GP object
# Make sure to include the mean function and log prior function if you're using them
gp = GP(kernel,  # Kernel object to use
        wav,     # Regression variable(s) along wavelength/vertical dimension
        times,     # Regression variable(s) along time/horizontal dimension
        mf = transit_light_curve_2D,  # (optional) mean function to use, defaults to zeros
        logPrior = logPrior           # (optional) log prior function, defaults to zero
       )

gp.plot(p_initial, spec_lc1)
plt.show()


# --------------------------------------------------------------------------
#               NumPyro model
# --------------------------------------------------------------------------

min_log_l_l = np.log(np.diff(wav).min())
max_log_l_l = np.log(50*(wav[-1] - wav[0]))
min_log_l_t = np.log(np.diff(times).min())
max_log_l_t = np.log(3*(times[-1] - times[0]))

param_bounds = {
                # Bounds from Kipping (2013) at just between 0 and 1
                'q1':[jnp.array([0.]*len(wav)), jnp.array([1.]*len(wav))],
                'q2':[jnp.array([0.]*len(wav)), jnp.array([1.]*len(wav))],
    
                # Can optionally include bounds on other mean function parameters but often they will be well constrained by the data
                'rprs':[jnp.array([0.]*len(wav)), jnp.array([1.]*len(wav))],
    
                # Sometimes prior bounds on hyperparameters are important for sampling
                # However their choice can sometimes affect the results so use with caution
                'log_h':   [jnp.log(1e-6)*np.ones(1), jnp.log(1)*jnp.ones(1)],
                'log_l_l': [min_log_l_l*jnp.ones(1), max_log_l_l*jnp.ones(1)],
                'log_l_t': [min_log_l_t*jnp.ones(1), max_log_l_t*jnp.ones(1)],
                'log_sigma':[jnp.log(1e-6)*jnp.ones(len(wav)), jnp.log(1e-2)*jnp.ones(len(wav))],
}

def transit_model(Yobs):
    # Makes of copy of any parameters to be kept fixed during sampling
    var_dict = deepcopy(p_initial)
    
    # Specify the parameters we've given bounds for
    var_dict['rprs'] = numpyro.sample('rprs', dist.Uniform(low = param_bounds['rprs'][0],
                                                           high = param_bounds['rprs'][1]))
    var_dict['log_h'] = numpyro.sample("log_h", dist.Uniform(low = param_bounds['log_h'][0],
                                                             high = param_bounds['log_h'][1]))
    var_dict['log_l_l'] = numpyro.sample("log_l_l", dist.Uniform(low = param_bounds['log_l_l'][0],
                                                                 high = param_bounds['log_l_l'][1]))
    var_dict['log_l_t'] = numpyro.sample("log_l_t", dist.Uniform(low = param_bounds['log_l_t'][0],
                                                                 high = param_bounds['log_l_t'][1]))
    var_dict['log_sigma'] = numpyro.sample("log_sigma", dist.Uniform(low = param_bounds['log_sigma'][0],
                                                                     high = param_bounds['log_sigma'][1]))
    var_dict['q1'] = numpyro.sample('q1', dist.Uniform(low = param_bounds['q1'][0],
                                                       high = param_bounds['q1'][1]))
    var_dict['q2'] = numpyro.sample('q2', dist.Uniform(low = param_bounds['q2'][0],
                                                       high = param_bounds['q2'][1]))
    
    # Specify the unbounded parameters
    var_dict['base'] = numpyro.sample('base', dist.ImproperUniform(dist.constraints.real, (),
                                                                   event_shape = (len(wav),)))
    var_dict['Tgrad'] = numpyro.sample('Tgrad', dist.ImproperUniform(dist.constraints.real, (),
                                                                     event_shape = (len(wav),)))

    numpyro.sample('log_like', LuasNumPyro(gp = gp, var_dict = var_dict), obs = Yobs)


# --------------------------------------------------------------------------
#               Initial optimization using SVD
# --------------------------------------------------------------------------

# Define step size and number of optimisation steps
step_size = 1e-3
num_steps = 5000

# Uses adam optimiser and a Laplace approximation calculated from the hessian of the log posterior as a guide
optimizer = numpyro.optim.Adam(step_size=step_size)
guide = AutoLaplaceApproximation(transit_model, init_loc_fn = init_to_value(values=p_initial))

# Create a Stochastic Variational Inference (SVI) object with NumPyro
svi = SVI(transit_model, guide, optimizer, loss=Trace_ELBO())

# Run the optimiser and get the median parameters
svi_result = svi.run(random.PRNGKey(0), num_steps, spec_lc1)
params = svi_result.params
p_fit = guide.median(params)

# Combine best-fit values with fixed values for log posterior calculation
p_opt = deepcopy(p_initial)
p_opt.update(p_fit)

print("Starting log posterior value:", gp.logP(p_initial, spec_lc1))
print("New optimised log posterior value:", gp.logP(p_opt, spec_lc1))

# Returns the covariance matrix returned by the Laplace approximation
# Also returns a list of parameters which is the order the array is in
# This matches the way jax.flatten_util.ravel_pytree will sort the parameter PyTree into
cov_mat, ordered_param_list = gp.laplace_approx_with_bounds(
    p_opt,               # Make sure to use best-fit values
    spec_lc1,            # The observations being fit
    param_bounds,        # Specify the same bounds that will be used for the MCMC
    return_array = True, # May optionally return a nested PyTree if set to False which can be more readable
    regularise = True,   # Often necessary to regularise values that return negative covariance
    large = False,       # Setting this to True is more memory efficient which may be needed for large data sets
)

# --------------------------------------------------------------------------
#               NumPyro sampling
# --------------------------------------------------------------------------


### -------   And sampling
# Random numbers in jax are generated like this:
rng_seed = 42
rng_keys = split(PRNGKey(rng_seed), cpu_cores)

# Define a sampler, using here the No U-Turn Sampler (NUTS)
# with a dense mass matrix:
sampler = NUTS(transit_model, # Our transit model specified earlier
               init_strategy = init_to_value(values = p_opt), # Often works well to initialise near best-fit values
               inverse_mass_matrix = cov_mat, # Inverse mass matrix is the same as the tuning covariance matrix
               adapt_mass_matrix=False, # Often Laplace approximation works better than trying to tune many parameters
               dense_mass = True,       # Need a dense mass matrix to account for correlations between parameters
               regularize_mass_matrix = False, # Mass matrix should already be regularised
               )

# Monte Carlo sampling for a number of steps and parallel chains:
mcmc = MCMC(sampler, num_warmup=2_500, num_samples=2_500, num_chains=cpu_cores)

# Run the MCMC
mcmc.run(rng_keys, spec_lc1)#, tim, fle, fl)

# Using arviz to extract results
# arviz converts a numpyro MCMC object to an `InferenceData` object based on xarray:
result = az.from_numpyro(mcmc)
pickle.dump(result, open(pout + '/res_numpyro.pkl','wb'))

# Trace plots
_ = az.plot_trace(result)#, var_names=all_var_names)
plt.tight_layout()
#plt.show()
plt.savefig(pout + '/trace.png', dpi=500)

# Result summary
summary = az.summary(result)#, var_names=all_var_names)
print(summary)

# Corner plot
#truth = dict(zip(['tc', 'duration', 'bb', 'rprs', 'u1', 'u2', 'sig_w'], np.array([0.27, 2.7, -4]),))
#_ = corner.corner(result)#, var_names=all_var_names);#, truths=truth,);
#plt.show()
#plt.savefig(pout + '/corner.pdf')#, dpi=500)