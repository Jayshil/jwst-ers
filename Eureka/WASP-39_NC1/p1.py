import numpy as np
import os
from transitspectroscopy import jwst as tjs
import pickle

f11 = os.getcwd() + '/Data/S0_NIRCam/jw01366002001_04103_00001-seg001_nrcalong_uncal.fits'
f22 = os.getcwd() + '/WASP-39_NC1/Outputs'

output_dict = tjs.stage1(datafile=f11, jump_threshold=10, get_times=True, get_wavelength_map=False,\
    maximum_cores='none', preamp_correction='loom', skip_steps=[], outputfolder=f22, quicklook=False,\
    uniluminated_mask=None, background_model=None, manual_bias=False, instrument='nircam')

out_data = {}
out_data['times'] = output_dict['times']
out_data['rateints'] = output_dict['rateints']
out_data['rateints_err'] = output_dict['rateints_err']
out_data['dq'] = output_dict['rateints_dq']

pickle.dump(out_data, open(f22 + '/rateints_seg001.pkl', 'wb'))