import numpy as np
import os
from jwst.pipeline import calwebb_detector1
from jwst.pipeline import calwebb_spec2
from jwst import datamodels
from path import Path
from tqdm import tqdm

# This file is to generate a new ramp-fitting file from recent updated pipeline/calibration files.
# Works for all segments

visit = 'NIRSpec'
segs = []
for i in range(4):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))
#segs = segs[-2:]
p1 = os.getcwd() + '/Data/' + visit
p2 = os.getcwd() + '/RateInts/Ramp_' + visit

for i in range(len(segs)):
    print('---------------------------------')
    print('Working on Segment: ', segs[i])
    print('')
    f12 = Path(p2 + '/jw01366004001_04101_00001-seg' + segs[i] + '_nrs1_0_rampfitstep.fits')
    f13 = Path(p2 + '/jw01366004001_04101_00001-seg' + segs[i] + '_nrs1_1_rampfitstep.fits')
    f14 = Path(p2 + '/bmap_seg' + segs[i] + '.npy')
    if f12.exists() and f13.exists() and f14.exists():
        print('>>>> --- The rampfitting files already exist...')
        print('         Continuing to the next segment...')
        continue
    else:
        pass
    uncal = datamodels.RampModel(p1 + '/jw01366004001_04101_00001-seg' + segs[i] + '_nrs1_uncal.fits')
    dq_results = calwebb_detector1.dq_init_step.DQInitStep.call(uncal, save_results=False)
    saturation_results = calwebb_detector1.saturation_step.SaturationStep.call(dq_results,
                                                                            save_results=False)
    superbias_results = calwebb_detector1.superbias_step.SuperBiasStep.call(saturation_results,
                                                                            save_results=False)
    refpix_results = calwebb_detector1.refpix_step.RefPixStep.call(superbias_results,
                                                                        odd_even_columns=True,
                                                                        odd_even_rows=True,
                                                                        save_results=False)
    linearity_results = calwebb_detector1.linearity_step.LinearityStep.call(refpix_results,
                                                                            save_results=False)
    darkcurrent_results = calwebb_detector1.dark_current_step.DarkCurrentStep.call(linearity_results,
                                                                                save_results=False)
    rampfitting_results = calwebb_detector1.ramp_fit_step.RampFitStep.call(darkcurrent_results,
                                                                        output_dir=p2,
                                                                        save_results=True)
    try:
        wmap_filename = 'wavelength_map_' + visit + '.npy'
        if not Path(os.getcwd() + '/Data/' + wmap_filename).exists():
            assign_wcs_results = calwebb_spec2.assign_wcs_step.AssignWcsStep.call(rampfitting_results[1],
                                                                                save_results=False)
            rows, columns = assign_wcs_results.data[0,:,:].shape
            wavelength_map = np.zeros([rows, columns])
            for row in tqdm(range(rows)):
                for column in range(columns):
                    wavelength_map[row, column] = assign_wcs_results.meta.wcs(column, row, 1)[-2]
            np.save(os.getcwd() + '/Data/' + wmap_filename, wavelength_map)
    except:
        continue
    # Saving the badpixel mask (2D array, listing bad-pixels as 0, and good pixel as 1)
    darkdq = darkcurrent_results.pixeldq
    np.save(p2 + '/bmap_seg' + segs[i] + '.npy', darkdq)