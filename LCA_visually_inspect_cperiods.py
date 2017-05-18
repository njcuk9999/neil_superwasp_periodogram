#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/04/17 at 2:04 PM

@author: neil

Takes the results of 'locate  probable peaks' and allows visual inspection
of the top 3 phase folds (i.e. asks the user - "Did we find a period that looks
correct?"

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
from astropy.stats import LombScargle
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
import light_curve_analysis as lca
from matplotlib_select import AddButtons
try:
    import periodogram_functions2 as pf2
except ImportError:
    raise Exception("Program requires 'periodogram_functions.py'")


# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = "/Astro/Projects/RayPaul_Work/SuperWASP"
DPATH = WORKSPACE + ("/Data/ls_analysis_run_3/light_curve_analysis_periods_"
                     "regions_groups_5.fits")
DPATH1 = WORKSPACE + ("/Data/ls_analysis_run_3/light_curve_analysis_periods_"
                     "regions_groups_5_post_visual.fits")
# column definitions
NAMECOL = "group_name"
MASKCOL = "has_period_1_05_to_50"
SELECTIONCOL = "visual_rank"
FLAGCOL = "passes_visual_check"
PERIODCOLS = ['Period_A', 'Period_B', 'Period_C']
# database columns
DATACOL = 'TAMMAG2'
EDATACOL = 'TAMMAG2_ERR'

# record results
RECORD = False
# do a single object
SINGLE_OBJECT = True
SID = 'BPC_46A'
# -----------------------------------------------------------------------------
# Phase offset
OFFSET = (-0.1, 0.1)

# --------------------------------------------------------------------------
# set database settings
HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = '1234'
DATABASE = 'swasp'
TABLE = 'swasp_sep16_tab'
TIMECOL = 'HJD'
# -----------------------------------------------------------------------------
# Data cleaning
UNCERTAINTY_CLIP = 0.005
SIGMA_CLIP = 3.0


# =============================================================================
# Define functions
# =============================================================================
# -------------------------------------------------------------------------
def calculation(inputs, pp):

    time, data, edata, period = inputs

    # format time (days from first time)
    time, data, edata, day0 = lca.format_time_days_from_first(*inputs[:-1],
                                                              mask=None)
    pp['day0'] = day0
    # zero data
    data = data - np.median(data)

    results = dict()

    ls = LombScargle(time, data, edata, fit_mean=True)
    # calcuate phase data
    print('\n Computing phase curve...')
    phase, phasefit, powerfit = pf2.phase_data(ls, time, period)
    results['phase'] = phase
    results['phasefit'] = phasefit
    results['powerfit'] = powerfit

    inputs = time, data, edata, period
    return inputs, results, pp


def plot_phase_curve(frame, inputs, results):
    # plot phased periodogram
    time, data, edata, period = inputs
    phase = results['phase']
    phasefit = results['phasefit']
    powerfit = results['powerfit']
    args = [frame, phase, data, edata, phasefit, powerfit, OFFSET]
    kwargs = dict(title='Phase Curve, period={0:.3f} days'.format(period),
                  ylabel='Magnitude', plotsigma=None)
    frame = pf2.plot_phased_curve(*args, **kwargs)
    frame.set_ylim(*frame.get_ylim()[::-1])
    return frame


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # load data
    print('\n Loading data...')
    data = fits.getdata(DPATH, ext=1)
    # ----------------------------------------------------------------------
    # mask
    if MASKCOL in data.columns.names:
        mask = ~np.array(data[MASKCOL])
    else:
        mask = np.ones(len(data), dtype=bool)
    # get the names
    names = np.array(data[NAMECOL])
    # get the periods
    periods = []
    for col in PERIODCOLS:
        periods.append(np.array(data[col]))
    # ----------------------------------------------------------------------
    # set up parameter dictionary
    pp = dict(FROM_DATABASE=True, HOSTNAME=HOSTNAME, DATABASE=DATABASE,
              TABLE=TABLE, USERNAME=USERNAME, PASSWORD=PASSWORD,
              TIMECOL=TIMECOL, DATACOL=DATACOL, EDATACOL=EDATACOL,
              UNCERTAINTY_CLIP=UNCERTAINTY_CLIP, SIGMA_CLIP=SIGMA_CLIP)
    # ----------------------------------------------------------------------
    # set up options and add buttons
    odict = dict(close=True)
    options = dict(Keep=1.0, Maybe=0.5, Reject=0.0)
    blabels = list(options.keys())
    # ----------------------------------------------------------------------
    # loop around each name
    uinputs = []
    for row in range(len(names)):
        # if not single object mode
        if not SINGLE_OBJECT:
            # row mask False skip
            if ~mask[row]:
                continue
        else:
            if names[row] != SID:
                continue
        # get SID/name
        pp['SID'] = names[row]
        pargs = ['='*50, pp['SID'], row+1, len(names)]
        print('{0}\n Analsising {1} \t {2} of {3} \n{0}'.format(*pargs))
        # get data
        time_arr, data_arr, edata_arr, pp = lca.load_data(pp)
        # set up figure
        plt.close()
        fig, frames = plt.subplots(nrows=1, ncols=len(periods))
        fig.set_size_inches(16, 10)
        # loop round each period
        for p_it, period in enumerate(periods):
            # set up inputs
            inputs = time_arr, data_arr, edata_arr, period[row]
            # calculate phase folds
            inputs, results, pp = calculation(inputs, pp)
            # plot phase folds
            frame = frames[p_it]
            frame = plot_phase_curve(frame, inputs, results)
        a = AddButtons(ax=frames[-1],
                       button_labels=blabels,
                       button_actions=['OPTION']*len(blabels),
                       button_params=[odict]*len(blabels))
        plt.suptitle(names[row])
        plt.show()
        plt.close()
        # use options dict to read result into uinputs
        if a.result == 0:
            uinputs.append(np.nan)
        else:
            selected_option = options[a.result]
            uinputs.append(selected_option)
    if RECORD:
        # print
        print('\n Saving to file...')
        # load up table
        table = Table.read(DPATH)
        if SELECTIONCOL in table.colnames:
            table[SELECTIONCOL][mask] = uinputs
        elif len(uinputs) == len(table):
            table[SELECTIONCOL] = uinputs
        else:
            raise ValueError("number of user inputs is not equal to number"
                             "of rows in table")
        table[FLAGCOL] = np.array(table[SELECTIONCOL] == 1.0, dtype=bool)
        table.write(DPATH1, format='fits', overwrite=True)

# =============================================================================
# End of code
# =============================================================================
