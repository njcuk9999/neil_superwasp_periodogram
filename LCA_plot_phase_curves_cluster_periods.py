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
DPATH = WORKSPACE + "/Data/ls_analysis_run_2/LCA_cluster_periods_0.05.fits"
DPATH = WORKSPACE + "/Data/ls_analysis_run_2/LCA_cluster_periods_0.05_post_visual_all.fits"
# column definitions
NAMECOL = "group_name"
MASKCOL = "has_period_1_05_to_50"
SELECTIONCOL = "visual_rank"
FLAGCOL = "passes_visual_check"
PERIODCOLS = ['Period_A', 'Period_B', 'Period_C']
# database columns
DATACOL = 'MAG2'
EDATACOL = 'MAG2_ERR'

# record results
RECORD = True
# do a single object
SINGLE_OBJECT = False
SID = 'BPC_46A'
# -----------------------------------------------------------------------------
# Phase offset
OFFSET = (-0.5, 0.5)

# --------------------------------------------------------------------------
# set database settings
HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = '1234'
DATABASE = 'swasp'
TABLE = 'swasp_sep16_tab'
TIMECOL = 'HJD'


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
                  ylabel='Magnitude', plotsigma=3.0)
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
    if MASKCOL in data.column.names:
        mask = ~np.array(data[MASKCOL])
    else:
        mask = np.zeros(len(data), dtype=bool)
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
              TIMECOL=TIMECOL,  DATACOL=DATACOL, EDATACOL=EDATACOL)
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
        print('{0}\n Analsising {1} \n{0}'.format('='*50, pp['SID']))
        # get data
        time_arr, data_arr, edata_arr, pp = lca.load_data(pp)
        # set up figure
        plt.close()
        fig, frames = plt.subplots(nrows=1, ncols=len(periods))
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
        # load up table
        table = Table.read(DPATH)
        table[SELECTIONCOL][mask] = uinputs
        table[FLAGCOL] = np.array(table[SELECTIONCOL] == 1.0, dtype=bool)
        table.write(DPATH, format='fits', overwrite=True)

# =============================================================================
# End of code
# =============================================================================
