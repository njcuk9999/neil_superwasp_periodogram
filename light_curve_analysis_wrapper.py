#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 05/04/17 at 2:48 PM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm
import os
try:
    import periodogram_functions2 as pf2
except ModuleNotFoundError:
    raise Exception("Program requires 'periodogram_functions.py'")


# =============================================================================
# Define variables
# =============================================================================
# Deal with choosing a target and data paths
WORKSPACE = "/Astro/Projects/RayPaul_Work/SuperWASP/"
# location of folder to plot files to
PLOTPATH = WORKSPACE + '/Plots/ls_analysis_run/'
# file to save periods to
PERIODPATH = WORKSPACE + '/Data/ls_analysis_run/'
PERIODPATH += 'light_curve_analysis_periods_regions.fits'
# if True and periodpath file exists we will skip entries that exist
SKIP_DONE = True
# Write to file (if false does not save to file)
WRITE_TO_FILE = True
# Reset - resets all files (i.e. deletes fits and graphs)
RESET = True
# -----------------------------------------------------------------------------
# set database settings
FROM_DATABASE = True
HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = '1234'
DATABASE = 'swasp'
TABLE = 'swasp_sep16_tab'
# program to run
COMMAND = 'python light_curve_analysis.py '
# -----------------------------------------------------------------------------
# whether to show the graph
SHOW = False
# size in inches of the plot
FIGSIZE = (20, 16)
# decide whether to plot nan periods (saves time)
PLOT_NAN_PERIOD = True
# whether to log progress to standard output (print)
LOG = True
# -----------------------------------------------------------------------------
# minimum time period to be sensitive to (5 hours)
TMIN = 5/24.0
# maximum time period to be sensitive to (100 days)
TMAX = 100
# number of samples per peak
SPP = 5
# -----------------------------------------------------------------------------
# random seed for bootstrapping
RANDOM_SEED = 9999
# number of bootstraps to perform
N_BS = 100
# Phase offset
OFFSET = (-0.5, 0.5)
# define the FAP percentiles
PERCENTILES = [pf2.sigma2percentile(1)*100,
               pf2.sigma2percentile(2)*100,
               pf2.sigma2percentile(3)*100]
# -----------------------------------------------------------------------------
# number of peaks to find
NPEAKS = 5
# number of pixels around a peak to class as same peak
BOXSIZE = 5
# percentage around noise peak to rule out true peak
THRESHOLD = 5.0
# percentile (FAP) to cut peaks at (i.e. any below are not used)
CUTPERCENTILE = pf2.sigma2percentile(1.0)*100
# -----------------------------------------------------------------------------
# minimum number of data points to define a sub region
MINPOINTS = 50   # points
# maximum gap between data points to define a sub region
MAXGAP = 20  # days


# =============================================================================
# Define functions
# =============================================================================
def reset(do_reset):
    if do_reset and os.path.exists(PERIODPATH):
        print('\n Removing {0}'.format(PERIODPATH))
        os.remove(PERIODPATH)
    if do_reset and os.path.exists(PLOTPATH):
        files = os.listdir(PLOTPATH)
        for filename in files:
            loc = PLOTPATH + '/' + filename
            print('\n Removing {0}'.format(loc))
            os.remove(loc)


def load_pre_existing_data(sids, dpath):
    """
    Load pre-exisiting data

    :param dpath: string, filepath to the pre-existing data
    :return:
    """
    if os.path.exists(PERIODPATH) and SKIP_DONE and WRITE_TO_FILE:
        print("\n Loading pre existing files...")
        atable = Table.read(dpath)
        done_ids = list(atable['name'])
        del atable
        # ---------------------------------------------------------------------
        # skip sids if the are in table (assume this means they are done)
        do_ids, skips = [], 0
        for done_id in done_ids:
            raw_id = done_id.split('Full')[0].split('Region')[0]
            if raw_id not in sids:
                do_ids.append(raw_id)
            else:
                skips += 1

        # Print statement for how many files skipped due to pre existing data
        print('\n Skipping {0} sids'.format(skips))
    else:
        print('\n Nothing skipped.')
        do_ids = list(sids)
    return do_ids


def sort_data_in_dict(s_id, sarr, earr, start, end):
    """
    Sort the data based on the id and the start and end points (into the sarr
    and earr dictionaries

    :param s_id: string, ID of this object (will be used as key)
    :param sarr: dict, start point list (list of time series data for each
                 segments i.e. if there are two segments list would be
                 [x pos start of segment 1, x pos start of segment 2]
    :param earr: dict, end point list (list of time series data for each
                 segments i.e. if there are two segments list would be
                 [x pos end of segment 1, x pos end of segment 2]
    :param start: float, the x starting position of this segment
    :param end: float, the x ending position of this segment
    :return:
    """
    if s_id not in sarr:
        sarr[s_id], earr[s_id] = [start], [end]
    else:
        sarr[s_id].append(start), earr[s_id].append(end)
    return sarr, earr


def get_arguments_from_constants():
    argstring = ''
    for arg in list(globals().keys()):
        if arg.isupper():
            argstring += '{0}="{1}" '.format(arg, globals()[arg])
    return argstring


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # deal with reset
    reset(RESET)
    # get list of unique ids (for selecting each as a seperate curve)
    sql_kwargs = dict(host=HOSTNAME, db=DATABASE, table=TABLE,
                      user=USERNAME, passwd=PASSWORD, conn_timeout=100000)
    sids, conn = pf2.get_list_of_objects_from_db(conn=None, **sql_kwargs)
    # -------------------------------------------------------------------------
    # load file if it exists (to save on repeating on exit)
    do_sids = load_pre_existing_data(sids, PERIODPATH)
    # -------------------------------------------------------------------------
    # construct python command and arguments
    argumentstring = get_arguments_from_constants()
    # -------------------------------------------------------------------------
    # loop around SIDs
    for sid in do_sids:
        # add SID to argument string
        argumentstring += ' SID="{0}"'.format(sid)
        # run python program for file
        os.system(COMMAND + argumentstring)
        # input('Enter to continue. Control+C to cancel')


# =============================================================================
# End of code
# =============================================================================
