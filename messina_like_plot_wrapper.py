#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/03/17 at 12:03 PM

@author: neil

Program description here

Version 0.0.0
"""
import numpy as np
from astropy.io import fits
import os
import subprocess
try:
    from periodogram_functions import load_db
    from periodogram_functions import get_list_of_objects_from_db
    from periodogram_functions import get_lightcurve_data
except ModuleNotFoundError:
    raise ModuleNotFoundError("Program requires 'periodogram_functions.py'")
# =============================================================================
# Define variables
# =============================================================================
# wrapped program name
PROGRAM_NAME = 'messina_like_plot.py'
# set file paths
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'
DPATH = WORKSPACE + '/Data/messina_match_from_paul.fits'
PLOTPATH = WORKSPACE + '/Plots/Messina_like_plots/'
# -----------------------------------------------------------------------------
SID = 'BPC_46A'
SKIP = True
# -----------------------------------------------------------------------------
# Column info
TIMECOL = 'HJD'
DATACOL = 'MAG2'
EDATACOL = 'MAG2_ERR'
# -----------------------------------------------------------------------------
# whether to bin data
BINDATA = True
BINSIZE = 0.1
# -----------------------------------------------------------------------------
# sigma clipping options
SIGMACLIP = True
# size in pixels (actual size will be median separation between points * SIZE)
SIZE = 100
# sigma of the clip i.e. median(DATACOL) + sigma*std(DATACOL)
SIGMA = 2.0
# whether to use a weighted median based on uncertainties
WEIGHTED = True
# -----------------------------------------------------------------------------
# uncertainty filter
ERRORCLIP = True
PERCENTAGE = 0.5
# -----------------------------------------------------------------------------
# periodogram constants
SAMPLES_PER_PEAK = 4
NYQUIST_FACTOR = 100
FAP_LEVELS = [0.01]
PEAK_RANGE = [0.1, 100.0]
PEAK_THRESHOLD_PERCENTAGE = 1.0
# -----------------------------------------------------------------------------
# Time to subtract off the time vector
TIME_CONST = 2453800
# Period from messina
MESSINA_PERIOD = 3.237
# -----------------------------------------------------------------------------
# database variables
HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = '1234'
DATABASE = 'swasp'
TABLE = 'swasp_sep16_tab'
# -----------------------------------------------------------------------------
# whether to load from full catalogue of lightcurves (True) or from
# Messina matches list (False)
fullcat = False

# =============================================================================
# Define functions
# =============================================================================
def make_commandline_args(**kwargs):
    keys = ['DPATH', 'PLOTPATH', 'SID', 'TIMECOL', 'DATACOL', 'EDATACOL',
            'BINDATA', 'BINSIZE', 'SIGMACLIP', 'SIZE', 'SIGMA', 'WEIGHTED',
            'ERRORCLIP', 'PERCENTAGE', 'SAMPLES_PER_PEAK', 'NYQUIST_FACTOR',
            'FAP_LEVELS', 'PEAK_RANGE', 'PEAK_THRESHOLD_PERCENTAGE',
            'TIME_CONST', 'MESSINA_PERIOD', 'SKIP']
    dvalues = [DPATH, PLOTPATH, SID, TIMECOL, DATACOL, EDATACOL, BINDATA,
               BINSIZE, SIGMACLIP, SIZE, SIGMA, WEIGHTED, ERRORCLIP,
               PERCENTAGE, SAMPLES_PER_PEAK, NYQUIST_FACTOR, FAP_LEVELS,
               PEAK_RANGE, PEAK_THRESHOLD_PERCENTAGE, TIME_CONST,
               MESSINA_PERIOD, SKIP]
    params = dict(zip(keys, dvalues))
    # override defaults with kwargs
    for key in kwargs:
        params[key] = kwargs[key]
    # make arguments in form ' key1=value1 key2=value2'
    stringargs = []
    for key in params:
        stringargs.append('{0}="{1}"'.format(key, params[key]))
    return stringargs


def load_messina_matches():
    data = fits.getdata(DPATH, ext=1)
    systemid = np.array(data['systemid'])
    comp = np.array(data['comp'])
    periods = np.array(data['P'])
    s_ids = []
    for sit in range(len(systemid)):
        if str(systemid[sit].strip()) == 'NULL':
            continue
        s_id = '{0}{1}'.format(systemid[sit].strip(), comp[sit].strip())
        s_ids.append(s_id.strip())
    usid, indices = np.unique(s_ids, return_index=True)
    return usid, periods[indices]


def command(process, ignore_error=True):

    path = '/tmp/'
    outfilepath = 'pyout'
    errfilepath = 'pyerr'
    if outfilepath in os.listdir(path):
        os.remove(path + outfilepath)
    if errfilepath in os.listdir(path):
        os.remove(path + errfilepath)
    outfile = open(path + outfilepath, 'w')
    errfile = open(path + errfilepath, 'w')
    proc = subprocess.Popen(process, bufsize=0,
                            stdout=outfile, stderr=errfile)
    cond = True
    while proc.wait() and cond:
        outfile.close()
        errfile.close()

        errfile = open(path + errfilepath, 'r')
        for line in errfile.readlines():
            print(line)
        errfile.close()

        errfile = open(path + errfilepath, 'r')
        for line in errfile.readlines():
            if "Error" in line:
                if not ignore_error:
                    _ = input('Press enter to continue, ctrl+c to exit.')
                    ignore_error=True
                    cond = False
            if "Skipping due to file existing" in line:
                cond = False
        errfile.close()
        outfile = open(path + outfilepath, 'w')
        errfile = open(path + errfilepath, 'w')


    if outfilepath in os.listdir(path):
        os.remove(path + outfilepath)
    if errfilepath in os.listdir(path):
        os.remove(path + errfilepath)


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    if fullcat:
        sql_kwargs = dict(host=HOSTNAME, db=DATABASE, table=TABLE,
                          user=USERNAME, passwd=PASSWORD, conn_timeout=100000)
        c, conn = load_db(**sql_kwargs)
        sids = get_list_of_objects_from_db(conn, **sql_kwargs)
        mperiods = np.repeat(np.nan, len(sids))
    else:
        sids, mperiods = load_messina_matches()
    # ----------------------------------------------------------------------
    # loop around SIDs
    for sit, sid in enumerate(sids):
        pargs = ['='*50, sit+1, len(sids), sid]
        print('\n{0}\n {1}/{2} Processing sid = {3} \n{0}\n'.format(*pargs))
        # ----------------------------------------------------------------------
        # if plot exists then skip (if SKIP = True)
        tname = sid
        if SKIP:
            filename = 'Messina_plot_' + tname.replace(' ', '_') + '.png'
            if filename in os.listdir(PLOTPATH):
                print('Skipping due to file existing')
                continue
        # ----------------------------------------------------------------------
        # run command
        args = make_commandline_args(SID=sid, DPATH=None,
                                     MESSINA_PERIOD=mperiods[sit])
        cmdstring = ['python', PROGRAM_NAME] + args
        command(cmdstring, ignore_error=False)


# =============================================================================
# End of code
# =============================================================================
