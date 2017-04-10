#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/03/17 at 11:40 AM

@author: neil

Program description here

Version 0.0.0
"""

import light_curve_analysis as lca
try:
    import periodogram_functions2 as pf2
except ModuleNotFoundError:
    raise Exception("Program requires 'periodogram_functions.py'")

# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = "/Astro/Projects/RayPaul_Work/SuperWASP/"
# Deal with choosing a target and data paths
Elodie = True
if Elodie:
    # SID = 'GJ1289'
    # SID = 'GJ793'
    SID = 'ARG_54'
    TIMECOL = "time"
    DATACOL = "flux"
    EDATACOL = "eflux"
    # for GJ1289
    if SID == 'GJ1289':
        DPATH = WORKSPACE + "Data/Elodie/bl_gj1289.fits"
    elif SID == 'GJ793':
        DPATH = WORKSPACE + "Data/Elodie/bl_gj793.fits"
    elif SID == 'ARG_54':
        DPATH = WORKSPACE + 'Data/Elodie/ARG_54_lightcurve.fits'
else:
    # set file paths
    DPATH = WORKSPACE + 'Data/from_exoplanetarchive/'
    DPATH += '1SWASP J192338.19-460631.5.fits'
    PLOTPATH = WORKSPACE + '/Plots/Messina_like_plots_from_exoarchive/'
    # Column info
    TIMECOL = 'HJD'
    DATACOL = 'MAG2'
    EDATACOL = 'MAG2_ERR'
# -----------------------------------------------------------------------------
TEST_RUN = False
TEST_PERIOD = 3.28
DT = None
# -----------------------------------------------------------------------------
# minimum time period to be sensitive to
TMIN = 0.1
# maximum time period to be sensitive to
TMAX = 100
# number of samples per peak
SPP = 5
# random seed for bootstrapping
RANDOM_SEED = 9
# number of bootstraps to perform
N_BS = 100
# Phase offset
OFFSET = (-0.5, 0.5)
# define the FAP percentiles
PERCENTILES = [pf2.sigma2percentile(1)*100,
               pf2.sigma2percentile(2)*100,
               pf2.sigma2percentile(3)*100]
# number of peaks to find
NPEAKS = 5
# number of pixels around a peak to class as same peak
BOXSIZE = 5
# percentage around noise peak to rule out true peak
THRESHOLD = 5.0
# percentile (FAP) to cut peaks at (i.e. any below are not used)
CUTPERCENTILE = pf2.sigma2percentile(1.0)*100
# whether to normalise
NORMALISE = False


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    pp = lca.get_params()
    # -------------------------------------------------------------------------
    # Load data
    time_arr, data_arr, edata_arr, pp = lca.load_data(pp)
    # -------------------------------------------------------------------------
    # find sub regions
    m, pp['EXT'] = None, ''
    # ---------------------------------------------------------------------
    # print progress if logging on
    lca.update_progress(pp)
    # ---------------------------------------------------------------------
    # Calculation
    inp = time_arr, data_arr, edata_arr
    inp, res, pp = lca.calculation(inp, pp, m)
    # ---------------------------------------------------------------------
    # plotting
    lca.plot_graph(inp, res, pp)
    # ---------------------------------------------------------------------
    # save periods to file
    lca.save_to_fit(res, pp)


# =============================================================================
# End of code
# =============================================================================
