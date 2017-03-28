#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28/03/17 at 1:47 PM

@author: neil

Program description here

Version 0.0.0
"""

from . import clean_periodogram
from . import fastDFT
from . import neil_clean
from . import periodogram_functions


LombScarglePeriodogram = periodogram_functions.lombscargle_periodogram
PhaseFold = periodogram_functions.phase_fold
iFAP = periodogram_functions.iFAP
FapMCMC = periodogram_functions.fap_montecarlo
FindYpeaks = periodogram_functions.find_y_peaks
GetLightcurveData = periodogram_functions.get_lightcurve_data
RelativeCluster = periodogram_functions.relative_cluster