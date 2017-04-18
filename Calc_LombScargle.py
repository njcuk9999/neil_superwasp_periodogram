#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/04/17 at 1:59 PM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
from astropy.stats import LombScargle
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm

# =============================================================================
# Define variables
# =============================================================================


# -----------------------------------------------------------------------------



# =============================================================================
# Define functions
# =============================================================================

def calc_LombScagle_manually(time, freq, data):

    powers = []
    for f in tqdm(freq):
        omega = 2*np.pi*f
        #calculate tau
        sin2ot = np.sum(np.sin(2*omega*time))
        cos2ot = np.sum(np.cos(2*omega*time))
        tau = (1/(2.0*omega))*np.arctan2(sin2ot, cos2ot)
        # calculate power
        cosott = np.cos(omega*(time-tau))
        sinott = np.sin(omega*(time-tau))
        part1 = np.sum(data*cosott)**2/np.sum(cosott**2)
        part2 = np.sum(data*sinott)**2/np.sum(sinott**2)

        power = 0.5*(part1 + part2)
        powers.append(power)


def calc_LombScargle_astropy(time, freq, data):
    ls = LombScargle(time, data, fit_mean=True)
    lspower = ls.power(freq, normalization='psd')
    return lspower


def monte_carlo_ls(time, freq, data, edata):
    # keep the sampling times the same and randomly select data
    rng = np.random.RandomState(9)
    # sample with replacement
    num = 100
    # generate samples
    print('\n Generating samples...')
    monte_carlos = np.zeros((num, len(time)))
    for i in tqdm(range(len(time))):
         monte_carlos[:, i] = np.random.normal(data[i], edata[i], size=num)

    for j in tqdm(range(num)):
        # sample with replacement
        resample = rng.randint(0, len(data), len(data))
        # define the Lomb Scargle with resampled data and using frequency_grid
        monte_carlos[j] = data[resample]
    # do the lombscargles
    maxpowers = []
    print('\n Running monte carlo')
    for j in tqdm(range(num)):
        # define the Lomb Scargle with resampled data and using frequency_grid
        mcpower = calc_LombScargle_astropy(time, freq, monte_carlos[j])
        plt.plot(1./freq, mcpower)
        # get the maximum power
        maxpowers.append(np.max(mcpower))
    # plot CDF
    X2 = np.sort(maxpowers)
    F2 = np.array(range(num))/float(num)
    plt.plot(X2, 1 - F2)
    plt.show()
    plt.close()

# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":

# ----------------------------------------------------------------------

# =============================================================================
# End of code
# =============================================================================
