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
try:
    import periodogram_functions2 as pf2
except ModuleNotFoundError:
    raise Exception("Program requires 'periodogram_functions.py'")

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


    # real lombscargle
    lspower = calc_LombScargle_astropy(time, freq, data)

    # do the lombscargles
    maxpowers, maxfreqs = [], []
    print('\n Running monte carlo')

    for j in tqdm(range(num)):
        newdata = [np.random.normal(data[i], edata[i], size=1)[0]
                   for i in range(len(time))]
        newdata = np.array(newdata)
        #newdata = data
        resample = rng.randint(0, len(newdata), len(newdata))
        # define the Lomb Scargle with resampled data and using frequency_grid
        mcpower = calc_LombScargle_astropy(time, freq, newdata)
        mcpower = calc_LombScargle_astropy(time, freq, newdata[resample])
        # get the maximum power
        argmax = np.argmax(mcpower)
        maxpowers.append(mcpower[argmax])
        maxfreqs.append(freq[argmax])
    # plot CDF
    X2 = np.sort(maxpowers)
    CDF = 1 - np.array(range(num))/float(num)

    import matplotlib.pyplot as plt
    plt.close()
    fig, frames = plt.subplots(ncols=1, nrows=2)
    frames[0].plot(1.0/freq, lspower, color='r' ,zorder=1, linewidth=0.5)
    [frames[0].vlines(1.0/maxfreqs[i], 0, maxpowers[i], zorder=2)
     for i in range(len(maxpowers))]
    frames[0].set(xscale='log', yscale='log',
                  xlabel='Time / days', ylabel='Power',
                  title='MCMC magnitudes')

    for sig in [1, 2, 3]:
        sigma = pf2.sigma2percentile(sig)
        fap = np.percentile(maxpowers, sigma*100)
        frames[0].axhline(fap, linestyle='--', color='blue')

    frames[1].plot(X2, 1 - CDF)
    frames[1].set(xlabel='Power', ylabel='Cumulatative Probability')
    plt.show()
    plt.close()

# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    pass
# ----------------------------------------------------------------------

# =============================================================================
# End of code
# =============================================================================
