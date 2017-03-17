#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/03/17 at 10:47 AM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.table import Table
from astropy.stats import LombScargle



# =============================================================================
# Define variables
# =============================================================================
np.random.seed(9)

# -----------------------------------------------------------------------------


# =============================================================================
# Define periodogram functions
# =============================================================================
def lombscargle_periodogram(time, data, edata=None, freqs=None,
                            samples_per_peak=5, nyquist_factor=5):
    """
    Calculates the Lombscargle periodogram using astropy.stats


    :param time: numpy array or list, input time(independent) vector

    :param data: numpy array or list, input dependent vector

    :param edata: numpy array or None, uncerainties associated with data vector
                  if None LombScargle uses flat weights

    :param freqs: numpy array or None, frequency vector if None uses
                 astropy.stats.Lombscargle.autopower() to generate frequencies

    :return freqs: numpy array, frequency vector

    :return power: numpy array, power spectrum
    """
    # Calculate lombscargle
    kwargs = dict(samples_per_peak=samples_per_peak,
                  nyquist_factor=nyquist_factor)
    if freqs is None:
        freqs = LombScargle(time, data, dy=edata).autofrequency(**kwargs)
    power = LombScargle(time, data, dy=edata).power(freqs, normalization='psd')

    npower = power * (len(time)-1)/2

    return freqs, npower


def probability(Pn, N):
    """
      Returns the probability to obtain a power *Pn* or larger from the noise,
      which is assumes to be Gaussian.

    from http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/pyTimingDoc/
         pyPeriodDoc/periodograms.html

      Parameters:
        - `Pn` - float, Power threshold.

        - `N` - int, length of time/data vector

      .. note::
        *LombScargle* calculates the quantity (N-1)/2.*p=p' (in the formalism of
        [ZK09]_), which is de facto the normalization
        prescription of [HB86]_. In this
        scheme the probability P(p'>Pn) is given by the following statement:

        .. math::
          P(p'>Pn) = \\left(1 - 2 \\frac{Pn}{N-1} \\right)^{(N-3)/2}

        If properly normalized to the population variance of the time series,
        which must be known a priori (usually not the case),

        the power :math:`p/p_n=p"` is a direct measure of the SNR as proposed
        by [Scargle82]_:

        .. math::
          P(p">Pn) = exp(-Pn) \\; .

        This formula is often used erroneously in this context.
    """
    return (1. - 2. * Pn / (N - 1.)) ** ((N - 3.) / 2.)


def iprobability(Prob, N):
    """
      Inverse of `Prob(Pn)`. Returns the minimum power
      for a given probability level `Prob`.

    from http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/pyTimingDoc/
         pyPeriodDoc/periodograms.html


      Parameters:
        - `Prob` - float, probability
    """
    return (N - 1.) / 2. * (1. - Prob** (2. / (N - 3.)))


def FAP(Pn, N, ofac, hifac):
    """
      Obtain the false-alarm probability (FAP).

    from http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/pyTimingDoc/
         pyPeriodDoc/periodograms.html


      The FAP denotes the probability that at least one out of M
      independent power values in a prescribed search band of a
      power spectrum computed from a white-noise time series is
      as large as or larger than the threshold, `Pn`.
      It is assessed through

      .. math:: FAP(Pn) = 1 - (1-Prob(P>Pn))^M \\; ,

      where "Prob(P>Pn)" depends on the type of periodogram
      and normalization and is
      calculated by using the *prob* method;
      *M* is the number of independent power
      values and is computed internally.

      Parameters
      ----------
      Pn : float
          Power threshold.

       ofac - int, Oversampling factor.

       hifac - float, Maximum frequency `freq` = `hifac` *
               (average Nyquist frequency).

      Returns
      -------
      FAP : float
          False alarm probability.
    """
    nout = int(ofac*hifac*N/2.0)
    M = 2.*nout/ofac

    prob = M * probability(Pn)
    if prob > 0.01:
        return 1. - (1. - probability(Pn)) ** M
    else:
        return prob


def iFAP(FAPlevel, N, ofac, hifac):
    """
      Power threshold for FAP level.

    from http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/pyTimingDoc/
         pyPeriodDoc/periodograms.html


      Parameters
      ----------
      FAPlevel : float or array
            "False Alarm Probability" threshold

       ofac - int, Oversampling factor. (samples per peak)

       hifac - float, Maximum frequency `freq` = `hifac` *
               (average Nyquist frequency).

      Returns
      -------
      Threshold : float or array
          The power threshold pertaining to a specified
          false-alarm probability (FAP). Powers exceeding this
          threshold have FAPs smaller than FAPlevel.
    """
    nout = int(ofac*hifac*N/2.0)
    M = 2.*nout/ofac
    Prob = 1. - (1. - FAPlevel) ** (1. / M)
    return iprobability(Prob, N)


def fap_montecarlo(periodfunction, fargs, fkwargs, N=1000, log=False,
                   samples_per_peak=5, nyquist_factor=5):
    """

    :param periodfunction:
    :param fargs:
    :param fkwargs:
    :param N:
    :param log:
    :param samples_per_peak:
    :param nyquist_factor:
    :return:
    """

    fname = periodfunction.__name__
    if fname == 'clean_periodogram':
        time, data = fargs
        # force turn off logging
        fkwargs['log'] = False
        # force sampling and maximum peak
        fkwargs['fmax'] = nyquist_factor
        fkwargs['ppb'] = samples_per_peak
        # set edata to none
        edata = None
    elif fname == 'lombscargle_periodogram':
        time, data, edata = fargs
        # force sampling and maximum peak
        fkwargs['nyquist_factor'] = nyquist_factor
        fkwargs['samples_per_peak'] = samples_per_peak
    else:
        raise Exception('Period function needs to be "lombscargle_periodogram"'
                        'or "clean_periodogram"')
    # Extract the full days and fractional days
    days = time // 1
    fdays = np.mod(time, 1.0)
    tsize = len(time)
    powers, freq = [], []
    if log:
        print('\n Computing Monte Carlo periodogram for {0}...'.format(fname))
    for ni in __tqdmlog__(range(N), log):
        # randomise the days (but not the times)
        randomdays = np.random.choice(days, tsize)
        # add back in the fractional part of each day
        randomtimes = randomdays + fdays
        # Now insert these times into fargs (from period function)
        if fname == 'clean_periodogram':
            rfargs = randomtimes, data
        elif fname == 'lombscargle_periodogram':
            rfargs = randomtimes, data, edata
        # run the given period function
        output = periodfunction(*rfargs, **fkwargs)
        # deal with differing output
        if fname == 'clean_periodogram':
            freq, _, _, power = output
        elif fname == 'lombscargle_periodogram':
            freq, power = output
        else:
            raise Exception("No way in here.")
        # combine power from monticarlo
        powers.append(power)
    # Assume Gaussian statistics median is True value
    # and can assign uncertainties on each power pixel
    median = np.percentile(powers, 50, axis=0)
    upper = np.percentile(powers, 68.2689492137 / 2.0 + 50, axis=0)
    lower = np.percentile(powers, 50 - 68.2689492137 / 2.0)

    return freq, median, upper, lower


def phase_fold(time, data, period):
    # fold the xdata at given period
    tfold = (time / period) % 1
    # commute the lomb-scargle model at given period
    tfit = np.array(np.linspace(0.0, time.max(), 1000))
    yfit = LombScargle(time, data).model(tfit, 1.0/period)
    tfitfold = np.array((tfit / period) % 1)
    fsort = np.argsort(tfitfold)
    return tfold, tfitfold[fsort], yfit[fsort]

# =============================================================================
# Define other mathematical functions
# =============================================================================
def quantile_1D(data, weights, quantile):
    """
    Compute the weighted quantile of a 1D numpy array.

    Taken from:
    https://github.com/nudomarinero/wquantiles/blob/master/wquantiles.py

    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.
    Returns
    -------
    quantile_1D : float
        The output value.
    """
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if (quantile > 1.) or (quantile < 0.):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    #assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (sn-0.5*sorted_weights)/np.sum(sorted_weights)
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)


def bin_data(time, data, edata=None, binsize=None, log=False):
    """
    Bin time and data vectors by binsize (using a median combine of points in
    each bin (weight median if edata is not None).

    :param time: numpy array or list, input time(independent) vector

    :param data: numpy array or list, input dependent vector

    :param edata: None or numpy array, uncertainties associated with "data"

    :param binsize: float, size of each bin (in units of "time")

    :param log: boolean, if True prints progress to standard output
                         if False silent

    :return binnedtime: numpy array, binned "time" array

    :return binneddata: numpy array, binned "data" array

    """
    # Deal with bin size, if None, rebin to 1000 elements or don't bin
    # if len(time) is less than 1000
    if binsize is None:
        maxbins = np.min([len(time), 1000])
        bins = np.linspace(min(time), max(time), maxbins)
    else:
        bins = np.arange(min(time), max(time), binsize)

    # Now bin the data
    binnedtime = []
    binneddata = []
    binnederror = []
    # Loop round each bin and median the time and the data for all values
    # within that bin
    if log:
        print('\n\t Binning data...')
    for ibin in __tqdmlog__(bins, log):
        # mask values within this iteration bin
        mask = (time >= ibin) & (time < ibin+binsize)
        # if there are no values in this bin do not bin it
        if np.sum(mask) == 0:
            continue
        # if there are values in this bin take the median or weighted median
        # if we have uncertainties
        else:
            # No uncertainties with time so just take the median
            # btime = bin
            btime = np.median(time[mask])
            # We have no uncertainties don't weight points
            if edata is None:
                bdata = np.median(data[mask])
                berror = np.nan
            # We have uncertainties so weight the medians
            else:
                weights = np.array(1.0 / edata[mask] ** 2)
                bdata = quantile_1D(data[mask], weights, 0.50)
                berror = 1.0/np.sqrt(np.sum(weights))
                # Finally add the binned data to array
            binnedtime.append(btime)
            binneddata.append(bdata)
            binnederror.append(berror)

    return np.array(binnedtime), np.array(binneddata), np.array(binnederror)


# =============================================================================
# Define auxiliary functions
# =============================================================================
def save_to_file(coldata, savename, savepath, exts=None):
    # ---------------------------------------------------------------------
    # Convert to astropy table
    atable = Table()
    for col in coldata:
        dtype = type(coldata[col][0])
        atable[col] = np.array(coldata[col], dtype=dtype)
    # ---------------------------------------------------------------------
    # Save as fits file
    print('\n Saving to file...')
    if exts is None:
        exts = ['.fits']
    formats = dict(fits='fits', dat='ascii',  csv='csv')
    for ext in exts:
        fmt = formats[ext]
        path = '{0}{1}.{2}'.format(savepath, savename, ext)
        atable.write(path, format=fmt, overwrite='True')


def __tqdmlog__(x_input, log):
    """
    Private function for dealing with logging

    :param x_input:  any iterable object

    :param log: bool, if True and module tqdm exists use logging

    :return:
    """
    # deal with importing tqdm
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = (lambda x: x)
    # deal with logging
    if log:
        rr = tqdm(x_input)
    else:
        rr = x_input
    return rr


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    print("\n Done.")
    # ----------------------------------------------------------------------

# =============================================================================
# End of code
# =============================================================================
