#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/04/17 at 11:55 AM

@author: neil

Program description here

Version 0.0.0
"""
import MySQLdb
import numpy as np
import pandas
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.stats import LombScargle
from scipy.special import erf, erfinv
from photutils import find_peaks
import mpmath
import sys

# =============================================================================
# Define variables
# =============================================================================
SQRTTWO = np.sqrt(2)

# sigma to plot to (away from median)
PLOTSIGMA = 5


# =============================================================================
# Define Periodogram functions
# =============================================================================
def make_frequency_grid(t, fmin, fmax, samples_per_peak):
    f0 = fmin
    df = 1 / (t.max() - t.min()) / samples_per_peak
    Nf = int(np.ceil((fmax - f0) / df))
    f = f0 + df * np.arange(Nf)
    return f


def lombscargle(time, data, edata=None, fit_mean=True, fmin=100, fmax=1.0,
                samples_per_peak=10, norm='standard', freq=None):
    """
    Produce a lombscargle periodogram

    :param time: numpy array, the time vector

    :param data: numpy array, the data vector

    :param edata: numpy array, the uncertainty vector associated with the data
                  vector

    :param fit_mean: boolean, if True uses a floating mean periodogram
                          (generalised Lomb-Scargle periodogram) else uses
                          standard Lomb-Scargle periodogram

    :param fmin: float, the minimum frequency to compute frequency grid to
                 maximum time period able to find = 1/fmin

    :param fmax: float, the maximum frequency to compute frequency grid to
                 minimum time period able to find = 1/fmax

    :param samples_per_peak: int, number of samples per peak to use

    :param norm: string (optional, default='standard')
                 Normalization to use for the periodogram.
                 Options are 'standard', 'model', or 'psd'

    :param freq: numpy array or None, if defined the freqeuncy grid to be used
                 in the lombscargle (overrides fmin, fmax and samples_per_peak)

    :return:
    """
    norm = "standard"
    method = "fast"
    # Set up an instance of the LombScargle class
    ls = LombScargle(time, data, edata, fit_mean=fit_mean)
    if freq is None:
        # Auto generate the frequencies and calculate the Lomb Scargle
        freq, power = ls.autopower(minimum_frequency=fmin,
                                   maximum_frequency=fmax,
                                   samples_per_peak=samples_per_peak,
                                   normalization=norm, method=method)
    else:
        power = ls.power(freq, normalization=norm, method=method)

    # un-normalise
    # if norm == "psd":
    #     power = power / (0.5 * time.size)
    # elif norm == "standard":
    #     y, dy = data, edata
    #     w = dy ** -2.0
    #     w /= w.sum()
    #     y = y - np.dot(w, y)
    #     YY = np.dot(w, y ** 2)
    #     power *= YY

    # # undo normalisation
    # w = 1 / edata ** 2
    # w /= w.sum()
    # y = data - np.dot(w, data**2)
    # YY = np.dot(w, y**2)
    # power = power * YY
    #
    # # normalise by inverse variance
    # power = power / np.std(data)**2

    # normalise power by the sum of all powers
    power = power/np.sum(power)

    # Return the auto generated frequencies and the normalised power
    # and the lombscargle instance
    return freq, power, ls


def compute_window_function(time, fmin=100, fmax=1.0, samples_per_peak=10,
                            freq=None):
    """
    Get the window function from the time vector (using Lomb-Scargle)

    :param time: numpy array, time vector

    :param fmin: float, the minimum frequency to compute frequency grid to
                 maximum time period able to find = 1/fmin

    :param fmax: float, the maximum frequency to compute frequency grid to
                 minimum time period able to find = 1/fmax

    :param samples_per_peak: int, number of samples per peak to use

    :return:
    """
    ls = LombScargle(time, 1, fit_mean=False, center_data=False)
    if freq is None:
        freq, power = ls.autopower(minimum_frequency=fmin,
                                   maximum_frequency=fmax,
                                   samples_per_peak=samples_per_peak)
    else:
        power = ls.power(freq)
    return freq, power


def phase_data(ls, time, period, offset=(-1, 1)):
    """
    Produce phased time data and fit the period to the ls model

    :param ls: instances of LombScargle (from astropy.stats)

    :param time: numpy array, the time vector

    :param period: float, the period selected from Lomb-Scarlge periodogram

    :param offset: tuple, phase beyond 0 to 1 to extend (i.e. phase curve
                   will run from - offset[0] to 1 + offset[1]
    :return:
    """
    if hasattr(period, '__len__'):
        period = period[0]
    # phase the time vector
    phase = (time * (1.0 / period)) % 1
    # Make model phase fit data (evenly spaced)
    phase_fit = np.linspace(offset[0], 1 + offset[1], 1000)
    # Calculate the power fit data give the used Lomb Scargle, phase_fit and
    # frequency_peak
    power_fit = ls.model(phase_fit * period, 1.0 / period)
    # return the phase, phase_fit and power_fit
    return phase, phase_fit, power_fit


def fap_from_theory(time, power):
    N = len(time)
    power = np.array(power)
    Meff = -6.363 + 1.193*N + 0.00098*N**2
    prob = (1 - power)**mpmath.mpf((N-3)/2)

    FAP = 1 - (1-prob)**mpmath.mpf(Meff)

    FAP[FAP < (sys.float_info.min * 10)] = 0

    return np.array(FAP, dtype=float)

    # M = len(freq)
    # # probability p of observing a power less than or equal to P0
    # p = 1 - mpmath.exp(-maxpower)
    # # probability of seeing at least one sapmle exceeding this value
    # pv = 1 - p**mpmath.mpf(M)
    # # return pv


def power_from_prob(time, faps=None, percentiles=None):
    if faps is None and percentiles is None:
        raise ValueError("Need to define either faps or percentiles")
    if faps is None:
        faps = 1 - np.array(percentiles)/100.0

    N = len(time)
    faps = np.array(faps)
    Meff = -6.363 + 1.193*N + 0.00098*N**2
    prob = 1 - (1 - faps)**mpmath.mpf(1/Meff)
    power = 1 - (prob)**mpmath.mpf(2/(N-3))

    power[power < (sys.float_info.min * 10)] = 0

    return np.array(power, dtype=float)





# =============================================================================
# Define Bootstrap functions
# =============================================================================
def lombscargle_bootstrap(time, data, edata, frequency_grid, n_bootstraps=100,
                          random_seed=None, full=False, norm='standard',
                          fit_mean=True, log=False):
    """
    Perform a bootstrap analysis that resamples the data/edata keeping the
    temporal (time vector) co-ordinates constant

    modified from:
    https://github.com/jakevdp/PracticalLombScargle/blob/master
          /figures/Uncertainty.ipynb

    :param time: numpy array, the time vector

    :param data: numpy array, the data vector

    :param edata: numpy array, the uncertainty vector associated with the data
                  vector

    :param frequency_grid: numpy array, the frequency grid to use on each
                           iteration

    :param n_bootstraps: int, number of bootstraps to perform

    :param random_seed: int, random seem to use

    :param full: boolean, if True return freq at maximum power and maximum
                 powers, else return powers

    :param norm: Lomb-Scargle normalisation
                          (see astropy.stats.LombScargle)

    :param fit_mean: boolean, if True uses a floating mean periodogram
                          (generalised Lomb-Scargle periodogram) else uses
                          standard Lomb-Scargle periodogram

    :param log: boolean, if true displays progress to standard output (console)

    :return:
    """
    rng = np.random.RandomState(random_seed)

    kwargs = dict(fit_mean=fit_mean, freq=frequency_grid, norm=norm)

    def bootstrapped_power():
        # sample with replacement
        resample = rng.randint(0, len(data), len(data))
        # define the Lomb Scargle with resampled data and using frequency_grid
        ff, xx, _ = lombscargle(time, data[resample], edata[resample], **kwargs)
        # return frequency at maximum and maximum
        return ff, xx

    # run bootstrap
    f_arr, d_arr, x_arr = [], [], []
    for _ in __tqdmlog__(range(n_bootstraps), log):
        f, x = bootstrapped_power()
        argmax = np.argmax(x)
        x_arr.append(x)
        f_arr.append(f[argmax]), d_arr.append(x[argmax])
    # sort
    sort = np.argsort(f_arr)
    x_arr = np.array(x_arr)
    f_arr, d_arr = np.array(f_arr)[sort], np.array(d_arr)[sort]

    median = np.percentile(x_arr, 50, axis=0)
    # upper = np.percentile(x_arr, sigma2percentile(2.0) / 2.0 + 50, axis=0)
    # lower = np.percentile(x_arr, sigma2percentile(2.0) / 2.0, axis=0)

    # return
    if full:
        return frequency_grid, median, f_arr, d_arr
    else:
        return d_arr


def false_alarm_probability_from_bootstrap(ppeaks, percentiles):
    """
    Calculate and return the false alarm probabilities from given percentiles

    modified from:
    https://github.com/jakevdp/PracticalLombScargle/blob/master
          /figures/Uncertainty.ipynb

    :param ppeaks: numpy array, maximum power of each resample from the
                   bootstrap process

    :param percentiles: float/list, float in range of [0,100]
                        (or sequence of floats) Percentile to compute, which
                        must be between 0 and 100 inclusive
    :return:

    """
    faps = np.percentile(ppeaks, percentiles)
    return faps


def inverse_fap_from_bootstrap(ppeaks, fap, dp=3):
    """
    Brute force approach to inversely calculating the ifap

    calculates the fap over a regular grid of resolution 1/10^dp and returns
    the closest percentile to the fap value

    :param ppeaks: numpy array, maximum power of each resample from the
                   bootstrap process

    :param fap: float, the power of a peak on the Lomb-Scargle periodogram

    :param dp: int, number of decimal places to find (warning the more decimal
               points the larger the grid and slower this will run
    :return:
    """
    if not hasattr(fap, '__len__'):
        fap = [fap]

    quant = 10**dp
    tries = np.linspace(1, quant-1/quant, quant)/quant
    faps = np.percentile(ppeaks, 100*tries)

    ifaps = []
    for fap_it in range(len(fap)):
        if np.isfinite(fap[fap_it]):
            position = np.argmin(abs(faps-fap[fap_it]))
            ifaps.append(tries[position])
        else:
            ifaps.append(np.nan)
    return np.round(ifaps, dp)


# =============================================================================
# Define MCMC functions
# =============================================================================
def probability(power, num):
    """
      Returns the probability to obtain a power *power* or larger from the
      noise, which is assumes to be Gaussian.

    from http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/pyTimingDoc/
         pyPeriodDoc/periodograms.html

      Parameters:
        - `power` - float, Power threshold.

        - `num` - int, length of time/data vector

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
    if num == 1:
        return np.nan

    return (1. - 2. * power / (num - 1.)) ** ((num - 3.) / 2.)


def iprobability(prob, num):
    """
      Inverse of `Prob(Pn)`. Returns the minimum power
      for a given probability level `Prob`.

    from http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/pyTimingDoc/
         pyPeriodDoc/periodograms.html


      Parameters:
        - `Prob` - float, probability
    """
    if num == 3:
        return np.nan

    return (num - 1.) / 2. * (1. - prob ** (2. / (num - 3.)))


def false_alarm_probability(power, num, ofac, hifac):
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
      :param power: float, Power threshold.

      :param num: int, length of the time vector

      :param ofac: int, Oversampling factor.

       :param hifac: float, Maximum frequency `freq` = `hifac` * (average
                     Nyquist frequency).

      Returns
      -------
      :return FAP : float, False alarm probability.
    """
    nout = int(ofac * hifac * num / 2.0)
    mnum = 2. * nout / ofac

    prob = mnum * probability(power, num)
    if prob > 0.01:
        return 1. - (1. - probability(power, num)) ** mnum
    else:
        return prob


def ifalse_alarm_probability(faplevel, num, ofac, hifac):
    """
      Power threshold for FAP level.

    from http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/pyTimingDoc/
         pyPeriodDoc/periodograms.html


      Parameters
      ----------
      :param faplevel : float or array, "False Alarm Probability" threshold

      :param num: int, length of the time vector

      :param ofac: int, Oversampling factor. (samples per peak)

      :param hifac: float, Maximum frequency `freq` = `hifac` * (average
                    Nyquist frequency).

      Returns
      -------
      :return Threshold : float or array
                          The power threshold pertaining to a specified
                          false-alarm probability (FAP). Powers exceeding this
                          threshold have FAPs smaller than FAPlevel.
    """
    nout = int(ofac * hifac * num / 2.0)
    mnum = 2. * nout / ofac
    prob = 1. - (1. - faplevel) ** (1. / mnum)
    return iprobability(prob, num)


def ls_noiseperiodogram(time, data, edata, frequency_grid, n_iterations=100,
                        random_seed=None, norm='standard', fit_mean=True,
                        log=False, randomize='times'):
    rng = np.random.RandomState(random_seed)

    if log:
        print('\n Computing noise LS periodogram...')
    kwargs = dict(fit_mean=fit_mean, freq=frequency_grid, norm=norm)
    # loop around N_iterations
    powers = []

    days = time // 1
    fdays = np.mod(time, 1.0)
    tsize = len(time)

    f_arr, d_arr = [], []
    for _ in __tqdmlog__(range(n_iterations), log):
        if randomize == 'mag':
            # randomise the mags but not the times
            resample = rng.randint(0, len(data), len(data))
            # run lombscargle
            _, power, _ = lombscargle(time, data[resample], edata[resample],
                                      **kwargs)
        else:
            # randomise the days (but not the times)
            randomdays = np.random.choice(days, tsize)
            # add back in the fractional part of each day
            rtime = randomdays + fdays
            # run lombscargle
            _, power, _ = lombscargle(rtime, data, edata, **kwargs)

        if len(power) > 0:
            # combine power from monticarlo
            powers.append(power)
            # add maximums
            argmax = np.argmax(power)
            f_arr.append(frequency_grid[argmax])
            d_arr.append(power[argmax])

    # Assume Gaussian statistics median is True value
    # and can assign uncertainties on each power pixel
    if len(power) > 0:
        median = np.percentile(powers, 50, axis=0)
    else:
        median = np.repeat([np.nan], len(frequency_grid))
    # onesig_per = float(sigma2percentile(2.0))

    return frequency_grid, median, f_arr, d_arr


# =============================================================================
# Define period finding functions
# =============================================================================
def find_period(lsargs, bsargs=None, msargs=None):
    """

    :param lsargs: dict, dictionary containing the frequency and power for the
                   computed lombscargle periodogram

            required keyword arguments are:

                   freq = numpy array, frequency grid

                   power = numpy array, power at each frequency grid point

                   boxsize = int, size in pixels around the peak to define
                              as still being part of the peak

                   number = int, number of peaks to find (if less then peak
                            will be a Nan value


    :param bsargs: dict, dictionary containing variables for filtering by the
                   bootstrap method

            required keyword arguments are:

                   ppeaks = numpy array, maximum power peaks from each resample
                            of the bootstrap

                   percentile = float, between 0 and 100, the percentile below
                                which peaks will be rejected


    :param msargs: dict, dictionary containing variables for filtering by the
                   monte carlo method

            required keyword arguments are:

                   freq = numpy array, frequency grid

                   power = numpy array, power at each frequency grid point

                   boxsize = int, size in pixels around the peak to define
                              as still being part of the peak

                   number = int, number of peaks to find (if less then peak
                            will be a Nan value

                   threshold = float, the percentage threshold around a
                               noise peak, if a peak is inside +/- this then
                               it will be rejected

    :return:
    """
    ls_freq, ls_power = lsargs['freq'], lsargs['power']
    boxsize, number = lsargs['boxsize'], lsargs['number']
    ls_time = np.array(1.0 / ls_freq)
    sort = np.argsort(ls_time)
    ls_time, ls_power = ls_time[sort], ls_power[sort]
    try:
        # ls_time, ls_power = filter_by_bootstrap(ls_time, ls_power, bsargs)
        mcmcr = filter_by_mcmc(ls_time, ls_power, msargs)
        ls_time, ls_power, nperiod, npower = mcmcr
        if len(ls_time) == 0:
            raise KeyError()
        elif len(ls_time) == 1:
            period, power = [ls_time[0]], [ls_power[0]]
        else:
            period, power = find_y_peaks(ls_time, ls_power, boxsize=boxsize,
                                         number=number)
    except KeyError:
        argmax = np.argmax(lsargs['power'])
        period = [1.0/lsargs['freq'][argmax]]
        power = [-999]
        nperiod, npower = np.repeat([-999], number), np.repeat([-999], number)

    temp_n = len(period)
    if temp_n < number:
        period = period + list(np.repeat([np.nan], number - temp_n))
        power = power + list(np.repeat([np.nan], number - temp_n))
    temp_n = len(nperiod)
    if temp_n < number:
        nperiod = nperiod + list(np.repeat([np.nan], number - temp_n))
        npower = npower + list(np.repeat([np.nan], number - temp_n))

    return period, power, nperiod, npower


def filter_by_bootstrap(lstime, lspower, bsargs):
    # if we have bootstrap data then work out the FAP at given percentile
    # and only look at periods above this
    if bsargs is not None:
        try:
            ppeaks = bsargs['ppeaks']
            percentile = bsargs['percentile']
            fap = false_alarm_probability_from_bootstrap(ppeaks, percentile)
            fapmask = lspower > fap
            lstime, lspower = lstime[fapmask], lspower[fapmask]
        except KeyError:
            print('Keyword arguments for bootstrap filter not satisfied')
    return lstime, lspower


def filter_by_mcmc(lstime, lspower, msargs):
    # if we have mcmc data then disregard peaks near noise peaks
    sort = np.argsort(lstime)
    lstime, lspower = lstime[sort], lspower[sort]
    if msargs is not None:
        try:
            msfreq, mspower = msargs['freq'], msargs['power']
            mstime = np.array(1.0 / msfreq)
            sort = np.argsort(mstime)
            mstime, mspower = mstime[sort], mspower[sort]
            boxsize, number = msargs['boxsize'], msargs['number']
            threshold = msargs['threshold'] / 100.0
            msperiods, mspowers = find_y_peaks(mstime, mspower, number=number,
                                               boxsize=boxsize)
            for msperiod in msperiods:
                if np.isnan(msperiod):
                    continue
                lowlimit = (msperiod * (1 - threshold))
                highlimit = (msperiod * (1 + threshold))
                mask = (lstime > lowlimit) & (lstime < highlimit)
                lstime, lspower = lstime[~mask], lspower[~mask]
        except KeyError:
            print('Keyword arguments for mcmc filter not satisfied')
            msperiods, mspowers = [], []
    return lstime, lspower, msperiods, mspowers


def find_y_peaks(x=None, y=None, x_range=None, kind='binpeak', number=1,
                 boxsize=5):
    """
    Finds the maximum peak in y (at point x if defined)


    :param y: numpy array of floats, vector to find the peak(s) of

    :param x: numpy array or None, if not None locates the point in x (same
              shape as y) where the peak is located

    :param x_range: list of two floats or None, if not None defines the range
                    of x values in which to search for peaks
                    x_range = [xmin, xmax]    by default

    :param kind: string, the kind of peak finding to use

                 currently supported are:

                 binpeak - uses the numpy max function to find bin peak

    :param number: int, the number of peaks to find (i.e. 1 is the highest peak,
                   2 would be the two highest peaks etc)

    :param boxsize: int, the number of pixels around a peak to include as part
                    of that peak

    :return:
    """
    if y is None:
        print('\n No y data present - cannot find peaks')
        return np.repeat(np.nan, number), np.repeat(np.nan, number)
    # -------------------------------------------------------------------------
    # deal with no x axis value
    if x is None:
        x = np.arange(0, len(y), 1.0)
        no_x_value = True
    else:
        no_x_value = False
    # -------------------------------------------------------------------------
    # make sure y and x are numpy arrays
    x = np.array(x)
    y = np.array(y)
    # -------------------------------------------------------------------------
    # make sure x and y are the same length
    if len(x) != len(y):
        raise ValueError("y and x arrays must be the same length")
    # -------------------------------------------------------------------------
    # remove non finite numbers
    nanmask = np.isfinite(x) & np.isfinite(y)
    x, y = x[nanmask], y[nanmask]
    # -------------------------------------------------------------------------
    # sort by x
    sort = np.argsort(x)
    x, y = x[sort], y[sort]
    # -------------------------------------------------------------------------
    if kind == 'binpeak':
        xpeaks, ypeaks = binmax(x, y, num=number, boxsize=boxsize)
    else:
        print('\n No peaks in xrange={0}'.format(x_range))
        raise ValueError("Kind {0} not supported.".format(kind))
    # mask by x_range
    xpeaks, ypeaks = np.array(xpeaks), np.array(ypeaks)
    if x_range is None:
        x_range = [x.min(), x.max()]
    xmask = (xpeaks > x_range[0]) & (xpeaks < x_range[1])
    xpeaks, ypeaks = xpeaks[xmask], ypeaks[xmask]
    for i in range(number):
        if i >= len(xpeaks):
            xpeaks = np.append(xpeaks, np.nan)
            ypeaks = np.append(ypeaks, np.nan)
    # -------------------------------------------------------------------------
    # return
    if no_x_value:
        return ypeaks
    else:
        return xpeaks, ypeaks


def binmax(x, y, num=1, boxsize=5):
    # use photutils to get peak
    data = np.array([y, y])
    threshold = np.median(y)
    tbl = find_peaks(data, threshold, box_size=boxsize)
    if type(tbl) == Table:
        # mask out one of the rows
        mask = tbl['y_peak'] == 0
        tbl = tbl[mask]
        # get the columns from the table and sort the tbl by peak value
        ypeak = np.array(tbl['peak_value'])
        xindices = np.array(tbl['x_peak'])
        xpeak = x[xindices]
        peaksort = np.argsort(ypeak)[::-1]
        xpeak, ypeak = xpeak[peaksort], ypeak[peaksort]
        # deal with there being less than N peaks
        if len(xpeak) < num:
            xpeaki, ypeaki = np.array(xpeak), np.array(ypeak)
            xpeak, ypeak = [], []
            for xpi in range(num):
                if xpi < len(xpeaki):
                    xpeak.append(xpeaki[xpi])
                    ypeak.append(ypeaki[xpi])
                else:
                    xpeak.append(np.nan)
                    ypeak.append(np.nan)
        # return the N largest peaks
        return xpeak[:num], ypeak[:num]
    else:
        return np.repeat([np.nan], num), np.repeat([np.nan], num),


# =============================================================================
# Define Plot functions
# =============================================================================
def plot_rawdata(frame, time, data, edata, **kwargs):
    # deal with keyword arguments
    color = kwargs.get('color', 'gray')
    ecolor = kwargs.get('ecolor', 'lightgray')
    alpha = kwargs.get('alpha', 0.5)
    xlabel = kwargs.get('xlabel', 'phase')
    ylabel = kwargs.get('ylabel', 'mag')
    xlim = kwargs.get('xlim', (0, time.max()))
    ylim = kwargs.get('ylim', None)
    title = kwargs.get('title', None)
    # plot errorbar
    frame.errorbar(time, data, yerr=edata, color=color, marker='o', ms=2,
                   ecolor=ecolor, alpha=alpha, linestyle='None')
    # deal with limits as None
    if xlabel is not None:
        frame.set_xlabel(xlabel)
    if ylabel is not None:
        frame.set_ylabel(ylabel)
    if xlim is not None:
        frame.set_xlim(*xlim)
    if ylim is not None:
        frame.set_ylim(*ylim)
    else:
        median, std = np.median(data), np.std(data)
        frame.set_ylim(median - PLOTSIGMA * std, median + PLOTSIGMA * std)
    if title is not None:
        frame.set_title(title)
    # return frame
    return frame


def plot_periodogram(frame, time, power, **kwargs):
    # deal with keyword arguments
    color = kwargs.get('color', 'k')
    xlabel = kwargs.get('xlabel', 'Time / days')
    ylabel = kwargs.get('ylabel', 'Lomb-Scargle Power')
    xlim = kwargs.get('xlim', (time.min(), time.max()))
    ylim = kwargs.get('ylim', (0, 1.2 * power.max()))
    title = kwargs.get('title', None)
    zorder = kwargs.get('zorder', 2)
    alpha = kwargs.get('alpha', 1)
    # plot periodogram plot
    frame.plot(time, power, color=color, zorder=zorder, alpha=alpha)
    # deal with limits as None
    if xlabel is not None:
        frame.set_xlabel(xlabel)
    if ylabel is not None:
        frame.set_ylabel(ylabel)
    if xlim is not None:
        frame.set_xlim(*xlim)
    if ylim is not None:
        frame.set_ylim(*ylim)
    if title is not None:
        frame.set_title(title)
    # return frame
    return frame


def add_arrows(frame, periods, power, **kwargs):
    # deal with keyword arguments
    firstcolor = kwargs.get('firstcolor', 'r')
    normalcolor = kwargs.get('normalcolor', 'b')
    arrowstart = kwargs.get('arrowstart', 1.1)
    arrowend = kwargs.get('arrowend', 1.2)
    zorder = kwargs.get('zorder', 2)

    # if periods is a non length object push it into a list
    if not hasattr(periods, '__len__'):
        periods = [periods]
    # scale to the data
    maxpower = np.max(power)
    arrowstart, arrowend = arrowstart * maxpower, arrowend * maxpower
    # loop around each period and plot an arrow
    for p, period in enumerate(periods[::-1]):
        if p == len(periods) - 1:
            color = firstcolor
            zorder += 20
        else:
            color = normalcolor
        # set up the arrow props from keyword arguments
        arrowprops = dict(arrowstyle="->", color=color)
        frame.annotate('', (period, arrowstart), (period, arrowend),
                       arrowprops=arrowprops, zorder=zorder)
    # return frame
    return frame


def add_fap_to_periodogram(frame, time, peaks=None, percentiles=95.0, **kwargs):
    # deal with keyword arguments
    color = kwargs.get('color1', 'b')
    linestyle = kwargs.get('linestyle', 'dotted')
    zorder = kwargs.get('zorder', 2)
    if not hasattr(percentiles, '__len__'):
        percentiles = [percentiles]
    if peaks is not None:
        # calculate faps from bootstrap
        faps = false_alarm_probability_from_bootstrap(peaks, percentiles)
    else:
        # calculate faps theoretically
        faps = power_from_prob(time, percentiles=percentiles)

    sigmas = []
    # plot faps
    for f, fap in enumerate(faps):
        frame.axhline(fap, color=color, linestyle=linestyle, zorder=zorder)
        # plot label
        sigma = '{0:.2f}'.format(percentile2sigma(percentiles[f] / 100.0))
        sigmas.append(sigma)
    xmin, xmax, ymin, ymax = frame.axis()
    frame1 = frame.twinx()
    frame1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    frame1.set_yticks(faps)
    frame1.set_yticklabels([i + '$\sigma$' for i in sigmas])


    # return frame
    return frame


def plot_phased_curve(frame, phase, data, edata, phase_fit, power_fit, offset,
                      **kwargs):
    # deal with keyword arguments
    datacolor = kwargs.get('datacolor', 'gray')
    edatacolor = kwargs.get('edatacolor', 'lightgray')
    dataalpha = kwargs.get('dataalpha', 0.5)
    datalabel = kwargs.get('datalabel', None)
    modelcolor = kwargs.get('modelcolor', 'red')
    modellabel = kwargs.get('modellabel', None)
    xlabel = kwargs.get('xlabel', 'phase')
    ylabel = kwargs.get('ylabel', 'mag')
    xlim = kwargs.get('xlim', (offset[0], offset[1] + 1))
    ylim = kwargs.get('ylim', None)
    title = kwargs.get('title', None)
    plotsigma = kwargs.get('plotsigma', None)
    # Plot the data with an offset (i.e between -1 and 2)
    for oset in (-1, 0, 1):
        if oset != 0:
            label = None
        else:
            label = datalabel

        frame.errorbar(phase + oset, data, yerr=edata,
                       color=datacolor, ecolor=edatacolor, alpha=dataalpha,
                       label=label, linestyle='None', marker='o', ms=2,
                       zorder=1)
    # Plot the model
    frame.plot(phase_fit, power_fit, color=modelcolor, label=modellabel,
               zorder=2)
    # highlight the 0 to 1 phase region
    frame.axvline(0.0, color='k', lw=1.5)
    frame.axvline(1.0, color='k', lw=1.5)
    # deal with limits as None
    if xlabel is not None:
        frame.set_xlabel(xlabel)
    if ylabel is not None:
        frame.set_ylabel(ylabel)
    if xlim is not None:
        frame.set_xlim(*xlim)
    if ylim is not None:
        frame.set_ylim(*ylim)
    else:
        median, std = np.median(data), np.std(data)
        if plotsigma is not None:
            frame.set_ylim(median - plotsigma * std, median + plotsigma * std)
    if title is not None:
        frame.set_title(title)
    # return
    return frame


def add_fap_lines_to_periodogram(frame, sigmas, faps, **kwargs):
    # deal with keyword arguments
    color = kwargs.get('color', 'b')
    linestyle = kwargs.get('linestyle', 'dotted')
    zorder = kwargs.get('zorder', 2)
    # plot faps
    for f, fap in enumerate(faps):
        xmin, xmax, ymin, ymax = frame.axis()
        frame.axhline(fap, color=color, linestyle=linestyle, zorder=zorder)
        # if fap < ymin:
        #     frame.set_ylim(0.9*fap, ymax)
        # if fap > ymax:
        #     frame.set_ylim(ymin, 1.1*fap)
    # then add a second axis
    xmin, xmax, ymin, ymax = frame.axis()

    for i, sigma in enumerate(sigmas):
        text = str(sigma) + '$\sigma$'
        frame.text(xmax*1.05, faps[i], text, color=color)

    # frame1 = frame.twinx()
    # frame1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    # frame1.set_yticks(faps)
    # frame1.set_yticklabels([str(i) + '$\sigma$' for i in sigmas])
    # frame1.tick_params(axis='y', colors=color)
    # return frame
    return frame


# =============================================================================
# Define data retrieval functions
# =============================================================================
def find_sys_ids(conn=None, **mysqlkwarg):
    # load database
    # Must have database running
    # mysql -u root -p
    if conn is None:
        print("\nConnecting to database...")
        c, conn = load_db(**mysqlkwarg)
    # ----------------------------------------------------------------------
    # find all systemids
    sids = get_list_of_objects_from_db(conn)
    return sids


def load_db(**kwargs):
    """
    Connect to the database

    :param kwargs: keyword arguments are as follows:

    - host,     string, the hostname (default: "localhost")
    - db        string, the datebase to connect to (default: "swasp")
    - table     string, the table to connect to (default: "swasp_sep16_tab")
    - user,     string, the username (default: "root")
    - passwd    string, the password (default: "1234")
    - conn_timeout  int, connection timeout in millseconds? (default: 100000)

    :return:
    """
    host = kwargs.get('host', 'localhost')
    db_name = kwargs.get('db', 'swasp')
    uname = kwargs.get('user', 'root')
    pword = kwargs.get('passwd', '1234')
    conn_timeout = kwargs.get('conn_timeout', 100000)

    # set database settings
    conn1 = MySQLdb.connect(host=host, user=uname, db=db_name,
                            connect_timeout=conn_timeout, passwd=pword)
    c1 = conn1.cursor()
    return c1, conn1


def get_list_of_objects_from_db(conn=None, **kwargs):
    """
    Gets a list of objects from the database using keyword arg query
    :param conn: the connection to the database

    :param kwargs: keyword arguments are as follows:

    - host,     string, the hostname (default: "localhost")
    - db        string, the datebase to connect to (default: "swasp")
    - table     string, the table to connect to (default: "swasp_sep16_tab")
    - user,     string, the username (default: "root")
    - passwd    string, the password (default: "1234")
    - query     string or None, if defined the query to use
    - conn_timeout  int, connection timeout in millseconds? (default: 100000)
    :return:
    """

    query = kwargs.get('query', None)
    # ----------------------------------------------------------------------
    if conn is None:
        c, conn = load_db(**kwargs)
    # find all systemids
    print("\nGetting list of objects...")
    if query is None:
        query = "SELECT CONCAT(c.systemid,c.comp)"
        query += " AS sid FROM {0} AS c".format(kwargs['table'])
        query += " where c.systemid is not null and c.systemid <> ''"
    rawdata = pandas.read_sql_query(query, conn)
    rawsystemid = np.array(rawdata['sid'])
    # get list of unique ids (for selecting each as a seperate curve)
    sids = np.array(np.unique(rawsystemid), dtype=str)
    # return list of objects
    return sids, conn


def get_lightcurve_data(conn=None, sid=None, sortcol=None, replace_infs=True,
                        **kwargs):
    """

    :param conn: connection to the database

    :param sid: string or None, if string and query is None uses default
                query to get data

    :param sortcol: string or None, if string use this column to sort by
                    (must be in sql database) if not don't sort

    :param replace_infs: boolean, if True infinities are replaced with NaNs

    :param kwargs: keyword arguments are as follows:

    - host,     string, the hostname (default: "localhost")
    - db        string, the datebase to connect to (default: "swasp")
    - table     string, the table to connect to (default: "swasp_sep16_tab")
    - user,     string, the username (default: "root")
    - passwd    string, the password (default: "1234")
    - query     string or None, if defined the query to use
    - conn_timeout  int, connection timeout in millseconds? (default: 100000)

    :return:
    """
    # Connect to database if not already connected
    if conn is None:
        c, conn = load_db(**kwargs)
    query = kwargs.get('query', None)
    # if we have no sid and no query raise exception
    if query is None and sid is None:
        raise ValueError("Must have either an SID defined or a query defined.")
    # get data using SQL query on database
    if query is None:
        query = 'SELECT * FROM {0} AS c '.format(kwargs['table'])
        query += 'WHERE CONCAT(c.systemid,c.comp) = "{0}"'.format(sid)
    else:
        query = kwargs['query']
    # use pandas to real sql query
    pdata = pandas.read_sql_query(query, conn)
    # sort by HJD column
    if sortcol is not None:
        pdata = pdata.sort_values(sortcol)
    # Replace infinities with nans
    if replace_infs:
        pdata = pdata.replace([np.inf, -np.inf], np.nan)
    # return pdata
    return pdata


# =============================================================================
# Define sub region functions
# =============================================================================
def get_subregion(data, min_points, max_gap):
    # use relative clustering algorithm to group periods (by percentage)
    groups, group_indices = cluster(data, maxgap=max_gap, return_indices=True)

    # max sure all groups have at least min_points in group
    kept_groups, kept_group_indices = [], []
    for g, group in enumerate(groups):
        if len(group) >= min_points:
            kept_groups.append(group)
            kept_group_indices.append(group_indices[g])

    return kept_groups, kept_group_indices


def subregion_mask(data, min_points, max_gap):
    # get subregions
    groups, group_indices = get_subregion(data, min_points, max_gap)
    # define data indices
    indices = np.indices(data.shape)[0]
    # define group masks
    groupmasks = []
    for groupindex in group_indices:
        # use in 1d to create a mask of all indices in data that are in group
        groupmask = np.in1d(indices, groupindex)
        # append group mask to list
        groupmasks.append(groupmask)
    # return list of group masks
    return groupmasks


def subregion_bounds(data, min_points, max_gap):
    # get subregions
    groups, group_indices = get_subregion(data, min_points, max_gap)
    # define group masks
    groupbounds = []
    for group in group_indices:
        # find the bounds of the group
        gmin, gmax = np.min(group), np.max(group)
        # append group bounds to list
        groupbounds.append([gmin, gmax])
    # return list of group masks
    return groupbounds


def cluster(data, maxgap=None, percentagemaxgap=None, return_indices=False):
    """

    Arrange data into groups where successive elements
    differ by no more than *maxgap* or *percentagemaxgap*

    :param data: numpy array, vector to sort into groups

    :param maxgap: float or None, the maximum gap between groups of points

    :param percentagemaxgap: float or None, if not None overrides maxgap, the
                             relative maximum gap between groups defined as a
                             percetnage away from the data element

    :param return_indices: boolean, if True returns indices of the clustered
                           points

    modified from
       http://stackoverflow.com/questions/14783947/
               grouping-clustering-numbers-in-python

    example:

        cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)

        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

        cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)

        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

    """
    if maxgap is None and percentagemaxgap is None:
        raise ValueError("Must define either maxgap or percentagemaxgap")

    data = np.array(data)
    argsort = np.argsort(data)
    data = data[argsort]

    groups = [[data[0]]]
    group_index = [[argsort[0]]]
    for i, x in enumerate(data[1:]):
        if percentagemaxgap is not None:
            gap = x * percentagemaxgap / 100.0
        else:
            gap = maxgap

        if abs(x - groups[-1][-1]) <= gap:
            groups[-1].append(x)
            group_index[-1].append(argsort[i + 1])
        else:
            groups.append([x])
            group_index.append([argsort[i + 1]])
    if return_indices:
        return groups, group_index
    else:
        return groups


# =============================================================================
# Define auxiliary functions
# =============================================================================
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


def create_data(num: int, timeamp=4, signal_to_noise=5, period=1.0,
                random_state=None, dt=None):
    """
    Create test data

    modified from:
    https://github.com/jakevdp/PracticalLombScargle/blob/master
          /figures/Uncertainty.ipynb

    :param num: int, number of data points

    :param timeamp: float, scale factor for time series (if T=1 will run from
                    t = 0 to t = 1

    :param signal_to_noise: float, the required signal to noise ratio for the
                            data (used to compute uncertainties)

    :param period: float, the period of the signal

    :param random_state: int, random seem to use

    :param dt: None or float, used as minimum spacing, i.e. if dt = 1.0
               data will be spaced to nearest 1.0
               (used to affect the window function)
    :return:
    """
    rng = np.random.RandomState(random_state)
    t = timeamp * rng.rand(num)

    if dt is not None:
        t = np.array(t // dt, dtype=int) * dt

    dy = 0.5 / signal_to_noise * np.ones_like(t)
    y = np.sin(2 * np.pi * t / period) + dy * rng.randn(num)
    return t, y, dy


def normalise(x, kind):
    if len(x) == 0:
        return x
    if kind == 'max':
        return x / np.max(x)
    else:
        return x


def sigma2percentile(sigma):
    """
    Percentile calculation from sigma
    i.e. 1 sigma == 0.68268949213708585 (68.27%)
    :param sigma: [float]     sigma value
    :return:
    """
    # percentile = integral of exp{-0.5x**2}
    percentile = erf(sigma / SQRTTWO)
    return percentile


def percentile2sigma(percentile):
    """
    Sigma calcualtion from percentile
    i.e. 0.68268949213708585 (68.27%) == 1 sigma
    :param percentile: [float]     percentile value
    :return:
    """
    # area = integral of exp{-0.5x**2}
    sigma = SQRTTWO * erfinv(percentile)
    return sigma

# =============================================================================
# End of code
# =============================================================================
