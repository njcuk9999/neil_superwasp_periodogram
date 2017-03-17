#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/03/17 at 3:53 PM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from scipy.signal import convolve
from astropy.io import fits
import time as tt
try:
    from fastDFT import dft_l, dft_l_ne
    USE = "FAST"
except ModuleNotFoundError:
    USE = "SLOW"
try:
    from non_clean_periodogram_functions import bin_data
    from non_clean_periodogram_functions import quantile_1D
    from non_clean_periodogram_functions import save_to_file
    from non_clean_periodogram_functions import __tqdmlog__
except ModuleNotFoundError:
    raise Exception("Program requires 'non_clean_periodogram_functions.py'")


# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'

# Test data 1
TESTPATH = WORKSPACE + './CLEAN_periodogram_IDL/test.fits'

# test data 2
# TESTPATH = WORKSPACE + '/Data/Elodie/ARG_54_lightcurve.fits'

# whether to bin data
BINDATA = True
BINSIZE = 0.1


# =============================================================================
# Define functions
# =============================================================================
def dfourt(time, data, df=None, fmax=None, ppb=None, dtmin=None, log=False):
    """
    The frequency grid, "freq", on which the spectral window function "wfn"
    and "dft" are computed, controlled by "df", "fmax" and "ppb".

    Python conversion of dfourt.pro http://www.arm.ac.uk/~csj/idl/CLEAN/

    :param time: numpy array or list, input time(independent) vector

    :param data: numpy array or list, input dependent vector

    :param df:   float, frequency increment for the FT (default: 1/T)
                 See Note 2 below

    :param fmax: float, max frequency in the FT        (default: 1/min(dt))
                 See Note 3 below

    :param ppb:  float, points per restoring beam      (default: 4)

    :param dtmin: float, minimum size between time elements (default: 1e-4)
                  See Note 4 below

    :param log: boolean, if True prints progress to standard output
                         if False silent

    :return freq:

    The frequency grid, "freq", on which the spectral window function "wfn"
    and "dft" are computed, controlled by "df", "fmax" and "ppb".

    Note that this implementation is completely general, and therefore slow,
    since it cannot make use of the timing enhancements of the FFT.

    The IDL implementation of the DFT is based on a suite of FORTRAN routines
    developed by Roberts et al.  For more information concerning the algorithm,
    please refer to:
        Roberts, D.H., Lehar, J., & Dreher, J. W. 1987, AJ, 93, 968
        "Time Series Analysis with CLEAN. I. Derivation of a Spectrum"

    Note 1: The frequency resolution element "df" is oversampled by "ppb" to
            ensure accurate determination of the location of peaks in the
            Fourier Transform.

    Note 2: T = total time spanned = max(time) - min(time)

    Note 3: dt = 2. * [minimum time separation]

    Note 4: the number of frequencies can get very large if the minimum time
            separation is small

            number of frequencies = fmax*ppb/df

            therefore "dtmin" sets the minimum time separation allowable
            which caps the value of fmax = 1/dtmin if fmax is greater than this
            value, this is to allow the code to run even if the minimum
            separation in points is zero, note even at the default dtmin
            the code can take a significant amount of time id df is small

    :return freq: numpy array of floats, frequency vector

    """
    # deal with logging
    if log:
        print('\n\t Calculating Frequency Grid...')
    # -------------------------------------------------------------------------
    # Code starts here
    # -------------------------------------------------------------------------
    # Make time and data numpy arrays and test that lengths are equal
    if type(time) not in [np.ndarray]:
        try:
            time = np.array(time)
        except:
            emsg = "Error: 'time' array cannot be convert to numpy array"
            raise Exception(emsg)
    if type(data) not in [np.ndarray]:
        try:
            data = np.array(data)
        except:
            emsg = "Error: 'data' array cannot be convert to numpy array"
            raise Exception(emsg)
    if data.shape != time.shape:
        emsg = "Error: 'time' array and 'data' array must be the same length"
        raise Exception(emsg)
    # remove all infinities and NaNs
    nanmask = np.isfinite(time) & np.isfinite(data)
    time = time[nanmask]
    data = data[nanmask]
    # Sort time vector
    sort = np.argsort(time)
    time = time[sort]
    # Subtract mean time and mean data value from respective vectors
    tvec = time - np.mean(time)
    # get the frequency grid
    freq = calc_freq(tvec, df, fmax, ppb)
    # return frequency grid
    return freq


def calc_freq(time, df=None, fmax=None, ppb=None, dtmin=None):
    """
    Calculates the "time" vector and computes the frequency grid that will be
    used to calculate the discrete Fourier Transform of the time series.

    :param time: numpy array or list, input time(independent) vector

    :param df:   float, frequency increment for the FT (default: 1/T)
                 See Note 2 below

    :param fmax: float, max frequency in the FT        (default: 1/min(dt))
                 See Note 3 below

    :param ppb:  float, points per restoring beam      (default: 4)

    :param dtmin: float, minimum size between time elements (default: 1e-4)
                  See Note 4 below

    The frequency grid, "freq", on which the spectral window function "wfn"
    and "dft" are computed, controlled by "df", "fmax" and "ppb".

    Note 1: The frequency resolution element "df" is oversampled by "ppb" to
            ensure accurate determination of the location of peaks in the
            Fourier Transform.

    Note 2: T = total time spanned = max(time) - min(time)

    Note 3: dt = 2. * [minimum time separation]

    Note 4: the number of frequencies can get very large if the minimum time
            separation is small

            number of frequencies = fmax*ppb/df

            therefore "dtmin" sets the minimum time separation allowable
            which caps the value of fmax = 1/dtmin if fmax is greater than this
            value, this is to allow the code to run even if the minimum
            separation in points is zero, note even at the default dtmin
            the code can take a significant amount of time id df is small


    :return freq: numpy array, frequency grid calculated from the time vector
    """


    # need to set a dt min threshold (and thus a maximum fmax and maximum
    # number of frequencies (this number can be huge if the separation between
    # times is small (or worse is zero --> infinite frequencies)
    if dtmin is None:
        # Default is "1e-3" which can still lead to problems is "df" is
        # very small as number of frequencies = fmax*ppb/df
        dtmin = 1.0e-4
    # Freqeuncy resolution: default is 1./(total time spanned)
    if df is None:
        df = 1./(max(time) - min(time))
    # Maximum frequency: default is 1. / (2. * [minimum time separation])
    # Must ignore two data values at same time point (otherwise frequency
    # separation is infinite
    if fmax is None:
        fmax = 1./(2 * min(time[1: -1] - time[0: -2]))
    if fmax > 1.0/dtmin:
        fmax = 1.0/dtmin
    # Points per beam: default is 4
    if ppb is None:
        ppb = 4.0
    # size of frequency increment
    dfreq = df/ppb
    # Number of frequencies
    mfreq = (fmax/dfreq) + 1
    # frequency vector
    freq = np.arange(0, 2*mfreq, 1.0)*dfreq
    # return frequency vector
    return freq


def run_discrete_fourier_transform(freq, time, data, log=False, use=USE,
                                   maxsize=None):
    """
    Computes the dirty discrete Fourier Transform, for a 1-D time series, "data",
    which is samples at arbitrarily spaced time intervals given by "time"

    Python conversion of dfourt.pro http://www.arm.ac.uk/~csj/idl/CLEAN/

    :param time: numpy array or list, input time(independent) vector

    :param data: numpy array or list, input dependent vector

    :param freq: numpy array, frequency grid calculated from the time vector

    :param log: boolean, if True prints progress to standard output
                         if False silent

    :param use: string, if "FAST" will attempt to use numexpr to speed up
                the discrete fourier transform (requires python module
                numexpr to run) else tries to run a very using numpy
                (around 6 times slower)

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    Note that this implementation is completely general, and therefore slow,
    since it cannot make use of the timing enhancements of the FFT.

    The IDL implementation of the DFT is based on a suite of FORTRAN routines
    developed by Roberts et al.  For more information concerning the algorithm,
    please refer to:
        Roberts, D.H., Lehar, J., & Dreher, J. W. 1987, AJ, 93, 968
        "Time Series Analysis with CLEAN. I. Derivation of a Spectrum"


    """
    # remove all infinities and NaNs
    nanmask = np.isfinite(time) & np.isfinite(data)
    time = time[nanmask]
    data = data[nanmask]
    # Sort time vector
    sort = np.argsort(time)
    time = time[sort]
    data = data[sort]
    # Subtract mean time and mean data value from respective vectors
    tvec = time - np.mean(time)
    dvec = data - np.mean(data)
    # Calculate the Fourier transform
    # The DFT is normalised to have the mean value of the data at zero frequency
    if use == 'FAST':
        kwargs = dict(log=log, maxsize=maxsize)
        wfn, dft = discrete_fourier_transform2(freq, tvec, dvec, **kwargs)
    else:
        wfn, dft = discrete_fourier_transform1(freq, tvec, dvec, log=log)
    # return frequency vector, spectral window function and "dirty" discrete
    # Fourier transform
    return wfn, dft


def discrete_fourier_transform1(freq, tvec, dvec, log=False):
    """
    Calculate the Discrete Fourier transform (slow scales with N^2)
    The DFT is normalised to have the mean value of the data at zero frequency

    :param freq: numpy array, frequency grid calculated from the time vector

    :param tvec: numpy array or list, input time(independent) vector, normalised
                 by the mean of the time vector

    :param dvec: numpy array or list, input dependent vector, normalised by the
                 mean of the data vector

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    :param log: boolean, if True prints progress to standard output
                         if False silent

    """
    # deal with logging
    if log:
        print('\n\t Calculating Discrete Fourier Transform...')
    # -------------------------------------------------------------------------
    # Code starts here
    # -------------------------------------------------------------------------
    wfn = np.zeros(len(freq), dtype=complex)
    dft = np.zeros(int(len(freq)/2), dtype=complex)
    for i in __tqdmlog__(range(len(freq)), log):
        phase = -2*np.pi*freq[i]*tvec
        phvec = np.array(np.cos(phase) + 1j * np.sin(phase))
        if i < int(len(freq)/2):
            wfn[i] = np.sum(phvec)/len(tvec)
            dft[i] = np.sum(dvec*phvec)/len(tvec)
        # complete the spectral window function
        else:
            wfn[i] = np.sum(phvec)/len(tvec)
    return wfn, dft


def discrete_fourier_transform2(freq, tvec, dvec, log=False, maxsize=None):
    """
    Calculate the Discrete Fourier transform (slow scales with N^2)
    The DFT is normalised to have the mean value of the data at zero frequency
    uses a optimized method requiring numexpr

    :param freq: numpy array, frequency grid calculated from the time vector

    :param tvec: numpy array or list, input time(independent) vector, normalised
                 by the mean of the time vector

    :param dvec: numpy array or list, input dependent vector, normalised by the
                 mean of the data vector

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    :param log: boolean, if True prints progress to standard output
                         if False silent

    :param maxsize: int, maximum number of frequency rows to processes,
                  default is 10,000 but large tvec/dvec array will use
                  a large amount of RAM (64*len(tvec)*maxsize bits of data)
                  If the program is using too much RAM, reduce "maxsize" or
                  bin up tvec/dvec

    """
    # deal with logging
    if log:
        print('\n\t Calculating fast Discrete Fourier Transform...')
    # Make sure user has numexpr if not use slower (but faster than original
    # DFT)
    try:
        import numexpr
        # Return DFT (faster than discrete_fourer_transform1 essentially
        # the same idea)
        return dft_l_ne(freq, tvec, dvec, log, maxsize)
    except ModuleNotFoundError:
        warn = '\n\nWarning: numexpr module needed to run fast DFT'
        print(warn + ' using slower DFT')
        # Return DFT (faster than discrete_fourer_transform1 but slower than
        # numexpr by ~ factor of 6 essentially the same idea as slow DFT)
        return dft_l(freq, tvec, dvec, log, maxsize)


def clean(freq, wfn, dft, gain=0.5, ncl=100, log=False):
    """
    Deconvolve the spectral window function "wfn" from the "dirty" discrete
    Fourier Transform "dft" by using a 1-D version of the interactive CLEAN
    algorithm [Hoegbom, J. A. 1974, A&AS, 15, 417]

    Python conversion of clean.pro http://www.arm.ac.uk/~csj/idl/CLEAN/

    :param freq: frequency vector

    :param wfn: numpy array of complex numbers, spectral window function

    :param dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    :param gain: fraction of window function to subtract per iteration
                 (default: 0.5)

    :param ncl: number of CLEAN iterations to perform (default: 100)

    :param log: boolean, if True prints progress to standard output
                         if False silent

    :return cdft: numpy array of complex numbers, the CLEANed perodogram

    Through  successive subtractions of fractions of WFN, the couplings
    between physical frequencies and their aliases or pseudo-aliases will be
    removed, while preserving the information contained in the peak
    (i.e., the frequency, amplitude, phase of the cosinusoidal component).

    The routine proceeds as follows:
       1) During each iteration, the WFN is centered on the peak that currently
          has the  maximum amplitude.
       2) A fraction of the amplitude of this peak, specified by the GAIN
          parameter, is entered into the complex array of "CLEAN components",
          CCOMP, at the location corresponding to the peak.  Eventually, the
          entire amplitude of the peak will be restored to this CLEAN component.
       3) The WFN is scaled by the current CCOMP, and subtracted from the input
          DFT to form a residual Fourier spectrum, RDFT.
       4) The process is repeated, with the RDFT from the previous iteration
          used as the input spectrum to the current iteration.
       5) After NCL iterations, CCOMP is convolved with the Gaussian "beam",
          truncated at 5*b_sigma.  The standard deviation, b_sigma, is
          determined from the HWHM of the amplitude of the primary (0-frequency)
          peak of  WFN.  In this suite of routines, the beam is a purely real
          function  because the time vector is symmetrized [
          i.e., tvec = tvec - tmean]
          prior to the computation of WFN and DFT by the procedure DFOURT.
       6) The residual Fourier transform is added to this convolution to
          produce the CLEANed discrete Fourier Transform, CDFT.

    Since the Fourier Transforms of real data are Hermitian, only the
    non-negative frequencies need to be dealt with explicitly.  The negative
    frequency elements may be recovered by using the function CL_CVAL, which
    returns the complex conjugate for negative frequencies, and zero for
    frequencies outside the defined range.  However, for essentially all
    practical purposes, the  negative component can be accounted for by
    doubling the amplitude of  positive component determined directly from CDFT.

    The IDL implementation of the CLEAN algorithm is based on a suite of
    FORTRAN routines developed by Roberts et al.  For details of the
    concerning algorithm and practical advice regarding its use, please
    refer to:  Roberts, D.H., Lehar, J., & Dreher, J. W. 1987, AJ, 93, 968
               "Time Series Analysis with CLEAN. I. Derivation of a Spectrum"

    HISTORY:
        Jan. 91: translated to IDL                         [A.W.Fullerton, BRI]
                 Source:  FORTRAN code distributed by Roberts et al.
        Apr. 96: reorganized code for more efficient execution  [AWF, USM]
                 replaced slow cl_convolv with much faster intrinsic IDL
                 function reorganized I/O list, improved documentation

    COMMENT:
        The IDL implementation of the convolution is very efficient,
        because it makes use of an intrinsic IDL function.  However, there are
        small disagreements between this version and the Roberts et al. code
        for the first MB indices of CDFT.  These difference are due to a minor
        bug in the Roberts et al. code, which incorrectly includes data with
        *negative* indices in the convolution summation for indices 0 < i < MB.
        The ultimate culprit is the routine CVAL, which returns the complex
        conjugate of C when given a negative index. In this context, "0"
        should be returned.

    """
    # deal with logging
    if log:
        print('\n\t Running clean algorithm...')
    # -------------------------------------------------------------------------
    # Code starts here
    # -------------------------------------------------------------------------
    # Set up array of CLEAN components
    ccomp = np.zeros_like(dft)
    # -------------------------------------------------------------------------
    # compute Gaussian restoring beam
    beam = clean_beam(wfn)
    # -------------------------------------------------------------------------
    # CLEAN loop (iterate NCL times)
    for i in __tqdmlog__(range(ncl), log):
        # find current maximum in dft
        pk = np.argmax(abs(dft))
        # estimate CLEAN component
        cc = gain * clean_alpha(wfn, dft, pk)
        # shift and scale wfn and cubstract cc
        dft = clean_subtract_ccomp(wfn, dft, cc, pk)
        # store component in ccomp
        ccomp[pk] = ccomp[pk] + cc
    # -------------------------------------------------------------------------
    # Convolve "ccomp" with the "beam" to produce the "clean FT"
    # -------------------------------------------------------------------------
    # elements per half beam
    mb = int((len(beam)-1)/ 2)
    # define padding
    pad = np.repeat([0+0j], mb)
    # pad the data
    input_array = np.concatenate([pad, ccomp, pad])
    # convolve and recenter
    cdft = np.roll(convolve(input_array, beam), -mb)
    # strip padding
    cdft = cdft[mb: len(input_array) - mb]
    # Return
    return cdft


def clean_beam(wfn, b_sigma=None):
    """
    Function return the "CLEAN" beam, which will subsequently be convolved
    with the CLEAN components to obtain the CLEANed Fourier Transform.

    Python conversion of cl_beam.pro http://www.arm.ac.uk/~csj/idl/CLEAN/

    :param wfn: numpy spectral window function

    :param b_sigma: float, optional - the sigma of the Gaussian restoring beam,
                    in "index units" of the frequency vector.

    The beam is modelled by a Gaussian function truncated
    at +/- 5*b_sigma, where b_sigma is the standard deviation determined
    from the HWHM of the amplitude of the peak at 0-frequency in the spectral
    window function WFN.  Since the time vector was symmetrized by subtraction
    of the mean value (so that t=0 corresponds to the middle point of the
    time series) prior to the calculation of WFN by the routine DFOURT, the
    beam will be a purely real entity.  For details, refer to  Roberts et al.
    1987, AJ, 93, 968.

    HISTORY:
        Apr. 1996:  Written by A. W. Fullerton [Uni-Sternwarte Muenchen]
                    Based on FORTRAN routines of Roberts et al.
                    Concatenation of routines CL_HWHM and CL_FILLB

    :return beam:
    """
    if b_sigma is None:
        #  First, estimate the half width at half max of the specyral window
        # function. Assume that the maximum is located at the 0th element
        hmax = 0.5*abs(wfn[0])

        mask = np.where(abs(wfn) <= hmax)[0]
        i2 = mask[0]
        w2 = wfn[i2]
        # lineraly interpolate to get a more accurate estimate
        if w2 < hmax:
            i1 = mask[0] - 1
            w1 = abs(wfn[i1])
            # interpolating function
            q = (hmax - w1)/(w2 - w1)
            # HWHM of WFN (frequency index units)
            hwidth = i1 + q*(i2 - i1)
        else:
            # HWHM of WFN (frequency index units)
            hwidth = i2

        # beam sigma in index units
        b_sigma = hwidth / np.sqrt(2. * np.log(2.0))

    # Now fill the restoring beam with a Gaussian characterized by HWHM = hwidth
    # (in frequency index units). The beam will be a purely real function,
    # since the time vector was symmetrized (tvec = tvec - tmean) prior to the
    # calculation of WFN by the dfourt function

    # Gaussian normalization constant
    const = 1.0/(2*b_sigma**2)

    # size of restoring beam (truncated at 5*b_sigma)
    n_beam = int(5*b_sigma) + 2

    # "one-sided" beam, purely real
    x = np.arange(0, n_beam, 1.0)
    # Gaussian function
    y = np.exp(-const*x**2)

    # define real part of beam
    # append(negative frequencies, positive frequencies)
    realpart = np.append(y[::-1], y[1: n_beam])
    beam = np.array(realpart + 0j)

    return beam


def clean_alpha(wfn, dft, l, err=1.0e-4):
    """
    This function returns an estimate for the component ALPHA, which produces
    the DFT at frequency index L via the relation below

    Python conversion of cl_alpha.pro http://www.arm.ac.uk/~csj/idl/CLEAN/

    :param wfn: numpy array of complex numbers, spectral window function

    :param dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    :param l: int, the current maximum in dft

    :param err: float, the allowed error in "wnorm"

    :return alpha: complex number, the amplitude "alpha" in
                   equations below

    Relation:

    dft(l) = alpha*wfn(0) + conj(alpha) * wfn(2*l)

    alpha is given by:

                dft(l) - conj(dft(l))*wfn(2*l)
    alpha =     ------------------------------
                            wnorm

    where:

            wnorm = 1 - abs(wfn(2*l))^2

    See Section III b) [especially equation (24)] of Roberts et al.
    (1987, AJ, 93, 968).

    HISTORY:
        Jan. 91: translated for FORTRAN code by Roberts et al.  [AWF, Bartol]
        Apr. 96: recoded for efficiency and added documentation [AWF, USM]
    """
    # Find the (L, -L) component which produce dft(l) through wfn
    win2l = wfn[2*l]
    wnorm = 1.0 - abs(win2l)**2

    # Trap to avoid singularities
    if (wnorm < err):
        alpha = 0.5*dft[l]
    else:
        alpha = (dft[l] - win2l*np.conjugate(dft[l]))/wnorm
    # return alpha
    return alpha


def clean_subtract_ccomp(wfn, dft, ccomp, l):
    """
    This procedure compute and removes the influence of CLEAN component "ccomp"
    at frequency index L on the "dirty" DFT.

    Python conversion of cl_subcmp.pro http://www.arm.ac.uk/~csj/idl/CLEAN/

    :param wfn: numpy array of complex numbers, spectral window function,
                size = (2* number of frequencies)
    :param dft: numpy array of complex numbers, "dirty" discrete Fourier
                transform, size = number of frequencies

    :param ccomp: complex number, the CLEAN component to remove

    :param l: int, the frequency index of the CLEAN Component

    :return dft: numpy array of complex numbers, modified "dirty" discrete
                 Fourier transform, via subtraction of the frequency couplings
                 associated with the CLEAN component "ccomp" at frequency
                 index "l"

    NOTE: [AWF, April 22, 1996]

        ; The IDL version of CL_SUBCMP does not make the basic mathematical
        operations  as clear as the previous version, but is nonetheless
        preferable because it avoids do loops and extra calls to functions.
        For completeness, and also  to help illuminate the present version of
        the code, here is a stipped  down version of the old procedure:

        PRO CL_SUBCMP, WFN, DIRTY, L, CCOMP
            cneg=conj(ccomp)
            for i=0L,n_elements(dirty)-1 do begin
                dirty(i)=dirty(i) - ccomp*ncl_cval(wfn,i-l) - cneg*wfn(i+l)
            endfor
            return
            end

        where CL_CVAL is a function with the following characteristics:
            FUNCTION NCL_CVAL, ARRAY, I
                This function returns CVAL, the value of the complex ARRAY
                at the index  location I, subject to the following rules:

                (a) if  0 < I < n_elements(array)-1, CVAL = ARRAY(I)

                (b) if      I > n_elements(array)-1, CVAL = 0.0

                (c) if  I < 0                      , CVAL = CONJ(ARRAY(I))

                Item (c) implies that ARRAY is Hermitian,
                        since ARRAY(I) = CONJ(ARR(-I)).

                ii=abs(i)
                if (ii le n_elements(array)-1) then begin ; i is defined

                    if (i ge 0) then cval=array(ii) $    ; i > 0, take array(i)
 	                else cval=conj(array(ii))            ; i < 0, take conjugate

                    endif else begin                     ; i is undefined
                        cval=complex(0.,0.)
                    endelse
                return, cval
                end

    HISTORY:
        Jan. 91: translated from FORTRAN code by Roberts et al.   [AWF, BRI]
        Apr. 96: vectorized and upgraded documentation            [AWF, USM]

    """
    # max element = nwind - 1
    nwind= len(wfn)
    # max element = ndirt - 1
    ndirt = len(dft)
    # -------------------------------------------------------------------------
    # Compute the effect of +l component
    # -------------------------------------------------------------------------
    cplus = np.zeros(ndirt, dtype=complex)
    # index for wfn, shifted to +l comp
    index = np.arange(ndirt) - l
    # for indices less than zero take the conj(wfn[index[mask1]])
    mask1 = index < 0
    cplus[mask1] = np.conjugate(wfn[abs(index[mask1])])
    # for indices between 0 and nwind-1 take wfn[index[mask2]]
    mask2 = (index >= 0) & (index <= nwind-1)
    cplus[mask2] = wfn[index[mask2]]
    # for indices greater than equal to nwind set to zero
    mask3 = index >= nwind
    cplus[mask3] = np.repeat([0+0j], len(mask3[mask3]))
    # -------------------------------------------------------------------------
    # Compute the effect of -l component
    # -------------------------------------------------------------------------
    cminus = np.zeros(ndirt, dtype=complex)
    # index for wfn, shifted to -l comp
    index = np.arange(ndirt) + l
    # for indices less than zero take the conj(wfn[index[mask1]])
    mask1 = index < 0
    cminus[mask1] = np.conjugate(wfn[abs(index[mask1])])
    # for indices between 0 and nwind-1 take wfn[index[mask2]]
    mask2 = (index >= 0) & (index <= nwind-1)
    cminus[mask2] = wfn[index[mask2]]
    # for indices greater than equal to nwind set to zero
    mask3 = index >= nwind
    cminus[mask3] = np.repeat([0+0j], len(mask3[mask3]))
    # -------------------------------------------------------------------------
    # return realigned, rescaled window function .
    dft = dft - ccomp*cplus - np.conjugate(ccomp)*cminus
    return dft


def clean_periodogram(time, data, **kwargs):
    """
    Takes a time and data vector, computes a frequency grid, dirty discrete
    Fourier Transform, and cleans it using the CLEAN algorithm

    :param time: numpy array or list, input time(independent) vector

    :param data: numpy array or list, input dependent vector

    :param kwargs: keyword arguments (if left out will use default values)

    kwargs are as follows:

        - freq:   numpy array, frequency grid calculated from the time vector
                  ONLY uses smallest 50% for DFt and CFT - See Note 1

        - df      float, frequency increment for the FT
                  (default: 1/T See Note 2)

        - fmax    float, max frequency in the FT
                  (default: 1/min(dt) See Note 3)

        - ppb     float, points per restoring beam      (default: 4)

        - gain    fraction of window function to subtract per iteration
                  (default: 0.5)

        - ncl     number of CLEAN iterations to perform (default: 100)

        - log     boolean, if True prints progress to standard output
                  if False silent

        - full    boolean, if True

        - use     string, if "FAST" will attempt to use numexpr to speed up
                  the discrete fourier transform (requires python module
                  numexpr to run) else tries to run a very using numpy
                  (around 6 times slower)

        - maxsize int, maximum number of frequency rows to processes,
                  default is 10,000 but large tvec/dvec array will use
                  a large amount of RAM (64*len(tvec)*maxsize bits of data)
                  If the program is using too much RAM, reduce "maxsize" or
                  bin up tvec/dvec

    Definitions and explanations for these keywords are found in
    "dfourt" and "clean" (see below)

    Uses the dfourt and clean functions.
    Total conversion and update of IDL routines located at:
    http://www.arm.ac.uk/~csj/idl/CLEAN/

    Note 1: by design we don't use the smallest 50% of times or the largest
            50% of frequencies in the DFT so MUST define frequencies accordingly

    Note 2: T = total time spanned = max(time) - min(time)

    Note 3: dt = 2. * [minimum time separation]

    ---------------------------------------------------------------------------
    dfourt:
    ---------------------------------------------------------------------------

    Computes the dirty discrete Fourier Transform, for a 1-D time series,
    "data", which is samples at arbitrarily spaced time intervals
    given by "time"

    Python conversion of dfourt.pro http://www.arm.ac.uk/~csj/idl/CLEAN/

    :param time: numpy array or list, input time(independent) vector

    :param data: numpy array or list, input dependent vector

    :param df:   float, frequency increment for the FT (default: 1/T)

    :param fmax: float, max frequency in the FT        (default: 1/min(dt))

    :param ppb:  float, points per restoring beam      (default: 4)

    :param log: boolean, if True prints progress to standard output
                         if False silent

    The frequency grid, "freq", on which the spectral window function "wfn"
    and "dft" are computed, controlled by "df", "fmax" and "ppb".

    Note that this implementation is completely general, and therefore slow,
    since it cannot make use of the timing enhancements of the FFT.

    The IDL implementation of the DFT is based on a suite of FORTRAN routines
    developed by Roberts et al.  For more information concerning the algorithm,
    please refer to:
        Roberts, D.H., Lehar, J., & Dreher, J. W. 1987, AJ, 93, 968
        "Time Series Analysis with CLEAN. I. Derivation of a Spectrum"

    Note 1: The frequency resolution element "df" is oversampled by "ppb" to
            ensure accurate determination of the location of peaks in the
            Fourier Transform.

    Note 2: T = total time spanned = max(time) - min(time)

    Note 3: dt = 2. * [minimum time separation]

    :return freq: numpy array of floats, frequency vector

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    ---------------------------------------------------------------------------
    clean:
    ---------------------------------------------------------------------------

    This function returns an estimate for the component ALPHA, which produces
    the DFT at frequency index L via the relation below

    Python conversion of cl_alpha.pro http://www.arm.ac.uk/~csj/idl/CLEAN/

    :param wfn: numpy array of complex numbers, spectral window function

    :param dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    :param l: int, the current maximum in dft

    :param err: float, the allowed error in "wnorm"

    :return alpha: complex number, the amplitude "alpha" in
                   equations below

    Relation:

    dft(l) = alpha*wfn(0) + conj(alpha) * wfn(2*l)

    alpha is given by:

                dft(l) - conj(dft(l))*wfn(2*l)
    alpha =     ------------------------------
                            wnorm

    where:

            wnorm = 1 - abs(wfn(2*l))^2

    See Section III b) [especially equation (24)] of Roberts et al.
    (1987, AJ, 93, 968).

    HISTORY:
        Jan. 91: translated for FORTRAN code by Roberts et al.  [AWF, Bartol]
        Apr. 96: recoded for efficiency and added documentation [AWF, USM]

    """
    # -------------------------------------------------------------------------
    # Deal with keyword arguments
    for fname in ['frequency', 'freqs', 'freq']:
        if fname in kwargs:
            kwargs['freq'] = kwargs[fname]
    freq = kwargs.get('freq', None)
    df = kwargs.get('df', None)
    fmax = kwargs.get('fmax', None)
    ppb = kwargs.get('ppb', 4)
    gain = kwargs.get('gain', 0.5)
    ncl = kwargs.get('ncl', 100)
    log = kwargs.get('log', False)
    full = kwargs.get('full', False)
    use = kwargs.get('use', USE)
    maxsize = kwargs.get('maxsize', None)
    # -------------------------------------------------------------------------
    # Use the default frequency parameters to describe the frequency grid.
    # The defaults are selected by leaving df, fmax and ppb out of the function
    if freq is None:
        freq = dfourt(time, data, df, fmax, ppb, log)
    # input frequencies must be in order from low to high!
    else:
        freq = np.sort(freq)
    # -------------------------------------------------------------------------
    # Compute the "dirty" discrete Fourier transform.
    start1 = 0.0
    if log:
        print('\n Computing "dirty" discrete Fourier transform...')
        start1 = tt.time()
    wfn, dft = run_discrete_fourier_transform(freq, time, data, log, use,
                                              maxsize)
    if log:
        end1 = tt.time()
        print('\n\t Took {0} s'.format(end1 - start1))
    # -------------------------------------------------------------------------
    # Clean the DFT. For this demonstration, use a gain of 0.5 and continue
    # for 100 iterations
    start2 = 0.0
    if log:
        print('\n Computing clean periodogram...')
        start2 = tt.time()
    cdft = clean(freq, wfn, dft, gain, ncl, log)
    if log:
        end2 = tt.time()
        print('\n\t Took {0} s'.format(end2 - start2))
    # -------------------------------------------------------------------------
    # If full return frequency, the spectral window function, the dirty DFT
    # and the CLEANed DFT
    if full:
        return freq, wfn, dft, cdft
    # else return only the CLEANed DFT
    else:
        return cdft


def plot_test_graph(time, data, freq, cdft, logged=True):
    """
    Plots a matplotlib test plot of the raw data and the CLEANed periodogram
    compares it to a lombscargle periodogram using the same frequencies

    :param time: numpy array or list, input time(independent) vector

    :param data: numpy array or list, input dependent vector

    :param freq: frequency vector

    :param cdft: numpy array of complex numbers, the CLEANed periodogram

    :param logged: boolean, whether to log the x axis on frequency and time
                   graphs

    :return:
    """
    # Calculate lombscargle
    freq, power = lombscargle_periodogram(time, data, freq)

    # plot
    import matplotlib.pyplot as plt
    plt.close()
    fig, frame = plt.subplots(ncols=1, nrows=3)

    # plot data
    frame[0].scatter(time, data, s=5)

    # plot lombscargle
    frame[1].plot(freq, power/np.nanmax(power),
                  color='red', label='LombScargle')

    # plot clean periodogram
    amps = np.array(2.0*abs(cdft))
    frame[1].plot(freq[0: len(cdft)], amps/np.nanmax(amps),
                  color='b', label='CLEAN perioogram')

    # plot lombscargle
    frame[2].plot(1.0/freq, power/np.nanmax(power),
                  color='red', label='LombScargle')

    # plot clean periodogram
    amps = np.array(2.0*abs(cdft))
    frame[2].plot(1.0/freq[0: len(cdft)], amps/np.nanmax(amps),
                  color='b', label='CLEAN perioogram')

    # finalise graph
    frame[0].set_xlabel('Time')
    frame[0].set_xlabel('Flux')

    frame[1].set_xlabel('Frequency')
    frame[1].set_ylabel('Normalised Amplitude/Power Spectrum')
    frame[1].legend(loc=1, numpoints=1, scatterpoints=1)

    frame[2].set_xlabel('Time / days')
    frame[2].set_ylabel('Normalised Amplitude/Power Spectrum')
    frame[2].legend(loc=1, numpoints=1, scatterpoints=1)

    if logged:
        frame[1].set_xscale('log')
        frame[2].set_xscale('log')

    plt.show()
    plt.close()


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Load data
    print('\n Loading data...')
    fitsrec = fits.getdata(TESTPATH, ext=1)
    time_arr = np.array(fitsrec['time'])
    data_arr = np.array(fitsrec['flux'])
    if 'eflux' in fitsrec.columns.names:
        edata_arr = np.array(fitsrec['eflux'])
    else:
        edata_arr = None
    # ----------------------------------------------------------------------
    # bin data
    if BINDATA:
        if edata_arr is None:
            time_arr, data_arr = bin_data(time_arr, data_arr, binsize=BINSIZE,
                                          log=True)
        else:
            time_arr, data_arr = bin_data(time_arr, data_arr, edata_arr,
                                          binsize=BINSIZE, log=True)
    # ----------------------------------------------------------------------
    # Run clean
    results = clean_periodogram(time_arr, data_arr, log=True, full=True)
    freqs, wfn_arr, dft_arr, cdft_arr = results
    # ----------------------------------------------------------------------
    # Plot the CLEANed amplitude spectrum. The factor of 2 allows for the
    # "mirror image" of the DFT at nagative frequencies
    print('\n Plotting graph...')
    plot_test_graph(time_arr, data_arr, freqs, cdft_arr)


# =============================================================================
# End of code
# =============================================================================
