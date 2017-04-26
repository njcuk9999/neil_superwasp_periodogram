#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/04/17 at 3:51 PM

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

# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = "/Astro/Projects/RayPaul_Work/SuperWASP/"
DPATH1 = WORKSPACE + "/Data/from_exoplanetarchive/tmp1/"
DPATH2 = WORKSPACE + "/Data/from_exoplanetarchive/ra_dec_list/"
DFILE2 = "SuperWASP_id_tile_id_lookup"
# -----------------------------------------------------------------------------
WGET1 = "wget -O '{0}' 'http://exoplanetarchive.ipac.caltech.edu:80"
WGET2 = "/data/ETSS//SuperWASP/FITS/DR1/tile{1}/"
WGET3 = "{0}.fits' -a search_521424856.log"
WGET = WGET1 + WGET2 + WGET3

CHUNKSIZE = 1000000

# =============================================================================
# Define functions
# =============================================================================
def get_tile_id(filename):
    tile_id = filename.split('SuperWASP_tile')[1].split('_ids')[0]
    return str(tile_id)


def construct_query(tile, name):
    return WGET.format(name, tile)


def get_ra_dec(filename):
    raw = filename.split("J")[1]
    if '+' in raw:
        part1, part2 = raw.split('+')
    else:
        part1, part2 = raw.split('-')
    ra = 15*num_from_dms(part1)
    dec = num_from_dms(part2)
    return ra, dec


def num_from_dms(string):
    hours = float(string[:2])
    minutes = float(string[2:4])
    seconds = float(string[4:])
    return hours + minutes/60.0 + seconds/3600.0


def concat_tables():
    # get files to merge
    filenames = os.listdir(DPATH2)
    stilts = 'java -jar /home/neil/bin/topcat/topcat-full.jar -stilts'
    # construct command
    outpath = DPATH2 + DFILE2 + '.fits'
    command = '{0} tcat ifmt="fits" out="{1}" in='.format(stilts, outpath)
    # get in command from filelist
    instr = '"'
    for filename in filenames:
        if DFILE2 in filename and filename != DFILE2:
            instr += "{0}{1} ".format(DPATH2, filename)
    instr += '"'
    # run command
    os.system(command + instr)


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    filenames = os.listdir(DPATH1)
    # ----------------------------------------------------------------------
    # loop around files
    print('\n Extracting rows...')
    names_array, tiles_array = [], []
    for filename in tqdm(filenames):
        # only consider .tbl files
        if '.tbl' not in filename:
            continue
        # extract names from data
        data = Table.read(DPATH1 + filename, format='ascii.ipac')
        names = np.array(data['sourceid'])
        del data
        # extract tile id from filename
        tile_id = get_tile_id(filename)
        # add individual files to names_array
        names_array = np.append(names_array, names)
        tiles_array = np.append(tiles_array, np.repeat(tile_id, len(names)))
    # process rows
    print('\n Processing rows...')
    ra_array, dec_array, query_array = [], [], []
    for row in tqdm(range(len(names_array))):
        # get ra, dec and query
        ra, dec = get_ra_dec(names_array[row])
        query = construct_query(tiles_array[row], names_array[row])
        # append to arrays
        ra_array.append(ra), dec_array.append(dec)
        query_array.append(query)

    # construct new fits file
    print('\n Saving to file...')
    chunks = np.arange(0, len(names_array)+CHUNKSIZE, CHUNKSIZE)
    for c in tqdm(range(len(chunks[:-1]))):
        start, end = chunks[c], chunks[c+1]
        print(start, end)
        table = Table()
        table['sourceid'] = names_array[start: end]
        table['tiles_array'] = tiles_array[start: end]
        table['ra'] = ra_array[start: end]
        table['dec'] = dec_array[start: end]
        table['query'] = query_array[start: end]
        table.write(DPATH2 + DFILE2 + '_{0}.fits'.format(c),
                    format='fits', overwrite=True)
        del table

# =============================================================================
# End of code
# =============================================================================
