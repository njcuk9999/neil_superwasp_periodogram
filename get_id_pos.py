#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/06/17 at 2:41 PM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
import MySQLdb
import pandas
from astropy.table import Table
from tqdm import tqdm


# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = "/Astro/Projects/RayPaul_Work/SuperWASP"
SAVEPATH = WORKSPACE + "/Data/Swasp_sep16_all_unique.fits"

# --------------------------------------------------------------------------
# set database settings
HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = '1234'
DATABASE = 'swasp'
TABLE = 'swasp_sep16_tab'
TIMECOL = 'HJD'


# =============================================================================
# Define functions
# =============================================================================
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
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Code here
    # ----------------------------------------------------------------------
    sql_kwargs = dict(host=HOSTNAME, db=DATABASE, table=TABLE, user=USERNAME,
                      passwd=PASSWORD)
    # get sids
    sids, conn = get_list_of_objects_from_db(**sql_kwargs)

    # loop around each sid
    datadict = dict()
    for sid in tqdm(sids):
        pdata = get_lightcurve_data(conn=conn, sid=sid, **sql_kwargs)
        for col in pdata.columns:
            value = pdata[col][0]
            if col not in datadict:
                datadict[col] = [value]
            else:
                datadict[col].append(value)

    # convert to table
    table = Table()
    for col in list(datadict.keys()):
        table[col] = datadict[col]
    table.write(SAVEPATH, overwrite=True)

# =============================================================================
# End of code
# =============================================================================
