#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:09:12 2017

@author: jbrlod
"""

#%% init
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from datatools import ts_to_dec

datadir = '../data'
fname = 'sst_data.nc'
climname = 'sst_climato.nc'
normname = 'sst_norm.nc'

refdate = datetime(1950,1,1)
timerange = 30/365.25 #fraction of year 

#%%load data
data = xr.open_dataset(os.path.join(datadir,fname))
clim = xr.open_dataset(os.path.join(datadir,climname))
dates = pd.DatetimeIndex([refdate + timedelta(d/24) for d in data.dates.values])
frac_year = ts_to_dec(dates)
data_names = [k for k in data.data_vars]


#%%processing
Larr = []
for fname in data_names:
    sst_norm = data[fname].copy()
    
    for fyear,sst in zip(frac_year,sst_norm):
        sst[:,:] = sst - clim[fname].sel(frac_year=fyear,method='nearest',tolerance=0.002).values
        #sst is a ref to sst_norm so modifying sst is modifying sst_norm
        #IMPORTANT [:,:] otherwise it does not work
    Larr.append(sst_norm)
    
#%% save
out = xr.merge(Larr)
out.to_netcdf(os.path.join(datadir,normname))
