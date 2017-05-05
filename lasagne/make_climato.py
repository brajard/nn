#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:52:41 2017

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
refdate = datetime(1950,1,1)
timerange = 30/365.25 #fraction of year 



#%%load data
data = xr.open_dataset(os.path.join(datadir,fname))
dates = pd.DatetimeIndex([refdate + timedelta(d/24) for d in data.dates.values])

data_names = [k for k in data.data_vars]
clim2 = np.linspace(0,1,366)
clim2 = clim2[:-1]
frac_year = ts_to_dec(dates)


#%% process
#clim values are computed in fonction on year fraction for a non bissectile year

Larr = []
for fname in data_names:
    sst_mean = np.zeros((clim2.size,data.yind.size,data.xind.size))
    sst_xr = xr.DataArray(sst_mean,
                          coords=[clim2,data.yind,data.xind],
                          dims=['frac_year','yind','xind'],
                          name=fname)
    
    
    
    for data_arr in sst_xr:
        #selecti time in the timerange
        dfy = data_arr.frac_year.values
        sel = np.logical_or((frac_year-dfy)%1<timerange/2,(dfy-frac_year)%1<timerange/2)
        sst = data[fname].loc[sel].mean(axis=0)
        data_arr[:,:]=sst 
    #data_arr is a ref to sst_xr so modifying data_arr is modifying sst_xr
    #IMPORTANT [:,:] otherwise it does not work
    Larr.append(sst_xr)
#%% save
out = xr.merge(Larr)
out.to_netcdf(os.path.join(datadir,climname))


