#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:32:19 2017
Extract surface data (one file per day)
To use this script you can modify:
    outdir : directory to save the extraction
    datadir : directory where to find the daily data
    prefix : prefix of the ncfiles in the format 'prefix*.nc'
    fieldname : name of the parameter to download
Prerequisites :
    all nc files must have the same grid
    all nc files must contain depth,longitude,latitude, time 
The files should have the same grid 
@author: jbrlod
"""

#%% init
import netCDF4 as nc
from os.path import join
import glob
import numpy as np
import xarray as xr

datadir = '/usr/home/jbrlod/these/postdoc/collaborations/bigdata/lip6/cmems/CMEMS_DATA'
outdir = '../data'
prefix = 'NATL_AN_'
fieldname = 'thetao'
depth0 = 0.49402499
Lat0 = 39.33
Lon0 = range (-63,-53)
geo = [(Lat0,lon) for lon in Lon0]
n = 31
d = n//2
xind,yind = range(n), range(n)

#%% load data
L = glob.glob(join(datadir,prefix+'*.nc'))
xdata = []
#Look for info in L[0]
with nc.Dataset(L[0],'r') as ncf:
    depth = ncf.variables['depth'][:]
    lon = ncf.variables['longitude'][:]
    lat = ncf.variables['latitude'][:]
    
#%% Extract data
for ig,(lat0,lon0) in enumerate(geo):
    idepth = np.argmin((depth-depth0)**2)
    ilon = np.argmin((lon-lon0)**2)
    ilat = np.argmin((lat-lat0)**2)
    slon = slice(ilon-d,ilon-d+n)
    slat = slice(ilat-d,ilat-d+n)
    
    time = np.zeros(len(L)) #hours since 1950-01-01 00:00:00
    data = np.zeros((len(L),n,n))
    
    for i,f in enumerate(L):
        with nc.Dataset(f,'r') as ncf:
            data[i,:,:] = ncf.variables[fieldname][0,idepth,slat,slon]
            time[i] = ncf.variables['time'][:]
    itime = np.argsort(time)
    xdata.append( xr.DataArray(data[itime],coords=[time[itime],yind,xind],dims=['dates','yind','xind'],name = fieldname+'_'+str(ig)))

# %% Save data  
outdata = xr.Dataset({data.name:data for data in xdata})
outdata.to_netcdf(join(outdir,'sst_data.nc'))