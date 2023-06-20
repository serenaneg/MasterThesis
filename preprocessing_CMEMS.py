#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:23:00 2023

@author: serena
"""

############## UNIFICATION OF CMEMS DATASET######################################
import xarray as xr
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt

path = "/home/serena/Scrivania/Magistrale/thesis/data/"

data_path = path + "CMEMS/"

path_bathy = "/home/serena/Scrivania/Magistrale/thesis/data/"
#%%
grep = xr.open_mfdataset(data_path + "grep.*.nc", combine = "nested", concat_dim="time")

grep.load().to_netcdf(data_path + "cmems_grep.nc")

#%%BATHY INTERPOLATION
ds = xr.open_dataset(data_path + 'cmems_grep.nc').load()

lons, lats = ds.longitude, ds.latitude

bathy = xr.open_dataset(path_bathy + "bathymetry.nc")

bathy = bathy.interp(lon = lons, lat = lats, method = "nearest")

#new netcdf file comned and interpolated
bathy.to_netcdf(data_path + "bathymetry_interpolated_cmems.nc")

