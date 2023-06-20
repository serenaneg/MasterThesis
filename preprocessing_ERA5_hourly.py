#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:59:05 2023

@author: serena
"""

############## UNIFICATION OF WIND DATA######################################
import xarray as xr
import numpy as np
import datetime
import pandas as pd

path = "/home/serena/Scrivania/Magistrale/thesis/data/"

wind_path = path + "ERA5/"

#%%combine NetCDF together
#open all yearly u and v dataset
u_wind = xr.open_mfdataset(wind_path + "uwind.*.nc", combine = 'nested', concat_dim="time")

v_wind = xr.open_mfdataset(wind_path + "vwind.*.nc", combine = 'nested', concat_dim="time")


uwind_daily = u_wind.resample(time = "D").mean()

uwind_daily.load().to_netcdf(wind_path + "u_wind_daily_era5.nc")

vwind_daily = v_wind.resample(time = "D").mean()

vwind_daily.load().to_netcdf(wind_path + "v_wind_daily_era5.nc")
