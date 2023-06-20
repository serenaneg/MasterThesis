#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:16:55 2023

@author: serena
"""

############## UNIFICATION OF DAILY DATA######################################
import xarray as xr
import numpy as np
from datetime import datetime

path = "/home/serena/Scrivania/Magistrale/thesis/data/"

chl_path = path + "MODIS_CHL_DAILY/"

data_path = path + "MODIS_INTERPOLATED_DATA/"

#%%
#open all file as dataset
chl = xr.open_mfdataset(chl_path + "AQUA_MODIS.*.L3m.DAY.CHL.x_chlor_a.nc", combine = "nested", concat_dim="time")

# BASH COMMAND TO USE FOR dates.txt CREATION
# printf '%s\n' * | cut -d . -f2 >> dates.txt

with open(chl_path + "dates.txt", 'r') as file:
     dates_list = file.read().split("\n")

#!!LIST 6351 ELEMENTS
# Remove empty strings from the dates_list
dates_list = [date for date in dates_list if date]


chl["time"] = [datetime.strptime(i, "%Y%m%d") for i in dates_list[1:]]

#new netcdf file combined
chl.chlor_a.load().to_netcdf(data_path + "MODIS_chl.nc") #daily dataset
#%%
#lat, lon, bathy
lons, lats = chl.lon, chl.lat

bathy = xr.open_dataset(path + "../data/bathymetry.nc")

bathy = bathy.interp(lon = lons, lat = lats, method = "nearest")

#new netcdf file comned and interpolated
bathy.to_netcdf(data_path + "bathymetry_interpolated.nc")

#%%CHL DAILY NO MISSING DATES
chlorophyll = xr.open_dataset(data_path + "MODIS_chl.nc").load()

#add missing dates 
chlorophyll = chlorophyll.resample(time = 'D').asfreq()

chlorophyll.load().to_netcdf(data_path + "MODIS_chl_1D_nomissing.nc")

#%%

chlorophyll = xr.open_dataset(data_path + "MODIS_chl.nc").load().convert_calendar("noleap")

chlorophyll = chlorophyll.sel(time=~((chlorophyll.time.dt.month == 12) & (chlorophyll.time.dt.day >= 27))).resample(time = "1D").asfreq()

c = chlorophyll.where(chlorophyll["time.year"] == 2003, drop = True).resample(time = "8D").mean()
for y in range(2004,2021):
    c = xr.merge([c, chlorophyll.where(chlorophyll["time.year"] == y, drop = True).resample(time = "8D").mean()])

datetimeindex = c.indexes['time'].to_datetimeindex()

c["time"] = datetimeindex

c.load().to_netcdf(data_path + "MODIS_chl_8D.nc")

#%%################ MONTHLY ANOMALIES #################################################
chlor = xr.open_mfdataset(data_path + "MODIS_chl_1D_nomissing.nc").chlor_a
chlor_month = chlor.groupby('time.month').mean()
month_list = chlor_month['month'].values.tolist()

# chlorophyll = chlorophyll.assign_coords(day_of_year = chlorophyll.time.dt.strftime("%d-%m"))
# chl_anomalies = ((chlorophyll.groupby("day_of_year") - chlorophyll.groupby("day_of_year").mean("time")).groupby("day_of_year")) / chlorophyll.groupby("day_of_year").std("time")


chl_anomalies = xr.Dataset()
for i in month_list:
    a = chlor_month.where(chlor_month['month'] == 5,  drop = True) - chlor_month.mean
    chl_anomalies = xr.merge([chl_anomalies, a])

chl_anomalies.load().to_netcdf(data_path + "MODIS_chl_anomalies_yearly.nc")
