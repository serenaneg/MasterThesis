#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:03:15 2023

@author: serena
"""

import xarray as xr
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import matplotlib.cm as cm
from scipy import signal 
from matplotlib.path import Path
from scipy import interpolate
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap
import datetime

#%% LOAD DAILY MODIS DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/MODIS_INTERPOLATED_DATA/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/chlor/enhancements/"
#monthly data
ds = xr.open_dataset(path +"MODIS_chl_1D_nomissing.nc")

#%%
# #choose ARRAY coordinate
lat_range = slice(41, 38) #alto-basso
lon_range = slice(-72.6, -68)  #dx-sx

ds = ds.sel(lon=lon_range)
ds = ds.sel(lat=lat_range)
#%% BATHYMETRY ALREADY INTERPOLATED ON CHL
#lat, lon, bathy
lons, lats = ds['lon'], ds['lat']
bathy = xr.open_dataset(path + "bathymetry_interpolated.nc")

depth = bathy['elevation'] 
depth = depth.sel(lat = lats, lon = lons)
depth_limits = depth.where((depth <= -50) & (depth >= -3000), drop = True)




#%%POLYGON MASK
poly_verts = [(-70.6, 38.5), (-72.6, 39.8), (-72.6, 41), (-68, 41), (-68, 38.5)]

# Create vertex coordinates for each grid cell...
lon2d, lat2d = np.meshgrid(lons, lats)
lon2d, lat2d = lon2d.flatten(), lat2d.flatten()

points = np.vstack((lon2d, lat2d)).T

path = Path(poly_verts)
grid = path.contains_points(points)
grid_bool = grid.reshape((72, 110)) #72 = lons length, 110 lat lenghts
#coverto boolean grid into 0-1 grid
grid_int = grid_bool*1


#%%MASK BATHYMETRY
depth_polyg = depth_limits * grid_int
#substitude 0 with nan beacuse depth_limits is a greater area then grid_int
depth_zeros = depth_polyg.where(depth_polyg != 0, np.nan)
depth_ones = depth_zeros.where(np.isnan(depth_zeros), 1)

#%%FIND ENHANCEMENT
#enhancemente = value in the shelf is greater 
#shelf_break = 75 - 1000, shelf = 50, 74, slope = 1001 - 3000
#create masks
shelf = depth_zeros.where((depth_zeros <= -50) & (depth_zeros > -75), drop = False)
shelf = shelf.where(np.isnan(shelf), 1)

shelf_break = depth_zeros.where((depth_zeros <= -75) & (depth_zeros > -1000), drop = False)
shelf_break = shelf_break.where(np.isnan(shelf_break), 1)

slope = depth_zeros.where((depth_zeros <= -1000) & (depth_zeros >= -3000), drop = False)
slope = slope.where(np.isnan(slope), 1)


#%%UPLOAD CHLORD DATA
chlor = ds['chlor_a']
time = chlor.time.values
# #Choose only those map that have more than 50 % of data

valid_list = []
valid_index = []

for i in time:
    f = chlor.sel(time=i)
    if f.count(axis=(0, 1)) >= 3960:
        print(i)
        valid_list.append(f)
        valid_index.append(i)

valid = xr.concat(valid_list, dim='time') #1238 valid
#%%
a = valid[0,:,:] * shelf
b = valid[0,:,:] * shelf_break
c = valid[0,:,:] * slope
for i in range(1, len(valid)): #valid shape
    a = xr.concat([a, valid[i,:,:] * shelf], dim='time')
    b = xr.concat([b, valid[i,:,:] * shelf_break], dim='time')
    c = xr.concat([c, valid[i,:,:] * slope], dim='time')
    print(i)

chl_shelf = xr.DataArray(a)
chl_shelf_break = xr.DataArray(b)
chl_slope = xr.DataArray(c)

#%%Enhance if chl_shelf_break is greater than slope and shelf
mean_s = chl_shelf.mean(axis=(1,2), skipna=True)
mean_sb = chl_shelf_break.mean(axis=(1,2), skipna=True)
mean_sl = chl_slope.mean(axis=(1,2), skipna=True)
#%%
enhance_list = []  # Initialize an empty list to store the DataArrays
index = []

for i in range(0, len(valid)):
    if np.logical_and((mean_sb.values[i] > mean_s.values[i]), (mean_sb.values[i] > mean_sl.values[i])):
        enhance_list.append(mean_sb[i])
        index.append(i)
# Concatenate the DataArrays in the list along the 'time' dimension
enhance = xr.concat(enhance_list, dim='time')

enhance_count = enhance.groupby('time.month').count()
print(enhance_count)
#<xarray.DataArray (month: 8)>
# array([ 1,  7, 17, 23,  4,  1,  3, 10])
enhance_year = enhance.groupby('time.year').count()
print(enhance_year)
# <xarray.DataArray (year: 17)>
# array([ 8,  3,  2,  4,  5,  1,  4,  1,  9,  1,  3,  4,  2,  2,  5, 10,  2])
#%%
month_full = pd.Series(np.nan, index=range(1, 12 + 1))
month_full[enhance_count.month] = enhance_count.values

year_full = np.full(18, np.nan)

year_full = pd.Series(np.nan, index=range(2003, 2020 + 1))
year_full[enhance_year.year] = enhance_year.values
#%%
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(16, 10))

ax.vlines(year_full.index, 0, year_full.values, linewidths=53, alpha = .8, color = 'steelblue')
ax.yaxis.set_tick_params(labelsize = 20)
ax.set_xticks(year_full.index)
ax.set_xticklabels(['2003','2004','2005','2006','2007','2008','2009','2010','2011', 
                    '2012','2013','2014','2015','2016','2017','2018','2019', '2020'], fontsize = 20)
ax.set_ylabel('Number of enhancements per year', fontsize=20)
ax.set_xlabel('Year', fontsize = 20)
ax.grid(linewidth=0.5, color='gray', linestyle='--')
fig.suptitle("Number of shelf-break's enhancements per year", fontsize = 30, y = 1.0)
fig.tight_layout()
plt.show()
fig.savefig(plot_path + 'num_enhancement_year.png', bbox_inches='tight', dpi = 300)

#%%
fig, ax = plt.subplots(figsize=(12, 10))

ax.vlines(month_full.index, 0, month_full.values, linewidths=58, alpha = .8, color = 'steelblue')
ax.yaxis.set_tick_params(labelsize = 20)
ax.set_xticks(month_full.index)
ax.set_xticklabels(month_list, fontsize = 20)
ax.set_ylabel('Number of enhancements', fontsize=20)
ax.set_xlabel('Month', fontsize = 20)
ax.grid(linewidth=0.5, color='gray', linestyle='--')
fig.suptitle("Number of shelf-break's enhancements per month", fontsize = 30, y = 1.0)
fig.tight_layout()
plt.show()
fig.savefig(plot_path + 'num_enhancement_month.png', bbox_inches='tight', dpi = 300)
#%%
#datatime to integer index
start_date = pd.to_datetime('2003-01-02')
end_date = pd.to_datetime('2020-12-31')
enhance_full = enhance.reindex(time=pd.date_range(start_date, end_date, freq='D'), copy = True)
values = enhance_full.values
index = np.where(~np.isnan(values))[0]

index_ext = np.full(6574, np.nan)
index_ext[index] = index
#%%
year_list = ['2003','2004', '2005','2006','2007','2008','2009','2010','2011', 
                   '2012','2013','2014','2015','2016','2017','2018','2019', '2020']

fig, (ax2, ax1) = plt.subplots(ncols=2, nrows = 1, figsize=(32,10), gridspec_kw={'width_ratios': [2, 1]})
cmap = cm.tab20
for i in np.arange(0, 6205, 364):
    color = cmap(((i)%6205)/6205)
    ax1.scatter(range(0, 365), values[i:i+365], marker  = "x", s= 120, linewidths = 3, color = color)
    # ax.vlines(range(0, 365), 0, extend_values[i:i+365], linewidths=73, alpha = .8, color = color)

# ax.set_xticks(np.arange(0, 6500, 365.2))
# ax.set_xticklabels(['2003','2004', '2005','2006','2007','2008','2009','2010','2011', 
#                     '2012','2013','2014','2015','2016','2017','2018','2019', '2020'])
ax1.xaxis.set_tick_params(labelsize = 25)
ax1.yaxis.set_tick_params(labelsize = 25)
ax1.grid(lw=.5, linestyle = '--')

ax1.set_xlabel('Day of the year', fontsize = 30)
# ax1.set_ylabel('Ehnanced Chlor-a concentration [mg/m$^3$]', fontsize = 20)
ax1.grid(lw = .5, linestyle='--')

for i in np.arange(0, 6205, 364):
    color = cmap(((i)%6205)/6205)
    ax2.scatter(index_ext[i:i+365], enhance_full.values[i:i+365],  marker  = "x", s= 120, linewidths = 3,
               color = color)

ax2.set_xticks(np.arange(0, 6500, 365.2))
ax2.set_xticklabels(['2003','2004', '2005','2006','2007','2008','2009','2010','2011', 
                    '2012','2013','2014','2015','2016','2017','2018','2019', '2020'])
ax2.xaxis.set_tick_params(labelsize = 25)
ax2.yaxis.set_tick_params(labelsize = 25)
ax2.grid(lw=.5, linestyle = '--')

ax2.set_ylabel('Ehnanced Chlor-a concentration [mg/m$^3$]', fontsize = 30)

fig.suptitle("Enhanced Chlorophyll days", fontsize = 40, y = 1.)
fig.tight_layout()
fig.savefig(plot_path + "enhancement.png", bbox_inches='tight')  
#%%
#PRINT DATE FOR ENHANCMENT WITH CHL > 6
# Open the file in write mode
with open(plot_path + 'ehnancements.txt', 'w') as file:
    
    for i in range(0, len(enhance)):
        # if enhance.values[i] > 1 :
            print(enhance.time.values[i])
            print(enhance.values[i])
            
            result = enhance.time.values[i], enhance.values[i]
            
            file.write(str(result) + '\n')

#%%SPECTRUM
sr = 1/86400 

# Perform Welch's periodogram
segment = 1800 #1800 = sesonal #365semi stagionale
print(segment)
myhann = signal.get_window('hann', segment) #overlapping window

# obtain simply Power (amplitude^2) 
myparams = dict(fs = sr, nperseg = segment, window = np.ones(segment), detrend ='linear',
                noverlap = segment/2, scaling = 'spectrum')
#%%
chlor_bloom = enhance_full.to_numpy()
#nan to zeros
chlor_bloom = np.nan_to_num(chlor_bloom, nan=0)

freq, ps = signal.welch(x = chlor_bloom, **myparams)

tau = (1/freq[1:])/86400
energy = ps[1:]
            
#%%PLOT
fig, ax = plt.subplots(1, 1, figsize = (16, 12))

ax.plot(tau, energy, linewidth=1.5, color='darkred')
ax.set_xscale('log')
# ax.set_yscale('log')

ax.set_xticks([10, 30, 60, 120, 180, 365, 730])
# ax.set_xticks([10, 30, 60, 120, 180])
ax.set_yticks(np.linspace(energy.min(), energy.max(), 10))

ax.xaxis.set_tick_params(labelsize = 25)
ax.yaxis.set_tick_params(labelsize = 25)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

# ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(minor_thresholds=(2, 0.4)))
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))


ax.set_ylabel( "Chlor-a concentration  [$(mg/m^{3})^2$]", fontsize = 30)
ax.set_xlabel('Period [Days]', fontsize = 30)
ax.grid(linewidth = .75, linestyle='--')

fig.suptitle("Power Spectrum of Chlorophyll enhacements", fontsize = 35, y = 1)
fig.tight_layout()
plt.show()
fig.savefig(plot_path + "spectrum_ehnancements_seasonal.png", bbox_inches='tight')
# fig.savefig(plot_path + "spectrum_ehnancements_annual.png", bbox_inches='tight')


            