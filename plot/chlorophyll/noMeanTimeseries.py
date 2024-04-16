#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:14:22 2024

@author: serena
"""

import xarray as xr
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from scipy import signal 
from scipy import interpolate
from matplotlib.path import Path

#%%FUNTIONS
def mean_values_list(variable):
            
    #create a new coodinates = day count
    # Calculate the start date
    variable['time'] = pd.to_datetime(variable['time'], format='%Y-%m-%d')
    start_date = variable['time'].min()
    
    # Calculate the time difference in days and store it in a new column
    variable['day_count'] = (variable['time'] - start_date).dt.days
    
    days = variable['day_count'].values.tolist()
    variable['time'] = variable['day_count']
       
    #daily mean of chl values => 1 value per day
    # variable_mean = variable.mean(dim=('longitude', 'latitude')) #for winds
    variable_mean = variable.mean(dim = ('lon', 'lat')) #for chl
 
    #remouve nan
    variable_mean = variable_mean.where(~np.isnan(variable_mean), 0)
    variable_list = variable_mean.values.tolist()
    
    return variable.day_count, variable_list

#%% LOAD DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/MODIS_INTERPOLATED_DATA/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/chlor/"

ds = xr.open_dataset(path + "MODIS_chl_1D_nomissing.nc")

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

#%%MASK BATHYMETRY AND CHLOROPHYLL
depth_polyg = depth_limits * grid_int
#substitude 0 with nan beacuse depth_limits is a greater area then grid_int
depth_zeros = depth_polyg.where(depth_polyg != 0, np.nan)
depth_ones = depth_zeros.where(np.isnan(depth_zeros), 1)

#%%APPLY MASK TO CHL
chl = ds['chlor_a']

chl_sel = []
for i in range(0, 6574):
    a = chl[i,:,:] * depth_ones
    chl_sel.append(a)
    print(i)
    
chl_masked = xr.DataArray(
    np.array(chl_sel),
    dims=('time', 'lat', 'lon'),  
    coords={'time': chl['time'], 'lat': chl['lat'], 'lon': chl['lon']}
) #masked chl only on the box region for the spectrum

#%% INTERPOLATED TIMESERIES 1D WITH MEAN
chl_days, chl_list = mean_values_list(chl_masked)
data = np.vstack([chl_days, chl_list])

#mask zeros
ma_data = np.ma.masked_equal(data,0)

#interp on complete timeserie
chl_ma = ma_data[1,:]
#delete masked array
chl_mask = chl_ma[~chl_ma.mask]
interp = interpolate.interp1d(np.arange(chl_mask.size), chl_mask , kind='nearest')
chl_stretch = interp(np.linspace(0, chl_mask.size-1, len(chl_list)))
chl_noGlobalMean = chl_stretch - np.mean(chl_stretch)
#%%
year = np.array(range(0, 6574 + 1, 366))

fig2, bx = plt.subplots(1, 1, figsize = (42, 15))
bx.set_title("Timeseries of spatially avegraged Chl-a (01-01-2003 / 31-12-2020)", fontsize = 60, y = 1.05)
bx.plot(chl_days, chl_noGlobalMean, "-", label = 'Chl(t) - global mean', color = "seagreen") 
bx.set_xlim(0,6600)
bx.set_ylabel('Chl-a concentration  [mg/m$^{3}$]', fontsize = 50)
bx.set_xlabel('Years', fontsize= 50)
#bx.set_yscale('log')
bx.yaxis.set_major_formatter(ScalarFormatter())
bx.set_xticks(year)
bx.set_xticklabels(['2003','2004', '2005','2006','2007','2008','2009','2010','2011', 
                    '2012','2013','2014','2015','2016','2017','2018','2019', '2020'])
bx.xaxis.set_tick_params(labelsize = 40)
bx.yaxis.set_tick_params(labelsize = 40)
bx.legend(loc='upper left', fontsize = 40)
bx.set_facecolor('whitesmoke')
bx.grid(linewidth = .75, linestyle='--',  color = 'gray')

fig2.tight_layout()
plt.show()
fig2.savefig(plot_path + "chl_NoMean_global.png", bbox_inches='tight')
#%%MONTHLY MEAN TO REMOUVE FROM THE TIMESERIE
ds_month = chl_masked.groupby('time.month').mean(skipna=True).load()
chl_mean_month = np.nanmean(ds_month, axis = (1,2))

chl_NoMean = chl_masked.copy()  # Create a copy of the original dataset to preserve it
for i in range(1, 13):  # Loop through each month (1 to 12)
    month_data = chl_masked.where(chl_masked['time.month'] == i, drop=True)  # Select data for the current month
    chl_NoMean.loc[{'time': month_data['time']}] -= ds_month.sel(month=i)  # Subtract the monthly mean from the current month

#%%
chl_days, chl_list = mean_values_list(chl_NoMean)
data = np.vstack([chl_days, chl_list])

#mask zeros
ma_data = np.ma.masked_equal(data,0)

#interp on complete timeserie
chl_ma = ma_data[1,:]
#delete masked array
chl_mask = chl_ma[~chl_ma.mask]
interp = interpolate.interp1d(np.arange(chl_mask.size), chl_mask , kind='nearest')
chl_stretch_nomean = interp(np.linspace(0, chl_mask.size-1, len(chl_list)))

#%%
april = np.array(range(120, 6574 + 1, 365))
dec = np.array(range(365, 6574 + 1, 365))

#TIMESERES NO MEAN
fig, bx = plt.subplots(1, 1, figsize = (42, 15))
bx.set_title("Time series of spatially averaged Chl-a (01-01-2003 / 31-12-2020)", fontsize = 60, y = 1.05)
bx.plot(chl_days, chl_stretch_nomean, "-", label = 'Chl(t) - monthly mean', color = "seagreen", linewidth = 2.5)
bx.plot(chl_days, chl_noGlobalMean, "-", label = 'Chl(t) - global mean', color = "gold", linewidth = 2.5, alpha = .7)  
bx.plot(chl_days, chl_stretch, "-", label = 'Chl(t) raw', color = "red", linewidth = 2.5, alpha = .4, zorder = -1)  
bx.set_xlim(0,6600)
bx.set_ylabel('Chl-a concentration  [mg/$m^{3}$]', fontsize = 50)
bx.set_xlabel('Years', fontsize= 50)
bx.yaxis.set_major_formatter(ScalarFormatter())
bx.set_xticks(year)
bx.set_xticklabels(['2003','2004', '2005','2006','2007','2008','2009','2010','2011', 
                    '2012','2013','2014','2015','2016','2017','2018','2019', '2020'])
bx.xaxis.set_tick_params(labelsize = 40)
bx.yaxis.set_tick_params(labelsize = 40)
bx.set_facecolor('whitesmoke')
bx.grid(linewidth = .75, linestyle='--', color = 'gray')
bx.legend(loc='upper left', fontsize = 40)

# Iterate over each year
# for yy in range(0, 18):
#     march_start = yy * 365 + 59  
#     april_end = yy * 365 + 120   
#     bx.fill_between((march_start, april_end), 0, np.max(chl_stretch), color='red', alpha=0.2)
#     half_oct = yy * 365 + 289 #half oct  
#     nov_end = yy * 365 + 335
#     bx.fill_between((half_oct, nov_end), 0, np.max(chl_stretch), color='blue', alpha=0.2)
# for j in dec:  
#     bx.vlines(x=j, ymin = 0, ymax = np.max(chl_stretch), colors='k', lw=3, linestyle = '--')
fig.tight_layout()
plt.show()
fig.savefig(plot_path + "chl_noMean_monthly.png", bbox_inches='tight')

#%%
#SPECTRUM OF THIS FILTERED TIMESERIES
sr = 1/86400 

# Perform Welch's periodogram
segment = 1800 #1800 = sesonal  #365 annual
print(segment)
myhann = signal.get_window('hann', segment) #overlapping window

# obtain simply Power (amplitude^2) 
myparams = dict(fs = sr, nperseg = segment, window = np.ones(segment), detrend ='linear',
                noverlap = segment/2, scaling = 'spectrum', nfft = 1800)
#%%
range_1 = np.linspace(50, 100, 16)
range_2= np.linspace(100, 1000, 4)
range_3 = np.linspace(1000, 2000, 4)
range_4 = np.linspace(2000, 2500, 5)
range_5 = np.linspace(2500, 3000, 26)

slices = np.concatenate((range_1[:-1], range_2[:-1], range_3[:-1], range_4[:-1], range_5))
#%%
energy = []
bins = []

for i in slices:
    if i <=100:
         sel_depth = depth_zeros.where((depth_zeros <= -i) & (depth_zeros > (-i-3.125)), drop = False)
         print([i])
    elif 100 < i <= 1000:
         sel_depth = depth_zeros.where((depth_zeros <= -i) & (depth_zeros > (-i-225)), drop = False)
         print([i])
                
    elif 1000 < i <= 2000:
         sel_depth = depth_zeros.where((depth_zeros <= -i) & (depth_zeros > (-i-250)), drop = False)
         print([i])
    
    elif 2000 < i <= 2500:
         sel_depth = depth_zeros.where((depth_zeros <= -i) & (depth_zeros > (-i-100)), drop = False)
         print([i])
         
    elif 2500 < i <= 3000:
         sel_depth = depth_zeros.where((depth_zeros <= -i) & (depth_zeros > (-i-19.2)), drop = False)
         print([i])
    #subsitute non nan values with 1
    sel_depth = sel_depth.where(np.isnan(sel_depth), 1)
    #nan = 0
    # sel_depth = sel_depth.where(~np.isnan(sel_depth), 0)
    
    bins.append(i)
    
    chl_bin = []
     #no global mean
    for j in range(0, 6574):
         b = chl_masked[j,:,:] * sel_depth
         chl_bin.append(b)
    
    chl_bins = np.array(chl_bin)
                 
    chl_mean = np.nanmean(chl_bins, axis = (1,2))
    print(chl_mean.shape)
     #mask zeros
    chl_no_nan = np.nan_to_num(chl_mean, nan=0)
    print(chl_no_nan.shape)
     #mean ignoring nan
    chl_ma = np.ma.masked_equal(chl_no_nan, 0)
     
     #delete masked values => reduce array length
    chl_mask = chl_ma[~chl_ma.mask]
    print(chl_mask.shape)
     
    interp = interpolate.interp1d(np.arange(chl_mask.size), chl_mask , kind='nearest')
    chl_interp = interp(np.linspace(0, chl_mask.size-1, len(chl_mean)))
    chl_filt = chl_interp - np.mean(chl_interp)
     
     #power spectrum
    freq, ps = signal.welch(x = chl_filt, **myparams)
     
     #!! first element ps, much more little than the others => deleted
    energy.append(ps[1:])
     
    dfreq = freq[1]
    print('Spectral resolution = %2.9f Hz'%dfreq)
    
#%%
tau = (1/freq[1:])/86400
energy = np.array(energy)

fig, ax = plt.subplots(1, 1, figsize = (20, 16))
minim = np.mean(energy) -2*np.std(energy)
maxim = np.mean(energy) +2*np.std(energy)
levels = np.linspace(0.000, 0.01, 100) #seasonl 180
#levels = np.linspace(0, 0.01, 100)
scalar = ax.contourf(np.abs(bins), tau, energy.T, levels = levels, extend = 'max', cmap = "nipy_spectral")
ax.set_xscale('log')
ax.set_yscale('log')

ax.vlines(100, np.min(tau), np.max(tau), color = 'white', linewidths = 3)
ax.vlines(1000, np.min(tau), np.max(tau), color = 'white', linewidths = 3)

ax.set_xticks([50, 75, 100, 200, 500, 1000, 1500, 2000])
ax.set_yticks([10, 30, 60, 120, 180])
#ax.set_yticks([10, 30, 60, 120, 180, 365])
#ax.set_yticks([10, 30, 60, 120, 180, 365, 730])
ax.set_xlim([70, 2300])
ax.xaxis.set_tick_params(labelsize = 38)
ax.yaxis.set_tick_params(labelsize = 40)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel('Depth [m]', fontsize = 40)
ax.set_ylabel('Period [Days]', fontsize = 40)
ax.grid(linewidth = .5, linestyle='--')

fmt = ScalarFormatter(useMathText=True)
fmt.set_useMathText(True)
cbar = fig.colorbar(scalar, orientation = 'horizontal', location = 'bottom', pad = 0.1, format=fmt)
cbar.formatter.set_powerlimits((0,0))
cbar.ax.xaxis.get_offset_text().set_fontsize(30)
cbar.ax.tick_params(labelsize = 30)
cbar.set_label( label = "Chl-a concentration  [$(mg/m^{3})^2$]", fontsize = 35)

fig.suptitle("2D Power Spectrum surface Chl-a", fontsize = 45, y = 1)
fig.tight_layout()
fig.savefig(plot_path + "spectrum_2D_seasonal_noMean.png", bbox_inches='tight', dpi = 100)

#%%

