#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:31:06 2023

@author: serena
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
import matplotlib.cm as cm
import matplotlib.colors as colors
import cmocean
from scipy import stats
#%% LOAD DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/MODIS_INTERPOLATED_DATA/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/chlor/"

#monthly data
ds = xr.open_dataset(path +"MODIS_chl_1D_nomissing.nc") 

#%%CCORDS
lat_range = slice(41, 38)
lon_range = slice(-72.6, -68)

ds = ds.sel(lon=lon_range)
ds = ds.sel(lat=lat_range)

#%%#%% BATHYMETRY ALREADY INTERPOLATED ON CHL
#lat, lon, bathy
lons, lats = ds['lon'], ds['lat']
bathy = xr.open_dataset(path + "bathymetry_interpolated.nc")

depth = bathy['elevation'] 
depth = depth.sel(lat = lats, lon = lons)

#%%CHLOROPHYLL ANOMALIES
chl = ds['chlor_a']

#media mensile => 12 mesi medi
clim_month = chl.groupby('time.month').mean(dim='time', skipna=True)
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
std_month = chl.groupby('time.month').std()

anomalies = [] 
for y in range(2003,2021):
    year = chl.where(chl['time.year'] == y, drop = True)
    month = year.groupby('time.month')
    for i in np.arange(1, 13, 1):
        anom = (month[i].mean(axis = 0) - clim_month[i-1, :, :]) / std_month[i-1, :, :]       
        anomalies.append(anom)

plt.contourf(lons, lats, anomalies[120], cmap = cmocean.cm.curl)

dev = clim_month/std_month
#%%PLOT MEAN ANOMALIY
proj = ccrs.PlateCarree()

fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw = dict(projection=proj), figsize=(55, 35))
depths = (75, 500, 1000, 2500)
cmap = cm.winter

for j, ax in enumerate(axes.flat):        
    scalar = ax.contourf(lons, lats, dev[j], vmin = np.min(dev), mvmax = np.max(dev), cmap=cmap, zorder = 1)
    lines = ax.contour(lons, lats, -depth, levels = depths, colors='black', linewidths = .75)
    ax.clabel(lines, inline=1, fontsize=50, colors = 'black')
    lines2 = ax.contour(lons, lats, -depth, levels = [200], colors='black', linewidths = 3)
    # ax.clabel(lines2, inline=2, fontsize=30, colors = 'black')
    ax.set_title(str(month_list[j]), fontsize=60)
               
for ax in axes.flat:    
    ax.tick_params(axis = "both", labelsize = 15)
    ax.axes.axis("tight")

    #add coastlines
    res = '10m'
    ax.coastlines(resolution = res, linewidths = 0.5)
    
    #grid and axes
    if ax == axes.flat[0] or ax == axes.flat[4]:
        gl = ax.gridlines(linewidth=0.5, color='gray', alpha=0.5,
                          linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabels_bottom = False
        gl.xlabel_style = {'fontsize': 60}
        gl.ylabel_style = {'fontsize': 60}
        
    if ax == axes.flat[1] or ax == axes.flat[2] or ax == axes.flat[3] or ax == axes.flat[5] or ax == axes.flat[6] or ax == axes.flat[7]:
        gl = ax.gridlines(linewidth=0.5, color='gray', alpha=0.5,
                          linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabels_bottom = False
        
    if ax == axes.flat[8]:
        gl = ax.gridlines(linewidth=0.5, color='gray', alpha=0.5,
                          linestyle='--',draw_labels=True, zorder = 2)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'fontsize': 60}
        gl.ylabel_style = {'fontsize': 60}
    
              
    if ax == axes.flat[9] or ax == axes.flat[10] or ax == axes.flat[11]:
        gl = ax.gridlines(linewidth=0.5, color='gray', alpha=0.5,
                          linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabel_style = {'fontsize': 60}
        gl.ylabel_style = {'fontsize': 60}
    
    #add continent shading
    ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray', zorder = 1)
        
# #color bar
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8], anchor='C')
cbar = plt.colorbar(scalar, format='%.1f', spacing='proportional', cax=cax,
                    shrink=0.9, orientation = 'vertical', location = "right", pad = 0.08)
cbar.set_label( label = "$\sigma_{Chl-a}$  [mg/m$^3$]", fontsize = 90, y = 0.5, labelpad = 30)
cbar.ax.tick_params(which='minor', size=35, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=55, width=1, color='k', direction='in', labelsize = 60)
# Adjust the position and size of the colorbar axis
cax.set_position([1.02, 0.1, 0.02, 0.8])


# set the title of the figure
fig.suptitle("Monthly Chlorophyll-a standard deviation (01/01/2003 - 31/12/2020)", fontsize = 100, y = 0.98)
fig.tight_layout(pad = 5.0)
plt.show()
#fig.savefig(plot_path + 'chl_anomalies.png', bbox_inches='tight')

#%%HIST MEAN CHL
ds_month = np.nanmean(clim_month, axis=(1,2))
std_month = np.nanstd(np.array(clim_month), axis = (1,2))
#std_month = np.nanstd(np.array(ds_month))


chl_month = chl.groupby('time.month')

months = []
std = []
for i in range(1,13):
    a = chl.where(chl['time.month'] == i, drop = True).mean(dim=('lat','lon'), skipna=True)
    b = chl.where(chl['time.month'] == i, drop = True).std(dim=('lat','lon'), skipna=True)
    months.append(a)
    std.append(b)
months = xr.DataArray(months)
std = xr.DataArray(std)

#concat by month
flat = xr.concat(months.values, dim='time')
flat_std  = xr.concat(std.values, dim='time')

#%%
fig, ax = plt.subplots(figsize=(14, 10))
# for i in range (0, 12):
ax.vlines(range(0, 12), 0, ds_month, linewidths=71, alpha = .8, color = 'seagreen')
# ax.errorbar(range(0, 12), ds_month, xerr=0, yerr=std_month,
#             marker='o', markersize=8, linewidth = 2, color = 'k',
#             linestyle='none')
ax.set_xticks(range(0,12))
ax.yaxis.set_tick_params(labelsize = 30)
# Set the y-axis label
ax.set_ylabel('Chl-a concentration [mg/m$^3$]', fontsize=30)
ax.set_xticklabels(month_list, fontsize = 30)
ax.grid(linewidth=.75, color='gray', linestyle='--')
ax.set_facecolor('whitesmoke')
fig.suptitle("Mean of monthly spatially averaged Chl-a", fontsize = 40, y = 1.0)
fig.tight_layout()
plt.show()
fig.savefig(plot_path + 'chl_hist.png', bbox_inches='tight')
