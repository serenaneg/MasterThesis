#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:07:24 2023

@author: serena
"""

#%%########################## ANNUAL MEAN ######################################
import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm, ListedColormap
from matplotlib import ticker, cm
import matplotlib.colors as colors

path = "/home/serena/Scrivania/Magistrale/thesis/data/"
data_path = path + "MODIS_INTERPOLATED_DATA/"
plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/chlor/"

data = xr.open_dataset(data_path + "MODIS_chl.nc").load()

#%%ARRAY
# #choose ARRAY coordinate
lat_range = slice(41, 38)
lon_range = slice(-72.6, -68)

data = data.sel(lon=lon_range)
data = data.sel(lat=lat_range)
#%%
#lat, lon, bathy
lons, lats = data['lon'], data['lat']
bathy = xr.open_dataset(data_path + "bathymetry_interpolated.nc")

depth = bathy['elevation']
depth= depth.sel(lat = lats, lon = lons)

#%%ANNUAL MEAN
ds_year = data.groupby('time.year').mean(skipna=True)
year_list = ds_year['year'].values.tolist()
chlor = ds_year['chlor_a']

#%%cmap
top = cm.get_cmap('YlGnBu_r', 128) # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)# combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')

#%%#################PLOT#######################################################
proj = ccrs.PlateCarree()
fig, axes = plt.subplots(nrows=3, ncols=6, subplot_kw = dict(projection=proj), figsize=(90, 40),
                         tight_layout=True)

#Set x-y axis label and title the axes.set_title() function
depths = (50, 75, 100, 200, 500, 1000, 2500)
levels = np.linspace(np.nanmin(chlor), 4.5, 100)
cmap = orange_blue

# Set the y-axis view limits for each Axes with the axes.set_ylim() function
for i, ax in enumerate(axes.flat):
    chl = ax.contourf(data['lon'], data['lat'], chlor[i], levels = levels, norm = 'log',
                       cmap=cmap, zorder = 1, extend = 'max')
    lines = ax.contour(lons, lats, -depth, levels = depths, colors='black', linewidths = .75)
    ax.clabel(lines, inline=1, fontsize=40, colors = 'black')
    ax.set_title(str(year_list[i]), fontsize=50)
    
    norm= LogNorm(vmin=0.1, vmax=4.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap = chl.cmap)
    sm.set_array([])

for ax in axes.flat:  
    ax.set_xticks(lons, crs=ccrs.PlateCarree())
    ax.set_xticks(lats, crs=ccrs.PlateCarree())
    ax.tick_params(axis ='both', labelsize=50)
    ax.axes.axis("tight")

    #add coastlines
    res = '10m'
    ax.coastlines(resolution = res, linewidths = 0.5)
    
    #grid and axes
    if ax == axes.flat[0] or ax == axes.flat[6]:
        gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabels_bottom = False
        gl.xlabel_style = {'fontsize': 50}
        gl.ylabel_style = {'fontsize': 50}
        
    if ax == axes.flat[1] or ax == axes.flat[2] or ax == axes.flat[3] or ax == axes.flat[4] or ax == axes.flat[5] or ax == axes.flat[7] or ax == axes.flat[8] or ax == axes.flat[9] or ax == axes.flat[10] or ax == axes.flat[11]:
        gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
    
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabels_bottom = False
        
    if ax == axes.flat[12]:
        gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'fontsize': 50}
        gl.ylabel_style = {'fontsize': 50}
    
              
    if ax == axes.flat[13] or ax == axes.flat[14] or ax == axes.flat[15] or ax == axes.flat[16] or ax == axes.flat[17]:
        gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabel_style = {'fontsize': 50}
        gl.ylabel_style = {'fontsize': 50}
    
    #add continent shading
    ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray', zorder = 2)
    
# #color bar
skip = (slice(None, None, 10))
# Create a new axis for the colorbar
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8], anchor='C')
cbar = plt.colorbar(sm, format='%.1f', spacing='proportional', cax=cax,
                    shrink=0.9, orientation = 'vertical', location = "right", pad = 0.1)
cbar.set_label(label = "Chlorophyll [mg/m$^3$]", fontsize = 60, y = 0.5)
cbar.ax.tick_params(which='minor', size=35, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=55, width=1, color='k', direction='in', labelsize = 50)
# Adjust the position and size of the colorbar axis
cax.set_position([1.02, 0.1, 0.02, 0.8])
# set the title of the figure
fig.suptitle("Annual Mean Chlorophyll-a Concentration (01-01-2003 / 12-31-2020)", fontsize = 80, y = 0.98)
fig.tight_layout(pad=5.0)
plt.show()
# fig.savefig(plot_path +'chl_annual_extend.png', bbox_inches='tight')
fig.savefig(plot_path + 'chl_annual_log.png', bbox_inches='tight')

#%%
chlor = data['chlor_a']
count_year = chlor.groupby('time.year').count(dim= 'time').to_numpy()

perc_year = (count_year/365)*100
#%%
proj = ccrs.PlateCarree()
fig, axes = plt.subplots(nrows=3, ncols=6, subplot_kw = dict(projection=proj), figsize=(80, 35))

#Set x-y axis label and title the axes.set_title() function
depths = (50, 75, 100, 200, 500, 1000, 2500)

# Set the y-axis view limits for each Axes with the axes.set_ylim() function
for i, ax in enumerate(axes.flat):
    im = ax.pcolormesh(lons, lats, perc_year[i], cmap="jet", vmin = 10)
    lines = ax.contour(lons, lats, -depth, levels = depths, colors='black', linewidths = .75)
    ax.clabel(lines, inline=2, fontsize=30, colors = 'black')
    ax.set_title(str(year_list[i]), fontsize=40)

for ax in axes.flat:  
    ax.set_xticks(lons, crs=ccrs.PlateCarree())
    ax.set_xticks(lats, crs=ccrs.PlateCarree())
    ax.tick_params(axis ='both', labelsize=35)
    ax.axes.axis("tight")

    #add coastlines
    res = '10m'
    ax.coastlines(resolution = res, linewidths = 0.5)
    
    #grid and axes
    if ax == axes.flat[0] or ax == axes.flat[6]:
        gl = ax.gridlines(linewidth=0, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabels_bottom = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
        
    if ax == axes.flat[1] or ax == axes.flat[2] or ax == axes.flat[3] or ax == axes.flat[4] or ax == axes.flat[5] or ax == axes.flat[7] or ax == axes.flat[8] or ax == axes.flat[9] or ax == axes.flat[10] or ax == axes.flat[11]:
        gl = ax.gridlines(linewidth=0, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
    
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabels_bottom = False
        
    if ax == axes.flat[12]:
        gl = ax.gridlines(linewidth=0, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
    
              
    if ax == axes.flat[13] or ax == axes.flat[14] or ax == axes.flat[15] or ax == axes.flat[16] or ax == axes.flat[17]:
        gl = ax.gridlines(linewidth=0, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
    
    #add continent shading
    ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray', zorder = 2)
    
# #color bar
cbar = fig.colorbar(im, format='%.0f', spacing='proportional',
                    orientation = 'horizontal', location = "bottom", pad = -0.15)
cbar.set_label(label = "% of not cloudy days per pixel per year", fontsize = 60, y = 0.5)
cbar.ax.tick_params(which='minor', size=25, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=55, width=1, color='k', direction='in', labelsize = 55)

# adjust bottom margin and position colorbar at the bottom
fig.subplots_adjust(bottom=0.2)
cbar.ax.set_position([0.35, 0.0, 0.35, 0.08])

# set the title of the figure
fig.suptitle("Annual Chlorophyll-a concentration available data (01-01-2003 / 12-31-2020)", fontsize = 70, y = 0.95)
fig.tight_layout()
plt.show()
fig.savefig(plot_path + 'chl_annual_count_year.png', bbox_inches='tight')






