
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:45:45 2023

@author: serena
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm, ListedColormap
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
import cmocean


#%% LOAD MONTHLY CMEMS DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/MODIS_INTERPOLATED_DATA/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/chlor/"

#monthly data
ds = xr.open_dataset(path + "MODIS_chl.nc").load()

#%%ARRAY
# #choose ARRAY coordinate
lat_range = slice(41, 38)
lon_range = slice(-72.6, -68)

ds = ds.sel(lon=lon_range)
ds = ds.sel(lat=lat_range)

#%%
#lat, lon, bathy
lons, lats = ds['lon'], ds['lat']
bathy = xr.open_dataset(path + "bathymetry_interpolated.nc")

depth = bathy['elevation']
depth= depth.sel(lat = lats, lon = lons)

#%% CHLOROPHYLL
ds_month = ds.groupby('time.month').mean(skipna=True).load()
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

chl = ds_month['chlor_a']

#%%cmap
top = cm.get_cmap('YlGnBu_r', 128) # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)# combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')

#%%  #%% CHLOROFYLL AT THE SURFACE
proj = ccrs.PlateCarree()
fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw = dict(projection=proj), figsize=(55, 35))

#Set x-y axis label and title the axes.set_title() function
levels = np.linspace(np.nanmin(chl), 4.5, 100)
depths = (50, 100, 200, 500, 1000, 2500)
cmap = orange_blue
    
    
# Set the y-axis view limits for each Axes with the axes.set_ylim() function
for i, ax in enumerate(axes.flat):
    scalar = ax.contourf(lons, lats, chl[i],  norm = 'log', levels = levels,
                         extend = 'max', cmap=cmap, zorder = 1)
    lines = ax.contour(lons, lats, -depth, levels = depths, colors='black', linewidths = .75)
    ax.clabel(lines, inline=2, fontsize=35, colors = 'black')
    ax.set_title(str(month_list[i]), fontsize=45)
    
    norm= LogNorm(chl.min(), vmax=4.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])

for ax in axes.flat:  
    ax.set_xticks(lons, crs=ccrs.PlateCarree())
    ax.set_xticks(lats, crs=ccrs.PlateCarree())
    ax.tick_params(axis ='both', labelsize=35)
    ax.axes.axis("tight")

#add coastlines
    res = '10m'
    ax.coastlines(resolution = res, linewidths = 0.5)

    #grid and axes
    if ax == axes.flat[0] or ax == axes.flat[4]:
        gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabels_bottom = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
        
    if ax == axes.flat[1] or ax == axes.flat[2] or ax == axes.flat[3] or ax == axes.flat[5] or ax == axes.flat[6] or ax == axes.flat[7]:
        gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
    
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabels_bottom = False
        
    if ax == axes.flat[8]:
        gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
    
              
    if ax == axes.flat[9] or ax == axes.flat[10] or ax == axes.flat[11]:
        gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
    
    #add continent shading
    ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray', zorder = 1)
    
# #color bar
skip = (slice(None, None, 10))
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8], anchor='C')
cbar = plt.colorbar(sm, format='%.1f', spacing='proportional', cax=cax,
                    shrink=0.9, orientation = 'vertical', location = "right", pad = 0.1)
cbar.set_label(label = "Chlorophyll [mg m^-3]", fontsize = 60, y = 0.5, labelpad = 30)
cbar.ax.tick_params(which='minor', size=35, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=55, width=1, color='k', direction='in', labelsize = 50)
# Adjust the position and size of the colorbar axis
cax.set_position([1.02, 0.1, 0.02, 0.8])

# set the title of the figure
fig.suptitle("Monthly Chlorophyll-a concentration (01-01-2003 / 31-12-2020)", fontsize = 60, y = 0.97)
fig.tight_layout(pad=5)
plt.show()
# fig.savefig(plot_path + 'chl_M_ARRAY_LIN.png', bbox_inches='tight')
fig.savefig(plot_path + 'chl_M_LOG.png', bbox_inches='tight')

#%%COUNT VALID DATA FOR MEAN
chlor = ds['chlor_a']
a = chlor.to_numpy()
#DataArray.count() counts the non NaN values along one dimension
count = (chlor.count(axis=0).to_numpy() / chlor.shape[0]) * 100

count_nan = np.count_nonzero(~np.isnan(a), axis = 0)
count_nan = count_nan / chlor.shape[0] * 100
#%% Heat map
depths = (50, 100, 200, 500, 1000, 2500, 3000)
proj = ccrs.PlateCarree()

fig, ax = plt.subplots(subplot_kw = dict(projection=proj), figsize=(15, 10))
im = ax.pcolormesh(lons, lats, count_nan, cmap="jet", vmin = 10)
# im = ax.contourf(lons, lats, a , cmap="jet")
lines = ax.contour(lons, lats, -depth, levels = depths, colors='black', linewidths = .75)
ax.clabel(lines, inline=2, fontsize=30, colors = 'black')


# Show all ticks and label them with the respective list entries
ax.axes.get_xaxis().set_ticklabels([])
ax.axes.get_yaxis().set_ticklabels([])
ax.axes.axis("tight")
ax.set_xlabel("")

res = '10m'
ax.coastlines(resolution = res, linewidths = 0.5)

gl = ax.gridlines(linewidth=0.5, color='gray', alpha=0.5,
                  linestyle='--', draw_labels=False)
gl.xlabels_top = False
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabels_bottom = True
gl.xlabel_style = {'fontsize': 25}
gl.ylabel_style = {'fontsize': 25}

ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray', zorder = 1)

cbar = plt.colorbar(im, format='%.0f', spacing='proportional', ax=ax, shrink=0.9, pad = 0.03,
                    orientation = 'vertical', location = "right")
cbar.set_label( label = "% of not cloudy days per pixel", fontsize = 25, y = 0.5, x =1)
cbar.ax.tick_params(which='minor', size=10, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=15, width=1, color='k', direction='in', labelsize = 25)
# fig.subplots_adjust(bottom=0.25)
# adjust bottom margin and position colorbar at the bottom
fig.subplots_adjust(bottom=0.2)
cbar.ax.set_position([0.2, 0.07, 0.6, 0.07])

ax.set_title("Number of useful days for calculations", {'fontsize': 30})
fig.tight_layout()
plt.show()
fig.savefig(plot_path + 'count_days.png',  bbox_inches='tight')

#%% x%%
plt.plot(chlor.time.values, range(0, 6351))

