#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:04:28 2024

@author: serena
"""

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

path = "/home/serena/Scrivania/Magistrale/thesis/data/NOAA/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/chlor/"
figures = "/home/serena/Scrivania/Magistrale/thesis/figures_paper/"

data = xr.open_dataset(path + "2019_V2019142_A1_WW00_chlora_2km.nc").load()

top = cm.get_cmap('YlGnBu_r', 128) # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)# combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')
#%%
lat_ext = slice(38, 43.5)
lon_ext = slice(-75, -67.5)

bathy = xr.open_dataset(bathy_path + "bathymetry.nc")

depth_ext = bathy['elevation']
depth_ext= depth_ext.sel(lat = lat_ext, lon = lon_ext)
lon_ext = depth_ext['lon']
lat_ext = depth_ext['lat']
#%%
lat_range = slice(41.5, 38)   # for example, select between 40 and 50 degrees latitude
lon_range = slice(-73, -67.5)

data = data.sel(lon=lon_range)
data = data.sel(lat=lat_range)
#%%
lons, lats = data['lon'], data['lat']

bathy = xr.open_dataset(bathy_path + "bathymetry.nc")

bathy = bathy.interp(lon = lons, lat = lats, method = "nearest")
depth = bathy['elevation']
depth= depth.sel(lat = lats, lon = lons)

lon_stat = np.array([-70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83])
lat_stat = np.array([40.475, 40.4, 40.3361, 40.2722, 40.2083, 40.1444, 40.0806, 40.0167, 
                    39.9528, 39.8889, 39.825, 39.75, 39.6861, 39.6222])
                    
lables =['A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18']
#%%
chlor = data.chlor_a
chlor = chlor[0, 0, :, :]
#%%
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw = dict(projection=proj), figsize=(30, 15), tight_layout=True)

#Set x-y axis label and title the axes.set_title() function #(25, 15)
depths = (100, 500, 3000)
label_depth = ( 75, 1000, 2000)
levels = np.linspace(np.nanmin(chlor), 5, 50)
cmap = orange_blue

chl = ax.contourf(chlor.lon, chlor.lat, chlor, levels = levels, norm = 'log', cmap=cmap, extend = 'max', zorder = 0)
ax.contour(lon_ext, lat_ext, -depth_ext, levels = depths, colors='black', linewidths = 1)
lines = ax.contour(lon_ext, lat_ext, -depth_ext, levels = label_depth, colors='black', linewidths = 1, zorder = 1)
lines2 = ax.contour(lon_ext, lat_ext, -depth_ext, levels = [200], colors='black', linewidths = 4, zorder = 1)
#ax.clabel(lines, inline = True, fontsize=45, colors = 'black')
norm= LogNorm(vmin=0.1, vmax=4.5)
sm = plt.cm.ScalarMappable(norm=norm, cmap = chl.cmap)
sm.set_array([])

y = [ 38.5, 39.8, 41, 41, 38.5, 38.5, 38.5] #lats
x = [ -70.6, -72.6, -72.6, -68, -68, -68, -70.6]
ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'b', linewidth = 5, zorder = 3)

ax.scatter(lon_stat, lat_stat, s = 150, marker = 'D',  color = 'b', zorder = 3)
ax.text(lon_stat[0]-0.35, lat_stat[0], s = 'A5', color = 'b', fontsize = 45)
ax.text(lon_stat[13]-0.48, lat_stat[13]-0.14, s = 'A18', color = 'b',  fontsize = 45)
ax.set_xticks(lons, crs=ccrs.PlateCarree())
ax.set_xticks(lats, crs=ccrs.PlateCarree())
ax.tick_params(axis ='both', labelsize=40)
ax.axes.axis("tight")

ax.scatter(-70.6, 41.6, s = 300, marker = 'o', color = 'purple', zorder = 3)
ax.text(-70.5, 41.6, s = 'Cape Cod', c = 'purple', fontsize = 45)
ax.scatter(-74.6, 39.5, s = 300, marker = 'o', color = 'purple', zorder = 3)
ax.text(-74.6, 39.6, s = 'New Jersey', c = 'purple', fontsize = 45)

#add coastlines
res = '10m'
ax.coastlines(resolution = res, linewidths = 0.5)
gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'fontsize': 40}
gl.ylabel_style = {'fontsize': 40}
    
ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray', zorder = 2)
    
cbar = plt.colorbar(chl, format='%.2f', spacing='proportional', orientation = 'vertical', location = "right", pad = 0.05)
cbar.set_label(label = "[mg/m$^3$]", fontsize = 40, y = 0.5, labelpad = 30)
cbar.ax.tick_params(which='minor', size=15, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=20, width=1, color='k', direction='in', labelsize = 35)
cbar.ax.locator_params(axis='y', nbins=8)
fig.suptitle(r'Shelfbreak Chlorophyll-$\alpha$ enhancement, April 22th 2019', fontsize = 40, y = 1)
fig.tight_layout()
plt.show()
fig.savefig(figures + 'chl_enhanced.png', dpi = 300, bbox_inches='tight')
