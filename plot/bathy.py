#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:26:35 2023

@author: serena
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from scipy import interpolate
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import xarray as xr
import matplotlib.axes
import matplotlib.patches as patches
import cmocean
from sympy import Point, Line
  

############### PLOT FUNTIONS ################################################
def set_plot():
    fig, ax = plt.subplots(subplot_kw = dict(projection = ccrs.PlateCarree()), figsize = [12,12])
    
    font_size = 25

    ax.coastlines(resolution="10m", linewidths=0.5)
    ax.add_feature(cfeature.LAND.with_scale("10m"),
               edgecolor='lightgray',facecolor='lightgray',
               zorder=0)

    ax.tick_params(axis = "both", labelsize = 15)

    gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5,
                  linestyle='--',draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False

    gl.xlabel_style = {'fontsize': font_size}
    gl.ylabel_style = {'fontsize': font_size}

    return (fig, ax)

def set_cbar(c, title):
    cbar = plt.colorbar(c, pad = 0.05, orientation = "horizontal")
    cbar.ax.tick_params(labelsize = 15)
    cbar.set_label(label = title, size = 20)
    return cbar

def title_set(ax, title):
    ax.set_title(title, fontsize = 30, y=1.05)

#%%############# PLOT BATHY #####################################################    
path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/"
file_nc = "bathymetry.nc"

ds = xr.open_dataset(path + "/" + file_nc)

# #read lat, lon e temp
#%%ARRAY

lat_range = slice(37, 43)   # for example, select between 40 and 50 degrees latitude
lon_range = slice(-74, -67)

ds = ds.sel(lat=lat_range, lon=lon_range)
#%%
lon, lat = ds.lon, ds.lat

el = ds.variables['elevation']
depth = np.where(el>0, float('nan'), ds.variables['elevation'][:,:])
levels = (40, 3000)
depths = (50, 75, 100, 200, 500, 1000, 2500, 3000)
#%%

fig, ax = set_plot()

scalar1 = ax.contourf(lon, lat, -depth, cmap = 'cmo.deep', alpha = .9, transform = ccrs.PlateCarree()) #20 levels of colors
lines = ax. contour(lon, lat, -depth, levels = depths, colors='black', linewidths = .5)
ax.clabel(lines, inline=2, fontsize=20, colors = 'black')
ax.tick_params(which='major', labelsize = 25)


las,lan=41,38 #alto-basso
low,loe=-68,-72.6  #dx-sx

y = [ lan, las, las, lan, lan ] #lats
x = [ loe, loe, low, low, loe ]

ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'red', linewidth = 2.5)

cbar_ax = fig.add_axes([0.12, 0.05, 0.75, 0.035]) #left, bottom, width, height
cbar = fig.colorbar(scalar1, orientation = 'horizontal', location = "bottom", cax = cbar_ax, pad = 0.05)
cbar.set_label(label = "Depth [m]", fontsize = 20)
cbar.ax.tick_params(which='major', labelsize = 20)
title_set(ax, "Bathymetry")


# ax.text(x = -81.05, y = 25.5, s = "FLORIDA", transform = ccrs.PlateCarree(), size = 20)
# ax.text(x = -82, y = 22.8, s = "CUBA", transform = ccrs.PlateCarree(), size = 25)
# ax.text(x = -81.45, y = 24.8, s = "FLORIDA KEYS", transform = ccrs.PlateCarree(), size = 20)

fig.savefig(plot_path + 'bathymetry.png', bbox_inches='tight', dpi = 400)

#%%MASK
depth_mask = el.where((el <= -50) & (el >= -3000))

p1, p2, p3 = Point(73, 40.5), Point(70, 41.5), Point(70.6, 41)
  
l1 = Line(p1, p2)
  
# using perpendicular_line() method
l2 = l1.perpendicular_line(p3)
print(l2.parameter_value)

#%%
fig, ax = set_plot()

scalar = ax.contourf(lon, lat, -depth_mask, cmap = 'cmo.deep', levels = scalar1.levels, alpha = .9, transform = ccrs.PlateCarree()) #20 levels of colors
lines = ax. contour(lon, lat, -depth, levels = depths, colors='black', linewidths = .5)
ax.clabel(lines, inline=2.5, fontsize=20, colors = 'black')

y1 = [ 40.5, 41.5] #lats
x1 = [ -73, -70]

ax.plot(x1, y1, transform = ccrs.PlateCarree(), color = 'blue', linewidth = 2.5)

#from parametr_value of l3
x_perp = [-353/5, -348/5]
y_perp = [41, 38]

ax.plot(x_perp, y_perp, transform = ccrs.PlateCarree(), color = 'blue', linewidth = 2.5)


y = [ 38.5, 39.8, 41, 41, 38.5, 38.5, 38.5] #lats
x = [ -70.6, -72.6, -72.6, -68, -68, -68, -70.6]

ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'red', linewidth = 2.5)

p1 = [41., -70.6]
p2 = [38.85 ,-69.9]
ax.plot(p2[1], p2[0], marker = 'o', markersize = 20, color = 'yellow')
ax.plot(x_perp[0], y_perp[0], marker = 'o', markersize = 20, color = 'yellow')

cbar_ax = fig.add_axes([0.12, 0.05, 0.75, 0.035]) #left, bottom, width, height
cbar = fig.colorbar(scalar, orientation = 'horizontal', location = "bottom", cax = cbar_ax, pad = 0.05)
cbar.ax.tick_params(labelsize=20) 
cbar.set_label(label = "Depth [m]", fontsize = 20)
title_set(ax, "Bathymetry masked")

fig.savefig(plot_path + 'bathymetry_mask_lines.png', dpi = 500, bbox_inches='tight')

#%%DISTANCE

