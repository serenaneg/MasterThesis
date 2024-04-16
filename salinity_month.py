#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:45:45 2023

@author: serena
"""

import xarray as xr
import numpy as np
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
import pandas as pd
import datetime
import cmocean


#%% LOAD MONTHLY CMEMS DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/CMEMS/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/salinity/"

#monthly data
ds = xr.open_dataset(path +"cmems_grep.nc")

#%%ARRAY
lat_range = slice(38, 41)
lon_range = slice(-72.6, -68)

ds = ds.sel(latitude=lat_range, longitude=lon_range)
#%%

#lat, lon, bathy
lons, lats = ds.longitude, ds.latitude
bathy = xr.open_dataset(path + "GLO-MFC_001_030_mask_bathy.nc")

depth = bathy['deptho']
depth= depth.sel(latitude = lats, longitude = lons)

#%% SALINITY

ds_month = ds.groupby('time.month').mean()
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

sal = ds_month['so_mean']

#%%  #%% SALINITY AT THE SURFACE
sal_surf = sal.isel(depth=0)

proj = ccrs.PlateCarree()
fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw = dict(projection=proj), figsize=(55, 35))
depths = (75, 1000, 2500)
#Set x-y axis label and title the axes.set_title() function
levels = np.linspace(32, 36, 20)
cmap = 'cmo.haline'
    
    
# Set the y-axis view limits for each Axes with the axes.set_ylim() function
for i, ax in enumerate(axes.flat):
    scalar = ax.contourf(lons, lats, sal_surf[i], levels = levels, extend = 'both', cmap=cmap, zorder = -1)
    lines = ax.contour(lons, lats, depth, levels = depths, colors='black', linewidths = .75)
    ax.clabel(lines, inline = 2, fontsize=50, colors = 'black')
    ax.set_title(str(month_list[i]), fontsize=60)
    
    lines2 = ax.contour(lons, lats, depth, levels = [200], colors='black', linewidths = 3)
    #ax.clabel(lines2, inline = 2, fontsize=30, colors = 'black')
    
    norm= Normalize(vmin=32, vmax=36)
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])
    
for ax in axes.flat:    
    ax.set_xticks(lons, crs=ccrs.PlateCarree())
    ax.set_xticks(lats, crs=ccrs.PlateCarree())
    ax.tick_params(axis ='both', labelsize=60)
    ax.axes.axis("tight")

    #add coastlines
    res = '10m'
    ax.coastlines(resolution = res, linewidths = 0.5)
    gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)

    
    #grid and axes
    if ax == axes.flat[0] or ax == axes.flat[4]:
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabels_bottom = False
        gl.xlabel_style = {'fontsize': 60}
        gl.ylabel_style = {'fontsize': 60}
        
    if ax == axes.flat[1] or ax == axes.flat[2] or ax == axes.flat[3] or ax == axes.flat[5] or ax == axes.flat[6] or ax == axes.flat[7]:
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabels_bottom = False
        
    if ax == axes.flat[8]:
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'fontsize': 60}
        gl.ylabel_style = {'fontsize': 60}
    
              
    if ax == axes.flat[9] or ax == axes.flat[10] or ax == axes.flat[11]:
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabel_style = {'fontsize': 60}
        gl.ylabel_style = {'fontsize': 60}
    
    #add continent shading
    ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray')
    
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8], anchor='C')
cbar = plt.colorbar(scalar, format='%.1f', spacing='proportional', cax=cax, 
                    shrink=0.9, orientation = 'vertical', location = "right", pad = 0.1)
cbar.set_label( label = "Salinity [PSU]", fontsize = 90, y = 0.5, labelpad = 40)
cbar.ax.tick_params(which='minor', size=55, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=55, width=1, color='k', direction='in', labelsize = 60)
# Adjust the position and size of the colorbar axis
cax.set_position([1.02, 0.1, 0.02, 0.8])


# set the title of the figure
fig.suptitle("Monthly Sea Surface salinity (01-01-2003 / 31-12-2020)", fontsize = 100, y = 0.98)
fig.tight_layout(pad = 5.0)
plt.show()
fig.savefig(plot_path + 'sal_surf_month.png', bbox_inches='tight')

#%%  #%% SALINITY AT DIFFERENT DEPTHS
 #surf, 75, 100, 200, 500, 1000
isobathy = np.array([20, 22, 26, 31, 35], dtype = 'int64')

salinity = xr.Dataset()
z = np.zeros(5)

proj = ccrs.PlateCarree()

for j, n in enumerate(isobathy):
    salinity = sal.isel(depth=isobathy[j])

    fig2, axes = plt.subplots(nrows=3, ncols=4, subplot_kw = dict(projection=proj), figsize=(60, 35))

    #Set x-y axis label and title the axes.set_title() function
    cmap = cm.GnBu
            
            
        # Set the y-axis view limits for each Axes with the axes.set_ylim() function
    for i, ax in enumerate(axes.flat):
        levels = np.linspace(sal[i].min(), sal[i].max(), 10)
        scalar = ax.contourf(lons, lats, salinity[i], levels = levels, cmap=cmap, zorder = -1)
        lines = ax.contour(lons, lats, depth, levels = depths, colors='black', linewidths = .75)
        ax.clabel(lines, inline=1, fontsize=50, colors = 'black')
        ax.set_title(str(month_list[i]), fontsize=60)

        # Set the x-y axis labels for each Axes with the axes.set_xlabel() function
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.label_outer()
        
    for ax in axes.flat:    
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        ax.axes.axis("tight")
        ax.set_xlabel("")

        #add coastlines
        res = '10m'
        ax.coastlines(resolution = res, linewidths = 0.5)
        
        #grid and axes
        gl = ax.gridlines(linewidth=0.5, color='gray', alpha=0.5, linestyle='--',draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'fontsize': 20}
        gl.ylabel_style = {'fontsize': 20}
        
        #add continent shading
        ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray')
        
    # #color bar
    cbar = fig2.colorbar(scalar, orientation = 'horizontal', ticks = levels,  location = "bottom", pad = -0.15)
    cbar.set_label( label = "Salinity [PSU]", fontsize = 40, y = 0.5)
    cbar.ax.tick_params(labelsize = 30)

    # adjust bottom margin and position colorbar at the bottom
    fig2.subplots_adjust(bottom=0.2)
    cbar.ax.set_position([0.2, 0.07, 0.6, 0.07])
    
    z = salinity.depth.values

    # set the title of the figure
    fig2.suptitle("Monthly Sea Water Salinity at z = " + "{:3f}".format(z) + " m", fontsize = 60, y = 0.93)
    fig2.tight_layout()
    plt.show()
    fig2.savefig(plot_path + 'sal_month_' + str(n-1)+'.png', bbox_inches='tight')
    plt.clf()
    

