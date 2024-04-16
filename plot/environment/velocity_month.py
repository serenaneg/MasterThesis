#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:30:21 2023

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
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
import pandas as pd
import datetime
import cmocean

#%% LOAD DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/CMEMS/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/currents/"

#monthly data
ds = xr.open_dataset(path +"cmems_grep.nc")

#%%ARRAY
# lat_range = slice(37, 42)  
# lon_range = slice(-73, -67)

lat_range = slice(38, 41)
lon_range = slice(-72.6, -68)


ds = ds.sel(latitude=lat_range, longitude=lon_range)

#%%

#lat, lon, bathy
lons, lats = ds.longitude, ds.latitude
bathy = xr.open_dataset(path + "GLO-MFC_001_030_mask_bathy.nc")

depth = bathy['deptho']
depth= depth.sel(latitude = lats, longitude = lons)

#%% VELOCITY
ds_month = ds.groupby('time.month').mean("time")
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

v = ds_month['vo_mean']
u = ds_month['uo_mean']

# data_mask = xr.open_dataset(path + 'GLO-MFC_001_030_mask_bathy.nc')
speed = np.sqrt(np.square(u) + np.square(v))

#%%PLOT PARAMETERS
colors = [tuple(np.array((235,245,255))/255), # light blue
          tuple(np.array((157,218,247))/255), # blue
          tuple(np.array((72,142,202))/255),  # dark blue
          tuple(np.array((73,181,70))/255),   # green
          tuple(np.array((250,232,92))/255),  # yellow
          tuple(np.array((245,106,41))/255),  # orange
          tuple(np.array((211,31,40))/255),   # red
          tuple(np.array((146,21,25))/255)]   # dark red
mycmap = LinearSegmentedColormap.from_list('WhiteBlueGreenYellowRed', colors, N=100)


#%% VELOCITY AT THE SURFACE CURLY VECTORS
vel_surf = speed.isel(depth=0)
u_surf = u.isel(depth=0)
v_surf = v.isel(depth=0)

#%%
proj = ccrs.PlateCarree()
depths = (50, 100, 200,1000,2500)
levels = np.linspace(vel_surf.min(), vel_surf.max(), 15)
cmap = mycmap

#create 2d grid lat-lon
lon2d, lat2d = np.meshgrid(lons, lats)

#%%SURF CURRENT
fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw = dict(projection=proj), figsize=(55, 35))

# las,lan=41,38 #alto-basso
# low,loe=-68,-72.6  #dx-sx

# y = [ lan, las, las, lan, lan ] #lats
# x = [ loe, loe, low, low, loe ]
    
for i, ax in enumerate(axes.flat):
    # Define the starting point 
    speed_flat = v_surf[i].values.ravel()
    scale = 0.2/np.max(v_surf[i]) 
    
    # if ax == axes.flat[0]:
    #     ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'red', linewidth = 2.5)
    
    skip = (slice(None, None, 2), slice(None, None, 2))

   # plot a quiver with axes.quiver() from that point. Make the quiver small enough so that only the arrow is visible
    scalar = ax.contourf(lon2d, lat2d, vel_surf[i], levels = levels,
                        transform = proj, cmap = cmap)

    q = ax.quiver(lon2d[skip], lat2d[skip], u_surf[i][skip]/vel_surf[i][skip], v_surf[i][skip]/vel_surf[i][skip],
                transform=ccrs.PlateCarree())
    # q = ax.quiver(lon2d[skip], lat2d[skip], u_surf[i][skip], v_surf[i][skip],
    #             transform=ccrs.PlateCarree())
    
    norm= Normalize(vmin=scalar.cvalues.min(), vmax=scalar.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])
    
    lines = ax.contour(lon2d, lat2d, depth, levels = depths, colors='black', linewidths = .55, linestyle = '--')
    ax.clabel(lines, inline=1, fontsize=30, colors = 'black')
    ax.set_title(str(month_list[i]), fontsize=50)
    
    norm= Normalize(vmin=scalar.cvalues.min(), vmax=scalar.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])
 
    # Set the x-y axis labels for each Axes with the axes.set_xlabel() function
    ax.set_xticks(lons, crs=ccrs.PlateCarree())
    ax.set_xticks(lats, crs=ccrs.PlateCarree())
    ax.tick_params(axis ='both', labelsize=35)
    ax.axes.axis("tight")
    
for ax in axes.flat:    
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.axis("tight")
    ax.set_xlabel("")

    #add coastlines
    res = '10m'
    ax.coastlines(resolution = res, linewidths = 0.5)
    gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)

    #grid and axes
    if ax == axes.flat[0] or ax == axes.flat[4]:
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabels_bottom = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
        
    if ax == axes.flat[1] or ax == axes.flat[2] or ax == axes.flat[3] or ax == axes.flat[5] or ax == axes.flat[6] or ax == axes.flat[7]:
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabels_bottom = False
        
    if ax == axes.flat[8]:
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
    
              
    if ax == axes.flat[9] or ax == axes.flat[10] or ax == axes.flat[11]:
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
     
    
    #add continent shading
    ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray', zorder = 1)

cax = fig.add_axes([0.92, 0.1, 0.02, 0.8], anchor='C')
cbar = plt.colorbar(sm, format='%.2f', spacing='proportional', cax=cax,
                    shrink=0.9, orientation = 'vertical', location = "right", pad = 0.1)
cbar.set_label( label = "Velocity [m/s]", fontsize = 60, y = 0.5, labelpad = 30)
cbar.ax.tick_params(which='minor', size=35, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=55, width=1, color='k', direction='in', labelsize = 50)
# Adjust the position and size of the colorbar axis
cax.set_position([1.02, 0.1, 0.02, 0.8])


# set the title of the figure
fig.suptitle("Monthly Sea Surface currents (01-01-2003 / 31-12-2020)", fontsize = 60, y = 0.98)
fig.tight_layout(pad = 5.0)
plt.show()
# fig.savefig(plot_path + 'vel_surf_month_extend_norm.png', bbox_inches='tight')
fig.savefig(plot_path + 'vel_surf_month_box_norm.png', bbox_inches='tight')
#%%VELOCITY AT DIFFERENT DEPTHS
 #surf, 75, 100, 200, 500, 1000
isobathy = np.array([20, 22, 26, 31, 35], dtype = 'int64')

vel = xr.Dataset()
u_2d = xr.Dataset()
v_2d = xr.Dataset()
z = np.zeros(5)

proj = ccrs.PlateCarree()

for j, n in enumerate(isobathy):
    vel = speed.isel(depth=isobathy[j])
    u_2d = u.isel(depth=isobathy[j])
    v_2d = v.isel(depth=isobathy[j])

    fig2, axes = plt.subplots(nrows=3, ncols=4, subplot_kw = dict(projection=proj), figsize=(40, 35))

                        
        # Set the y-axis view limits for each Axes with the axes.set_ylim() function
    for i, ax in enumerate(axes.flat):
        scalar = ax.contourf(lons, lats, vel[i], levels = levels, cmap=cmap, zorder = -1)
        lines = ax.contour(lons, lats, depth, levels = depths, colors='black', linewidths = .75, linestyle = '--', zorder = -1)
        ax.clabel(lines, inline=1, fontsize=20, colors = 'black')

        ax.set_title(str(month_list[i]), fontsize=40)

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
    cbar = fig2.colorbar(scalar, orientation = 'horizontal', location = "bottom", pad = -0.15)
    cbar.set_label( label = "Velocity [m/s]", fontsize = 40, y = 0.5)
    cbar.ax.tick_params(labelsize = 30)

    # adjust bottom margin and position colorbar at the bottom
    fig2.subplots_adjust(bottom=0.2)
    cbar.ax.set_position([0.2, 0.07, 0.6, 0.07])
    
    z = vel.depth.values

    # set the title of the figure
    fig2.suptitle("Monthly Sea currents velocity at z = " + "{:3f}".format(z) + " m", fontsize = 60, y = 0.93)
    fig2.tight_layout()
    plt.show()
    fig2.savefig(plot_path + 'vel_M_' + str(n-1)+'.png', bbox_inches='tight')
    plt.clf()
    
