#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:28:58 2023

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

#at the surface
v = ds_month['vo_mean'].isel(depth=0).to_numpy()
u = ds_month['uo_mean'].isel(depth=0).to_numpy()

# data_mask = xr.open_dataset(path + 'GLO-MFC_001_030_mask_bathy.nc')
speed = np.sqrt(np.square(u) + np.square(v))
#%%
proj = ccrs.PlateCarree()
depths = (50, 200, 1000,2500)
levels = np.linspace(speed.min(), speed.max(), 20)
# cmap = 'cmo.speed'

#create 2d grid lat-lon
lon2d, lat2d = np.meshgrid(lons, lats)

#%%
colors = [tuple(np.array((235,245,255))/255), # light blue
          tuple(np.array((157,218,247))/255), # blue
          tuple(np.array((72,142,202))/255),  # dark blue
          tuple(np.array((73,181,70))/255),   # green
          tuple(np.array((250,232,92))/255),  # yellow
          tuple(np.array((245,106,41))/255),  # orange
          tuple(np.array((211,31,40))/255),   # red
          tuple(np.array((146,21,25))/255)]   # dark red
mycmap = LinearSegmentedColormap.from_list('WhiteBlueGreenYellowRed', colors, N=100)


#%%SURF CURRENT
fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw = dict(projection=proj), figsize=(55, 35))
    
for i, ax in enumerate(axes.flat):    
    # Downsampling a 2d array
    step=2
    lon2d_sub = lon2d[1:len(lats)-1:step,1:len(lons)-1:step]
    lat2d_sub = lat2d[1:len(lats)-1:step,1:len(lons)-1:step]
    u_sub = u[i][1:len(lats)-1:step,1:len(lons)-1:step]
    v_sub = v[i][1:len(lats)-1:step,1:len(lons)-1:step]
    speed_sub = speed[i][1:len(lats)-1:step,1:len(lons)-1:step]
    
    
    # Define the starting point
    speed_flat = speed_sub.flatten()
    start_points = np.array([lon2d_sub.flatten(),lat2d_sub.flatten()]).T


   # plot a quiver with axes.quiver() from that point. Make the quiver small enough so that only the arrow is visible
    scalar = ax.contourf(lon2d, lat2d, speed[i], levels = levels,
                        transform = proj, cmap = mycmap)

    # make streamplot with axes.streamplot() with no arrows that is traced backward from a given point
    scale = 0.1/np.max(speed_sub)
    
    uu = u[i]
    vv = v[i]
    
    for j in range(start_points.shape[0]):
        ax.streamplot(lon2d,lat2d,uu,vv,
            color='black',
            start_points=np.array([start_points[j,:]]),
            minlength=1.2*speed_flat[j]*scale,
            maxlength=1.5*speed_flat[j]*scale,
            integration_direction='backward',
            density=20,
            arrowsize=0.0,
            linewidth=2.)

# plot a quiver with axes.quiver() from that point. Make the quiver small enough so that only the arrow is visible
    ax.quiver(lon2d_sub,lat2d_sub,u_sub/speed_sub, v_sub/speed_sub,scale=30)           
    
    norm= Normalize(vmin=scalar.cvalues.min(), vmax=scalar.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])
    
    lines = ax.contour(lon2d, lat2d, depth, levels = depths, colors='black', linewidths = .55, linestyles = '--')
    ax.clabel(lines, inline=2, fontsize=30, colors = 'black')
    ax.set_title(str(month_list[i]), fontsize=40)
 
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
    
# #color bar
skip = (slice(None, None, 2))
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8], anchor='C')
cbar = plt.colorbar(sm, format='%.2f', spacing='proportional', cax=cax,
                    shrink=0.9, orientation = 'vertical', location = "right", pad = 0.1)
cbar.set_label( label = "Velocity [m/s]", fontsize = 60, y = 0.5, labelpad = 30)
cbar.ax.tick_params(which='minor', size=35, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=55, width=1, color='k', direction='in', labelsize = 50)
# Adjust the position and size of the colorbar axis
cax.set_position([1.02, 0.1, 0.02, 0.8])

# set the title of the figure
fig.suptitle("Monthly Sea Surface currents velocity", fontsize = 60, y = 0.93)
fig.tight_layout()
plt.show()
# fig.savefig(plot_path + 'vel_surf_month_extend_norm.png', bbox_inches='tight')
fig.savefig(plot_path + 'vel_surf_month_curvy.png', bbox_inches='tight')

