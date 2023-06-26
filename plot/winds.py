#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:50:16 2023

@author: serena
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import datetime
from matplotlib.colors import  LinearSegmentedColormap, Normalize
import cmocean

#%% LOAD DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/ERA5/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/wind/"

#monthly data
ds = xr.open_dataset(path +"winds_monthly_era5.nc") 
#216 months in 18 years
#%%#%% BATHYMETRY 
#lat, lon, bathy
lons, lats = ds['longitude'], ds['latitude']

bathy = xr.open_dataset(path + "bathymetry.nc")
bathy = bathy.interp(lon = lons, lat = lats, method = "nearest")

# lat_range = slice(40.6, 39)
# lon_range = slice(-72.6, -69.4)

bathy = bathy.sel(latitude = lats, longitude = lons)

depth = bathy['elevation'] 
#%%
ds_month = ds.groupby('time.month').mean()
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

u_wind = ds_month['u10']
v_wind = ds_month['v10']

speed = np.sqrt(u_wind*u_wind + v_wind*v_wind)

#%%COLORS
colors = [tuple(np.array((255,255,255))/255), # white
          tuple(np.array((157,218,247))/255), # blue
          tuple(np.array((72,142,202))/255),  # dark blue
          tuple(np.array((73,181,70))/255),   # green
          tuple(np.array((250,232,92))/255),  # yellow
          tuple(np.array((245,106,41))/255),  # orange
          tuple(np.array((211,31,40))/255),   # red
          tuple(np.array((146,21,25))/255)]   # dark red
mycmap = LinearSegmentedColormap.from_list('WhiteBlueGreenYellowRed', colors, N=100)


#%%WINDS WITH BARBS
proj = ccrs.PlateCarree()
depths = (50, 100, 200, 1000, 2500)
levels = np.linspace(speed.min(), speed.max(), 100)

lon2d, lat2d = np.meshgrid(lons, lats)

fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw=dict(projection=proj), figsize=(55,35))

for i, ax in enumerate(axes.flat):
    # Plot the 2D vector field of arrows with the axes.streamplot() function    
    skip = (slice(None, None, 2), slice(None, None, 2))
    scalar = ax.contourf(lon2d[skip], lat2d[skip], speed[i][skip], cmap = 'turbo', alpha=.8)
    barba = ax.barbs(lon2d[skip], lat2d[skip], u_wind[i][skip], v_wind[i][skip], color = 'k',#speed[i][skip], cmap = 'turbo',
                     transform=proj,length=7, barb_increments=dict(half=0.4, full=0.8, flag=3.6), linewidth = 3.5,
                     fill_empty=True, rounding=True, sizes=dict(emptybarb=0.2, spacing=0.5, height=0.5, width = 1))
    lines = ax.contour(lons, lats, -depth, levels = depths, colors='black', linewidths = .55, linestyles='--')
    ax.clabel(lines, inline=3, fontsize=30, colors = 'black')
    ax.set_title(str(month_list[i]), fontsize=45)
    # ax.set_facecolor("whitesmoke")
    norm= Normalize(vmin=scalar.cvalues.min(), vmax=scalar.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])
    
    # Set the x-y axis labels for each Axes with the axes.set_xlabel() function
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.label_outer()
    
for ax in axes.flat:    
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.axis("tight")
    #add coastlines
    res = '10m'
    ax.coastlines(resolution = res, linewidths = 0.75)
    
    #grid and axes
    if ax == axes.flat[0] or ax == axes.flat[4]:
        gl = ax.gridlines(linewidth=0.5, color='gray', alpha=0.5,
                          linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabels_bottom = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
        
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
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
    
              
    if ax == axes.flat[9] or ax == axes.flat[10] or ax == axes.flat[11]:
        gl = ax.gridlines(linewidth=0.5, color='gray', alpha=0.5,
                          linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
    
    #add continent shading
    ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray', zorder = -1)
   
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8], anchor='C')
cbar = plt.colorbar(sm, format='%.2f', spacing='proportional', cax=cax,
                    shrink=0.9, orientation = 'vertical', location = "right", pad = 0.1)
cbar.set_label( label = "Wind Speed [m/s]", fontsize = 60, y = 0.5, labelpad = 30)
cbar.ax.tick_params(which='minor', size=35, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=55, width=1, color='k', direction='in', labelsize = 50)
# Adjust the position and size of the colorbar axis
cax.set_position([1.02, 0.1, 0.02, 0.8])

# set the title of the figure
fig.suptitle("Monthly Wind speed and direction (01-01-2003 / 31-12-2020)", fontsize = 60, y = 0.98)
fig.tight_layout(pad = 5.0)
plt.show()  
fig.savefig(plot_path + 'wind_barbs.png', bbox_inches='tight')

#%%EASTWARD WINDS
levels = np.linspace(u_wind.min(), u_wind.max(), 15)

lon2d, lat2d = np.meshgrid(lons, lats)

fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw=dict(projection=proj), figsize=(60,35))

for i, ax in enumerate(axes.flat):
    # Plot the 2D vector field of arrows with the axes.streamplot() function
    skip = (slice(None, None, 1), slice(None, None, 1))
    
    scalar = ax.contourf(lon2d, lat2d, u_wind[i], levels = levels, transform = proj, cmap = 'turbo')

    q = ax.quiver(lon2d[skip], lat2d[skip], u_wind[i][skip], v_wind[i][skip],
                transform=ccrs.PlateCarree())
    
    lines = ax.contour(lons, lats, -depth, levels = depths, colors='black', linewidths = .55, linestyles='--')
    ax.clabel(lines, inline=1, fontsize=20, colors = 'black')
    ax.set_title(str(month_list[i]), fontsize=40)
    # ax.set_facecolor("whitesmoke")
    
    # Set the x-y axis labels for each Axes with the axes.set_xlabel() function
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.label_outer()
    
for ax in axes.flat:    
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
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
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
        
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
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
    
              
    if ax == axes.flat[9] or ax == axes.flat[10] or ax == axes.flat[11]:
        gl = ax.gridlines(linewidth=0.5, color='gray', alpha=0.5,
                          linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.xlabel_style = {'fontsize': 40}
        gl.ylabel_style = {'fontsize': 40}
    
    
    #add continent shading
    ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray', zorder = -1)
   
# #color bar
cbar = fig.colorbar(scalar, format='%.2f', spacing='proportional',
                    orientation = 'horizontal', location = "bottom", pad = -0.15)
cbar.set_label( label = "Eastward wind intensity [m/s]", fontsize = 40, y = 0.5)
cbar.ax.tick_params(which='minor', size=30, width=1, color='white', direction='in')
cbar.ax.tick_params(which='major', size=45, width=1, color='white', direction='in', labelsize = 40)

# adjust bottom margin and position colorbar at the bottom
fig.subplots_adjust(bottom=0.2)
cbar.ax.set_position([0.2, 0.07, 0.6, 0.07])

# set the title of the figure
fig.suptitle("Monthly zonal wind \n (01-01-2003 / 31-12-2020)", fontsize = 60, y = 0.95)
fig.tight_layout()
plt.show()  
fig.savefig(plot_path + 'wind_eastward.png', dpi=500, bbox_inches='tight')

