#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 17:37:33 2023

@author: serena
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm, ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
import cmocean

path = "/home/serena/Scrivania/Magistrale/thesis/data/CMEMS/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/currents/"

#monthly data
ds = xr.open_dataset(path + 'cmems_grep.nc')
#%%
lons, lats = ds['longitude'], ds['latitude']
bathy = xr.open_dataset(path + "bathymetry_interpolated_cmems.nc")

bathy = bathy['elevation']
bathy= bathy.sel(latitude = lats, longitude = lons)

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
#%%
u = ds['uo_mean'].sel(time = ds.time.dt.month.isin([2, 3]), depth = 0.5, method='nearest')
v = ds['vo_mean'].sel(time = ds.time.dt.month.isin([2, 3]), depth = 0.5, method='nearest')
u = u.mean(axis=0, skipna=True).to_numpy()
v = v.mean(axis=0, skipna=True).to_numpy()
speed = np.sqrt(u*u + v*v)

#%%
proj = ccrs.PlateCarree()
depths = (50, 200, 1000, 2500)
levels = np.linspace(0.02, 0.4, 100)
# cmap = 'cmo.speed'

#create 2d grid lat-lon
lon2d, lat2d = np.meshgrid(lons, lats)
#%%

fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw = dict(projection=proj), figsize=(25, 18))
    
 # Downsampling a 2d array
step=3
lon2d_sub = lon2d[1:len(lats)-1:step,1:len(lons)-1:step]
lat2d_sub = lat2d[1:len(lats)-1:step,1:len(lons)-1:step]
u_sub = u[1:len(lats)-1:step,1:len(lons)-1:step]
v_sub = v[1:len(lats)-1:step,1:len(lons)-1:step]
speed_sub = speed[1:len(lats)-1:step,1:len(lons)-1:step]
 
 
# Define the starting point
speed_flat = speed_sub.flatten()
start_points = np.array([lon2d_sub.flatten(),lat2d_sub.flatten()]).T


# plot a quiver with axes.quiver() from that point. Make the quiver small enough so that only the arrow is visible
scalar = ax.contourf(lon2d, lat2d, speed, levels=levels, transform = proj, 
                     extend = 'both', cmap = mycmap)

 # make streamplot with axes.streamplot() with no arrows that is traced backward from a given point
scale = 0.1/np.nanmax(speed_sub)
    
for j in range(start_points.shape[0]):
    ax.streamplot(lon2d,lat2d,u,v,
        color='black',
        start_points=np.array([start_points[j,:]]),
        minlength=0.1*speed_flat[j]*scale,
        maxlength=1*speed_flat[j]*scale,
        integration_direction='backward',
        density=20,
        arrowsize=0.0,
        linewidth=3.)    


# plot a quiver with axes.quiver() from that point. Make the quiver small enough so that only the arrow is visible
ax.quiver(lon2d_sub,lat2d_sub,u_sub/speed_sub, v_sub/speed_sub,scale=40)           

norm= Normalize(vmin=0.02, vmax=0.4)
sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
sm.set_array([])

lines = ax.contour(lon2d, lat2d, -bathy, levels = depths, colors='black', linewidths = .55, linestyles = '--')
ax.clabel(lines, inline=2, fontsize=30, colors = 'black')
 
# Set the x-y axis labels for each Axes with the axes.set_xlabel() function
ax.set_xticks(lons, crs=ccrs.PlateCarree())
ax.set_xticks(lats, crs=ccrs.PlateCarree())
ax.tick_params(axis ='both', labelsize=35)
ax.axes.axis("tight") 
ax.axes.get_xaxis().set_ticklabels([])
ax.axes.get_yaxis().set_ticklabels([])
ax.set_xlabel("")

las,lan=41,38.05 #alto-basso
low,loe=-68,-72.6  #dx-sx

y = [ lan, las, las, lan, lan ] #lats
x = [ loe, loe, low, low, loe ]

ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'darkblue', linewidth = 4.5)

ax.text(x = -71.3, y = 40.5, s = "Shelfbreak Jet", transform = ccrs.PlateCarree(), size = 45, color = 'blue')
ax.text(x = -71.9, y = 39, s = "Gulf Stream", transform = ccrs.PlateCarree(), size = 45, color = 'blue')

 #add coastlines
res = '10m'
ax.coastlines(resolution = res, linewidths = 0.5)
gl = ax.gridlines(linewidth=0.7, color='gray', alpha=0.7, linestyle='--', draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'fontsize': 40}
gl.ylabel_style = {'fontsize': 40}
    
ax.add_feature(cfeature.LAND.with_scale(res), facecolor = 'lightgray', zorder = 1)
    
# adjust bottom margin and position colorbar at the bottom
cbar = fig.colorbar(sm, format='%.2f', spacing='proportional', ax=ax, shrink=0.9, pad = 0.03,
                    orientation = 'vertical', location = "right")
cbar.ax.tick_params(which='major', size=25, width=1, color='k', direction='in', labelsize = 30)
cbar.set_label(label = "Velocity [m/s]", fontsize = 35, y = 0.5)


# set the title of the figure
fig.suptitle("Location of the main surface currents", fontsize =50, y = 1.0)
fig.tight_layout()
plt.show()
fig.savefig(plot_path + 'currents_area.png', bbox_inches='tight')
