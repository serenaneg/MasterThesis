#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:09:11 2023

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

from scipy import ndimage

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

def dx_dy_geom(lat1, lat2, lon1, lon2):
    p_lats = [lat1, lat2]
    p_lons= [lon1 ,lon2]

    dlamba = np.abs(p_lons[1] - p_lons[0]) #long
    dtheta = np.abs(p_lats[1] - p_lats[0])#lans

    dx = 6371*(np.pi/180)*dlamba
    dy = 6371*(np.pi/180)*dtheta

    
    return dx, dy


def set_plot_u_currents(ncols, date, cmap, vmin, vmax):
    fig, axes = plt.subplots(ncols = ncols, nrows = 1, figsize = [18*ncols,10])
    levels = np.linspace(vmin, vmax, 20)
    cmap = cmap
    
    for ax in axes.flat:
        ax.set_xlabel('Latitude [°N]', fontsize = 35)
        ax.set_ylabel('Depth [m]', fontsize = 35)
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        ax.xaxis.set_tick_params(labelsize = 30)
        ax.yaxis.set_tick_params(labelsize = 30)
        
        if ax != axes.flat[0] :
            ax.yaxis.set_tick_params(labelsize = 0)
            ax.set_ylabel('')
            
    for i, ax in enumerate(axes.flat):
        scalar = ax.contourf(lats, depth, u_rotate.sel(time = date[i]), levels = levels,
                             extend = 'both', cmap=cmap, zorder = 1)
              
        ax.set_title('Zonal currents at ' + str(date[i]), fontsize = 35, y=1.02)
        norm= Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
        sm.set_array([])
   
    fig.tight_layout(pad = 2.3)  
       
    return (fig, axes, sm)

def set_plot_v_currents(ncols, date, cmap, vmin, vmax):
    fig, axes = plt.subplots(ncols = ncols, nrows = 1, figsize = [18*ncols,10])
    levels = np.linspace(vmin, vmax, 20)
    cmap = cmap
    
    for ax in axes.flat:
        ax.set_xlabel('Latitude [°N]', fontsize = 35)
        ax.set_ylabel('Depth [m]', fontsize = 35)
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        ax.xaxis.set_tick_params(labelsize = 30)
        ax.yaxis.set_tick_params(labelsize = 30)
        
        if ax != axes.flat[0] :
            ax.yaxis.set_tick_params(labelsize = 0)
            ax.set_ylabel('')
            
    for i, ax in enumerate(axes.flat):
        scalar = ax.contourf(lats, depth, v_rotate.sel(time = date[i]), levels = levels,
                             extend = 'both', cmap=cmap, zorder = 1)
              
        ax.set_title('Meridional currents at ' + str(date[i]), fontsize = 35, y=1.02)
        norm= Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
        sm.set_array([])
   
    fig.tight_layout(pad = 2.3)  
       
    return (fig, axes, sm)

def set_cbar(fig, c, title):    
    cbar = plt.colorbar(c, format='%.1f', spacing='proportional', ax=axes, shrink=0.9, pad = 0.01,
                        orientation = 'vertical', location = "right")
    cbar.set_label(label = title, fontsize = 35, y = 0.5, labelpad = 30)
    cbar.ax.tick_params(which='minor', size=15, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=25, width=1, color='k', direction='in', labelsize = 30)
    # fig.subplots_adjust(bottom=0.25)
    # cbar.ax.set_position([0.2, 0.08, 0.6, 0.08])
    return cbar
#%% LOAD MONTHLY CMEMS DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/CMEMS/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/chlor/enhancements/"

#monthly data
ds = xr.open_dataset(path + 'cmems_grep.nc')
#%%
# #choose ARRAY coordinate
lat_range = slice(38, 41)
lon_range = slice(-72.6, -68)

ds = ds.sel(longitude=lon_range)
ds = ds.sel(latitude=lat_range)
#%% BATHYMETRY ALREADY INTERPOLATED ON CHL
#lat, lon, bathy
lons, lats = ds['longitude'], ds['latitude']
bathy = xr.open_dataset(path + "bathymetry_interpolated_cmems.nc")

bathy = bathy['elevation']
bathy= bathy.sel(latitude = lats, longitude = lons)

#%%
#approximate bathy with a line
x_bathy = [-72.6, -68]
y_bathy = [39.8, 40.8]

#get the angle betweet this one and north-south direction
#convert spherical coordinates to geometric

#x3x1 = dy, x3x2 = dx
x3x2, x3x1 = dx_dy_geom(40.5, 40.8, -70.6, -69.4) #!!check the formula for the distance

angle = np.arctan(x3x2/x3x1)
print(angle)
grad = np.rad2deg(angle)
print(grad)  #75.96375653207353 degrees

#%%
fig, ax = set_plot()

depths1 = (50, 75, 100, 200, 500, 1000, 2500, 3000)
#20 levels of colors
lines = ax.contour(lons, lats, -bathy, levels = depths1, colors='black', linewidths = .5)
ax.clabel(lines, inline=1, fontsize=15, colors = 'black')

x_perp = [-72.6, -68]
y_perp = [39.8, 40.8]

ax.plot(x_perp, y_perp, transform = ccrs.PlateCarree(), color = 'blue', linewidth = 2.5)
ax.plot([-69.4, -69.4], [38.5, 40.5], color = 'green', linewidth = 3)

y = [ 38.5, 39.8, 41, 41, 38.5, 38.5, 38.5] #lats
x = [ -70.6, -72.6, -72.6, -68, -68, -68, -70.6]

ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'red', linewidth = 2.5)

fig.subplots_adjust(bottom=0.02)
# cbar_ax = fig.add_axes([0.13, 0.085, 0.8, 0.05]) #left, bottom, width, height
# cbar = fig.colorbar(scalar, orientation = 'horizontal', format = '%.0f', location = "bottom", cax = cbar_ax)
# cbar.ax.tick_params(labelsize=20) 
# cbar.set_label(label = "Depth [m]", fontsize = 20)
fig.tight_layout()
plt.show()


#%%DATES
date_18 = np.array(('2018-10-31', '2018-11-01', '2018-11-04', '2018-11-07'))
date_19_j = np.array(('2019-06-22', '2019-06-23'))
date_19_m = np.array(('2019-05-19', '2019-05-21', '2019-05-22', '2019-05-24'))
date_19_a = np.array(('2019-04-24', '2019-04-27'))

#%%CMAP
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
u = ds['uo_mean']
v = ds['vo_mean']

#rotate u of 90-grad
grad_u = 90 - grad #u zonal direction rotation = 14 degrees
print(grad_u)
grad_v = grad 

# Initialize empty arrays with the original dimensions
u_rot = np.empty((len(u.time), len(u.depth), u.shape[2], u.shape[3]))
v_rot = np.empty((len(u.time), len(u.depth), v.shape[2], v.shape[3]))

for j in range(len(u.time)):
    for i in range(len(u.depth)):
        rot_u = ndimage.rotate(u[j,i,:,:], 15, reshape=False)
        rot_v = ndimage.rotate(v[j,i,:,:], 15+90, reshape=False)
        u_rot[j, i, :, :] = rot_u
        v_rot[j, i, :, :] = rot_v


u_rotate = xr.DataArray(u_rot, dims=('time', 'depth', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': u.time.values, 'depth' : u.depth.values})
v_rotate = xr.DataArray(v_rot, dims=('time', 'depth', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': u.time.values, 'depth' : u.depth.values})

speed_rot = np.sqrt(rot_u*rot_u + rot_v*rot_v)

#%%TRANSECT
lon_array = -70.8
u_rotate = u_rotate.sel(longitude = lon_array, method='nearest') #-70.75
v_rotate = v_rotate.sel(longitude = lon_array, method='nearest') #-70.75
depth = u_rotate['depth']
#%%
fig, axes, sm = set_plot_u_currents(2,  date_18, mycmap, vmin = 0.01, vmax = 1.3)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'zonalC_2019_jun.png', bbox_inches='tight')

fig, axes, sm = set_plot_v_currents(2,  date_18, mycmap, vmin = 0.01, vmax = 1.3)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'zonalC_2019_jun.png', bbox_inches='tight')

plt.contourf(lons, lats, u_rotate[1,13,:,:])



