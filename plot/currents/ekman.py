#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:51:24 2023

@author: serena
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 09:59:50 2023

@author: serena
COMPUTE EKMAN PUMPING AND EKMAN TRANSPORT
"""

import xarray as xr
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from scipy import signal 
from matplotlib.path import Path
from scipy import interpolate
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap

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


#%% LOAD ERA5 DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/ERA5/"
ocean_path =  "/home/serena/Scrivania/Magistrale/thesis/data/CMEMS/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/currents/"

u = xr.open_dataset(path +"u_wind_daily_era5.nc") 
v = xr.open_dataset(path +"v_wind_daily_era5.nc") 
#%%LAT LON AND BATHY
lat_range = slice(41, 38) #alto-basso
lon_range = slice(-72.5, -68)  #dx-sx

u = u.sel(longitude=lon_range)
u = u.sel(latitude=lat_range)
v = v.sel(longitude=lon_range)
v = v.sel(latitude=lat_range)

lons, lats = u['longitude'], v['latitude']

bathy = xr.open_dataset(path + "bathymetry.nc")
bathy = bathy.interp(lon = lons, lat = lats, method = "nearest")
bathy = bathy.sel(latitude = lats, longitude = lons)

depth = bathy['elevation'] 
depth_limits = depth.where((depth <= -50) & (depth >= -3000), drop = True)

#%%
u10 = u.u10
v10 = v.v10
    
#constants
c_d = 1.5e-03 
rho_air = 1.225 #[kg/m^3]
f = 1e-4 

wind_module = np.sqrt(u10*u10 + v10*v10)
#%%WIND STRESS
#(tau_x, tau_y) = rho*cd*speed*(u,v)
tau_x = rho_air * c_d * wind_module * u10
tau_y = rho_air * c_d * wind_module * v10

tau_x_arr =np.array(tau_x)
tau_y_arr = np.array(tau_y)
#%%
#EKMANN TRANSPORT
#M = 1/f*(t_wind x z)
# Mx_ek = tau_y[0,:,:] / f2d
# My_ek = -tau_x[0,:,:] / f2d
# for i in range(1, 6575):
#     Mx_ek = xr.concat([Mx_ek, tau_y[i,:,:] / f2d], dim = 'time')
#     My_ek = xr.concat([My_ek, -tau_x[i,:,:] / f2d], dim = 'time')
#     print(i)
 
My = []
Mx = []
for i in range(0, 6575):
    my = -tau_x_arr[i,:,:]/f
    mx = tau_y_arr[i,:,:]/f
    My.append(my)
    Mx.append(mx)
    
Mx_ek = xr.DataArray(Mx)    
My_ek = xr.DataArray(My)
#%%EKMANN PUMPING = CONV/DIV
rho_ocean = 1.025 #[kg/m^3]
#partial derivates
#np.gradient = return derivatives along each array's dimension 
dx = np.diff(lons)[0]
dy = -np.diff(lats, axis = 0)[0]


w_ek = np.zeros(u10.shape)

dtau_x = np.zeros(u10.shape)
dtau_y = np.zeros(u10.shape)
w_ek = []
for i in range(0, 6575):
    dtau_x[i,1:,:] = tau_x_arr[i,1:,:] - tau_x_arr[i,:-1,:]
    dtau_y[i,:,1:] = tau_y_arr[i,:,1:] - tau_y_arr[i,:,:-1]
    # dtau_x[i,1:-1,:] = (tau_x[i,2:,:] - tau_x[i,:-2,:])/2
    # dtau_y[i,:,1:-1] = (tau_y[i,:,2:] - tau_y[i,:,:-2])/2 

    w = ((1/(rho_ocean*f)) * ((dtau_y[i,:,:]/dx) - (dtau_x[i,:,:]/dy)))/(np.pi*6371*1000)*180 
    w_ek.append(w)
    
w_ek = xr.DataArray(w_ek)
    
#%%VERIFY
fig, ax = set_plot()
depths1 = (50, 75, 100)
depths2 = (200, 500, 1000, 2500, 3000)
scalar = ax.contourf(lons, lats, wind_module[78],   cmap = 'jet', transform = ccrs.PlateCarree()) #20 levels of colors
lines = ax. contour(lons, lats, -depth, levels = depths1, colors='black', linewidths = .5)
lines2 = ax. contour(lons, lats, -depth, levels = depths2, colors='black', linewidths = .5)
ax.clabel(lines, inline=1, fontsize=15, colors = 'black')
ax.clabel(lines2, inline=1, fontsize=15, colors = 'black')

y = [ 38.5, 39.8, 41, 41, 38.5, 38.5, 38.5] #lats
x = [ -70.6, -72.6, -72.6, -68, -68, -68, -70.6]

ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'red', linewidth = 2.5)
fig.subplots_adjust(bottom=0.02)
cbar_ax = fig.add_axes([0.13, 0.085, 0.8, 0.05]) #left, bottom, width, height
cbar = fig.colorbar(scalar, orientation = 'horizontal', format = '%.8f', location = "bottom", cax = cbar_ax)
cbar.ax.tick_params(labelsize=20) 
cbar.set_label(label = "Depth [m]", fontsize = 20)
fig.tight_layout()
    
#%%ASSIGN COORDINATES LAT LON TO W_EK
#than merge
w_ek_coords = xr.DataArray(w_ek, dims=('time', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': u10.time})

Mx_ek_coords = xr.DataArray(Mx_ek, dims=('time', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': u10.time})
My_ek_coords = xr.DataArray(My_ek, dims=('time', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': u10.time})


w_ek_ds = w_ek_coords.to_dataset(name='w_ek')
Mx_ek_ds = Mx_ek_coords.to_dataset(name='Mx_ek')
My_ek_ds = My_ek_coords.to_dataset(name='My_ek')

ekman = xr.merge([Mx_ek_ds, My_ek_ds, w_ek_ds], compat='override')

ekman.load().to_netcdf(ocean_path + "EKMAN.nc") #daily dataset

#%%PLOT


