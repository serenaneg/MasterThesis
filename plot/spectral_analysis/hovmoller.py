#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:57:54 2023

@author: serena
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
from matplotlib.colors import ListedColormap, Normalize, LogNorm
import matplotlib.colors
import matplotlib.cm as cm

import cmocean

#%% LOAD DAILY MODIS DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/MODIS_INTERPOLATED_DATA/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/spectral_analysis/"
figures = plot_path + "/figures/"
#monthly data
ds = xr.open_dataset(path +"MODIS_chl_1D_nomissing.nc")

#%%ARRAY
# #choose ARRAY coordinate
lat_range = slice(41, 38) #alto-basso
lon_range = slice(-72.6, -68)  #dx-sx

ds = ds.sel(lon=lon_range)
ds = ds.sel(lat=lat_range)
#%% BATHYMETRY ALREADY INTERPOLATED ON CHL
#lat, lon, bathy
lons, lats = ds['lon'], ds['lat']
bathy = xr.open_dataset(path + "bathymetry_interpolated.nc")

depth = bathy['elevation'] 
depth = depth.sel(lat = lats, lon = lons)
depth_limits = depth.where((depth <= -50) & (depth >= -3000), drop = True)

#%%POLYGON MASK
poly_verts = [(-70.6, 38.5), (-72.6, 39.8), (-72.6, 41), (-68, 41), (-68, 38.5)]

# Create vertex coordinates for each grid cell...
lon2d, lat2d = np.meshgrid(lons, lats)
lon2d, lat2d = lon2d.flatten(), lat2d.flatten()

points = np.vstack((lon2d, lat2d)).T

path = Path(poly_verts)
grid = path.contains_points(points)
grid_bool = grid.reshape((72, 110)) #72 = lons length, 110 lat lenghts
#coverto boolean grid into 0-1 grid
grid_int = grid_bool*1

#%%MASK BATHYMETRY AND CHLOROPHYLL
depth_polyg = depth_limits * grid_int
#substitude 0 with nan beacuse depth_limits is a greater area then grid_int
depth_zeros = depth_polyg.where(depth_polyg != 0, np.nan)
depth_ones = depth_zeros.where(np.isnan(depth_zeros), 1)

chl = ds['chlor_a']

chl_sel = []
for i in range(0, 6574):
    a = chl[i,:,:] * depth_ones
    chl_sel.append(a)
    print(i)
    
chl_sel = np.array(chl_sel)


#%%
range_1 = np.linspace(50, 100, 16)
print(range_1)
range_2= np.linspace(100, 1000, 4)
print(range_2)
range_3 = np.linspace(1000, 2000, 4)
print(range_3)
range_4 = np.linspace(2000, 2500, 5)
range_5 = np.linspace(2500, 3000, 26)
slices = np.concatenate((range_1, range_2, range_3, range_4, range_5))

#%%

chlor = []
bins = []

for i in slices:
    if i <=100:
         sel_depth = depth_zeros.where((depth_zeros <= -i) & (depth_zeros > (-i-3.125)), drop = False)
         print([i])
    elif 100 < i <= 1000:
         sel_depth = depth_zeros.where((depth_zeros <= -i) & (depth_zeros > (-i-225)), drop = False)
         print([i])
                
    elif 1000 < i <= 2000:
         sel_depth = depth_zeros.where((depth_zeros <= -i) & (depth_zeros > (-i-250)), drop = False)
         print([i])
    
    elif 2000 < i <= 2500:
         sel_depth = depth_zeros.where((depth_zeros <= -i) & (depth_zeros > (-i-100)), drop = False)
         print([i])
         
    elif 2500 < i <= 3000:
         sel_depth = depth_zeros.where((depth_zeros <= -i) & (depth_zeros > (-i-19.2)), drop = False)
         print([i])
         
    #subsitute non nan values with 1
    sel_depth = sel_depth.where(np.isnan(sel_depth), 1)
    #nan = 0
    # sel_depth = sel_depth.where(~np.isnan(sel_depth), 0)
    
    bins.append(i)
     
    chl_bin = []
     #applay mask bathymetry (slice) on chl daily already masked on the area
    for j in range(0, 6574):
        b = chl_sel[j,:,:] * sel_depth
        chl_bin.append(b)
     
    chl_bins = np.array(chl_bin)
     
     #annual mean in the slice
    chl_year = [] 
    for k in  np.arange(0, 6205, 364):   #6205 17th year         
         year_mean = np.nanmean(chl_bins[k:k+364,:,:], axis = (1,2))
         chl_year.append(year_mean)
    
    chlor.append(chl_year)
        
chlor = np.array(chlor) 


#%%
# define top and bottom colormaps 
top = cm.get_cmap('YlGnBu_r', 128) # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)# combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')
#%%#################PLOT#######################################################
year_list = np.arange(2003, 2021, 1)
chlor_t = chlor.T

log_chlor = np.log10(chlor_t)
mini = np.nanmin(log_chlor)
maxi = np.nanmax(log_chlor)

levels =  np.linspace(0.1, 12., 25)

fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(70, 41))

for i, ax in enumerate(axes.flat):
   
    scalar = ax.contourf(np.abs(bins), range(0,364), chlor_t[:,i,:], norm='log',
                        levels = levels,  cmap = orange_blue)
    
    norm= LogNorm(vmin=0.1, vmax=12)
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])
            
    ax.set_title(str(year_list[i]), fontsize=60)
    ax.set_xscale('log')
    ax.set_facecolor("whitesmoke")

for ax in axes.flat:  
    ax.axes.axis("tight")
    
    ax.set_xticks([ 100, 200, 500, 1000, 2500])
    #y ticks feb, apr, jun, au, oct, dec
    ax.set_yticks([59, 120, 181, 243, 304])
    ax.set_yticklabels(['Feb', 'Apr', 'Jun', 'Aug', 'Oct'])

    #grid and axes
    if ax == axes.flat[0] or ax == axes.flat[6]:
        gl = ax.grid(linewidth=0.7, color='gray', alpha=0.7, linestyle='--')
        ax.xaxis.set_ticklabels([]) 
        ax.yaxis.set_tick_params(labelsize = 50)
        
    if ax == axes.flat[1] or ax == axes.flat[2] or ax == axes.flat[3] or ax == axes.flat[4] or ax == axes.flat[5] or ax == axes.flat[7] or ax == axes.flat[8] or ax == axes.flat[9] or ax == axes.flat[10] or ax == axes.flat[11]:
        gl = ax.grid(linewidth=0.7, color='gray', alpha=0.7, linestyle='--')
        ax.xaxis.set_ticklabels([]) 
        ax.yaxis.set_ticklabels([]) 
      
    if ax == axes.flat[12]:
        gl = ax.grid(linewidth=0.7, color='gray', alpha=0.7, linestyle='--')
        ax.set_xlabel('Depth [m]', fontsize = 60)
        ax.xaxis.set_tick_params(labelsize = 50)
        ax.yaxis.set_tick_params(labelsize = 50)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

                 
    if ax == axes.flat[13] or ax == axes.flat[14] or ax == axes.flat[15] or ax == axes.flat[16] or ax == axes.flat[17]:
        gl = ax.grid(linewidth=0.7, color='gray', alpha=0.7, linestyle='--')
        ax.set_xlabel('Depth [m]', fontsize = 60)
        ax.xaxis.set_tick_params(labelsize = 50)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_ticklabels([]) 

       
# #color bar
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8], anchor='C')
cbar = plt.colorbar(sm, format='%.1f', spacing='proportional', cax=cax,
                    shrink=0.9, orientation = 'vertical', location = "right", pad = 0.1)
cbar.set_label(label = "Chlorophyll [mg/m$^3$]", fontsize = 60, y = 0.5, labelpad = 30)
cbar.ax.tick_params(which='minor', size=35, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=55, width=1, color='k', direction='in', labelsize = 50)
# Adjust the position and size of the colorbar axis
cax.set_position([1.02, 0.1, 0.02, 0.8])

# set the title of the figure
fig.suptitle("", fontsize = 60, y = 0.95)
fig.tight_layout()
fig.subplots_adjust(hspace=0.2)
plt.show()
fig.savefig(plot_path + 'chl_slice_2d_log.png', bbox_inches='tight')
# fig.savefig(plot_path + 'chl_slice_2d.png', bbox_inches='tight')

#%%CHL CLIMATOLOGICAL
chl_clim = np.nanmean(chlor, axis = 1)
chl_median = np.nanmedian(chlor, axis = 1)
#%%
fig, ax = plt.subplots(figsize=(15, 25))

levels = np.linspace(0.1, 2.5, 9)
scalar = ax.contourf(np.abs(bins), range(0,364), chl_clim.T, levels = levels, extend = 'max', cmap = 'cmo.algae')
ax.set_title('Climatological Mean Chlor-a [mg/m$^3$]', fontsize=45, y = 1.05)
ax.set_xscale('log')

ax.set_xticks([75, 100, 200, 500, 1000, 2500])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
#y ticks feb, apr, jun, au, oct, dec
ax.set_yticks([0, 32, 60, 92, 122, 153, 183, 214, 245, 275, 305, 335])
ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

y = [111, 131, 131, 111, 111]
x = [75, 75, 2000, 2000, 75]
ax.plot(x, y, color = 'red', linewidth = 2.5)

gl = ax.grid(linewidth=0.7, color='gray', alpha=0.7, linestyle='--')
ax.xaxis.set_tick_params(labelsize = 40)
ax.yaxis.set_tick_params(labelsize = 40)

ax.set_xlabel('Depth [m]', fontsize = 35)
       
# #color bar
cbar = fig.colorbar(scalar, format='%.1f', spacing = 'uniform',  orientation = 'horizontal', ticks = levels,
                    location = "bottom", pad = 0.10)
cbar.set_label(label = "Chlorophyll [mg m^-3]", fontsize = 40, y = 0.5)
cbar.ax.tick_params(which='minor', size=20, width=1, color='white', direction='in')
cbar.ax.tick_params(which='major', size=45, width=1, color='white', direction='in', labelsize = 35)

# adjust bottom margin and position colorbar at the bottom
fig.subplots_adjust(bottom=0.2)
cbar.ax.set_position([0.2, 0.07, 0.6, 0.07])
# set the title of the figure
fig.suptitle("", fontsize = 60, y = 0.95)
fig.tight_layout()
plt.show()
# fig.savefig(plot_path +'chl_annual_extend.png', bbox_inches='tight')
fig.savefig(plot_path + 'chl_slice_clim.png', bbox_inches='tight')

#%%MEDIAN
fig, ax = plt.subplots(figsize=(15, 25))

levels = np.linspace(0.1, 3, 9)
scalar = ax.contourf(np.abs(bins), range(0,364), chl_median.T, levels = levels, extend = 'max', cmap = 'cmo.algae')
ax.set_title('Climatological Median Chlor-a [mg/m$^3$]', fontsize=45, y = 1.05)
ax.set_xscale('log')

ax.set_xticks([75, 100, 200, 500, 1000, 2500])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
#y ticks feb, apr, jun, au, oct, dec
ax.set_yticks([0, 32, 60, 92, 122, 153, 183, 214, 245, 275, 305, 335])
ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

y = [114, 134, 134, 114, 114]
x = [75, 75, 2400, 2400, 75]
ax.plot(x, y, color = 'red', linewidth = 2.5)

gl = ax.grid(linewidth=0.7, color='gray', alpha=0.7, linestyle='--')
ax.xaxis.set_tick_params(labelsize = 40)
ax.yaxis.set_tick_params(labelsize = 40)

ax.set_xlabel('Depth [m]', fontsize = 35)
       
# #color bar
cbar = fig.colorbar(scalar, format='%.1f', spacing = 'uniform',  orientation = 'horizontal', ticks = levels,
                    location = "bottom", pad = 0.10)
cbar.set_label(label = "Chlorophyll [mg m^-3]", fontsize = 40, y = 0.5)
cbar.ax.tick_params(which='minor', size=20, width=1, color='white', direction='in')
cbar.ax.tick_params(which='major', size=45, width=1, color='white', direction='in', labelsize = 35)

# adjust bottom margin and position colorbar at the bottom
fig.subplots_adjust(bottom=0.2)
cbar.ax.set_position([0.2, 0.07, 0.6, 0.07])
# set the title of the figure
fig.suptitle("", fontsize = 60, y = 0.95)
fig.tight_layout()
plt.show()
# fig.savefig(plot_path +'chl_annual_extend.png', bbox_inches='tight')
fig.savefig(plot_path + 'chl_median_clim.png', bbox_inches='tight')

