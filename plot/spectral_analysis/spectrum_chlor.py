#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:07:49 2023

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
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap
import cmocean

#%%FUNTIONS
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
    ax.set_title(title, fontsize = 25, y=1.05)
#%%   
colors = ['slateblue',   # red
          tuple(np.array((157,218,247))/255), # blue
          tuple(np.array((250,232,92))/255),  # yellow
          tuple(np.array((72,142,202))/255),  # dark blue
          tuple(np.array((73,181,70))/255),   # green         
          tuple(np.array((245,106,41))/255)]   # dark red

# mycmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"], N=52)
mycmap = ListedColormap(colors, N=17)
mycmap_1 = ListedColormap(colors[0:3], N=7)
mycmap_2 = ListedColormap(colors[3:7], N=7)
mycmap_3 = ListedColormap(colors, N=27)  
#%% LOAD DAILY MODIS DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/MODIS_INTERPOLATED_DATA/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/spectral_analysis/"
figures = plot_path + "/figures/"
#monthly data
ds = xr.open_dataset(path +"MODIS_chl_1D_nomissing.nc") #6351 days

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

#%%
range_1 = np.linspace(50, 100, 16)
print(range_1)
range_2= np.linspace(100, 1000, 4)
print(range_2)
range_3 = np.linspace(1000, 2000, 4)
print(range_3)
range_4 = np.linspace(2000, 2500, 5)
range_5 = np.linspace(2500, 3000, 26)


slices = np.concatenate((range_1[:-1], range_2[:-1], range_3[:-1], range_4[:-1], range_5))

#%%
fig, ax = set_plot()

depths1 = (50, 75, 100, 200, 500, 1000, 2500, 3000)
scalar = ax.contourf(lons, lats, -depth_zeros, levels = range_1, cmap = mycmap, transform = ccrs.PlateCarree(), zorder=-1) 
scalar = ax.contourf(lons, lats, -depth_zeros, levels = range_2, cmap = mycmap_1,  transform = ccrs.PlateCarree(), zorder=-1) #20 levels of colors
scalar = ax.contourf(lons, lats, -depth_zeros, levels = range_3, cmap = mycmap_2, transform = ccrs.PlateCarree(), zorder=-1) 
scalar = ax.contourf(lons, lats, -depth_zeros, levels = range_4, cmap = mycmap_1, transform = ccrs.PlateCarree(), zorder=-1) #20 levels of colors
scalar = ax.contourf(lons, lats, -depth_zeros, levels = range_5, cmap = mycmap_3, transform = ccrs.PlateCarree(), zorder=-1) #20 levels of colors
#20 levels of colors
lines = ax. contour(lons, lats, -depth, levels = depths1, colors='black', linewidths = .5)
ax.clabel(lines, inline=1, fontsize=15, colors = 'black')

# x_perp = [-143/2, -141/2]
# y_perp = [41, 38]

# ax.plot(x_perp, y_perp, transform = ccrs.PlateCarree(), color = 'blue', linewidth = 2.5)

y = [ 38.5, 39.8, 41, 41, 38.5, 38.5, 38.5] #lats
x = [ -70.6, -72.6, -72.6, -68, -68, -68, -70.6]

ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'red', linewidth = 2.5)

fig.subplots_adjust(bottom=0.02)
# cbar_ax = fig.add_axes([0.13, 0.085, 0.8, 0.05]) #left, bottom, width, height
# cbar = fig.colorbar(scalar, orientation = 'horizontal', format = '%.0f', location = "bottom", cax = cbar_ax)
# cbar.ax.tick_params(labelsize=20) 
# cbar.set_label(label = "Depth [m]", fontsize = 20)
title_set(ax, "Depth bins of bathymetry")
fig.tight_layout()
plt.show()
fig.savefig(plot_path + 'area_binned.png',  bbox_inches='tight')

#%%APPLAY AREA MASK ON BATHYMETRY
chl = ds['chlor_a']

chl_sel = []
for i in range(0, 6574):
    a = chl[i,:,:] * depth_ones
    chl_sel.append(a)
    print(i)
    
chl_sel = np.array(chl_sel)
#%%POWER SPECTRUM PARAMETERS
#spectral resolution = sampling rate/tot num samples
sr = 1/86400 

# Perform Welch's periodogram
segment = 1800 #1800 = sesonal 
print(segment)
myhann = signal.get_window('hann', segment) #overlapping window

# obtain simply Power (amplitude^2) 
myparams = dict(fs = sr, nperseg = segment, window = np.ones(segment), detrend ='linear',
                noverlap = segment/2, scaling = 'spectrum')

#%%#%%CYCLE TIME SERIE FOR EACH BATHYMETR

energy = []
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
     #applay mask bathymetry (slice) on chl already masked on the area
    for j in range(0, 6574):
         b = chl_sel[j,:,:] * sel_depth
         chl_bin.append(b)
     
    chl_bins = np.array(chl_bin)
                 
    chl_mean = np.nanmean(chl_bins, axis = (1,2))
    print(chl_mean.shape)
     #mask zeros
    chl_no_nan = np.nan_to_num(chl_mean, nan=0)
    print(chl_no_nan.shape)
     #mean ignoring nan
    chl_ma = np.ma.masked_equal(chl_no_nan, 0)
     
     #delete masked values => reduce array length
    chl_mask = chl_ma[~chl_ma.mask]
    print(chl_mask.shape)
     
    interp = interpolate.interp1d(np.arange(chl_mask.size), chl_mask , kind='nearest')
    chl_interp = interp(np.linspace(0, chl_mask.size-1, len(chl_mean)))
     
     #power spectrum
    freq, ps = signal.welch(x = chl_interp, **myparams)
     
     #!! first element ps, much more little than the others => deleted
    energy.append(ps[1:])
     
    dfreq = freq[1]
    print('Spectral resolution = %2.9f Hz'%dfreq)
 
#%%PLOT VERIFY

fig, ax = set_plot()
# levels = np.arange(50, 3010, 58)
depths1 = (50, 75, 100, 200, 500, 1000, 2500, 3000)
scalar = ax.contourf(lons, lats, sel_depth,  cmap = 'flag', alpha=.8, transform = ccrs.PlateCarree()) #20 levels of colors
lines = ax. contour(lons, lats, -depth, levels = depths1, colors='black', linewidths = .5)
ax.clabel(lines, inline=1, fontsize=15, colors = 'black')

y = [ 38.5, 39.8, 41, 41, 38.5, 38.5, 38.5] #lats
x = [ -70.6, -72.6, -72.6, -68, -68, -68, -70.6]

ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'red', linewidth = 2.5)
fig.subplots_adjust(bottom=0.02)
cbar_ax = fig.add_axes([0.1, 0.085, 0.8, 0.05]) #left, bottom, width, height
cbar = fig.colorbar(scalar, orientation = 'horizontal', format = '%.2f', location = "bottom", cax = cbar_ax)
cbar.ax.tick_params(labelsize=20) 
cbar.set_label(label = "Depth [m]", fontsize = 20)
title_set(ax, "Bathymetry binned")
#%%     SUMMARY 2D PLOT POWER SPECTRUM
tau = (1/freq[1:])/86400
log_energy = np.log10(energy)

#%% plot log x-axes 
#log field
energy = np.array(energy)
#%%
fig, ax = plt.subplots(1, 1, figsize = (20, 16))
levels = np.linspace(-5.8, -1.08, 12)
scalar = ax.contourf(np.abs(bins), tau, log_energy.T, levels = levels, extend = 'both', cmap = "Spectral_r")
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xticks([50, 75, 100, 200, 500, 1000, 1500, 2500])
ax.set_yticks([10, 30, 60, 120, 180, 365, 730])

ax.xaxis.set_tick_params(labelsize = 30)
ax.yaxis.set_tick_params(labelsize = 30)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

ax.set_xlabel('Depth [m]', fontsize = 30)
ax.set_ylabel('Period [Days]', fontsize = 30)
ax.grid(linewidth = .5, linestyle='--')

cbar = fig.colorbar(scalar, orientation = 'horizontal', location = 'bottom', pad = 0.1, format='%.3f')
cbar.ax.tick_params(labelsize = 30)
cbar.set_label( label = "Chlor-a concentration  [$(mg/m^{3})^2$]", fontsize = 30)

fig.suptitle("2D Power Spectrum Chlor-a logarithmic concentration", fontsize = 35, y = 1)
fig.tight_layout()
plt.show()
# fig.savefig(plot_path + "spectrum_2D_logarithmic.png", bbox_inches='tight', dpi = 500)
fig.savefig(plot_path + "spectrum_2D.png", bbox_inches='tight', dpi = 500)

#%%linear
fig, ax = plt.subplots(1, 1, figsize = (20, 16))
levels = np.linspace(np.min(energy), 0.07, 12)
scalar = ax.contourf(np.abs(bins), tau, energy.T, levels = levels, extend = 'max', cmap = "Spectral_r")
ax.set_xscale('log')
ax.set_yscale('log')

norm= Normalize(vmin=np.min(energy), vmax=0.07)
sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
sm.set_array([])

ax.set_xticks([50, 75, 100, 200, 500, 1000, 1500, 2500])
ax.set_yticks([10, 30, 60, 120, 180, 365, 730])

ax.xaxis.set_tick_params(labelsize = 30)
ax.yaxis.set_tick_params(labelsize = 30)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

ax.set_xlabel('Depth [m]', fontsize = 30)
ax.set_ylabel('Period [Days]', fontsize = 30)
ax.grid(linewidth = .5, linestyle='--')

cbar = fig.colorbar(sm, orientation = 'horizontal', ticks = levels, location = 'bottom', pad = 0.1, format='%.3f')
cbar.ax.tick_params(labelsize = 30)
cbar.set_label( label = "Chlor-a concentration  [$(mg/m^{3})^2$]", fontsize = 30)
cbar.ax.tick_params(which='minor', size=25, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=20, width=1, color='k', direction='in', labelsize = 25)

fig.suptitle("2D Power Spectrum Chlor-a concentration", fontsize = 35, y = 1)
fig.tight_layout()
plt.show()
fig.savefig(plot_path + "spectrum_2D_linear.png", bbox_inches='tight', dpi = 500)


#%%FILTERED ROLLING MEAN
energy = []
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
     #applay mask bathymetry (slice) on chl already masked on the area
    for j in range(0, 6574):
         b = chl_sel[j,:,:] * sel_depth
         chl_bin.append(b)
     
    chl_bins = np.array(chl_bin)
                 
    chl_mean = np.nanmean(chl_bins, axis = (1,2))
    
    print(chl_mean.shape)
     #mask zeros
    chl_no_nan = np.nan_to_num(chl_mean, nan=0)
    print(chl_no_nan.shape)
     #mean ignoring nanchl_interp
    chl_ma = np.ma.masked_equal(chl_no_nan, 0)
     
     #delete masked values => reduce array length
    chl_mask = chl_ma[~chl_ma.mask]
    print(chl_mask.shape)
     
    interp = interpolate.interp1d(np.arange(chl_mask.size), chl_mask , kind='nearest')
    chl_interp = interp(np.linspace(0, chl_mask.size-1, len(chl_mean)))
    
    chl_mean_ds = pd.DataFrame({'Chlor': chl_interp})

    run_mean = chl_mean_ds.rolling(60, center=True).mean()
    run_mean = run_mean.Chlor[~np.isnan(run_mean.Chlor)]
     
     #power spectrum
    freq, ps = signal.welch(x = run_mean, **myparams)
     
     #!! first element ps, much more little than the others => deleted
    energy.append(ps[1:])
     
    dfreq = freq[1]
    print('Spectral resolution = %2.9f Hz'%dfreq)
    
#%%FILTERED
energy = np.array(energy)

fig, ax = plt.subplots(1, 1, figsize = (20, 16))
levels = np.linspace(np.min(energy), 0.07, 12)
scalar = ax.contourf(np.abs(bins), tau, energy.T, levels = levels, extend = 'max', cmap = "Spectral_r")
ax.set_xscale('log')
ax.set_yscale('log')

norm= Normalize(vmin=np.min(energy), vmax=0.07)
sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
sm.set_array([])

ax.set_xticks([50, 75, 100, 200, 500, 1000, 1500, 2500])
ax.set_yticks([10, 30, 60, 120, 180, 365, 730])

ax.xaxis.set_tick_params(labelsize = 30)
ax.yaxis.set_tick_params(labelsize = 30)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

ax.set_xlabel('Depth [m]', fontsize = 30)
ax.set_ylabel('Period [Days]', fontsize = 30)
ax.grid(linewidth = .5, linestyle='--')

cbar = fig.colorbar(sm, orientation = 'horizontal', ticks = levels, location = 'bottom', pad = 0.1, format='%.3f')
cbar.ax.tick_params(labelsize = 30)
cbar.set_label( label = "Chlor-a concentration  [$(mg/m^{3})^2$]", fontsize = 30)
cbar.ax.tick_params(which='minor', size=25, width=1, color='k', direction='in')
cbar.ax.tick_params(which='major', size=20, width=1, color='k', direction='in', labelsize = 25)

fig.suptitle("2D Power Spectrum Chlor-a, filtered with rolling window = 60D ", fontsize = 35, y = 1)
fig.tight_layout()
plt.show()
fig.savefig(plot_path + "spectrum_2D_filtered.png", bbox_inches='tight', dpi = 500)
