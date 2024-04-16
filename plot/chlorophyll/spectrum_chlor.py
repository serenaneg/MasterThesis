#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:07:49 2023

@author: serena
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal 
from matplotlib.path import Path
from scipy import interpolate
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.2f"  # Give format here
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
lon_stat = np.array([-70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83, -70.83])
lat_stat = np.array([40.475, 40.4, 40.3361, 40.2722, 40.2083, 40.1444, 40.0806, 40.0167, 
                    39.9528, 39.8889, 39.825, 39.75, 39.6861, 39.6222])
fig, ax = set_plot()

depths1 = (50, 75, 200, 500, 1000, 2500, 3000)
scalar = ax.contourf(lons, lats, -depth_zeros, levels = range_1, cmap = mycmap, transform = ccrs.PlateCarree(), zorder=-1) 
scalar = ax.contourf(lons, lats, -depth_zeros, levels = range_2, cmap = mycmap_1,  transform = ccrs.PlateCarree(), zorder=-1) #20 levels of colors
scalar = ax.contourf(lons, lats, -depth_zeros, levels = range_3, cmap = mycmap_2, transform = ccrs.PlateCarree(), zorder=-1) 
scalar = ax.contourf(lons, lats, -depth_zeros, levels = range_4, cmap = mycmap_1, transform = ccrs.PlateCarree(), zorder=-1) #20 levels of colors
scalar = ax.contourf(lons, lats, -depth_zeros, levels = range_5, cmap = mycmap_3, transform = ccrs.PlateCarree(), zorder=-1) #20 levels of colors
#20 levels of colors
lines = ax. contour(lons, lats, -depth, levels = depths1, colors='black', linewidths = .5)
ax.clabel(lines, inline=1, fontsize=20, colors = 'black')
lines2 = ax. contour(lons, lats, -depth, levels = [100], colors='black', linewidths = 2)
ax.clabel(lines2, inline=1, fontsize=20, colors = 'black')
ax.scatter(lon_stat, lat_stat, s = 30, marker = 'D',  color = 'b', zorder = 2)
ax.text(lon_stat[0]-0.22, lat_stat[0], s = 'A5', color = 'b', fontsize = 20)
ax.text(lon_stat[13]-0.28, lat_stat[13]-0.10, s = 'A18', color = 'b',  fontsize = 20)

# ax.plot(x_perp, y_perp, transform = ccrs.PlateCarree(), color = 'blue', linewidth = 2.5)

y = [ 38.5, 39.8, 41, 41, 38.5, 38.5, 38.5] #lats
x = [ -70.6, -72.6, -72.6, -68, -68, -68, -70.6]

ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'b', linewidth = 2.5)

fig.subplots_adjust(bottom=0.02)
# cbar_ax = fig.add_axes([0.13, 0.085, 0.8, 0.05]) #left, bottom, width, height
# cbar = fig.colorbar(scalar, orientation = 'horizontal', format = '%.0f', location = "bottom", cax = cbar_ax)
# cbar.ax.tick_params(labelsize=20) 
# cbar.set_label(label = "Depth [m]", fontsize = 20)
title_set(ax, "Depth bins of bathymetry")
fig.tight_layout()
plt.show()
fig.savefig(plot_path + 'area_binned.png',  bbox_inches='tight', dpi = 300)
#%%APPLAY AREA MASK ON BATHYMETRY
chlor = ds['chlor_a']
chlorophyll = chlor.assign_coords(day_of_year = chlor.time.dt.strftime("%d-%m"))
chl_anomalies = ((chlorophyll.groupby("day_of_year") - chlorophyll.groupby("day_of_year").mean("time")).groupby("day_of_year")) / chlorophyll.groupby("day_of_year").std("time")

chl_mask = []
anomalies_mask = []
for i in range(0, 6574):
    a = chlor[i,:,:] * depth_ones
    chl_mask.append(a)
    
    b = chl_anomalies[i, : :] * depth_ones
    anomalies_mask.append(b)
    print(i)
    
chl_mask = np.array(chl_mask)
anomalies_mask = np.array(anomalies_mask)
#%%POWER SPECTRUM PARAMETERS
#spectral resolution = sampling rate/tot num samples
sr = 1/86400 

# Perform Welch's periodogram
segment = 1643 #1800 = sesonal  #365 annual
myhann = signal.get_window('hann', segment) #overlapping window

# obtain simply Power (amplitude^2) 
myparams = dict(fs = sr, nperseg = segment, window = np.ones(segment), detrend ='linear',
                noverlap = segment/2, scaling = 'spectrum', nfft = 2048)

#%%#%%CYCLE TIME SERIE FOR EACH BATHYMETR

energy_chl = []
energy_anomalies = []
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
    anomalies_bin = []
     #applay mask bathymetry (slice) on chl already masked on the area
    for j in range(0, 6574):
         bb = chl_mask[j,:,:] * sel_depth
         chl_bin.append(bb)
         
         cc = anomalies_mask[j, :, :] *sel_depth
         anomalies_bin.append(cc)
         
    anomalies_bins = np.array(anomalies_bin)
    chl_bins = np.array(chl_bin)
                 
    chl_mean = np.nanmean(chl_bins, axis = (1,2))
    anomalies_mean = np.nanmean(anomalies_bins, axis = (1,2))

     #mask zeros
    chl_no_nan = np.nan_to_num(chl_mean, nan=0)
    anomalies_no_nan = np.nan_to_num(anomalies_mean, nan=0)
    print(anomalies_no_nan.shape)
     #mean ignoring nan
    chl_ma = np.ma.masked_equal(chl_no_nan, 0)
    anomalies_ma = np.ma.masked_equal(anomalies_no_nan, 0)
 
     #delete masked values => reduce array length
    chl_masked = chl_ma[~chl_ma.mask]
    anomalies_masked = anomalies_ma[~anomalies_ma.mask]
    print(anomalies_masked.shape)
  
    interp = interpolate.interp1d(np.arange(chl_masked.size), chl_masked , kind='nearest')
    chl_interp = interp(np.linspace(0, chl_masked.size-1, len(chl_mean)))
    
    interp2 = interpolate.interp1d(np.arange(anomalies_masked.size), anomalies_masked , kind='nearest')
    anomalies_interp = interp(np.linspace(0, anomalies_masked.size-1, len(anomalies_mean)))
     
     #power spectrum
    freq, ps_chl = signal.welch(x = chl_interp, **myparams)
    freq_anomalies, ps_anomalies = signal.welch(x = anomalies_interp, **myparams)
     
     #!! first element ps, much more little than the others => deleted
    energy_chl.append(ps_chl[1:])
    energy_anomalies.append(ps_anomalies[1:])
    
    dfreq = freq_anomalies[1]
    print('Spectral resolution = %2.9f Hz'%dfreq)
#%%
energy_chl = np.array(energy_chl)
energy_anomalies = np.array(energy_anomalies)
tau = (1/freq[1:])/86400
#%%linear
fig, (ax, bx) = plt.subplots(2, 1, gridspec_kw={'height_ratios' : [3,1]}, figsize=(20,25))

levels = np.linspace(0.000, 0.01, 100) #seasonl 180
#levels = np.linspace(0, 0.01, 100)
scalar = ax.contourf(np.abs(bins), tau, energy_chl.T, levels = levels, extend = 'max', cmap = "nipy_spectral")
ax.set_xscale('log')
ax.set_yscale('log')

ax.vlines(100, np.min(tau), np.max(tau), color = 'white', linewidths = 3)
ax.vlines(1000, np.min(tau), np.max(tau), color = 'white', linewidths = 3)

ax.set_xticks([50, 75, 100, 200, 500, 1000, 1500, 2000])
#ax.set_yticks([10, 30, 60, 120, 180])
#ax.set_yticks([10, 30, 60, 120, 180, 365])
ax.set_yticks([10, 30, 60, 120, 180, 365, 730])
ax.set_xlim([70, 2300])
ax.xaxis.set_tick_params(labelsize = 0)
ax.yaxis.set_tick_params(labelsize = 40)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.grid(linewidth = .5, linestyle='--')
fmt = ScalarFormatter(useMathText=True)
cbar = fig.colorbar(scalar, orientation = 'vertical', location = 'right',  format=fmt,
                    aspect=20)
cbar.formatter.set_powerlimits((0,0))
cbar.ax.yaxis.get_offset_text().set_fontsize(40)
cbar.ax.tick_params(labelsize = 40)
#cbar.set_label( label = "Chl-a concentration  [$(mg/m^{3})^2$]", fontsize = 35)

small= bx.contourf(np.abs(bins), tau, energy_chl.T, levels = np.linspace(0.000, 0.005, 100),
                   extend = 'max', cmap = "nipy_spectral")
bx.set_xscale('log')
bx.set_yscale('log')
bx.vlines(100, np.min(tau), np.max(tau), color = 'white', linewidths = 3.5)
bx.vlines(1000, np.min(tau), np.max(tau), color = 'white', linewidths = 3.5)
bx.set_xlabel('Depth [m]', fontsize = 40)
bx.set_xticks([50, 75, 100, 200, 500, 1000, 2000])
bx.set_yticks([5, 10, 15, 20])
bx.set_xlim([70, 2300])
bx.set_ylim([0, 25])
bx.xaxis.set_tick_params(labelsize = 38)
bx.yaxis.set_tick_params(labelsize = 40)
bx.xaxis.set_major_formatter(ticker.ScalarFormatter())
bx.yaxis.set_major_formatter(ticker.ScalarFormatter())
bx.set_xlabel('Depth [m]', fontsize = 45)
bx.grid(linewidth = .5, linestyle='--')

fmt = ScalarFormatterForceFormat()
fmt.set_useMathText(True)
cbar = fig.colorbar(small, orientation = 'vertical', location = 'right', 
                    format=fmt, aspect=7)
cbar.formatter.set_powerlimits((0,0))
cbar.ax.yaxis.get_offset_text().set_fontsize(40)
cbar.ax.tick_params(labelsize = 40)
cbar.ax.locator_params(axis='y', nbins=6)

fig.supylabel('Period [Days]', fontsize = 45)
fig.suptitle(r"2D Power Spectrum surface Chlorophyll-$\alpha$", fontsize = 45, y = 1)
fig.text(0.99, 0.5, 'Chlorophyll [(mg/m$^{3}$)$^2$]', ha='center', va='center', rotation='vertical', fontsize = 40)
fig.tight_layout()
plt.show()
fig.savefig(plot_path + "spectrum_2D_seasonal.png", bbox_inches='tight', dpi = 300)
#%%
fig, (ax, bx) = plt.subplots(2, 1, gridspec_kw={'height_ratios' : [3,1]}, figsize=(20,25))

levels = np.linspace(0.000, 0.01, 100) #seasonl 180
#levels = np.linspace(0, 0.01, 100)
scalar = ax.contourf(np.abs(bins), tau, energy_anomalies.T, levels = levels, extend = 'max', cmap = "nipy_spectral")
ax.set_xscale('log')
ax.set_yscale('log')

ax.vlines(100, np.min(tau), np.max(tau), color = 'white', linewidths = 3)
ax.vlines(1000, np.min(tau), np.max(tau), color = 'white', linewidths = 3)

ax.set_xticks([50, 75, 100, 200, 500, 1000,  2000])
#ax.set_yticks([10, 30, 60, 120, 180])
#ax.set_yticks([10, 30, 60, 120, 180, 365])
ax.set_yticks([10, 30, 60, 120, 180, 365, 730])
ax.set_xlim([70, 2300])
ax.xaxis.set_tick_params(labelsize = 0)
ax.yaxis.set_tick_params(labelsize = 40)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.grid(linewidth = .5, linestyle='--')
fmt = ScalarFormatter(useMathText=True)
cbar = fig.colorbar(scalar, orientation = 'vertical', location = 'right',  format=fmt,
                    aspect=20)
cbar.formatter.set_powerlimits((0,0))
cbar.ax.yaxis.get_offset_text().set_fontsize(40)
cbar.ax.tick_params(labelsize = 40)
#cbar.set_label( label = "Chl-a concentration  [$(mg/m^{3})^2$]", fontsize = 35)

small= bx.contourf(np.abs(bins), tau, energy_anomalies.T, levels = np.linspace(0.000, 0.005, 100),
                   extend = 'max', cmap = "nipy_spectral")
bx.set_xscale('log')
bx.set_yscale('log')
bx.vlines(100, np.min(tau), np.max(tau), color = 'white', linewidths = 3.5)
bx.vlines(1000, np.min(tau), np.max(tau), color = 'white', linewidths = 3.5)
bx.set_xlabel('Depth [m]', fontsize = 40)
bx.set_xticks([50, 75, 100, 200, 500, 1000, 2000])
bx.set_yticks([5, 10, 15, 20])
bx.set_xlim([70, 2300])
bx.set_ylim([0, 25])
bx.xaxis.set_tick_params(labelsize = 38)
bx.yaxis.set_tick_params(labelsize = 40)
bx.xaxis.set_major_formatter(ticker.ScalarFormatter())
bx.yaxis.set_major_formatter(ticker.ScalarFormatter())
bx.set_xlabel('Depth [m]', fontsize = 45)
bx.grid(linewidth = .5, linestyle='--')

fmt = ScalarFormatterForceFormat()
fmt.set_useMathText(True)
cbar = fig.colorbar(small, orientation = 'vertical', location = 'right', 
                    format=fmt, aspect=7)
cbar.formatter.set_powerlimits((0,0))
cbar.ax.yaxis.get_offset_text().set_fontsize(40)
cbar.ax.tick_params(labelsize = 40)
cbar.ax.locator_params(axis='y', nbins=6)

fig.supylabel('Period [Days]', fontsize = 45)
fig.suptitle(r"2D Power Spectrum surface Chlorophyll-$\alpha$ anomalies", fontsize = 45, y = 1)
fig.text(0.99, 0.5, 'Chlorophyll [mg m$^{-3}$]', ha='center', va='center', rotation='vertical', fontsize = 40)
fig.tight_layout()
plt.show()
fig.savefig(plot_path + "spectrum_2D_anomalies.png", bbox_inches='tight', dpi = 100)
#%%