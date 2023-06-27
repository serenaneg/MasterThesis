#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:01:47 2023

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
#%%FUNTIONS
def mean_values_list(variable):
            
    #create a new coodinates = day count
    # Calculate the start date
    variable['time'] = pd.to_datetime(variable['time'], format='%Y-%m-%d')
    start_date = variable['time'].min()
    
    # Calculate the time difference in days and store it in a new column
    variable['day_count'] = (variable['time'] - start_date).dt.days
    
    days = variable['day_count'].values.tolist()
    variable['time'] = variable['day_count']
       
    #daily mean of chl values => 1 value per day
    variable_mean = variable.mean(axis = (1, 2), skipna=True) #for chl
 
    #remouve nan
    variable_mean = variable_mean.where(~np.isnan(variable_mean), 0)
    variable_list = variable_mean.values.tolist()
    
    return variable.day_count, variable_list

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

#%% LOAD DAILY ERA5 DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/CMEMS/"
path_chl = "/home/serena/Scrivania/Magistrale/thesis/data/MODIS_INTERPOLATED_DATA/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/spectral_analysis/"

#monthly data
ds = xr.open_dataset(path +"EKMAN.nc") 

df = xr.open_dataset(path_chl + "MODIS_chl_1D_nomissing.nc")
#%%
lat_range = slice(41., 38.) #alto-basso
lon_range = slice(-72.5, -68.)

#chlorophyll
ds = ds.sel(longitude=lon_range)
ds = ds.sel(latitude=lat_range)

lons, lats = ds['longitude'], ds['latitude']

#%%wind is coarser than clorophyll => chlorophill on wind
df = df.interp(lon = lons, lat = lats, method = "nearest")
#%% BATHYMETRY ALREADY INTERPOLATED ON CHL
#lat, lon, bathy
bathy = xr.open_dataset(path_chl + "../bathymetry.nc")
bathy = bathy.interp(lon = lons, lat = lats, method = "nearest")

depth = bathy['elevation'] 
depth = depth.sel(latitude = lats, longitude = lons)
#ONLY SHELF-BREAK!!!
depth_limits = depth.where((depth <= -70) & (depth >= -1100), drop = False)

#%%POLYGON MASK
poly_verts = [(-70.6, 38.5), (-72.6, 39.8), (-72.6, 41), (-68, 41), (-68, 38.5)]

# Create vertex coordinates for each grid cell...
lon2d, lat2d = np.meshgrid(lons, lats)
lon2d, lat2d = lon2d.flatten(), lat2d.flatten()

points = np.vstack((lon2d, lat2d)).T

path = Path(poly_verts)
grid = path.contains_points(points)
grid_bool = grid.reshape((13, 19)) #72 = lons length, 110 lat lenghts
#coverto boolean grid into 0-1 grid
grid_int = grid_bool*1

#%%MASK BATHYMETRY AND CHLOROPHYLL
depth_polyg = depth_limits * grid_int
#substitude 0 with nan beacuse depth_limits is a greater area then grid_int
depth_zeros = depth_polyg.where(depth_polyg != 0, np.nan)
depth_ones = depth_zeros.where(np.isnan(depth_zeros), 1)
#%%VARIABLE CHLOROPHYLL

chl = df['chlor_a']

#chl on mask
chl_sel = []
for i in range(0, 6574):
    a = chl[i,:,:] * depth_ones
    chl_sel.append(a)
    print(i)
    
chl_coords = xr.DataArray(chl_sel, dims=('time', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': chl.time})

#mean values
chl_days, chl_list = mean_values_list(chl_coords)

#mask zeros
chl_ma = np.ma.masked_equal(chl_list,0)
#delete masked array
chl_mask = chl_ma[~chl_ma.mask]

interp = interpolate.interp1d(np.arange(chl_mask.size), chl_mask , kind='nearest')
chl_interp = interp(np.linspace(0, chl_mask.size-1, len(chl_list)))


#%%EKMAN
#mean values
Mx = []
My = []
w = []
for i in range(0, 6575):
    b = ds['Mx_ek'][i,:,:] * depth_ones
    My.append(b)
    c = ds['My_ek'][i,:,:] * depth_ones
    Mx.append(c)
    d = ds['w_ek'][i,:,:] * depth_ones
    w.append(d)
    print(i)

Mx_coords = xr.DataArray(Mx, dims=('time', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': ds.w_ek.time})
My_coords = xr.DataArray(My, dims=('time', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': ds.w_ek.time})
w_coords = xr.DataArray(w, dims=('time', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': ds.w_ek.time})


days, Mx_list = mean_values_list(Mx_coords)
days, My_list = mean_values_list(My_coords)
days, w_list = mean_values_list(w_coords)
#%%
mx = np.array(Mx_list)
my = np.array(My_list)
transport = np.sqrt(mx*mx + my*my)

w_pos = np.array(w_list)
w_pos = w_pos[w_pos > 0]

#%%VERIFY
fig, ax = set_plot()
depths1 = (50, 75, 100)
depths2 = (200, 500, 1000, 2500, 3000)
scalar = ax.contourf(lons, lats, depth_ones,  cmap = 'jet', transform = ccrs.PlateCarree()) #20 levels of colors
lines = ax. contour(lons, lats, -depth, levels = depths1, colors='black', linewidths = .5)
lines2 = ax. contour(lons, lats, -depth, levels = depths2, colors='black', linewidths = .5)
ax.clabel(lines, inline=1, fontsize=15, colors = 'black')
ax.clabel(lines2, inline=1, fontsize=15, colors = 'black')

y = [ 38.5, 39.8, 41, 41, 38.5, 38.5, 38.5] #lats
x = [ -70.6, -72.6, -72.6, -68, -68, -68, -70.6]

ax.plot(x, y, transform = ccrs.PlateCarree(), color = 'red', linewidth = 2.5)
fig.subplots_adjust(bottom=0.02)
cbar_ax = fig.add_axes([0.13, 0.085, 0.8, 0.05]) #left, bottom, width, height
cbar = fig.colorbar(scalar, orientation = 'horizontal', format = '%.6f', location = "bottom", cax = cbar_ax)
cbar.ax.tick_params(labelsize=20) 
cbar.set_label(label = "Depth [m]", fontsize = 20)
fig.tight_layout()


#%%COHERENCE ANALYSIS => WELCH METHOD BETWEEN CHLOROPHYLL ANDMERIDONAL TRANSPORT
sr = 1/86400
segment = 1800 #1800 = sesonal -> variability less than 6 months
myhann = signal.get_window('hann', segment)
myparams = dict(fs = sr, nperseg = segment, window = np.ones(segment), detrend ='linear',
                noverlap = segment/2)

#power spectrum
freq, ps_chl = signal.welch(chl_interp, **myparams, scaling = 'spectrum')
freq, ps_m = signal.welch(my, **myparams, scaling = 'spectrum')

#calculate coherence and cross power spetrum with welch method
freq, coherence = signal.coherence(chl_interp, my, **myparams)
freq, cps = signal.csd(chl_interp, my, **myparams, scaling = 'spectrum')

freq = freq[1:]
tau = (1/freq)/86400

#calculate phase and gain
phase = np.angle(cps)
grad = np.rad2deg(phase)
gain = np.sqrt(cps.real**2 + cps.imag**2) #or gain = np.abs(cps)

#%%PLOT
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,10))

ax1.plot(tau, ps_chl[1:], lw = 1.3, color = "seagreen", label = "Chlor-a")
ax1.plot(tau, ps_m[1:], lw = 1.3, color = "deepskyblue", label = "My_Ek")
ax1.set_yscale('log')
ax1.set_xscale('log')

ax1.set_xticks([10, 30, 60, 120, 180, 365, 730])

ax1.xaxis.set_tick_params(labelsize = 18)
ax1.yaxis.set_tick_params(labelsize = 18)
ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax1.set_xlabel('Period [Days]', fontsize = 20)
ax1.set_ylabel('[Amplitude$^2$]', fontsize = 20)
ax1.set_title("Power Spectrum", fontsize = 25)
ax1.grid(lw = .5, linestyle='--')
ax1.legend(loc = 'center right', fontsize=16)

ax2.plot(tau, coherence[1:], lw = 1.5, color = "teal")
ax2.set_xscale('log')

ax2.set_xticks([10, 30, 60, 120, 180, 365, 730])

ax2.xaxis.set_tick_params(labelsize = 18)
ax2.yaxis.set_tick_params(labelsize = 18)
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax2.set_xlabel('Period [Days]', fontsize = 20)
ax2.set_ylabel('')
ax2.set_title("Coherence", fontsize = 25)
ax2.grid(lw=.5, linestyle = '--')

ax3.plot(tau, grad[1:], lw = 1.5, color = "teal")
ax3.set_xscale('log')

ax3.set_xticks([10, 30, 60, 120, 180, 365, 730])
ax3.set_yticks([-180, -90, 0, 90, 180])

ax3.xaxis.set_tick_params(labelsize = 18)
ax3.yaxis.set_tick_params(labelsize = 18)
ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax3.set_xlabel('Period [Days]', fontsize = 20)
ax3.set_ylabel('Angle [Degree]', fontsize = 20)
ax3.set_title("Phase", fontsize = 25)
ax3.grid(lw=.5, linestyle = '--')

ax4.plot(tau, gain[1:], lw = 1.5, color = "teal")
ax4.set_xscale('log')
ax4.set_xticks([10, 30, 60, 120, 180, 365, 730])

ax4.xaxis.set_tick_params(labelsize = 18)
ax4.yaxis.set_tick_params(labelsize = 18)
ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax4.set_xlabel('Period [Days]', fontsize = 20)
ax4.set_ylabel('')
ax4.set_title("Gain", fontsize = 25)

ax4.grid(lw=.5, linestyle = '--')

fig.tight_layout()
fig.suptitle('Chlorophyll concentration vs Meridional Ekman transport', fontsize = 30, y = 1.05)
# fig.suptitle('Chlorophyll concentration vs  Ekman transport', fontsize = 30, y = 1.05)
# fig.savefig(plot_path + "analysis_chl_ekman.png", bbox_inches='tight')
fig.savefig(plot_path + "analysis_chl_my_ekman.png", bbox_inches='tight')


#%%ROTAY SPECTRUM
# #convert wind into complex number = wind = u + iv
# wind_complex = u_list + v_list*1j

# #power spectrum
# freq_rotary, ps_chl_rotary = signal.welch(chl_list, **myparams, scaling = 'spectrum', return_onesided=False)
# freq_rotary, ps_complex = signal.welch(wind_complex, **myparams, scaling = 'spectrum', return_onesided=False) #return_oneside False because complex input

# #calculate coherence and cross power spetrum with welch method
# #only real part for wind spectrum
# freq, coherence_rotary = signal.coherence(chl_list, wind_complex.real, **myparams)
# freq, cps_rotary = signal.csd(chl_list, wind_complex.real, **myparams, scaling = 'spectrum')

# freq_rotary = freq_rotary[1:]
# tau_rotary = (1/freq_rotary)/86400

# #calculate phase and gain
# phase_rotary = np.angle(cps_rotary)
# gain_rotary = np.sqrt(cps_rotary.real**2 + cps_rotary.imag**2) #or np.abs(cps)

#%%COHERENCE ANALYSIS => WELCH METHOD BETWEEN CHLOROPHYLL AND VERTICAL VELOCITY
sr = 1/86400
segment = 1800 #1800 = sesonal -> variability less than 6 months
myhann = signal.get_window('hann', segment)
myparams = dict(fs = sr, nperseg = segment, window = np.ones(segment), detrend ='linear',
                noverlap = segment/2)

#power spectrum
freq, ps_w = signal.welch(w_pos, **myparams, scaling = 'spectrum')

#calculate coherence and cross power spetrum with welch method
freq, coherence = signal.coherence(chl_interp, w_list, **myparams)
freq, cps = signal.csd(chl_interp, w_pos, **myparams, scaling = 'spectrum')

freq = freq[1:]
tau = (1/freq)/86400

#calculate phase and gain
phase = np.angle(cps)
grad = np.rad2deg(phase)
gain = np.sqrt(cps.real**2 + cps.imag**2) #or gain = np.abs(cps)

#%%PLOT
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,10))

ax1.plot(tau, ps_chl[1:], lw = 1.3, color = "seagreen", label = "Chlor-a")
ax1.plot(tau, ps_w[1:], lw = 1.3, color = "darkorange", label = "w_Ek")
ax1.set_yscale('log')
ax1.set_xscale('log')

ax1.set_xticks([10, 30, 60, 120, 180, 365, 730])

ax1.xaxis.set_tick_params(labelsize = 18)
ax1.yaxis.set_tick_params(labelsize = 18)
ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax1.set_xlabel('Period [Days]', fontsize = 20, labelpad = 15)
ax1.set_ylabel('[Amplitude$^2$]', fontsize = 20)
ax1.set_title("Power Spectrum", fontsize = 25)
ax1.grid(lw = .5, linestyle='--')
ax1.legend(loc = 'center right', fontsize=16)

ax2.plot(tau, coherence[1:], lw = 1.5, color = "teal")
ax2.set_xscale('log')

ax2.set_xticks([10, 30, 60, 120, 180, 365, 730])

ax2.xaxis.set_tick_params(labelsize = 18)
ax2.yaxis.set_tick_params(labelsize = 18)
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax2.vlines(8, np.nanmin(coherence[1:]), np.nanmax(coherence[1:]), color='red', linewidth = 1, linestyle = '--')
ax2.vlines(135, np.nanmin(coherence[1:]), np.nanmax(coherence[1:]), color='red', linewidth = 1, linestyle = '--')
ax2.vlines(364, np.nanmin(coherence[1:]), np.nanmax(coherence[1:]), color='red', linewidth = 1, linestyle = '--')

ax2.set_xlabel('Period [Days]', fontsize = 20, labelpad = 15)
ax2.set_ylabel('')
ax2.set_title("Coherence", fontsize = 25)
ax2.grid(lw=.5, linestyle = '--')

ax3.plot(tau, grad[1:], lw = 1.5, color = "teal")
ax3.set_xscale('log')

ax3.set_xticks([10, 30, 60, 120, 180, 365, 730])
ax3.set_yticks([-180, -90, 0, 90, 180])


ax3.xaxis.set_tick_params(labelsize = 18)
ax3.yaxis.set_tick_params(labelsize = 18)
ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax3.set_xlabel('Period [Days]', fontsize = 20)
ax3.set_ylabel('Angle [Degree]', fontsize = 20)
ax3.set_title("Phase", fontsize = 25)
ax3.grid(lw=.5, linestyle = '--')

ax4.plot(tau, gain[1:], lw = 1.5, color = "teal")
ax4.set_xscale('log')
ax4.set_xticks([10, 30, 60, 120, 180, 365, 730])
ax4.vlines(8, np.nanmin(gain[1:]), np.nanmax(gain[1:]), color='red', linewidth = 1, linestyle = '--')
ax4.vlines(135, np.nanmin(gain[1:]), np.nanmax(gain[1:]), color='red', linewidth = 1, linestyle = '--')
ax4.vlines(364, np.nanmin(gain[1:]), np.nanmax(gain[1:]), color='red', linewidth = 1, linestyle = '--')

ax4.xaxis.set_tick_params(labelsize = 18)
ax4.yaxis.set_tick_params(labelsize = 18)
ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax4.set_xlabel('Period [Days]', fontsize = 20)
ax4.set_ylabel('')
ax4.set_title("Gain", fontsize = 25)

ax4.grid(lw=.5, linestyle = '--')

fig.tight_layout(pad = 2.0)
fig.suptitle('Chlorophyll concentration vs Ekman pumping', fontsize = 30, y = 1.05)
fig.savefig(plot_path + "analysis_chl_w_ekman.png", bbox_inches='tight')
#%%TIMESERIES
My_ds = pd.DataFrame({'My_ek': My_list})

run_mean = My_ds.rolling(20, center=True).mean()

fig2, bx = plt.subplots(1, 1, figsize = (42, 15))
bx.set_title("Time series of Ekman transport (01-01-2003 / 31-12-2020)", fontsize = 40, y = 1.05)
bx.plot(days, my, "-", color = "deepskyblue", linewidth=1.5, label = 'All data') 
bx.plot(days, run_mean, "-", color = "darkred", linewidth=1.5, label = 'Rolling mean (20D)') 
bx.set_xlim(0,6600)
bx.set_ylabel('My_ek [m$^2$/s]', fontsize = 40)
bx.set_xlabel('Period [Days]', fontsize= 40)
bx.yaxis.set_major_formatter(ScalarFormatter())
bx.xaxis.set_tick_params(labelsize = 35)
bx.yaxis.set_tick_params(labelsize = 35)
bx.legend(loc='lower right', fontsize = 35)

bx.vlines(x=334, ymin = min(My_list), ymax = max(My_list), colors='saddlebrown', lw=3, linestyle = '--')
bx.vlines(x=120, ymin = min(My_list), ymax = max(My_list), colors='deeppink', lw=3, linestyle = '--')
bx.vlines(x=699, ymin = min(My_list), ymax = max(My_list), colors='saddlebrown', lw=3, linestyle = '--')
bx.vlines(x=485, ymin = min(My_list), ymax = max(My_list), colors='deeppink', lw=3, linestyle = '--')


bx.grid(linewidth = .5, linestyle='--')

fig2.tight_layout()
plt.show()
fig2.savefig(plot_path + "timeseries_ek.png", bbox_inches='tight', dpi = 400)
