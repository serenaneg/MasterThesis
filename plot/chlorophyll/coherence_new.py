#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:02:41 2024

@author: serena
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from scipy import signal 
from matplotlib.path import Path
from scipy import interpolate
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import t
#%%FUNTIONS
def mean_values_list(variable):
            
    #create a new coodinates = day count
    # Calculate the start date
    variable['time'] = pd.to_datetime(variable['time'], format='%Y-%m-%d')
    start_date = variable['time'].min()
    
    # Calculate the time difference in days and store it in a new column
    variable['day_count'] = (variable['time'] - start_date).dt.days
    
    variable['time'] = variable['day_count']
       
    #daily mean of chl values => 1 value per day
    variable_mean = variable.mean(axis = (1, 2), skipna=True) #for chl
 
    #remouve nan
    variable_mean = variable_mean.where(~np.isnan(variable_mean), 0)
    variable_list = variable_mean.values.tolist()
    
    return variable.day_count, variable_list

#%% LOAD DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/CMEMS/"
path_chl = "/home/serena/Scrivania/Magistrale/thesis/data/MODIS_INTERPOLATED_DATA/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/spectral_analysis/"

#monthly data
ds = xr.open_dataset(path +"EKMAN.nc") 
df = xr.open_dataset(path_chl + "MODIS_chl_1D_nomissing.nc")

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

#%%POLYGON MASK for chlorohyll and ekman
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

chl_fin = chl_interp - np.mean(chl_interp)

#%%EKMAN
My = []
w = []
ui = []
for i in range(0, 6575):
    c = ds['My_ek'][i,:,:] * depth_ones
    My.append(c)
    d = ds['w_ek'][i,:,:] * depth_ones
    w.append(d)
    e = ds['ui'][i,:,:] * depth_ones
    ui.append(e)
    print(i)

My_coords = xr.DataArray(My, dims=('time', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': ds.w_ek.time})
w_coords = xr.DataArray(w, dims=('time', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': ds.w_ek.time})
ui_coords = xr.DataArray(ui, dims=('time', 'latitude', 'longitude'),
                           coords={'longitude': lons.values, 'latitude': lats.values, 'time': ds.w_ek.time})

days, My_list = mean_values_list(My_coords)
days, w_list = mean_values_list(w_coords)
days, ui_list = mean_values_list(ui_coords)

my = np.array(My_list)
my_fin = my - np.mean(my)
ui = np.array(ui_list)
#%%%COHERENCE ANALYSIS => WELCH METHOD BETWEEN CHLOROPHYLL ANDMERIDONAL TRANSPORT
sr = 1/86400

#LOW FREQUENCIES
segment = 1643 #1800= sesonal -> variability less than 6 months, Length of each segmen
noverlap =  segment * 0.5
myparams = dict(fs = sr, nperseg = segment, window = 'hann', detrend ='linear',
                noverlap = noverlap, nfft = 2048) #2048


#power spectrum
freq, ps_chl_notscaled = signal.welch(chl_fin, **myparams, scaling = 'spectrum')
freq, ps_m_notscaled = signal.welch(my_fin[1:], **myparams, scaling = 'spectrum')

#calculate coherence and cross power spetrum with welch method
freq, coherence = signal.coherence(my_fin[1:], chl_fin, **myparams)
freq, cps = signal.csd(chl_fin, my_fin[1:], **myparams, scaling = 'spectrum')

freq = freq[1:]
tau = (1/freq)/86400

#calculate phase and gain
phase = np.angle(cps)
grad = np.rad2deg(phase)
gain = np.sqrt(cps.real**2 + cps.imag**2) #or gain = np.abs(cps)

#rescaling of the spectrum
ps_chl = np.square(8/3) * ps_chl_notscaled
ps_m = np.square(8/3) * ps_m_notscaled

#%%SIGNIFICANCE INTERVALS

from scipy.stats import chi2
 
probability = 0.95
alfa = 1 - probability

#coherence significance level for 95% probability
N = len(chl_fin)#*N/M but N = n. data point in the time series
M = segment/2 #half-width of the window in the time domain
EDof = 8/3 * N/M

cohe_level = 1 - alfa**(1/(EDof - 1))

print(cohe_level)
#P  is the number of estimates in welch function
#and also the degree of freedom.

# Calculate the critical values from the Chi-square distribution
v = (8/3 * N/M) / 2
c = chi2.ppf([1 - alfa / 2, alfa / 2], v)
c = v / c
Pxxc_lower = ps_chl[1:] * c[0]
Pxxc_upper = ps_chl[1:] * c[1]

Pxxc_lower_my = ps_m[1:] * c[0]
Pxxc_upper_my = ps_m[1:] * c[1]

#admittance
freq, cross = signal.csd(chl_fin, my_fin[1:], **myparams, scaling = 'spectrum', return_onesided=True)
adm = cross[1:]/ps_m[1:]

#%%PLOT
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,10))

ax1.plot(tau, ps_m[1:], lw = 1.3, color = "darkorange", label = "My_Ek")
ax1.fill_between(tau, Pxxc_lower_my, Pxxc_upper_my, color = "gold", alpha = 0.3, linestyle = '-',  label="95%")
ax1.plot(tau, ps_chl[1:], lw = 1.3, color = "seagreen", label = "Chl-a")
ax1.fill_between(tau, Pxxc_lower, Pxxc_upper, color = "lime", alpha = 0.3, linestyle = '-', label="95%")
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xticks([10, 30, 60, 120, 180, 365, 730])
ax1.invert_xaxis()
ax1.xaxis.set_tick_params(labelsize = 18)
ax1.yaxis.set_tick_params(labelsize = 18)
ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax1.set_facecolor("whitesmoke")
ax1.set_xlabel('Period [Days]', fontsize = 20)
ax1.set_ylabel('[Amplitude$^2$]', fontsize = 20)
ax1.set_title("Power Spectrum", fontsize = 25)
ax1.grid(lw = .5, linestyle='--')
ax1.legend(loc = 'center left', bbox_to_anchor=(0., 0.55), fontsize=20, ncol = 2)
ax1.set_xlim([np.max(tau), np.min(tau)])

ax2.plot(tau, coherence[1:], lw = 1.5, color = "royalblue")
ax2.set_xscale('log')
ax2.axhline(cohe_level, lw = 2, color = "red", linestyle = '--')
ax2.text(2080, cohe_level, s='95%', color = 'r', fontsize = 20, weight="bold")
ax2.set_xticks([10, 30, 60, 120, 180, 365, 730])
ax2.set_facecolor("whitesmoke")
ax2.xaxis.set_tick_params(labelsize = 18)
ax2.yaxis.set_tick_params(labelsize = 18)
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax2.invert_xaxis()
ax2.set_xlabel('Period [Days]', fontsize = 20)
ax2.set_ylabel('')
ax2.set_title("Coherence", fontsize = 25)
ax2.grid(lw=.5, linestyle = '--')
ax2.set_xlim([np.max(tau), np.min(tau)])

ax3.plot(tau, grad[1:], lw = 1.5, color = "royalblue", linestyle='-')
ax3.set_xscale('log')
ax3.set_facecolor("whitesmoke")
ax3.set_xticks([10, 30, 60, 120, 180, 365, 730])
ax3.set_yticks([-180, -90, 0, 90, 180])
ax3.invert_xaxis()
ax3.xaxis.set_tick_params(labelsize = 18)
ax3.yaxis.set_tick_params(labelsize = 18)
ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax3.set_xlim([np.max(tau), np.min(tau)])
ax3.set_xlabel('Period [Days]', fontsize = 20)
ax3.set_ylabel('Angle [Degree]', fontsize = 20)
ax3.set_title("Phase", fontsize = 25)
ax3.grid(lw=.5, linestyle = '--')

ax4.plot(tau, gain[1:], lw = 1.5, color = "royalblue")
ax4.set_xscale('log')
ax4.set_xticks([10, 30, 60, 120, 180, 365, 730])
ax4.set_facecolor("whitesmoke")
ax4.xaxis.set_tick_params(labelsize = 18)
ax4.yaxis.set_tick_params(labelsize = 18)
ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax4.invert_xaxis()
ax4.set_xlabel('Period [Days]', fontsize = 20)
ax4.set_ylabel('')
ax4.set_title("Gain", fontsize = 25)
ax4.grid(lw=.5, linestyle = '--')
ax4.set_xlim([np.max(tau), np.min(tau)])

fig.tight_layout()
fig.suptitle('Chlorophyll concentration vs Meridional Ekman transport', fontsize = 30, y = 1.05)
fig.savefig(plot_path + "coherence_LF", bbox_inches='tight')

#%%COHERENCE VS ADMITTANCE LOT
fig, ax2 = plt.subplots(figsize=(15,8))

ax2.plot(tau, coherence[1:], lw = 1.5, color = "royalblue", alpha = 0.5)
ax2.set_xscale('log')
ax2.axhline(cohe_level, lw = 2, color = "red", linestyle = '--')
ax2.text(2080, cohe_level, s='95%', color = 'r', fontsize = 20, weight="bold")
ax2.set_xticks([10, 30, 60, 120, 180, 365, 730])
ax2.set_facecolor("whitesmoke")
ax2.xaxis.set_tick_params(labelsize = 20)
ax2.yaxis.set_tick_params(labelsize = 20, labelcolor='royalblue')
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax2.invert_xaxis()
ax2.set_xlabel('Period [Days]', fontsize = 25)
ax2.set_ylabel('Coherence', color = 'royalblue', fontsize = 25)
ax2.set_title("Coherence and Admittance", fontsize = 25)
ax2.grid(lw=.5, linestyle = '--')

ax22 = ax2.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:purple'
ax22.set_ylabel('Admittance [mg/Sv m$^{3}$]', color=color, fontsize = 25)
ax22.plot(tau, adm, color=color, lw = 1.5)
ax22.tick_params(labelcolor=color, labelsize = 20)
ax22.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax22.yaxis.get_offset_text().set_fontsize(20)
fig.tight_layout()
fig.savefig(plot_path + "admittance_LF", bbox_inches='tight')
#%%
sr = 1/86400

#LOW FREQUENCIES
segment = 180 #1800= sesonal -> variability less than 6 months, Length of each segmen
myhann = signal.get_window('hamming', Nx = segment)
noverlap =  segment * 0.9
myparams = dict(fs = sr, nperseg = segment, window = myhann, detrend ='linear',
                noverlap = noverlap, nfft = 256)


#power spectrum
freq, ps_chl_notscaled = signal.welch(chl_fin, **myparams, scaling = 'spectrum')
freq, ps_m_notscaled = signal.welch(my_fin[1:], **myparams, scaling = 'spectrum')

#calculate coherence and cross power spetrum with welch method
freq, coherence = signal.coherence(my_fin[1:], chl_fin, **myparams)
freq, cps = signal.csd(chl_fin, my_fin[1:], **myparams, scaling = 'spectrum')

freq = freq[1:]
tau = (1/freq)/86400

#calculate phase and gain
phase = np.angle(cps)
grad = np.rad2deg(phase)
gain = np.sqrt(cps.real**2 + cps.imag**2) #or gain = np.abs(cps)

#rescaling of the spectrum
ps_chl = np.square(8/3) * ps_chl_notscaled
ps_m = np.square(8/3) * ps_m_notscaled
#%%
probability = 0.95
 
#P  is the number of estimates in welch function
#and also the degree of freedom.

# Calculate the critical values from the Chi-square distribution
alfa = 1 - probability
#coherence significance level for 95% probability
N = len(chl_fin) #*N/M but N = n. data point in the time series
M = segment/2 #half-width of the window in the time domain
EDof = 8/3 * N/M

cohe_level = 1 - alfa**(1/(EDof - 1))

print(cohe_level)
#P  is the number of estimates in welch function
#and also the degree of freedom.

# Calculate the critical values from the Chi-square distribution
v = (8/3 * N/M) / 2
c = chi2.ppf([1 - alfa / 2, alfa / 2], v)
c = v / c
Pxxc_lower = ps_chl[1:] * c[0]
Pxxc_upper = ps_chl[1:] * c[1]
print(np.min(Pxxc_lower), np.max(Pxxc_upper))

Pxxc_lower_my = ps_m[1:] * c[0]
Pxxc_upper_my = ps_m[1:] * c[1]
print(np.min(Pxxc_lower_my), np.max(Pxxc_upper_my))

#admittance
freq, cross = signal.csd(chl_fin, my_fin[1:], **myparams, scaling = 'spectrum', return_onesided=True)
adm = cross[1:]/ps_m[1:]

#%%PLOT
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,10))

ax1.plot(tau, ps_m[1:], lw = 1.3, color = "darkorange", label = "My_Ek")
ax1.fill_between(tau, Pxxc_lower_my, Pxxc_upper_my, color = "gold", alpha = 0.3, linestyle = '-',  label="95%")
ax1.plot(tau, ps_chl[1:], lw = 1.3, color = "seagreen", label = "Chl-a")
ax1.fill_between(tau, Pxxc_lower, Pxxc_upper, color = "lime", alpha = 0.3, linestyle = '-', label="95%")
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xticks([3, 4, 7, 10, 30, 60, 120, 180])
ax1.invert_xaxis()
ax1.set_xlim([180, 2])
ax1.xaxis.set_tick_params(labelsize = 18)
ax1.yaxis.set_tick_params(labelsize = 18)
ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax1.set_facecolor("whitesmoke")
ax1.set_xlabel('Period [Days]', fontsize = 20)
ax1.set_ylabel('[Amplitude$^2$]', fontsize = 20)
ax1.set_title("Power Spectrum", fontsize = 25)
ax1.grid(lw = .5, linestyle='--')
ax1.legend(loc = 'center left', bbox_to_anchor=(0., 0.55), fontsize=20, ncol = 2)

ax2.plot(tau, coherence[1:], lw = 1.5, color = "royalblue")
ax2.set_xscale('log')
ax2.axhline(cohe_level, lw = 2, color = "red", linestyle = '--')
ax2.set_xticks([3, 4, 7, 10, 30, 60, 120, 180])
ax2.text(150, cohe_level, s='95%', color = 'r', fontsize = 20, weight="bold")
ax2.set_facecolor("whitesmoke")
ax2.xaxis.set_tick_params(labelsize = 18)
ax2.yaxis.set_tick_params(labelsize = 18)
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax2.invert_xaxis()
ax2.set_xlim([180, 2])
ax2.set_xlabel('Period [Days]', fontsize = 20)
ax2.set_ylabel('')
ax2.set_title("Coherence", fontsize = 25)
ax2.grid(lw=.5, linestyle = '--')


ax3.plot(tau, grad[1:], lw = 0.5, color = "royalblue", linestyle='-', marker='o')
ax3.set_xscale('log')
ax3.set_facecolor("whitesmoke")
ax3.set_xticks([3, 4, 7, 10, 30, 60, 120, 180])
ax3.set_yticks([-180, -90, 0, 90, 180])
ax3.invert_xaxis()
ax3.xaxis.set_tick_params(labelsize = 18)
ax3.yaxis.set_tick_params(labelsize = 18)
ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax3.set_xlabel('Period [Days]', fontsize = 20)
ax3.set_ylabel('Angle [Degree]', fontsize = 20)
ax3.set_title("Phase", fontsize = 25)
ax3.set_xlim([180, 2])
ax3.grid(lw=.5, linestyle = '--')

ax4.plot(tau, gain[1:], lw = 1.5, color = "royalblue")
ax4.set_xscale('log')
ax4.set_xticks([3, 4, 7, 10, 30, 60, 120, 180])
ax4.set_facecolor("whitesmoke")
ax4.xaxis.set_tick_params(labelsize = 18)
ax4.yaxis.set_tick_params(labelsize = 18)
ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax4.invert_xaxis()
ax4.set_xlabel('Period [Days]', fontsize = 20)
ax4.set_ylabel('')
ax4.set_title("Gain", fontsize = 25)
ax4.set_xlim([180, 2])
ax4.grid(lw=.5, linestyle = '--')

fig.tight_layout()
fig.suptitle('Chlorophyll concentration vs Meridional Ekman transport', fontsize = 30, y = 1.05)
fig.savefig(plot_path + "coherence_HF.png", bbox_inches='tight')
#%%
fig, ax2 = plt.subplots(figsize=(15,8))

ax2.plot(tau, coherence[1:], lw = 1.5, color = "royalblue", alpha = 0.5 , marker = 'o')
ax2.set_xscale('log')
ax2.axhline(cohe_level, lw = 2, color = "red", linestyle = '--')
ax2.text(150, cohe_level, s='95%', color = 'r', fontsize = 20, weight="bold")
ax2.set_xticks([3, 4, 7, 10, 30, 60, 120, 180])
ax2.set_facecolor("whitesmoke")
ax2.xaxis.set_tick_params(labelsize = 20)
ax2.yaxis.set_tick_params(labelsize = 20, labelcolor='royalblue')
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax2.invert_xaxis()
ax2.set_xlabel('Period [Days]', fontsize = 25)
ax2.set_ylabel('Coherence', color = 'royalblue', fontsize = 25)
ax2.set_title("Coherence and Admittance", fontsize = 25)
ax2.grid(lw=.5, linestyle = '--')
ax2.set_xlim([180, 2])

ax22 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
ax22.set_xlim([180, 2])
color = 'tab:purple'
ax22.set_ylabel('Admittance [mg/Sv m$^{3}$]', color=color, fontsize = 25)
ax22.plot(tau, adm, color=color, lw = 1.5, marker = 'o')
ax22.tick_params(labelcolor=color, labelsize = 20)
ax22.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax22.yaxis.get_offset_text().set_fontsize(20)
fig.tight_layout()
fig.savefig(plot_path + "admittance_HF", bbox_inches='tight')

#%%CORRELATION COEFFICIENT
import scipy.stats as stats

corr, _ = stats.pearsonr(chl_interp, my[1:])
#CORRELATION BETWEEN TIMESRIES
# Pearson correlation = -0.02815 (linear)
# Spearman correlation = -0.09033 (monotonic)
# Kendall correlation = -0.06090 (monotonic)
 
corr, _ = stats.pearsonr(ps_chl, ps_m)
print('Pearson correlation = %1.5f'%corr)

spearman_corr, _ = stats.spearmanr(ps_chl, ps_m)
print('Spearman correlation = %1.5f'%spearman_corr)

kendall_corr, _ = stats.kendalltau(ps_chl, ps_m)
print('Kendall correlation = %1.5f'%kendall_corr)

#CORRELATION BETWEEN SPETRA
# Pearson correlation = 0.88352
# Spearman correlation = 0.71655
# Kendall correlation = 0.52066

#COVARIANCE
data = pd.DataFrame((chl_interp, my[1:]))
cov = np.cov(data)
print(cov)
#[[ 8.32396716e-01 -2.38731448e+01]
# [-2.38731448e+01  8.64108263e+05]] CORRELATION IS NEGATIVE

#%%INTEGRAL TIMESCALE

T_star = []
for i in range(0, 6573):
    summ = (chl_interp[i] + chl_interp[i+1])
    temp = 1/(2*np.var(chl_interp)) * summ
    T_star.append(temp)
    
T = np.sum(T_star)  #6529.326369986218 after that day the dataseries is decorrelated

