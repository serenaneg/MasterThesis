#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 08:44:53 2023

@author: serena
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean
from scipy import interpolate
import scipy.stats as ss


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface
    using the Haversine formula.
    """
    R = 6371  # Radius of the Earth in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate the differences between the latitudes and longitudes
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Apply the Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


def filter_values(array, lower_bound, upper_bound):
    filtered_values = array[np.logical_and(array[:, 0] >= lower_bound, array[:, 0] <= upper_bound)]
    return filtered_values

def my_interp(depth, variable):
    f = interpolate.interp1d(depth, variable)

    newx = np.linspace(np.min(depth), np.max(depth), len(depth)) #new depth
    ynew = f(newx)
    
    return ynew

def set_cbar(fig, c, title, ax):   
    # skip = (slice(None, None, 2))
    cbar = plt.colorbar(c, format='%.1f', spacing='proportional', ax=ax, shrink=0.9, pad = 0.01,
                        orientation = 'vertical', location = "right")
    cbar.set_label(label = title, fontsize = 25, y = 0.5, labelpad = 30)
    cbar.ax.tick_params(which='minor', size=5, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=10, width=1, color='k', direction='in', labelsize = 25)
    # fig.subplots_adjust(bottom=0.25)
    # cbar.ax.set_position([0.2, 0.08, 0.6, 0.08])
    return cbar
#%%
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2019/"

file = path + 'bottle_tn368_hydro.csv'
bottle = pd.read_csv(file)

file2 = path + 'ctd_tn_withlocation.csv'
ctd = pd.read_csv(file2)
#%%choose 14th july TRANSECT 37
# Parse datetime column
bottle['day'] = pd.to_datetime(bottle['day'])
ctd['day'] = pd.to_datetime(ctd['day'])
# Filter rows where day is 14
tn37 = bottle[(bottle['day'].dt.day == 14)]

ctd37 = ctd[ctd['day'].dt.day == 14]
#%%
lat = np.array(tn37.latitude, dtype = 'float')
lon = np.array(tn37.longitude, dtype = 'float')

#%%
sigma = np.array(tn37.density)

#format densiry values to only 1 decimal
sigma_1f = ["{:.1f}".format(x) for x in sigma]
sigma = np.array(np.column_stack((sigma_1f, lat, tn37.depth,  tn37.no3, tn37.chlor, tn37.fluor, tn37.PON)), dtype = 'float')

#nitrate along 26.0 isopicnal
iso25 = filter_values(sigma, 25.8, 25.8)
iso26 = filter_values(sigma, 26, 26)

#%%SAME FOR CTD
ctd_sig = np.array(ctd37.density)

ctd_sigma_1f = ["{:.1f}".format(x) for x in ctd_sig]
ctd_sigma = np.array(np.column_stack((ctd_sigma_1f, ctd37.lat, ctd37.depth, ctd37.fluorecence)), dtype = 'float')

#only 25.8 and 26
ctd25 = filter_values(ctd_sigma, 25.8, 25.8)
ctd26 = filter_values(ctd_sigma, 26, 26)
#%%INTERP BOTTLE DATA
#calculate the interpolated depth fotr the ispycnals
depth25 = iso25[:, 2]
f = interpolate.interp1d(depth25,  iso25[:,2])
newx = np.linspace(np.min(depth25), np.max(depth25), len(depth25)) #new depth
depth25 = f(newx)

depth26 = iso26[:, 2]
f2 = interpolate.interp1d(depth26,  iso26[:,2])
newx2 = np.linspace(np.min(depth26), np.max(depth26), len(depth26)) #new depth
depth26 = f2(newx2)

#use the inteprolated depth to interpolate nutrients
#NITRATE
nit25 = my_interp(depth25, iso25[:, 3])
nit26 = my_interp(depth26, iso26[:, 3])

#INTERP CHL
#chl may contain nan, delet
pon25 = my_interp(depth25, iso25[:, 6])
pon26 = my_interp(depth26, iso26[:, 6])

#FLUOR
flur25 = my_interp(depth25, iso25[:, 5])
flur26 = my_interp(depth26, iso26[:, 5])


#%%
fig, (ax3, ax2, ax1, ax) = plt.subplots(4,1, dpi = 200, figsize = (8,14))

ax.plot(iso25[:,1], depth25, '-o', color = 'lightskyblue', label = '25.8 isopycnal')
ax.plot(iso26[:,1], depth26, '-o', color = 'darkslateblue', label = '26.0 isopycnal')
ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.invert_yaxis()
ax.invert_xaxis()
ax.xaxis.set_tick_params(labelsize = 15)
ax.yaxis.set_tick_params(labelsize = 15) 
ax.set_facecolor("whitesmoke")
ax.set_xlabel('Latitude [$^\circ$N]', fontsize = 20)
ax.set_ylabel('Depth [m]', fontsize = 20)
ax.legend(loc='lower right', fontsize = 18)

ax1.plot(iso25[:,1], nit25, '-o', color = 'lightskyblue', label = '25.8 isopycnal')
ax1.plot(iso26[:,1], nit26, '-o', color = 'darkslateblue', label = '26.0 isopycnal')
ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax1.invert_xaxis()
ax1.set_facecolor("whitesmoke")
ax1.xaxis.set_tick_params(labelsize = 15)
ax1.yaxis.set_tick_params(labelsize = 15) 
ax1.set_ylabel('Nitrate [mmol/m$^3$]', fontsize = 20)

ax2.plot(iso25[:,1], pon25, '-o', color = 'lightskyblue', label = '25.8 isopycnal')
ax2.plot(iso26[:,1], pon26, '-o', color = 'darkslateblue', label = '26.0 isopycnal')
ax2.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax2.xaxis.set_tick_params(labelsize = 15)
ax2.yaxis.set_tick_params(labelsize = 15) 
ax2.set_facecolor("whitesmoke")
ax2.set_ylabel('PON [$\mu$mol/m$^3$]', fontsize = 20)
ax2.set_xlim(np.max(iso26[:,1])+0.167, np.min(iso26[:,1])-0.035)

ax3.plot(iso25[:,1], flur25, '-o', color = 'lightskyblue', label = '25.8 isopycnal')
ax3.plot(iso26[:,1], flur26, '-o', color = 'darkslateblue', label = '26.0 isopycnal')
ax3.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax3.xaxis.set_tick_params(labelsize = 15)
ax3.yaxis.set_tick_params(labelsize = 15) 
ax3.set_facecolor("whitesmoke")
ax3.set_ylabel('Chlor-a [mg/m$^3$]', fontsize = 20)
ax3.set_xlim(np.max(iso26[:,1])+0.167, np.min(iso26[:,1])-0.035)

fig.tight_layout() 
fig.savefig(plot_path + 'nitrate_profile_interpol.png')

#%%INTERP CTD DATA
depth25_ctd = ctd25[:, 2]
f = interpolate.interp1d(depth25_ctd,  ctd25[:,0])
newx = np.linspace(np.min(depth25_ctd), np.max(depth25_ctd), len(depth25_ctd)) #new depth
depth25_ctd = f(newx)

depth26_ctd = ctd26[:, 2]
f2 = interpolate.interp1d(depth26_ctd,  ctd26[:,0])
newx2 = np.linspace(np.min(depth26_ctd), np.max(depth26_ctd), len(depth26_ctd)) #new depth
depth26_ctd = f2(newx2)

chl25 = my_interp(depth25_ctd, ctd25[:, 3])
chl26 = my_interp(depth26_ctd, ctd26[:, 3])

#%%
fig, (ax3, ax) = plt.subplots(2,1, dpi = 200, figsize = (8,10))

ax.plot(ctd25[:,1], depth25_ctd, '-o', color = 'lightskyblue', label = '25.8 isopycnal')
ax.plot(ctd26[:,1], depth26_ctd, '-o', color = 'darkslateblue', label = '26.0 isopycnal')
ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.invert_yaxis()
ax.invert_xaxis()
ax.xaxis.set_tick_params(labelsize = 15)
ax.yaxis.set_tick_params(labelsize = 15) 
ax.set_facecolor("whitesmoke")
ax.set_xlabel('Latitude [$^\circ$N]', fontsize = 20)
ax.set_ylabel('Depth [m]', fontsize = 20)
ax.legend(loc='lower right', fontsize = 18)

ax3.plot(ctd25[:,1], chl25, '-o', color = 'lightskyblue', label = '25.8 isopycnal')
ax3.plot(ctd26[:,1], chl26, '-o', color = 'darkslateblue', label = '26.0 isopycnal')
ax3.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax3.xaxis.set_tick_params(labelsize = 15)
ax3.yaxis.set_tick_params(labelsize = 15) 
ax3.set_facecolor("whitesmoke")
ax3.set_ylabel('Chlor-a [mg/m$^3$]', fontsize = 20)
ax3.set_xlim(np.max(iso26[:,1])+0.167, np.min(iso26[:,1])-0.035)

fig.tight_layout() 
fig.savefig(plot_path + 'ctd_nitrate_profile.png')

#%%
fig, (ax, ax1) = plt.subplots(1,2, dpi = 200, figsize = ([25,12]))

scalar = ax.scatter(np.log10(tn37.fluor), np.log10(tn37.no3), c = tn37.depth,
                    s = 50, cmap = 'rainbow')
# lines = ax.contour(tn37.salinity, tn37.temperature, binned.statistic.T, colors = 'k', linestyles='dashed', linewidths = 3.5)
# # ax.clabel(lines, fontsize=25, colors = 'k')
ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.xaxis.set_tick_params(labelsize = 20)
ax.yaxis.set_tick_params(labelsize = 20) 
ax.set_xlabel('Chlorophyll [mg/m$^3$]', fontsize = 25)
ax.set_ylabel('Nitrate (NO$_{3^-}$) [$\mu$mol/m$^3$]', fontsize = 25)
ax.set_facecolor("whitesmoke")
set_cbar(fig, scalar, 'Depth [m]', ax)

scalar1 = ax1.scatter(np.log10(tn37.fluor), np.log10(tn37.no3), c = tn37.density,
                    s = 50, cmap = 'cmo.dense')
# lines = ax.contour(tn37.salinity, tn37.temperature, binned.statistic.T, colors = 'k', linestyles='dashed', linewidths = 3.5)
# # ax.clabel(lines, fontsize=25, colors = 'k')
ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax1.xaxis.set_tick_params(labelsize = 20)
ax1.yaxis.set_tick_params(labelsize = 20) 
# ax.set_xlim(0, 7)
ax1.set_xlabel('Chlorophyll [mg/m$^3$]', fontsize = 25)
ax1.set_facecolor("whitesmoke")
set_cbar(fig, scalar1, 'Density [kg/m$^3$]', ax1)

fig.suptitle('Chlorophyll-Nitrate Diagram (log scale)', fontsize = 35, y = 0.99)
fig.tight_layout() 
fig.savefig(plot_path + 'chl-no3_diagram_14july19.png')

#%%
#DIAGRAM USING CTD MEASUREMENTS, interp nitrate on the longer ctd fluorescence (gap filling)

