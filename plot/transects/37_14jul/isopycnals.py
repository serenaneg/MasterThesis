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
    f = interpolate.interp1d(depth, variable, kind='linear')

    newx = np.linspace(np.min(depth), np.max(depth), len(depth)) #new depth or depth
    ynew = f(newx)
    
    return ynew

def set_cbar(fig, c, title, ax):   
    # skip = (slice(None, None, 2))
    cbar = plt.colorbar(c, format='%.1f', spacing='proportional', ax=ax, shrink=0.9, pad = 0.01,
                        orientation = 'vertical', location = "right")
    cbar.set_label(label = title, fontsize = 25, y = 0.5)
    cbar.ax.tick_params(which='minor', size=5, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=10, width=1, color='k', direction='in', labelsize = 25)
    # fig.subplots_adjust(bottom=0.25)
    # cbar.ax.set_position([0.2, 0.08, 0.6, 0.08])
    return cbar

def round_array(arr):
    rounded_arr = []
    for num in arr:
        # rounded_num = round(num, 1)
        if 24.4490 <= num < 25.5500:
            num = 25.5
        elif 25.7490 <= num < 25.8500:
            num = 25.8
        elif 25.9400 <= num < 26.0500:
            num = 26.0
        rounded_arr.append(num)
    return rounded_arr

def round_column(matrix, column_index, decimal_places):
    rounded_column = np.round(matrix[:, column_index], decimal_places)
    matrix[:, column_index] = rounded_column
    return matrix

def average_rows_by_latitude(matrix):
    unique_latitudes = np.unique(matrix[:, 1])

    averaged_matrix = []
    for latitude in unique_latitudes:
        rows_with_latitude = matrix[matrix[:, 1] == latitude]
        average_row = np.nanmean(rows_with_latitude, axis=0)
        averaged_matrix.append(average_row)

    return np.array(averaged_matrix)
#%%
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2019/"

file = path + 'bottle_tn368_hydro.csv'
bottle = pd.read_csv(file)

#%%choose 14th july TRANSECT 37
# Parse datetime column
bottle['day'] = pd.to_datetime(bottle['day'])
# Filter rows where day is 14
tn37 = bottle[(bottle['day'].dt.day == 14)]
#%%
lat = np.array(tn37.latitude, dtype = 'float')
lon = np.array(tn37.longitude, dtype = 'float')

#%%
sigma = np.array(tn37.density)

sigma_1f = round_array(sigma)

#format densiry values to only 1 decimal
# sigma_1f = ["{:.1f}".format(x) for x in sigma]
sigma = np.array(np.column_stack((sigma_1f, lat, tn37.depth,  tn37.no3, tn37.chlor, tn37.fluor, tn37.PON)), dtype = 'float')

#nitrate along 26.0 isopicnal
iso255 = filter_values(sigma, 25.5, 25.5)
iso25 = filter_values(sigma, 25.8, 25.8)
iso26 = filter_values(sigma, 26, 26)

#%%INTERP BOTTLE DATA
#calculate the interpolated depth for the ispycnals
iso255 = round_column(iso255, 1, 2)
iso255 = average_rows_by_latitude(iso255)

xvals25 = np.arange(np.min(iso25[:,1]), np.max(iso25[:,1]), 0.032) 
depth25 = np.interp(xvals25, iso25[:,1], iso25[:,2])
nit25 = np.interp(xvals25, iso25[:,1], iso25[:,3])
pon25 = np.interp(xvals25, iso25[:,1], iso25[:,6])

xvals255 = np.arange(np.min(iso255[:,1]), np.max(iso255[:,1]), 0.032) 
depth255 = np.interp(xvals255, iso255[:,1], iso255[:,2])
nit255 = np.interp(xvals255, iso255[:,1], iso255[:,3])
pon255 = np.interp(xvals255, iso255[:,1], iso255[:,6])


xvals26 = np.arange(np.min(iso26[:,1]), np.max(iso26[:,1]), 0.032) 
depth26 = np.interp(xvals26, iso26[:,1], iso26[:,2])
nit26 = np.interp(xvals26, iso26[:,1], iso26[:,3])
pon26 = np.interp(xvals26, iso26[:,1], iso26[:,6])

#%%             
#use the inteprolated depth to interpolate nutrients
#NITRATE
# nit255 = my_interp(depth255, iso255[:, 3])
# nit25 = my_interp(depth25, iso25[:, 3])
# nit26 = my_interp(depth26, iso26[:, 3])

# #INTERP CHL
# #pon may contain nan, delet
# pon255 = my_interp(depth255, iso255[:, 6]) 
# pon25 = my_interp(depth25, iso25[:, 6])
# pon26 = my_interp(depth26, iso26[:, 6])
# pon255 = my_interp(iso255[:, 6], iso255[:, 6]) #
# pon25 = my_interp(iso25[:, 6], iso25[:, 6])
# pon26 = my_interp(iso26[:, 6], iso26[:, 6])

#%%CTD DATA
file2 = path + 'ctd_tn_withlocation.csv'
ctd = pd.read_csv(file2)

ctd['day'] = pd.to_datetime(ctd['day'])
ctd37 = ctd[ctd['day'].dt.day == 14] #select 14th july

#NOT NEEDED BECAUSE THE STATION ALREADY MATCH
# station = np.array(tn37.station, dtype = 'float')
# # Filter the DataFrame based on the array values in the 'Name' column
# filtered_ctd = ctd37[ctd37['station'].isin(station)]

#select ispopycnal
ctd_sig = np.array(ctd37.density)

ctd_sigma_1f = round_array(ctd_sig)
ctd_sigma = np.array(np.column_stack((ctd_sigma_1f, ctd37.lat, ctd37.depth, ctd37.fluorecence)), dtype = 'float')

#only 25.8 and 26
ctd255 = filter_values(ctd_sigma, 25.5, 25.5)
ctd25 = filter_values(ctd_sigma, 25.8, 25.8)
ctd26 = filter_values(ctd_sigma, 26, 26)
#%%
#DELETE MULTIPLE LATITUDE
ctd255 = round_column(ctd255, 1, 2)
ctd255 = average_rows_by_latitude(ctd255)

ctd25 = round_column(ctd25, 1, 2)
ctd25 = average_rows_by_latitude(ctd25)

ctd26 = round_column(ctd26, 1, 2)
ctd26 = average_rows_by_latitude(ctd26)
#%%INTERP CTD DATA
# depth255_ctd = ctd255[:, 2]
# f = interpolate.interp1d(depth255_ctd,  ctd255[:,2], kind='linear')
# newx = np.linspace(np.min(depth255_ctd), np.max(depth255_ctd), len(depth255_ctd)) #new depth
# depth255_ctd = f(newx)

# depth25_ctd = ctd25[:, 2]
# f = interpolate.interp1d(depth25_ctd,  ctd25[:,2], kind = 'linear')
# newx = np.linspace(np.min(depth25_ctd), np.max(depth25_ctd), len(depth25_ctd)) #new depth
# depth25_ctd = f(newx)

# depth26_ctd = ctd26[:, 2]
# f2 = interpolate.interp1d(depth26_ctd,  ctd26[:,2], kind = 'linear')
# newx2 = np.linspace(np.min(depth26_ctd), np.max(depth26_ctd), len(depth26_ctd)) #new depth
# depth26_ctd = f2(newx2)

# chlor255 = my_interp(depth255_ctd, ctd255[:,3])
# chlor25 = my_interp(depth25_ctd, ctd25[:,3])
# chlor26 = my_interp(depth26_ctd, ctd26[:,3])
#%%ITERPOLATE CTD 2 
ctdxvals25 = np.arange(np.min(ctd25[:,1]), np.max(ctd25[:,1]), 0.032) 
chlor25 = np.interp(ctdxvals25, ctd25[:,1], ctd25[:,3])

ctdxvals255 = np.arange(np.min(ctd255[:,1]), np.max(ctd255[:,1]), 0.032) 
chlor255 = np.interp(ctdxvals255, ctd255[:,1], ctd255[:,3])


ctdxvals26 = np.arange(np.min(ctd26[:,1]), np.max(ctd26[:,1]), 0.032) 
chlor26 = np.interp(ctdxvals26, ctd26[:,1], ctd26[:,3])

#%%
fig, (ax3, ax2, ax1, ax) = plt.subplots(4,1, dpi = 200, figsize = (8,15))


ax.plot(xvals255, depth255, '-o', color = 'lightskyblue', label = '25.5 isopycnal')
ax.plot(xvals25, depth25, '-o', color = 'royalblue', label = '25.8 isopycnal')
ax.plot(xvals26, depth26, '-o', color = 'purple', label = '26.0 isopycnal')
ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.invert_yaxis()
ax.invert_xaxis()
ax.xaxis.set_tick_params(labelsize = 15)
ax.yaxis.set_tick_params(labelsize = 15) 
ax.set_facecolor("whitesmoke")
ax.set_xlabel('Latitude [$^\circ$N]', fontsize = 20)
ax.set_ylabel('Depth [m]', fontsize = 20)
legend= ax.legend(loc='lower right', fontsize = 15, labelcolor = ['lightskyblue', 'royalblue', 'purple'],
                  handlelength=0, handletextpad=0)
for handle in legend.legendHandles:
    handle.set_marker('')

ax1.plot(xvals255, nit255, '-o', color = 'lightskyblue', label = '25.5 isopycnal')
ax1.plot(xvals25, nit25, '-o', color = 'royalblue', label = '25.8 isopycnal')
ax1.plot(xvals26, nit26, '-o', color = 'purple', label = '26.0 isopycnal')
ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax1.invert_xaxis()
ax1.set_facecolor("whitesmoke")
ax1.xaxis.set_tick_params(labelsize = 15)
ax1.yaxis.set_tick_params(labelsize = 15) 
ax1.set_ylabel('Nitrate [mmol/m$^3$]', fontsize = 20)

ax2.plot(xvals255, pon255, '-o', color = 'lightskyblue', label = '25.5 isopycnal')
ax2.plot(xvals25, pon25, '-o', color = 'royalblue', label = '25.8 isopycnal')
ax2.plot(xvals26, pon26, '-o', color = 'purple', label = '26.0 isopycnal')
ax2.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax2.xaxis.set_tick_params(labelsize = 15)
ax2.yaxis.set_tick_params(labelsize = 15) 
ax2.set_facecolor("whitesmoke")
ax2.set_ylabel('PON [$\mu$mol/m$^3$]', fontsize = 20)
ax2.set_xlim(np.max(iso26[:,1])+0.167, np.min(iso26[:,1])-0.035)

ax3.plot(ctdxvals255, chlor255, '-o', color = 'lightskyblue', label = '25.5 isopycnal')
ax3.plot(ctdxvals25, chlor25, '-o', color = 'royalblue', label = '25.8 isopycnal')
ax3.plot(ctdxvals26, chlor26, '-o', color = 'purple', label = '26.0 isopycnal')
ax3.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax3.xaxis.set_tick_params(labelsize = 15)
ax3.yaxis.set_tick_params(labelsize = 15) 
ax3.set_facecolor("whitesmoke")
ax3.set_ylabel('Chlor-a [mg/m$^3$]', fontsize = 20)
ax3.set_xlim(np.max(iso26[:,1])+0.167, np.min(iso26[:,1])-0.035)

fig.tight_layout() 
fig.savefig(plot_path + 'isopycnals_14July.png')

#%%CHECK INTERPOLATION
# fig, (ax3, ax2, ax1, ax) = plt.subplots(4,1, dpi = 200, figsize = (8,15))


# ax.plot(xvals255, depth255, '-o', color = 'lightskyblue', label = '25.5 isopycnal')
# ax.plot(xvals25, depth25, '-o', color = 'royalblue', label = '25.8 isopycnal')
# ax.plot(xvals26, depth26, '-o', color = 'purple', label = '26.0 isopycnal')
# ax.plot(iso255[:,1], iso255[:,2], 'v', color = 'red')
# ax.plot(iso25[:,1], iso25[:,2], 'v', color = 'orange')
# ax.plot(iso26[:,1], iso26[:,2], 'v', color = 'lime')
# ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax.invert_yaxis()
# ax.invert_xaxis()
# ax.xaxis.set_tick_params(labelsize = 15)
# ax.yaxis.set_tick_params(labelsize = 15) 
# ax.set_facecolor("whitesmoke")
# ax.set_xlabel('Latitude [$^\circ$N]', fontsize = 20)
# ax.set_ylabel('Depth [m]', fontsize = 20)
# legend= ax.legend(loc='lower right', fontsize = 15, labelcolor = ['lightskyblue', 'royalblue', 'purple'],
#                   handlelength=0, handletextpad=0)
# for handle in legend.legendHandles:
#     handle.set_marker('')
    
# ax1.plot(xvals255, nit255, '-o', color = 'lightskyblue', label = '25.5 isopycnal')
# ax1.plot(xvals25, nit25, '-o', color = 'royalblue', label = '25.8 isopycnal')
# ax1.plot(xvals26, nit26, '-o', color = 'purple', label = '26.0 isopycnal')
# ax1.plot(iso255[:,1], iso255[:,3], 'v', color = 'red')
# ax1.plot(iso25[:,1], iso25[:,3], 'v', color = 'orange')
# ax1.plot(iso26[:,1], iso26[:,3], 'v', color = 'lime')
# ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax1.invert_xaxis()
# ax1.set_facecolor("whitesmoke")
# ax1.xaxis.set_tick_params(labelsize = 15)
# ax1.yaxis.set_tick_params(labelsize = 15) 
# ax1.set_ylabel('Nitrate [mmol/m$^3$]', fontsize = 20)

# ax2.plot(xvals255, pon255, '-o', color = 'lightskyblue', label = '25.5 isopycnal')
# ax2.plot(xvals25, pon25, '-o', color = 'royalblue', label = '25.8 isopycnal')
# ax2.plot(xvals26, pon26, '-o', color = 'purple', label = '26.0 isopycnal')
# ax2.plot(iso255[:,1], iso255[:,6], 'v', color = 'red')
# ax2.plot(iso25[:,1], iso25[:,6], 'v', color = 'orange')
# ax2.plot(iso26[:,1], iso26[:,6], 'v', color = 'lime')
# ax2.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax2.xaxis.set_tick_params(labelsize = 15)
# ax2.yaxis.set_tick_params(labelsize = 15) 
# ax2.set_facecolor("whitesmoke")
# ax2.set_ylabel('PON [$\mu$mol/m$^3$]', fontsize = 20)
# ax2.set_xlim(np.max(iso26[:,1])+0.167, np.min(iso26[:,1])-0.035)

# ax3.plot(ctdxvals255, chlor255, '-o', color = 'lightskyblue', label = '25.5 isopycnal')
# ax3.plot(ctdxvals25, chlor25, '-o', color = 'royalblue', label = '25.8 isopycnal')
# ax3.plot(ctdxvals26, chlor26, '-o', color = 'purple', label = '26.0 isopycnal')
# ax3.plot(ctd255[:,1], ctd255[:,3], 'v', color = 'red')
# ax3.plot(ctd25[:,1], ctd25[:,3], 'v', color = 'orange')
# ax3.plot(ctd26[:,1], ctd26[:,3], 'v', color = 'lime')
# ax3.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax3.xaxis.set_tick_params(labelsize = 15)
# ax3.yaxis.set_tick_params(labelsize = 15) 
# ax3.set_facecolor("whitesmoke")
# ax3.set_ylabel('Chlor-a [mg/m$^3$]', fontsize = 20)
# ax3.set_xlim(np.max(iso26[:,1])+0.167, np.min(iso26[:,1])-0.035)
#%%COMPUTE DELTA NPO3 VS DELTA PON
delta_no3_25 = np.diff(nit25)
delta_no3_255 = np.diff(nit255)
delta_no3_26 = np.diff(nit26)

delta_pon_25 = np.diff(pon25)
delta_pon_255 = np.diff(pon255)
delta_pon_26 = np.diff(pon26)

#%%
# fig, (ax, ax1, ax2) = plt.subplots(1,3, dpi = 200, figsize = ([21,7]))

# ax.plot(delta_no3_25, delta_pon_25,'o', markersize = 15, alpha = .7, color = 'lightskyblue', label = '25.5 isopycnal')
# ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax.xaxis.set_tick_params(labelsize = 20)
# ax.yaxis.set_tick_params(labelsize = 20) 
# ax.set_xlabel('$\Delta$NO$_{3^-}$ [$\mu$mol/m$^3$]', fontsize = 25)
# ax.set_ylabel('$\Delta$PON [$\mu$mol/m$^3$]', fontsize = 25)
# ax.set_facecolor("whitesmoke")
# ax.set_title('25.5 isopycnal', fontsize = 25)

# ax1.plot(delta_no3_255, delta_pon_255, 'o', markersize = 15, alpha = .7,  color = 'royalblue', label = '25.8 isopycnal')
# ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax1.xaxis.set_tick_params(labelsize = 20)
# ax1.yaxis.set_tick_params(labelsize = 20) 
# ax1.set_xlabel('$\Delta$NO$_{3^-}$ [$\mu$mol/m$^3$]', fontsize = 25)
# ax1.set_facecolor("whitesmoke")
# ax1.set_title('25.8 isopycnal', fontsize = 25)


# ax2.plot(delta_no3_26, delta_pon_26, 'o', markersize = 15, alpha = .7, color = 'purple', label = '26.0 isopycnal')
# ax2.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax2.xaxis.set_tick_params(labelsize = 20)
# ax2.yaxis.set_tick_params(labelsize = 20) 
# ax2.set_facecolor("whitesmoke")
# ax2.set_xlabel('$\Delta$NO$_{3^-}$ [$\mu$mol/m$^3$]', fontsize = 25)
# ax2.set_title('26.0 isopycnal', fontsize = 25)

#%%
fig, ax = plt.subplots(1,1, dpi = 200, figsize = ([10.5,8]))

ax.plot(delta_no3_25, delta_pon_25,'o', markersize = 15, alpha = .7, color = 'lightskyblue', label = '25.5 isopycnal')
ax.plot(delta_no3_255, delta_pon_255, 'o', markersize = 15, alpha = .7,  color = 'royalblue', label = '25.8 isopycnal')
ax.plot(delta_no3_26, delta_pon_26, 'o', markersize = 15, alpha = .7, color = 'purple', label = '26.0 isopycnal')

ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.xaxis.set_tick_params(labelsize = 20)
ax.yaxis.set_tick_params(labelsize = 20) 
ax.set_xlabel('$\Delta$NO$_{3^-}$ [$\mu$mol/m$^3$]', fontsize = 25)
ax.set_ylabel('$\Delta$PON [$\mu$mol/m$^3$]', fontsize = 25)
ax.set_facecolor("whitesmoke")

legend= ax.legend(loc='lower right', fontsize = 25, labelcolor = ['lightskyblue', 'royalblue', 'purple'],
                  handlelength=0, handletextpad=0)
for handle in legend.legendHandles:
    handle.set_marker('')
    
fig.tight_layout() 
fig.savefig(plot_path + 'deltas_no3PON.png')

#%%PON VS NO3
# fig, (ax, ax1) = plt.subplots(1,2, dpi = 200, figsize = ([25,12]))

# scalar = ax.scatter(pon25, nit25,  c = depth25,
#                     s = 50, cmap = 'rainbow')
# ax.scatter(pon255, nit255,  c = depth255,
#                     s = 50, cmap = 'rainbow')
# ax.scatter(pon26, nit26,  c = depth26,
#                     s = 50, cmap = 'rainbow')
# ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax.xaxis.set_tick_params(labelsize = 25)
# ax.yaxis.set_tick_params(labelsize = 25) 
# ax.set_xlabel('PON [$\mu$mol/m$^3$]', fontsize = 25)
# ax.set_ylabel('Nitrate (NO$_{3^-}$) [$\mu$mol/m$^3$]', fontsize = 25)
# ax.set_facecolor("whitesmoke")
# set_cbar(fig, scalar, 'Depth [m]', ax)



# ax1.scatter(pon26, nit26, c = 'purple', s = 70, label = '26.0 isopycnal')
# ax1.scatter(pon25, nit25, c = 'royalblue', s = 70,  label = '25.8 isopycnal')
# ax1.scatter(pon255, nit255, c = 'lightskyblue', s = 70,  label = '25.5 isopycnal')
# ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax1.xaxis.set_tick_params(labelsize = 25)
# ax1.yaxis.set_tick_params(labelsize = 25) 
# # ax.set_xlim(0, 7)
# ax1.set_xlabel('PON [$\mu$mol/m$^3$]', fontsize = 25)
# ax1.set_facecolor("whitesmoke")
# legend= ax1.legend(loc='upper right', fontsize = 20, labelcolor = ['purple',  'royalblue','lightskyblue'])
# # for handle in legend.legendHandles:
# #     handle.set_marker('')

# fig.suptitle('Nitrate-PON Diagram', fontsize = 35, y = 0.99)
# fig.tight_layout() 
# fig.savefig(plot_path + 'no3-pon_diagram_14july19.png')

#%%
mat = np.column_stack((delta_no3_26, delta_pon_26, depth26[1:]))

fig, ax = plt.subplots(1,1, dpi = 200, figsize = ([10,6]))

scalar = ax.scatter(mat[:,0], mat[:,1], c = mat[:, 2],
                    s = 50, cmap = 'rainbow')
ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.xaxis.set_tick_params(labelsize = 25)
ax.yaxis.set_tick_params(labelsize = 25) 
ax.set_xlabel('PON [$\mu$mol/m$^3$]', fontsize = 25)
ax.set_ylabel('Nitrate (NO$_{3^-}$) [$\mu$mol/m$^3$]', fontsize = 25)
ax.set_facecolor("whitesmoke")
set_cbar(fig, scalar, 'Depth [m]', ax)


