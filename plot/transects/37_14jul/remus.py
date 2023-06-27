#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:57:31 2023

@author: serena
"""

import xarray as xr
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm, ListedColormap, LinearSegmentedColormap
import scipy.io as sio
import cmocean
import scipy.stats as ss
import matplotlib.cm as cm

#%%
 
def remove_values(arr, values):
    return [x for x in arr if x not in values]


def set_cbar(fig, c, title, ax):  
  
    cbar = plt.colorbar(c, format='%.1f', spacing='proportional', ax=ax, shrink=0.9, pad = 0.01,
                            orientation = 'vertical', location = "right")

    cbar.set_label(label = title, fontsize = 35, y = 0.5, labelpad = 30)
    cbar.ax.tick_params(which='minor', size=5, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=10, width=1, color='k', direction='in', labelsize = 30)
    # fig.subplots_adjust(bottom=0.25)
        # cbar.ax.set_position([0.2, 0.08, 0.6, 0.08])
    return cbar

top = cm.get_cmap('YlGnBu_r', 128) # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)# combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')
#%%
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2019/"

#data from mat file
file = sio.loadmat(path + 'REMUS_MSN005.mat')
print(file)

data = file['vprlog']
col_name = file['varnames']

matrix = np.concatenate((data, col_name.reshape(-1, 11)), axis=0)

#get variables
time = data[:,0]

lat = data[:,1]
depth = data[:,3]

temp = data[:,4]
salinity = data[:,5]
sigma = data[:,6]
fluor = data[:,7]
nitrate = data[:,10]

#%%SALINITY
#remouve outliers
sal = []
index = []
for i in range(len(salinity)):
    if salinity[i] <= 23:
        sal.append(salinity[i])
        index.append([i])
        
sal = np.array(sal)

sali = remove_values(salinity, sal)
sali = np.array(sali)

#remouve outliers also in lat and depth
indices_remove = [7253, 14494, 21739, 28099]
lat_reduce = np.delete(lat, indices_remove)
depth_reduce = np.delete(depth, indices_remove)
#%%
#plot
#temperature
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols = 2, nrows = 2, figsize = [38,20])

scalar = ax1.scatter(lat, depth, c = temp, marker = 'o', 
                    s = np.array([40 * temp[n] for n in range(len(temp))]), cmap = 'cmo.thermal')

ax1.set_ylabel('Depth [m]', fontsize = 35)
gl = ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax1.invert_yaxis()
ax1.invert_xaxis()
ax1.xaxis.set_tick_params(labelsize = 30)
ax1.yaxis.set_tick_params(labelsize = 30)
ax1.set_title('Temperature', fontsize = 40, y=1.02)
set_cbar(fig, scalar, '[$^\circ C$]', ax1)

#salinity
scalar = ax2.scatter(lat_reduce, depth_reduce, c = sali, marker = 'o', 
                     s = np.array([10 * sali[n] for n in range(len(sali))]), cmap = 'cmo.haline')
gl = ax2.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax2.invert_yaxis()
ax2.invert_xaxis()
ax2.xaxis.set_tick_params(labelsize = 30)
ax2.yaxis.set_tick_params(labelsize = 30)  
ax2.set_title('Salinity', fontsize = 40, y=1.02)
set_cbar(fig, scalar, '[PSU]', ax2)

#FLUORECENCE
scalar = ax4.scatter(lat, depth, c = fluor, marker = 'o', 
                     s = np.array([50 * fluor[n] for n in range(len(fluor))]), cmap = 'jet')

ax4.set_xlabel('Latitude [°N]', fontsize = 35)
gl = ax4.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax4.invert_yaxis()
ax4.invert_xaxis()
ax4.xaxis.set_tick_params(labelsize = 30)
ax4.yaxis.set_tick_params(labelsize = 30)   
ax4.set_title('Fluorecence', fontsize = 40, y=1.02)
set_cbar(fig, scalar, '[$\mu$g/m$^3$]', ax4) ##???UNITS

#NITRATE
scalar = ax3.scatter(lat, depth, c = nitrate, marker = 'o', 
                     s = np.array([15 * nitrate[n] for n in range(len(nitrate))]), cmap = orange_blue)

ax3.set_xlabel('Latitude [°N]', fontsize = 35)
ax3.set_ylabel('Depth [m]', fontsize = 35)
gl = ax3.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax3.invert_yaxis()
ax3.invert_xaxis()
ax3.xaxis.set_tick_params(labelsize = 30)
ax3.yaxis.set_tick_params(labelsize = 30)   
ax3.set_title('Nitrate', fontsize = 40, y=1.02)
set_cbar(fig, scalar, '[mmol/m$^3$]', ax3) ##???UNITS


ax4.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax2.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax3.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')

fig.suptitle('REMUS 5 transect at 70.8$^\circ$W, July 14th', fontsize = 50, y = 0.98)

fig.tight_layout(pad = 3.5) 
fig.savefig(plot_path + 'remus5_july14.png', bbox_inches='tight')

#%%PLOT
#EACH VARIABLE MUST BE INTERPOLATED
fig, ([ax,ax1],[ax2,ax3]) = plt.subplots(2,2, dpi = 200, figsize = ([36,20]))
 
# make a grid
# dx = np.diff(lat)
# print(dx)

x = np.arange(38, 41, 0.064) #0.064 get from diff
y = np.arange(0,  200, 1)
xx, yy = np.meshgrid(x, y)

#### FIRST COLUMN #####
###TEMP
binned = ss.binned_statistic_2d(lat, depth, temp, statistic='mean', 
                                expand_binnumbers = True, bins=[x, y])

# to do a contour plot, you need to reference the center of the bins, not the edges
# get the bin centers
xc = (x[:-1] + x[1:]) / 2
yc = (y[:-1] + y[1:]) / 2

# plot the data
vmin = 6
vmax = 25
levels = np.arange(vmin,vmax, 0.05)

sm = ax.contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = 'cmo.thermal',
                 extend ='both')
set_cbar(fig, sm, '[$^\circ$C]', ax)
ax.set_title('Temperature', fontsize = 40)

####FLUORECENCE
binned = ss.binned_statistic_2d(lat, depth, fluor, statistic='mean', bins=[x, y],
                                expand_binnumbers=True)

# plot the data
vmin = 0
vmax = 4
levels = np.arange(vmin,vmax, 0.05)

sm = ax2.contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, 
                  extend ='both', cmap = 'cmo.algae')
set_cbar(fig, sm, '[$\mu$g /m$^3$]', ax2)
ax2.set_title('Fluorescence', fontsize = 40)

####SECOND COLUMN
#SALINITY
binned = ss.binned_statistic_2d(lat_reduce, depth_reduce, sali, statistic='mean', bins=[x, y])

# plot the data
vmin = 31.8
vmax = 36
levels = np.arange(vmin,vmax, 0.05)

sm = ax1.contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.haline')
lines =  ax1.contour(xc, yc, binned.statistic.T, levels = [34.5], zorder = 2, colors = 'black', linestyles = '--')
ax1.clabel(lines, inline=1, fontsize=25, colors = 'black')
set_cbar(fig, sm, '[PSU]', ax1)
ax1.set_title('Salinity', fontsize = 40)


####NITRATE
binned = ss.binned_statistic_2d(lat, depth, nitrate, statistic='mean', bins=[x, y])

# plot the data
vmin = np.nanmin(nitrate)+2
vmax = np.nanmax(nitrate)-5
levels = np.arange(vmin,vmax, 0.05)
cmap = orange_blue

sm = plt.contourf(xc, yc, binned.statistic.T, vmin = vmin, vmax = vmax, levels = levels, cmap = cmap, extend ='both')
set_cbar(fig, sm, '[mmol/m$^3$]', ax3)
ax3.set_title('Nitrate (NO$_{3^-}$)', fontsize = 40)

###PLOT SET UP###
ax.set_xlim(39.68, 40.40)
ax1.set_xlim(39.68, 40.40)
ax2.set_xlim(39.68, 40.40)
ax3.set_xlim(39.68, 40.40)

ax.invert_yaxis()
ax.invert_xaxis()
ax1.invert_yaxis()
ax1.invert_xaxis()
ax2.invert_yaxis()
ax2.invert_xaxis()
ax3.invert_yaxis()
ax3.invert_xaxis()

ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax2.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax3.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')

ax1.xaxis.set_tick_params(labelsize = 30)
ax1.yaxis.set_tick_params(labelsize = 30) 
ax.xaxis.set_tick_params(labelsize = 30)
ax.yaxis.set_tick_params(labelsize = 30) 
ax2.xaxis.set_tick_params(labelsize = 30)
ax2.yaxis.set_tick_params(labelsize = 30) 
ax3.xaxis.set_tick_params(labelsize = 30)
ax3.yaxis.set_tick_params(labelsize = 30) 

ax2.set_xlabel('Latitude [$\degree$N]', fontsize = 40)
ax3.set_xlabel('Latitude [$\degree$N]', fontsize = 40)

ax.set_ylabel('Depth [m]', fontsize = 40)
ax2.set_ylabel('Depth [m]', fontsize = 40)

fig.suptitle('REMUS 5 transect at 70.8$^\circ$W, July 14th 2019', fontsize = 50, y = 0.98)
fig.tight_layout(pad = 4) 
fig.savefig(plot_path + 'remus_tn37_14july_section.png', bbox_inches='tight')

#%%
fig, ax = plt.subplots(1,1, dpi = 200, figsize = ([10,6]))
ax.scatter(lat, depth)
ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.invert_yaxis()
ax.invert_xaxis()
ax.xaxis.set_tick_params(labelsize = 20)
ax.yaxis.set_tick_params(labelsize = 20) 
ax.set_xlabel('Latitude [$\degree$N]', fontsize = 25)
ax.set_ylabel('Depth [m]', fontsize = 25)
ax.set_title('Remus data distribution', fontsize = 25)
fig.tight_layout() 
fig.savefig(plot_path + 'remus_distr.png')
