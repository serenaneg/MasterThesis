#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:45:26 2023

@author: serena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
import gsw
import scipy.stats as ss
import datetime
import scipy.io as sio 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.interpolate import CloughTocher2DInterpolator

def set_cbar(fig, c, title, ax):   
    # skip = (slice(None, None, 2))
    cbar = plt.colorbar(c, format='%.1f', spacing='proportional', ax=ax, shrink=0.9, pad = 0.01,
                        orientation = 'vertical', location = "right")
    cbar.set_label(label = title, fontsize = 35, y = 0.5, labelpad = 30)
    cbar.ax.tick_params(which='minor', size=5, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=10, width=1, color='k', direction='in', labelsize = 30)
    # fig.subplots_adjust(bottom=0.25)
    # cbar.ax.set_position([0.2, 0.08, 0.6, 0.08])
    return cbar
#%%

path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2019/"

file = path + 'ctd_tn_withlocation.csv'
ctd = pd.read_csv(file)

#%%BATHY LOAD
bathy = xr.open_dataset(bathy_path + "gebco_2023.nc")

lat_range = slice(39, 41.5)
lon_range = slice(-72.6, -69.5)

bathy = bathy.sel(lat=lat_range, lon = lon_range)
lon = bathy.lon
lat = bathy.lat
depth = bathy.variables['elevation']
#%%choose 14th july
# Parse datetime column
# ctd = ctd[(ctd.lon > 70.7)& (ctd.lon < 70.9)]
ctd['day'] = pd.to_datetime(ctd['day'])
# Filter rows where day is 14
tn37 = ctd[(ctd['day'].dt.day == 14)]
#%%TRANSECT 37 AREA
fig, ax = plt.subplots(1,1, subplot_kw = dict(projection = ccrs.PlateCarree()), figsize = (14, 8),
                       dpi = 300)
font_size = 20

ax.coastlines(resolution="10m", linewidths=0.5)
ax.add_feature(cfeature.LAND.with_scale("10m"),
           edgecolor='black',facecolor='saddlebrown',
           zorder=0)

ax.tick_params(axis = "both", labelsize = 15)

gl = ax.gridlines(linewidth=1, color='gray', alpha=0.7,
              linestyle='--',draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'fontsize': font_size}
gl.ylabel_style = {'fontsize': font_size}

depths = (50, 75, 100, 200, 500, 1000, 2500, 3000)
lines = ax. contour(lon, lat, -depth, levels = depths, colors='black', linewidths = .5, zorder = -1)
ax.clabel(lines, inline=2, fontsize=18, colors = 'black', zorder = -1)
ax.tick_params(which='major', labelsize = 25)

ax.set_facecolor("lightgrey")
ax.scatter(-tn37.lon, tn37.lat, c = 'dodgerblue', marker = 'D', s = 80, zorder = +1)
ax.set_title('Transect 37 location, July 14, 2019', fontsize = 25, y=1.02)
fig.tight_layout()
fig.savefig(plot_path + 'tn368_area', bbox_inches='tight')

#%%depth profile
bathy = xr.open_dataset(bathy_path + "gebco_3d.nc")

lat_range = slice(38.7, 40.40)
lon_range = -70.8

bathy = bathy.sel(lat=lat_range)
bathy = bathy.sel(lon = lon_range, method = 'nearest')
depth = bathy.Z_3d_interpolated
#%%PLOT
#EACH VARIABLE MUST BE INTERPOLATED
fig, ([ax,ax1],[ax3,ax2]) = plt.subplots(2,2, dpi = 200, figsize = ([36,20]))
 
# make a grid
# dx = np.diff(tn37.lat)
# print(dx)

x = np.arange(38, 42, 0.064) #0.064 get from diff
y = np.arange(0,  300, 1)
xx, yy = np.meshgrid(x, y)

#### FIRST COLUMN #####
###TEMP

binned = ss.binned_statistic_2d(tn37.lat, tn37.depth, tn37.temperature, statistic='mean', bins=[x, y])
binned_sal = ss.binned_statistic_2d(tn37.lat, tn37.depth, tn37.salinity, statistic='mean', bins=[x, y])

# to do a contour plot, you need to reference the center of the bins, not the edges
# get the bin centers
xc = (x[:-1] + x[1:]) / 2
yc = (y[:-1] + y[1:]) / 2

# plot the data
vmin = 6
vmax = 25
levels = np.linspace(vmin,vmax, 15)

sm = ax.contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = 'cmo.thermal',
                 extend ='both')
ax.contour(xc, yc, binned_sal.statistic.T, levels = [34.5], zorder = 2, colors = 'crimson', linestyles = '--', linewidths = 3)
ax.plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
set_cbar(fig, sm, '[$^\circ$C]', ax)
ax.set_title('Temperature', fontsize = 40)

####FLUORECENCE
binned = ss.binned_statistic_2d(tn37.lat, tn37.depth, tn37.fluorecence, statistic='mean', bins=[x, y],
                                expand_binnumbers=True)
binned_sigma = ss.binned_statistic_2d(tn37.lat, tn37.depth, tn37.density, statistic='mean', bins=[x, y])

# plot the data
vmin = 0
vmax = 5
# levels = np.arange(vmin,vmax, 0.05)
levels = np.linspace(vmin,vmax, 15)


sm = ax2.contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, 
                  extend ='both', cmap = 'cmo.algae')
ax2.contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'mediumspringgreen', linestyles = '--', linewidths = 3)
ax2.contour(xc, yc, binned_sigma.statistic.T, levels = [25.8], zorder = 2, colors = 'mediumspringgreen', linestyles = '--', linewidths = 3)
ax2.contour(xc, yc, binned_sal.statistic.T, levels = [34.5], zorder = 2, colors = 'crimson', linestyles = '--', linewidths = 3)
ax2.plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
set_cbar(fig, sm, '[$\mu$g /m$^3$]', ax2)
ax2.set_title('Chlorophyll', fontsize = 40)

####SECOND COLUMN
#SALINITY
binned = ss.binned_statistic_2d(tn37.lat, tn37.depth, tn37.salinity, statistic='mean', bins=[x, y])

# plot the data
vmin = 31.8
vmax = 36
# levels = np.arange(vmin,vmax, 0.05)
levels = np.linspace(vmin,vmax, 15)

sm = ax1.contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.haline')
lines =  ax1.contour(xc, yc, binned.statistic.T, levels = [34.5], zorder = 2, colors = 'crimson', linestyles = '--', linewidths = 3)
ax1.clabel(lines, inline=True, fontsize=25, colors = 'crimson')
ax1.plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
set_cbar(fig, sm, '[PSU]', ax1)
ax1.set_title('Salinity', fontsize = 40)


####SIGMA
binned = ss.binned_statistic_2d(tn37.lat, tn37.depth, tn37.density, statistic='mean', bins=[x, y])

# plot the data
vmin = 23
vmax = 27
# levels = np.arange(vmin,vmax, 0.05)
levels = np.linspace(vmin,vmax, 15)
cmap = 'cmo.dense'

sm = ax3.contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = cmap, extend ='both')
lines =  ax3.contour(xc, yc, binned.statistic.T, levels = [26.0], zorder = 2, colors = 'mediumspringgreen', linestyles = '--', linewidths = 3)
lines2 =  ax3.contour(xc, yc, binned.statistic.T, levels = [25.8], zorder = 2, colors = 'mediumspringgreen', linestyles = '--', linewidths = 3)
ax3.clabel(lines, fontsize=25, colors = 'mediumspringgreen')
ax3.clabel(lines2, fontsize=25, colors = 'mediumspringgreen')
ax3.plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
set_cbar(fig, sm, '[kg/m$^3$]', ax3)
ax3.set_title('Density ($\sigma_{\Theta}$)', fontsize = 40)

###PLOT SET UP## 
ax.set_ylim(0, 210)
ax1.set_ylim(0, 210)
ax2.set_ylim(0, 210)
ax3.set_ylim(0, 210)

ax.set_xlim(tn37.lat.min(), tn37.lat.max())
ax1.set_xlim(tn37.lat.min(), tn37.lat.max())
ax2.set_xlim(tn37.lat.min(), tn37.lat.max())
ax3.set_xlim(tn37.lat.min(), tn37.lat.max())

ax.invert_yaxis()
ax.invert_xaxis()
ax1.invert_yaxis()
ax1.invert_xaxis()
ax2.invert_yaxis()
ax2.invert_xaxis()
ax3.invert_yaxis()
ax3.invert_xaxis()

ax1.xaxis.set_tick_params(labelsize = 30)
ax1.yaxis.set_tick_params(labelsize = 30) 
ax.xaxis.set_tick_params(labelsize = 30)
ax.yaxis.set_tick_params(labelsize = 30) 
ax2.xaxis.set_tick_params(labelsize = 30)
ax2.yaxis.set_tick_params(labelsize = 30) 
ax3.xaxis.set_tick_params(labelsize = 30)
ax3.yaxis.set_tick_params(labelsize = 30) 

ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax2.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax3.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')

ax2.set_xlabel('Latitude [$\degree$N]', fontsize = 40)
ax3.set_xlabel('Latitude [$\degree$N]', fontsize = 40)

ax.set_ylabel('Depth [m]', fontsize = 40)
ax3.set_ylabel('Depth [m]', fontsize = 40)

fig.suptitle('CTD transect at 70.8$^\circ$W, July 14 2019', fontsize = 50, y = 0.98)
fig.tight_layout(pad = 4.3) 
fig.savefig(plot_path + 'ctd_tn37_14july.png', bbox_inches='tight')

#%%
fig, ax = plt.subplots(1,1, dpi = 200, figsize = ([10,6]))
ax.scatter(tn37.lat, tn37.depth)
ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.invert_yaxis()
ax.invert_xaxis()
ax.xaxis.set_tick_params(labelsize = 20)
ax.yaxis.set_tick_params(labelsize = 20) 
ax.set_xlabel('Latitude [$\degree$N]', fontsize = 25)
ax.set_ylabel('Depth [m]', fontsize = 25)
ax.set_title('CTD casts distribution', fontsize = 25)
fig.tight_layout() 
fig.savefig(plot_path + 'ctd_cast_distr.png')


