#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:16:20 2023

@author: serena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean
import scipy.stats as ss
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import xarray as xr


def set_cbar(fig, c, title, ax):
    # skip = (slice(None, None, 2))
    cbar = plt.colorbar(c, format='%.1f', spacing='proportional', ax=ax, shrink=0.9, pad=0.01,
                        orientation='vertical', location="right")
    cbar.set_label(label=title, fontsize=25, y=0.5)
    cbar.ax.tick_params(which='minor', size=5, width=1,
                        color='k', direction='in')
    cbar.ax.tick_params(which='major', size=10, width=1,
                        color='k', direction='in', labelsize=20)
    # fig.subplots_adjust(bottom=0.25)
    # cbar.ax.set_position([0.2, 0.08, 0.6, 0.08])
    return cbar


top = cm.get_cmap('YlGnBu_r', 128)  # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)  # combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))  # create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')

# %%
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2019/"

file = path + 'bottle_tn368.csv'
bottle = pd.read_csv(file)

# %%choose 14th july TRANSECT 37
# Parse datetime column
bottle['day'] = pd.to_datetime(bottle['day'])
# Filter rows where day is 14
tn37 = bottle[(bottle['day'].dt.day == 14)]

# remove nana
#TWO LINES WITH NO DATA => DELETE index 865 and 926
tn37 = tn37.drop([865, 926])
# %%
bathy = xr.open_dataset(bathy_path + "gebco_3d.nc")

lat_range = slice(38.7, 40.40)
lon_range = -70.8

bathy = bathy.sel(lat=lat_range)
bathy = bathy.sel(lon=lon_range, method='nearest')
depth = bathy.Z_3d_interpolated

#%%LOAD CTD TO ADD ISOPYCNAL
file = path + 'ctd_tn_withlocation.csv'
ctd = pd.read_csv(file)
ctd['day'] = pd.to_datetime(ctd['day'])
# Filter rows where day is 14
tn37_CTD = ctd[(ctd['day'].dt.day == 14)]


# %%
# EACH VARIABLE MUST BE INTERPOLATED
fig, ([ax, ax1], [ax2, ax3]) = plt.subplots(2, 2, dpi=200, figsize=([36, 20]))

# make a grid
# dx = np.diff(tn37.latitude)
# print(dx)

x = np.arange(38, 42, 0.064)  # 0.064 get from diff

y0 = np.arange(0, 15, 8)
y1 = np.arange(15, 34, 10)
y3 = np.arange(35, 100, 30)
y5 = np.arange(130, 150, 20)  # 150
# y5 = np.arange(141, 150, 20)
y6 = np.arange(160, 300, 50)  # 199

y = np.concatenate([y0, y1, y3,  y5, y6])
#


#### FIRST COLUMN #####
# NO3
binned = ss.binned_statistic_2d( tn37.latitude, tn37.depth, tn37.no3, statistic='mean', bins=[x, y])
binned_sigma = ss.binned_statistic_2d(tn37_CTD.lat, tn37_CTD.depth, tn37_CTD.density, statistic='mean', bins=[x, y])
binned_halin = ss.binned_statistic_2d(tn37_CTD.lat, tn37_CTD.depth, tn37_CTD.salinity, statistic='mean', bins=[x, y])


# to do a contour plot, you need to reference the center of the bins, not the edges
# get the bin centers
xc = (x[:-1] + x[1:]) / 2
yc = (y[:-1] + y[1:]) / 2

# plot the data
vmin = 0.0
vmax = 10
levels = np.linspace(vmin, vmax, 12)

sm = ax.contourf(xc, yc, binned.statistic.T, levels=levels, vmin=vmin, vmax=vmax, cmap=orange_blue,
                 extend='both')
lines =  ax.contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'mediumspringgreen', linestyles = '--', linewidths = 3)
lines1 =  ax.contour(xc, yc, binned_sigma.statistic.T, levels = [25.8], zorder = 2, colors = 'mediumspringgreen', linestyles = '--', linewidths = 3)
ax.contour(xc, yc, binned_halin.statistic.T,  levels = [34.5], zorder = 2, colors = 'crimson', linestyles = '--', linewidths = 3)
set_cbar(fig, sm, '[$\mu$mol/m$^3$]', ax)
ax.plot(bathy.lat, -depth.values, color='k', linewidth=3)
ax.set_title('Nitrate (NO$_{3^-}$)', fontsize=40)

# NH4
binned = ss.binned_statistic_2d(tn37.latitude, tn37.depth, tn37.nh4, statistic='mean', bins=[x, y],
                                expand_binnumbers=True)

# plot the data
vmin = 0.01
vmax = np.max(tn37.nh4)
levels = np.linspace(vmin, vmax, 12)

sm = ax2.contourf(xc, yc, binned.statistic.T, levels=levels, vmin=vmin, vmax=vmax,
                  extend='both', cmap=orange_blue)
ax2.contour(xc, yc, binned_halin.statistic.T,  levels = [34.5], zorder = 2, colors = 'crimson', linestyles = '--', linewidths = 3)
set_cbar(fig, sm, '[$\mu$mol/m$^3$]', ax2)
ax2.plot(bathy.lat, -depth.values, color='k', linewidth=3)
ax2.set_title('Ammonium (NH$_{4^+}$)', fontsize=40)

# SECOND COLUMN
# PO4
binned = ss.binned_statistic_2d(
    tn37.latitude, tn37.depth, tn37.po4, statistic='mean', bins=[x, y])

# plot the data
vmin = 0.0
vmax = 1.5
levels = np.linspace(vmin, vmax, 12)

sm = ax1.contourf(xc, yc, binned.statistic.T, levels=levels,
                  vmin=vmin, vmax=vmax, extend='both', cmap=orange_blue)
ax1.contour(xc, yc, binned_halin.statistic.T,  levels = [34.5], zorder = 2, colors = 'crimson', linestyles = '--', linewidths = 3)
set_cbar(fig, sm, '[$\mu$mol/m$^3$]', ax1)
ax1.plot(bathy.lat, -depth.values, color='k', linewidth=3)
ax1.set_title('Phosphate (PO$^{3^-}_4$)', fontsize=40)


# SI
binned = ss.binned_statistic_2d(
    tn37.latitude, tn37.depth, tn37.si, statistic='mean', bins=[x, y])

# plot the data
vmin = 0.0
vmax = 12
levels = np.linspace(vmin, vmax, 12)
cmap = orange_blue

sm = ax3.contourf(xc, yc, binned.statistic.T, levels=levels,
                  vmin=vmin, vmax=vmax, cmap=cmap, extend='both')
ax3.contour(xc, yc, binned_halin.statistic.T,  levels = [34.5], zorder = 2, colors = 'crimson', linestyles = '--', linewidths = 3)
set_cbar(fig, sm, '[$\mu$mol/m$^3$]', ax3)
ax3.plot(bathy.lat, -depth.values, color='k', linewidth=3)
ax3.set_title('Silicon (Si(OH)$_4$)', fontsize=45)

###PLOT SET UP##
ax.set_xlim(tn37.latitude.min(), tn37.latitude.max())
ax1.set_xlim(tn37.latitude.min(), tn37.latitude.max())
ax2.set_xlim(tn37.latitude.min(), tn37.latitude.max())
ax3.set_xlim(tn37.latitude.min(), tn37.latitude.max())

ax.set_ylim(0, 210)
ax1.set_ylim(0, 210)
ax2.set_ylim(0, 210)
ax3.set_ylim(0, 210)

ax.invert_yaxis()
ax.invert_xaxis()
ax1.invert_yaxis()
ax1.invert_xaxis()
ax2.invert_yaxis()
ax2.invert_xaxis()
ax3.invert_yaxis()
ax3.invert_xaxis()

ax1.xaxis.set_tick_params(labelsize=30)
ax1.yaxis.set_tick_params(labelsize=30)
ax.xaxis.set_tick_params(labelsize=30)
ax.yaxis.set_tick_params(labelsize=30)
ax2.xaxis.set_tick_params(labelsize=30)
ax2.yaxis.set_tick_params(labelsize=30)
ax3.xaxis.set_tick_params(labelsize=30)
ax3.yaxis.set_tick_params(labelsize=30)

ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=0)
ax1.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--',  zorder=0)
ax2.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=0)
ax3.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=0)

ax2.set_xlabel('Latitude [$\degree$N]', fontsize=40)
ax3.set_xlabel('Latitude [$\degree$N]', fontsize=40)

ax.set_ylabel('Depth [m]', fontsize=40)
ax2.set_ylabel('Depth [m]', fontsize=40)

fig.suptitle('Bottle data transect at 70.8$^\circ$W, July 14 2019',
             fontsize=50, y=0.98)
fig.tight_layout(pad=4)
fig.savefig(plot_path + 'bottle_tn37_14july.png', bbox_inches='tight')

# %%DATA DISTRIBUTION
fig, ax = plt.subplots(1, 1, dpi=200, figsize=([10, 6]))
ax.scatter(tn37.latitude, tn37.depth)
ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.invert_yaxis()
ax.invert_xaxis()
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.set_xlabel('Latitude [$\degree$N]', fontsize=25)
ax.set_ylabel('Depth [m]', fontsize=25)
ax.set_title('Bottle casts distribution', fontsize=25)
fig.tight_layout()
fig.savefig(plot_path + 'bottle_cast_distr.png')

# %%PON
#PON HAS MUCH MORE NAN VALUES, DELETE FROM TN37
tn37_nonan = tn37.dropna(axis=0)
#%%

fig, ax = plt.subplots(1, 1, dpi=200, figsize=([10, 6]))

x = np.arange(38, 42, 0.064)  # 0.064 get from diff

y0 = np.arange(0, 15, 8)
y1 = np.arange(15, 35, 10)
y3 = np.arange(38, 100, 30)
y5 = np.arange(130, 150, 20)  # 150
# y5 = np.arange(141, 150, 20)
y6 = np.arange(160, 300, 50)  # 199

y = np.concatenate([y0, y1, y3,  y5, y6])

binned = ss.binned_statistic_2d(tn37_nonan.latitude, tn37_nonan.depth, tn37_nonan.pon, statistic='mean', bins=[x, y])
binned_halin = ss.binned_statistic_2d(tn37_CTD.lat, tn37_CTD.depth, tn37_CTD.salinity, statistic='mean', bins=[x, y])
binned_sigma = ss.binned_statistic_2d(tn37_CTD.lat, tn37_CTD.depth, tn37_CTD.density, statistic='mean', bins=[x, y])

# to do a contour plot, you need to reference the center of the bins, not the edges
# get the bin centers
xc = (x[:-1] + x[1:]) / 2
yc = (y[:-1] + y[1:]) / 2

# plot the data
vmin = 0.0
vmax = np.max(tn37.pon)
levels = np.linspace(vmin, vmax, 12)

sm = ax.contourf(xc, yc, binned.statistic.T, levels=levels, vmin=vmin, vmax=vmax, cmap=orange_blue,
                 extend='both')
ax.contour(xc, yc, binned_halin.statistic.T,  levels = [34.5], zorder = 2, colors = 'crimson', linestyles = '--', linewidths = 3)
lines =  ax.contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'mediumspringgreen', linestyles = '--', linewidths = 3)
lines1 =  ax.contour(xc, yc, binned_sigma.statistic.T, levels = [25.8], zorder = 2, colors = 'mediumspringgreen', linestyles = '--', linewidths = 3)
ax.plot(bathy.lat, -depth.values, color='k', linewidth=3)
set_cbar(fig, sm, '[$\mu$mol/m$^3$]', ax)
ax.set_title('PON', fontsize=25)

ax.set_xlim(tn37.latitude.min(), tn37.latitude.max())
ax.set_ylim(0, 110)
ax.invert_yaxis()
ax.invert_xaxis()
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.set_xlabel('Latitude [$\degree$N]', fontsize=25)
ax.set_ylabel('Depth [m]', fontsize=25)
fig.tight_layout()
fig.savefig(plot_path + 'PON_14July.png')
