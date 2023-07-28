#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:26:37 2023

@author: serena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean
import scipy.stats as ss
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

def set_cbar(fig, c, title, j):
    cax = fig.add_axes([0.037 + j*0.246, -0.09, 0.2, 0.07]) #-0.01, 0.013   
    cbar = fig.colorbar(c, format='%.1f', spacing='proportional', cax=cax, shrink=0.9, pad = 6,
                        orientation = 'horizontal', location = "bottom")
    cbar.set_label(label = title, fontsize = 60, y = 0.5, labelpad = 30)
    cbar.ax.tick_params(which='minor', size=5, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=20, width=1, color='k', direction='in', labelsize = 50)
    return cbar
#%%

path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2019/july/"

file = path + 'ctd_tn_withlocation.csv'
ctd_data = pd.read_csv(file)
#%%choose
ctd_data['day'] = pd.to_datetime(ctd_data['day'])

#salinity check
ctd_data = ctd_data[ctd_data['salinity'] >= 25]
# ctd_data= ctd_data[(ctd_data['day'].dt.day == 6)]

#days 6, 7, 9, 14, 11-12, 16, 17-18
tn1 = ctd_data[(ctd_data['day'].dt.day == 6) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('00:07:10').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('17:58:56').time())]
tn3 = ctd_data[(ctd_data['day'].dt.day == 9) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('09:49:43').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('20:50:38').time())]
tn4 = ctd_data[(ctd_data['day'].dt.day == 11) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('18:21:50').time()) | (ctd_data['day'].dt.day == 12) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('13:51:23').time())]
tn5 = ctd_data[(ctd_data['day'].dt.day == 14) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('03:24:34').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('20:00:12').time())]
tn61 = ctd_data[(ctd_data['day'].dt.day == 16) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('00:39:25').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('13:18:41').time())]
tn62 = ctd_data[(ctd_data['day'].dt.day == 17) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('01:19:34').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('04:11:33').time())]

tn6 = pd.concat([tn61, tn62])
tn7 = ctd_data[(ctd_data['day'].dt.day == 17) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('12:39:06').time()) | (ctd_data['day'].dt.day == 18) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('05:45:55').time())]

combined_tn = pd.concat([tn1, tn3, tn4, tn5, tn6, tn7], ignore_index=True)

#save this dataframe as csv
combined_tn.to_csv(path + 'july19_onlygood.csv', index=False)
#%%depth profile
bathy = xr.open_dataset(bathy_path + "gebco_3d.nc")

lat_range = slice(38.7, 40.60)
lon_range = -70.8

bathy = bathy.sel(lat=lat_range)
bathy = bathy.sel(lon = lon_range, method = 'nearest')
depth = bathy.Z_3d_interpolated
#%%PLOT
#EACH VARIABLE MUST BE INTERPOLATED
transects = [tn1, tn3, tn4, tn5, tn6, tn7]
days = [6, 9, 11, 14, 16, 17]

# Loop over each DataFrame
# Create the figure and axes
fig, ax = plt.subplots(6, 4, dpi=50, figsize=([90, 84]))

for i, df in enumerate(transects):
    
    x = np.arange(38, 42, 0.064) #0.064 get from diff
    y = np.arange(0,  300, 1)
    xx, yy = np.meshgrid(x, y)
    
    ###TEMP
    
    binned = ss.binned_statistic_2d(df.lat, df.depth, df.temperature, statistic='mean', bins=[x, y])
    binned_sal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
    binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])

    # to do a contour plot, you need to reference the center of the bins, not the edges
    # get the bin centers
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    
    # plot the data
    vmin = 6
    vmax = 25
    levels = np.linspace(vmin,vmax, 15)
    
    sm = ax[i, 0].contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = 'cmo.thermal',
                     extend ='both')
    ax[i, 0].contour(xc, yc, binned_sal.statistic.T, levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 0].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 0].contour(xc, yc, binned_sigma.statistic.T, levels = [25.8], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 0].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 0].contour(xc, yc, binned_sigma.statistic.T, levels = [26.15], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 0].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
    ax[i, 0].text(x = 40.4, y = 200, s = "July " + str(days[i]), size = 60, color = 'k')
    ax[i, 0].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)

    ####FLUORECENCE
    binned = ss.binned_statistic_2d(df.lat, df.depth, df.fluorecence, statistic='mean', bins=[x, y],
                                    expand_binnumbers=True)    
    # plot the data
    vmin = 0
    vmax = 5
    # levels = np.arange(vmin,vmax, 0.05)
    levels = np.linspace(vmin,vmax, 15)
    
    
    sm1 = ax[2].contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, 
                      extend ='both', cmap = 'cmo.algae')
    ax[i, 2].contour(xc, yc, binned_sigma.statistic.T, levels = [26.15], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 2].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 2].contour(xc, yc, binned_sigma.statistic.T, levels = [25.8], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 2].contour(xc, yc, binned_sal.statistic.T, levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 2].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 2].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
    ax[i, 2].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
    
    ####SECOND COLUMN
    #SALINITY    
    # plot the data
    vmin = 31.8
    vmax = 36
    # levels = np.arange(vmin,vmax, 0.05)
    levels = np.linspace(vmin,vmax, 15)
    
    sm2 = ax[i, 1].contourf(xc, yc, binned_sal.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.haline')
    ax[i, 1].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 1].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    # ax1.clabel(lines,  inline_spacing = 0.001, fontsize=20, colors = 'crimson')
    # ax1.clabel(lines1,  manual = True, rightside_up = False,  fontsize=25, colors = 'crimson')
    ax[i, 1].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
    ax[i, 1].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
    ####SIGMA    
    # plot the data
    vmin = 23
    vmax = 27
    # levels = np.arange(vmin,vmax, 0.05)
    levels = np.linspace(vmin,vmax, 15)
    cmap = 'cmo.dense'
    
    sm3 = ax[i, 3].contourf(xc, yc, binned_sigma.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = cmap, extend ='both')
    ax[i, 3].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0],  zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 3].contour(xc, yc, binned_sigma.statistic.T, levels = [25.8], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 3].contour(xc, yc, binned_sigma.statistic.T, levels = [26.15], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    # ax3.clabel(lines2, fontsize=25, inline_spacing = 0.05,  colors = 'mediumspringgreen')
    ax[i, 3].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
    ax[i, 3].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
    ###PLOT SET UP## 
    ax[i, 0].set_ylim(0, 210)
    ax[i, 1].set_ylim(0, 210)
    ax[i, 2].set_ylim(0, 210)
    ax[i, 3].set_ylim(0, 210)
    
    ax[i, 0].set_xlim(39.6, 40.5)
    ax[i, 1].set_xlim(39.6, 40.5)
    ax[i, 2].set_xlim(39.6, 40.5)
    ax[i, 3].set_xlim(39.6, 40.5)
    
    ax[i, 0].invert_yaxis()
    ax[i, 0].invert_xaxis()
    ax[i, 1].invert_yaxis()
    ax[i, 1].invert_xaxis()
    ax[i, 2].invert_yaxis()
    ax[i, 2].invert_xaxis()
    ax[i, 3].invert_yaxis()
    ax[i, 3].invert_xaxis()
    
    ax[i, 1].xaxis.set_tick_params(labelsize = 40)
    ax[i, 1].yaxis.set_tick_params(labelsize = 40) 
    ax[i, 0].xaxis.set_tick_params(labelsize = 40)
    ax[i, 0].yaxis.set_tick_params(labelsize = 40) 
    ax[i, 2].xaxis.set_tick_params(labelsize = 40)
    ax[i, 2].yaxis.set_tick_params(labelsize = 40) 
    ax[i, 3].xaxis.set_tick_params(labelsize = 40)
    ax[i, 3].yaxis.set_tick_params(labelsize = 40) 
    
    ax[i, 0].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax[i, 1].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax[i, 2].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax[i, 3].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    ax[i, 0].set_ylabel('Depth [m]', fontsize = 60)
    
    if i == 5:
        ax[i, 2].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
        ax[i, 3].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
        ax[i, 0].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
        ax[i, 1].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
        
        set_cbar(fig, sm, '[$^\circ$C]', 0)
        set_cbar(fig, sm1, '[$\mu$g /m$^3$]', 2)
        set_cbar(fig, sm2, '[PSU]', 1) 
        set_cbar(fig, sm3, '[kg/m$^3$]', 3)
        
       
    if i == 0:
       ax[i, 0].set_title('Temperature', fontsize = 70)
       ax[i, 1].set_title('Salinity', fontsize = 70)
       ax[i, 2].set_title('Chlorophyll', fontsize = 70)
       ax[i, 3].set_title('Density ($\sigma_{\Theta}$)', fontsize = 70)
       
plt.subplots_adjust(hspace=0.5)      
fig.suptitle('CTD transect around 70.8$^\circ$W', fontsize = 90, y = 0.99)
fig.tight_layout(pad = 6) 
fig.savefig(plot_path + 'ctd_july19.png', bbox_inches='tight')

#%%17 july

fig, ax = plt.subplots(1, 4, dpi=50, figsize=([90, 15]))

x = np.arange(38, 42, 0.064) #0.064 get from diff
y = np.arange(0,  300, 1)
xx, yy = np.meshgrid(x, y)

###TEMP

binned = ss.binned_statistic_2d(tn6.lat, tn6.depth, tn6.temperature, statistic='mean', bins=[x, y])
binned_sal = ss.binned_statistic_2d(tn6.lat, tn6.depth, tn6.salinity, statistic='mean', bins=[x, y])
binned_sigma = ss.binned_statistic_2d(tn6.lat, tn6.depth, tn6.density, statistic='mean', bins=[x, y])

# to do a contour plot, you need to reference the center of the bins, not the edges
# get the bin centers
xc = (x[:-1] + x[1:]) / 2
yc = (y[:-1] + y[1:]) / 2

# plot the data
vmin = 6
vmax = 25
levels = np.linspace(vmin,vmax, 15)

sm = ax[0].contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = 'cmo.thermal',
                 extend ='both')
ax[0].contour(xc, yc, binned_sal.statistic.T, levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[0].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[0].contour(xc, yc, binned_sigma.statistic.T, levels = [25.8], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[0].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[0].contour(xc, yc, binned_sigma.statistic.T, levels = [26.15], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[0].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
ax[0].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)

####FLUORECENCE
binned = ss.binned_statistic_2d(tn6.lat, tn6.depth, tn6.fluorecence, statistic='mean', bins=[x, y],
                                expand_binnumbers=True)    
# plot the data
vmin = 0
vmax = 5
# levels = np.arange(vmin,vmax, 0.05)
levels = np.linspace(vmin,vmax, 15)


sm1 = ax[2].contourf(xc, yc, binned.statistic.T, levels = levels, vmin = vmin, vmax = vmax, 
                  extend ='both', cmap = 'cmo.algae')
ax[2].contour(xc, yc, binned_sigma.statistic.T, levels = [26.15], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[2].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[2].contour(xc, yc, binned_sigma.statistic.T, levels = [25.8], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[2].contour(xc, yc, binned_sal.statistic.T, levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[2].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[2].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
ax[2].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)

####SECOND COLUMN
#SALINITY    
# plot the data
vmin = 31.8
vmax = 36
# levels = np.arange(vmin,vmax, 0.05)
levels = np.linspace(vmin,vmax, 15)

sm2 = ax[1].contourf(xc, yc, binned_sal.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.haline')
ax[1].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[1].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
# ax1.clabel(lines,  inline_spacing = 0.001, fontsize=20, colors = 'crimson')
# ax1.clabel(lines1,  manual = True, rightside_up = False,  fontsize=25, colors = 'crimson')
ax[1].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
ax[1].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
####SIGMA    
# plot the data
vmin = 23
vmax = 27
# levels = np.arange(vmin,vmax, 0.05)
levels = np.linspace(vmin,vmax, 15)
cmap = 'cmo.dense'

sm3 = ax[ 3].contourf(xc, yc, binned_sigma.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = cmap, extend ='both')
ax[3].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0],  zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[3].contour(xc, yc, binned_sigma.statistic.T, levels = [25.8], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[3].contour(xc, yc, binned_sigma.statistic.T, levels = [26.15], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
# ax3.clabel(lines2, fontsize=25, inline_spacing = 0.05,  colors = 'mediumspringgreen')
ax[3].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
ax[3].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
###PLOT SET UP## 
ax[0].set_ylim(0, 210)
ax[1].set_ylim(0, 210)
ax[2].set_ylim(0, 210)
ax[3].set_ylim(0, 210)

ax[0].set_xlim(39.6, 40.5)
ax[1].set_xlim(39.6, 40.5)
ax[2].set_xlim(39.6, 40.5)
ax[3].set_xlim(39.6, 40.5)

ax[ 0].invert_yaxis()
ax[ 0].invert_xaxis()
ax[1].invert_yaxis()
ax[1].invert_xaxis()
ax[2].invert_yaxis()
ax[2].invert_xaxis()
ax[3].invert_yaxis()
ax[3].invert_xaxis()

ax[1].xaxis.set_tick_params(labelsize = 40)
ax[1].yaxis.set_tick_params(labelsize = 40) 
ax[0].xaxis.set_tick_params(labelsize = 40)
ax[ 0].yaxis.set_tick_params(labelsize = 40) 
ax[2].xaxis.set_tick_params(labelsize = 40)
ax[2].yaxis.set_tick_params(labelsize = 40) 
ax[3].xaxis.set_tick_params(labelsize = 40)
ax[3].yaxis.set_tick_params(labelsize = 40) 

ax[ 0].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax[1].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax[2].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax[3].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')

ax[0].set_ylabel('Depth [m]', fontsize = 60)


ax[2].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
ax[3].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
ax[0].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
ax[1].set_xlabel('Latitude [$\degree$N]', fontsize = 60)

set_cbar(fig, sm, '[$^\circ$C]', 0)
set_cbar(fig, sm1, '[$\mu$g /m$^3$]', 2)
set_cbar(fig, sm2, '[PSU]', 1) 
set_cbar(fig, sm3, '[kg/m$^3$]', 3)
 

ax[0].set_title('Temperature', fontsize = 70)
ax[1].set_title('Salinity', fontsize = 70)
ax[2].set_title('Chlorophyll', fontsize = 70)
ax[3].set_title('Density ($\sigma_{\Theta}$)', fontsize = 70)
   
plt.subplots_adjust(hspace=0.5)      
fig.suptitle('CTD transect around 70.8$^\circ$W, July 17th 2019', fontsize = 90, y = 0.99)
fig.tight_layout(pad = 6) 
fig.savefig(plot_path + 'ctd_17july1.png', bbox_inches='tight')

