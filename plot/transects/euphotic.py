#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:54:13 2024

@author: serena
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:58:02 2023

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
from matplotlib import ticker
from scipy.integrate import odeint
import math



def set_cbar(fig, c, title):
    cax = fig.add_axes([1.0, 0.15, 0.01, 0.8])  # Adjust the position and size as needed
    cbar = fig.colorbar(c, format='%.1f', spacing='proportional', cax=cax, shrink=0.9, pad=0.02,
                        orientation='vertical', location="right")
    cbar.set_label(label = title, fontsize = 60, y = 0.5, labelpad = 30)
    cbar.ax.tick_params(which='minor', size=5, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=20, width=1, color='k', direction='in', labelsize = 60)
    return cbar


def filter_daytime(df):
    start_time = pd.to_datetime('7:30:00').time()
    end_time = pd.to_datetime('18:30:00').time()
    filtered_df = df.between_time(start_time, end_time)
    return filtered_df

def filter_nighttime(df):
    start_time = pd.to_datetime('19:30:00').time()
    end_time = pd.to_datetime('6:00:00').time()
    filtered_df = df.between_time(start_time, end_time)
    return filtered_df


def dI_dz(I, kd):
    return -kd * I
#%%
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2018/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2018/"
figures = '/home/serena/Scrivania/Magistrale/thesis/figures_paper/'
boxes_path = '/home/serena/Scrivania/Magistrale/thesis/deltas/'

file = path + 'ctd_tn_withlocation_apr18.csv'
ctd_data = pd.read_csv(file)
ctd_data = ctd_data.dropna(subset=['station'])
#%%choose
ctd_data['day'] = pd.to_datetime(ctd_data['day'])
ctd_data = ctd_data[ctd_data['lon'] != 70.81]

tn1 = ctd_data[(ctd_data['day'].dt.day == 17) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('13:15:51').time()) | (ctd_data['day'].dt.day == 18) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('22:50:06').time())]
tn1 = tn1[tn1['Cast_num'] != 6]

tn2 = ctd_data[(ctd_data['day'].dt.day == 19) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('10:36:30').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('22:44:02').time())]
tn3 = ctd_data[(ctd_data['day'].dt.day == 21) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('06:43:32').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('18:51:18').time())]
tn4 = ctd_data[(ctd_data['day'].dt.day == 23) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('09:12:10').time()) | (ctd_data['day'].dt.day == 24) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('00:54:13').time())]
tn5 = ctd_data[(ctd_data['day'].dt.day == 25) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('15:33:00').time()) | (ctd_data['day'].dt.day == 26) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('10:27:27').time())]
tn6 = ctd_data[(ctd_data['day'].dt.day == 27) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('10:27:54').time()) | (ctd_data['day'].dt.day == 28) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('10:30:15').time())]

combined_tn = pd.concat([tn1, tn2, tn3, tn5, tn5, tn6], ignore_index=True)
transects = [tn1, tn2, tn3,  tn4, tn5, tn6]
#%%plot each station par profile to evaluate if there are some problematic features
days = [17, 19, 21,  23, 25, 27]

daytime_transects = {}
nighttime_transects = {}

# Apply the filtering function to each DataFrame and store the filtered DataFrame in the dictionary
for dd, idx in zip(transects, range(len(transects))):
    df = transects[idx]
    df.set_index('day', inplace=True)
    daytime_transects[idx] = filter_daytime(df)
    nighttime_transects[idx] = filter_nighttime(df)
    
 #%%   
light = {}

for df_name, j in zip(daytime_transects, range(len(daytime_transects))):
    tn = daytime_transects[j]
# for df_name, j in zip(transects, range(len(transects))):
#     tn = transects[j]


    station = np.array(np.unique(tn.Cast_num))
    station_data = []

    for i, s in enumerate(station):
        st = tn[tn['Cast_num'] == s]
        
        if len(st) < 10:
            continue
        
        depth = np.array(st['depth'])
        par = np.array(st['par'])
        spar = np.array(st['spar'])
        lat = np.array(st['lat'])
        
        # Sort depth and par based on the sorted index of depth
        sorted_indices = np.argsort(depth)
        depth = depth[sorted_indices]
        par = par[sorted_indices]
        spar = spar[sorted_indices]
        
        spar_surf = spar[0]
       

        # if spar_surf == -9.99e-29:
        while spar_surf < 20:
            # Move to the next cast by incrementing the cast index
            i += 1
            if i >= len(station):
                break  # Break if there are no more casts
            s = station[i]
            st = tn[tn['Cast_num'] == s]
            spar_surf = st['spar'].values[0]
  
        if spar_surf == -9.99e-29:
            # If there are no valid casts, you can handle it here
            continue
    
        euph = spar_surf * 0.01
        print(spar_surf)
        
        index_1_percent = min(range(len(par)), key=lambda i: abs(par[i] - euph))
        depth_euph = depth[index_1_percent]
        
        if depth_euph > 10:
        
            station_dict = {'Cast': s, 'depth_euph': depth_euph, 'latitude' : np.max(lat)}
            station_data.append(station_dict)

    light[j] = station_data
    
    #%%
light_2d = {}
lat_2d = {}
grid_size = 21


for k in range(len(transects)):
    print(k)
    df = light[k]  # Access the data for the current transect from the 'light' dictionary
    df = pd.DataFrame(df)
    
    tns = transects[k]
    cast_len = np.unique(tns.Cast_num)
    new_latitudes = np.linspace(np.min(tns.lat), np.max(tns.lat), len(cast_len))

    # Get all unique rawprofile field names
    stations = [field for field in df.columns if field.startswith('Cast')]

    light_grid = np.full((grid_size, len(stations)), np.nan)
    lat_grid = np.full((grid_size, len(stations)), np.nan)


    for i, profile_field in enumerate(stations):
        #profile = df[profile_field]    
        # Access other profile variables and reorder them based on sorted indices
        profile_light = df['depth_euph']
        
        profile_lat = df['latitude']

        #  Store the sorted variables in the grid
        light_grid[:len(profile_light), i] = profile_light
        lat_grid[:len(profile_light), i] = profile_lat
        
        # Remove NaN rows from both light_grid and lat_grid
        light_grid_without_nan = light_grid[~np.isnan(light_grid).all(axis=1)]
        lat_grid_without_nan = lat_grid[~np.isnan(lat_grid).all(axis=1)]

        # Sort both light_grid and lat_grid based on lat_grid values
        sort_indices = np.argsort(lat_grid_without_nan[:, i])
        lat_grid_sorted = lat_grid_without_nan[sort_indices, i]
        light_grid_sorted = light_grid_without_nan[sort_indices, i]
        
        # Perform interpolation using np.interp
        interpolated_light = np.interp(new_latitudes, lat_grid_sorted, light_grid_sorted)


    lat_2d[k] = new_latitudes
    light_2d[k] = interpolated_light
    
 #%%   
bathy = xr.open_dataset(bathy_path + "gebco_3d.nc")

lat_range = slice(38.7, 40.60)
lon_range = -70.83

bathy = bathy.sel(lat=lat_range)
bathy = bathy.sel(lon = lon_range, method = 'nearest')
depth_contour = bathy.Z_3d_interpolated
#%%READ COORDS BOXES
bottom = pd.read_excel(boxes_path + 'bottom_box_april.xlsx')
surface = pd.read_excel(boxes_path + 'surface_box_april.xlsx')

#%%
transects = [tn1, tn2, tn3,  tn4, tn5, tn6]
days = [17, 19, 21,  23, 25, 27]

fig, ax = plt.subplots(1, 6, dpi=50, figsize=([90,15]))

for (i, df), dfday in zip(enumerate(transects), transects):
    x = np.arange(38, 42, 0.064) #0.064 get from diff
    y = np.arange(0,  220, 3)
    xx, yy = np.meshgrid(x, y)
    
    fluor = 0.9995 * df.fluorecence + 0.3169
    binned_chlor = ss.binned_statistic_2d(df.lat, df.depth, fluor, statistic='mean', bins=[x,y], expand_binnumbers=True)
    binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
    binned_hal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
    # to do a contour plot, you need to reference the center of the bins, not the edges
    # get the bin centers
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    
    # plot the data
    vmin = 0
    vmax = 5
    levels = np.linspace(vmin,vmax, 15)
    
    
    sm = ax[i].contourf(xc, yc, binned_chlor.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = 'cmo.algae',
                     extend ='both', zorder = -1)
    ax[i].plot(bathy.lat, -depth_contour.values, color = 'k', linewidth = 3)
    ax[i].text(x = 40.5, y = 200, s = "April " + str(days[i]), size = 60, color = 'k')
    ax[i].fill_between(bathy.lat, -depth_contour.values[:, 0], 210, color='lightgrey', alpha=0.7)
    lines5 = ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [0.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    lines1 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6],  zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    lines2 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.5],  zorder = 2, colors = 'red', alpha = .4, linestyles = '-', linewidths = 5)
    lines3 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.4],  zorder = 2, colors = 'red', alpha = .4, linestyles = '-', linewidths = 5)
    ax[i].contourf(xc, yc, binned_sigma.statistic.T, levels = [26.3, 26.65], colors='pink', alpha=0.4)
    lines5 = ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
   # ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [34.0], zorder = 2, colors = 'blue', linestyles = '-', linewidths = 5)
    ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [34.75], zorder = 2, colors = 'k', linestyles = '--', linewidths = 5)
    ax[i].plot(lat_2d[i], light_2d[i], color = 'yellow', linewidth = 15, linestyle = 'dotted')
    labels = ['Target isopycnals:', '26.4 $\sigma$', '26.5 $\sigma$', '26.6 $\sigma$']#,  '26.2 $\sigma$']
    
    yt = [surface['depth min'][i], surface['depth max'][i], surface['depth max'][i], surface['depth min'][i], surface['depth min'][i]]
    xt = [surface['Lat max'][i], surface['Lat max'][i], surface['lat min'][i], surface['lat min'][i], surface['Lat max'][i]]
    ax[i].plot(xt, yt, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)

    yb = [bottom['depth min'][i], bottom['depth max'][i], bottom['depth max'][i], bottom['depth min'][i], bottom['depth min'][i]]
    xb = [bottom['Lat max'][i], bottom['Lat max'][i], bottom['lat min'][i], bottom['lat min'][i], bottom['Lat max'][i]]
    ax[i].plot(xb, yb, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)

    for j in range(len(labels)):
        ax[0].collections[j].set_label(labels[j])
        
    legend= fig.legend(loc='lower left', bbox_to_anchor=(0.02, -0.2), ncol = 5,
                      fontsize = 75, labelcolor = ['black', 'tomato', 'tomato', 'red'], facecolor='whitesmoke')

    ###PLOT SET UP## 
    ax[i].set_ylim(0, 210)
    
    ax[i].set_xlim(39.6, 40.6)
    
    ax[i].invert_yaxis()
    ax[i].invert_xaxis()
    
    ax[i].xaxis.set_tick_params(labelsize = 40)
    ax[i].yaxis.set_tick_params(labelsize = 40) 
    
    ax[i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')     
    
    if i == 0:
        ax[i].set_ylabel('Depth [m]', fontsize = 60) 
    
    ax[i].set_xlabel('Latitude [$\degree$N]', fontsize = 60) 
    set_cbar(fig, sm, 'Chlorophyll [mg/m$^3$]')
plt.subplots_adjust(hspace=0.5)      
# fig.suptitle('Position of the foot of 26.4 $\sigma$ vs 34.5 PSU (April 2018)', fontsize = 80, y = 0.95)
fig.tight_layout(pad = 4) 
fig.savefig(figures + 'figure_euphotic_2018_ext.png', bbox_inches='tight')

#%%
transects = [tn2, tn6]     
days = [19, 27]

fig, ax = plt.subplots(1, 2, dpi=100, figsize=([45,18]))

for (i, df), dfday in zip(enumerate(transects), transects):
    x = np.arange(38, 42, 0.064) #0.064 get from diff
    y = np.arange(0,  220, 3)
    xx, yy = np.meshgrid(x, y)
    
    fluor = 0.9995 * df.fluorecence + 0.3169
    binned_chlor = ss.binned_statistic_2d(df.lat, df.depth, fluor, statistic='mean', bins=[x,y], expand_binnumbers=True)
    binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
    binned_hal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
    # to do a contour plot, you need to reference the center of the bins, not the edges
    # get the bin centers
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    
    # plot the data
    vmin = 0
    vmax = 5
    levels = np.linspace(vmin,vmax, 15)
    
    
    sm = ax[i].contourf(xc, yc, binned_chlor.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = 'cmo.algae',
                     extend ='both', zorder = -1)
    ax[i].plot(bathy.lat, -depth_contour.values, color = 'k', linewidth = 3)
    ax[i].text(x = 40.5, y = 200, s = "April " + str(days[i]), size = 90, color = 'k')
    ax[i].fill_between(bathy.lat, -depth_contour.values[:, 0], 210, color='lightgrey', alpha=0.7)
    lines5 = ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [0.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    lines1 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6],  zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    lines2 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.5],  zorder = 2, colors = 'red', alpha = .4, linestyles = '-', linewidths = 5)
    lines3 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.4],  zorder = 2, colors = 'red', alpha = .4, linestyles = '-', linewidths = 5)
    ax[i].contourf(xc, yc, binned_sigma.statistic.T, levels = [26.3, 26.65], colors='pink', alpha=0.4)
    lines5 = ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [34.75], zorder = 2, colors = 'k', linestyles = '--', linewidths = 5)

    labels = ['Target isopycnals:', '26.4 $\sigma$', '26.5 $\sigma$', '26.6 $\sigma$']#,  '26.2 $\sigma$']
    
    if i ==0:
        ax[i].plot(lat_2d[1], light_2d[1], color = 'yellow', linewidth = 15, linestyle = 'dotted')
        ax[i].yaxis.set_tick_params(labelsize = 70) 
        xb = [40.35, 40.35, 40.23, 40.23, 40.35]
        yb = [70, 105, 105, 70, 70]
        ax[i].plot(xb, yb, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)
        xt = [40.0927232829477, 40.0927232829477, 40.0098333333333, 40.0098333333333, 40.0927232829477]
        yt = [30.6654721570575, 45, 45, 30.6654721570575, 30.6654721570575]
        ax[i].plot(xt, yt, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)
    else:
       ax[i].plot(lat_2d[5], light_2d[5], color = 'yellow', linewidth = 15, linestyle = 'dotted')
       ax[i].yaxis.set_tick_params(labelsize = 0)
       xb = [40.33, 40.33, 40.2444677584976, 40.2444677584976, 40.33]
       yb = [75, 96, 96, 75, 75]
       ax[i].plot(xb, yb, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)
       xt = [40.1497723886511, 40.149772388651, 40.0771441007566, 40.0771441007566, 40.149772388651]
       yt = [35, 60, 60, 35, 35]
       ax[i].plot(xt, yt, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)
       
    for j in range(len(labels)):
        ax[0].collections[j].set_label(labels[j])
        
    legend= fig.legend(loc='lower left', bbox_to_anchor=(0.02, -0.2), ncol = 5,
                      fontsize = 90, labelcolor = ['red', 'tomato', 'tomato','red'], facecolor='whitesmoke')

    ###PLOT SET UP## 
    ax[i].set_ylim(0, 210)
    
    ax[i].set_xlim(39.6, 40.6)
    
    ax[i].invert_yaxis()
    ax[i].invert_xaxis()
    
    ax[i].xaxis.set_tick_params(labelsize = 70) 
    
    ax[i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')     
    
    if i == 0:
        ax[i].set_ylabel('Depth [m]', fontsize = 90) 
    
    ax[i].set_xlabel('Latitude [$\degree$N]', fontsize = 90) 
    set_cbar(fig, sm, 'Chlorophyll [mg/m$^3$]')
plt.subplots_adjust(hspace=2)      
# fig.suptitle('Position of the foot of 26.4 $\sigma$ vs 34.5 PSU (April 2018)', fontsize = 80, y = 0.95)
fig.tight_layout(pad = 6) 
fig.savefig(figures + 'figure_euphotic_2018.png', bbox_inches='tight')

#%%
def filter_daytime(df):
    start_time = pd.to_datetime('8:30:00').time()
    end_time = pd.to_datetime('18:00:00').time()
    filtered_df = df.between_time(start_time, end_time)
    return filtered_df
#%%
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2019/july/"
figures = "/home/serena/Scrivania/Magistrale/thesis/figures_paper/"
file = path + 'ctd_tn_withlocation_jul19.csv'
ctd_data = pd.read_csv(file)
ctd_data = ctd_data.dropna(subset=['station'])

bottom = pd.read_excel(boxes_path + 'bottom_box_july.xlsx')
surface = pd.read_excel(boxes_path + 'surface_box_july.xlsx')
#%%choose
ctd_data['day'] = pd.to_datetime(ctd_data['day'])

#salinity check
ctd_data = ctd_data[ctd_data['salinity'] >= 25]

tn1 = ctd_data[(ctd_data['day'].dt.day == 6) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('00:07:10').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('17:58:56').time())]
tn2 = ctd_data[(ctd_data['day'].dt.day == 9) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('09:49:43').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('20:50:38').time())]
#delete cast 48 only two elements, wrong values in 1%light level
tn2 = tn2[tn2['Cast_num'] != 48]
tn2 = tn2[tn2['Cast_num'] != 42]
tn3 = ctd_data[(ctd_data['day'].dt.day == 11) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('18:21:50').time()) | (ctd_data['day'].dt.day == 12) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('13:51:23').time())]
tn4 = ctd_data[(ctd_data['day'].dt.day == 14) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('03:24:34').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('20:00:12').time())]
tn51 = ctd_data[(ctd_data['day'].dt.day == 16) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('00:39:25').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('13:18:41').time())]
tn52 = ctd_data[(ctd_data['day'].dt.day == 17) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('01:19:34').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('04:11:33').time())]

tn5 = pd.concat([tn51, tn52])
tn6 = ctd_data[(ctd_data['day'].dt.day == 17) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('12:39:06').time()) | (ctd_data['day'].dt.day == 18) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('05:45:55').time())]

combined_tn = pd.concat([tn1, tn2, tn3, tn4, tn5, tn6], ignore_index=True)

transects = [tn1, tn2, tn3, tn4, tn5, tn6]  
#%%
days = [6, 9, 11, 14, 16, 17]

clean_transects = {}
for df_name, j in zip(transects, range(len(transects))):
    tn = df_name

    station = np.array(np.unique(tn.Cast_num), dtype = 'int')

    for i, s in enumerate(station):
        st = tn[tn['Cast_num'] == s]
        
        depth = np.array(st['depth'])
        par = np.array(st['par'])
        
        # Sort depth and par based on the sorted index of depth
        sorted_indices = np.argsort(depth)
        depth = depth[sorted_indices]
        par = par[sorted_indices]
        
        # plt.figure() 
        # plt.plot(par, -depth)
        # plt.xlabel('PAR')
        # plt.ylabel('Depth')
        # plt.title(f"Profile" + str(days[j]) +" July, Cast" + str(s))
        
        df = tn.copy()
        #select only the good daytime transects and stations 
        # + new correction based on log(par) - depth
        if j == 0: #6 july
            mask = (df['Cast_num'] == 1) 
            df = df[~mask]
            mask = (df['Cast_num'] == 6) 
            df = df[~mask]
            mask = (df['Cast_num'] == 2) 
            df = df[~mask]
        elif j == 1:
            mask = (df['Cast_num'] == 51) 
            df = df[~mask]
            
        elif j == 2: #11 july
            mask = (df['Cast_num'] == 78) #ends at 30 m
            df = df[~mask]
            
        elif j == 3: #14 july
            mask = (df['Cast_num'] == 92)
            df = df[~mask]
            mask = (df['Cast_num'] == 94)
            df = df[~mask]
            
        elif j == 4: #16 july
            mask = (df['Cast_num'] == 11)
            df = df[~mask]
            
        elif j == 5:
            mask = (df['Cast_num'] == 109)
            df = df[~mask]
            
        clean_transects[j] = df
                
#%%choose only daytime cast
daytime_transects = {}

# Apply the filtering function to each DataFrame and store the filtered DataFrame in the dictionary
for dd, idx in zip(clean_transects, range(len(clean_transects))):
    df = clean_transects[idx]
    df.set_index('day', inplace=True)
    daytime_transects[f'tn{idx}'] = filter_daytime(df)
       
#%%
#light profile into the water I(x,t) = I0 exp(k_w + k_p*chl)
#where I(x, t) = PAR, I0 = SPAR, k_w = 0.04, chl = fluorescence
#need to estimate k_p 
#trail and errro method knowing that (from reference) k_d = k_w + k_p = [0.1 - 0.2]
#LHS from mesuremts must be equal to RHS esimated
best_kp = {}
for df, j in zip(daytime_transects, range(len(daytime_transects))):
    tn = daytime_transects[df]
    kz = 0.04 
    station = np.array(np.unique(tn.Cast_num), dtype = 'int')
    kplist = np.arange(0.01, 0.41, 0.01)
    
    
    bestk = np.full(len(station), np.nan)
    figure1 = plt.figure(1)
    plotcind = 0
    cmap = plt.cm.get_cmap('jet', len(station))
    stationlegend = []
    
    for i, s in enumerate(station):
        st = tn[tn['Cast_num'] == s]
        
        idpth = np.array(st['depth'])
        par = np.array(st['par'] / 4.6)
        iflor = np.array(st['fluorecence'] * 1.09 - 0.043)
        
        partest = par.copy()
        for d in range(1, len(idpth)):
            if not np.isnan(par[d - 1]):
                didz = partest[d - 1] * -kz
                partest[d] = partest[d - 1] + didz
    
        # Find phytn light attenuation coeff
        phytn = iflor
    
        partest2 = par.copy()
        ssd = np.zeros(len(kplist))
        for k_i, kp in enumerate(kplist):
            for d in range(1, len(idpth)):
                if not np.isnan(par[d - 1]):
                    partest2[d] = partest2[d - 1] + partest2[d - 1] * (-kz + -kp * phytn[d - 1])
    
            #X = par[dkey:60]/np.max(par) - partest2[dkey:60]/np.max(par);
            X = par/np.max(par) - partest2/np.max(par);
            ssd[k_i] = np.sum(X ** 2, where=~np.isnan(X))
    
        minki = np.argmin(ssd)
        bestk[i] = kplist[minki]
        plt.plot(kplist, np.log(ssd), linewidth=2, color=cmap(i), label = s)
        plt.legend( loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol = len(station)+2)

    best_kp[j] = bestk
    
    stationlegend.append('mean')
    stationlegend.append('median')
    # plt.xlabel('kp ([Chl]^-1 m^-1)')
    # plt.ylabel('log(sum sq. diff)')
    # plt.axvline(np.nanmean(bestk), color='r', linestyle='-', label='mean')
    # plt.axvline(np.nanmedian(bestk), color='b', linestyle='--', label='median')
    # plt.legend()
    # plt.show()
    
    # figure2 = plt.figure(2)
    # plt.hist(bestk[~np.isnan(bestk)], bins=10)
    # plt.xlabel('best kp')
    # plt.ylabel('number of casts')
    # plt.axvline(np.nanmean(bestk), color='r', linestyle='-', label='mean')
    # plt.axvline(np.nanmedian(bestk), color='b', linestyle='--', label='median')
    # plt.legend()
    # plt.show()


#%%
# Load CTD data again

light = {}
for k in range(len(clean_transects)):  
    tn = clean_transects[k]
    station = np.array(np.unique(tn.Cast_num), dtype = 'int')
    kw = 0.04
    kp = best_kp[k]
    kp = np.mean(kp)
    
    station_data = []
    
    for i, s in enumerate(station):
        st = tn[tn['Cast_num'] == s]
        
        if len(st) < 40:
            continue

        z = np.array(st['depth'])
        par = np.array(st['par'] / 4.6)
        spar = np.array(st['spar'])
        phytn = np.array(st['fluorecence']) * 1.09 - 0.043
        
        sorted_indices = np.argsort(z)
        z = z[sorted_indices]
        par = par[sorted_indices]
        spar = spar[sorted_indices]
        phytn = phytn[sorted_indices]

        kd = kw + kp * phytn
    
        # Findi I initial values corrwspondong to z = 0
        I_surf = par[0]
        
        #initial condition
        z_values = []
        I = []
        h = np.diff(z)
        
        z_ini = z[0]
        I_ini = I_surf *np.exp(-kd * z_ini)
             
        # Forward Euler method to numerically solve the ODE
        for zz, i in zip(z, range(1, len(z))):
            z_values.append(zz)
    
            # Euler's method
            I_ini[i] = I_ini[i] * np.exp(-kd[i] * h[i -1])
            I.append(I_ini)
            

        # Find the 1% I curve level
        one_percent_level = 0.01 * spar
        
        # Find the index where I is closest to the 1% I curve level
        index_closest = np.argmin(np.abs(I - one_percent_level))
        
        # Find the corresponding z value at the 1% I curve level
        z_at_1percent = z[index_closest]
        
        # Store the station data in a dictionary
        station_dict = {'Cast': s, 'z_at_1percent': z_at_1percent}
        station_data.append(station_dict)
    
    # Store the station data list for the current transect in the 'light' dictionary
    light[k] = station_data
#%%
light_2d = {}
grid_size = 18

sorted_indices = []  # To store the sorted indices for depth
sorted_in = []
for k in range(len(clean_transects)):
    print(k)
    df = light[k]  # Access the data for the current transect from the 'light' dictionary
    df = pd.DataFrame(df)

    # Get all unique rawprofile field names
    stations = [field for field in df.columns if field.startswith('Cast')]

    light_grid = np.full((grid_size, len(stations)), np.nan)

    for i, profile_field in enumerate(stations):
        #profile = df[profile_field]    
        # Access other profile variables and reorder them based on sorted indices
        profile_light = df['z_at_1percent']

        # Store the sorted variables in the grid
        light_grid[:len(profile_light), i] = profile_light
        
    light_2d[k] = light_grid
    # Filter out NaN values from each column of the light_grid
    light_grid_without_nan = light_grid[~np.isnan(light_grid).all(axis=1)]
    light_2d[k] = light_grid_without_nan


#%%
fig, ax = plt.subplots(1, 6, dpi=50, figsize=([90,15]))

for i, df in enumerate(transects):
    x = np.arange(38, 42, 0.064) #0.064 get from diff
    y = np.arange(0, 300, 1)
    xx, yy = np.meshgrid(x, y)
    
    fluor = 1.0932*df.fluorecence -0.04250
    binned_chlor = ss.binned_statistic_2d(df.lat, df.depth, fluor, statistic='mean', bins=[x,y], expand_binnumbers=True)
    binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
    binned_hal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
    # to do a contour plot, you need to reference the center of the bins, not the edges
    # get the bin centers
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    
    # plot the data
    vmin = 0
    vmax = 5
    levels = np.linspace(vmin,vmax, 15)
       
    sm = ax[i].contourf(xc, yc, binned_chlor.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = 'cmo.algae',
                     extend ='both', zorder = -1)
    ax[i].plot(bathy.lat, -depth_contour.values, color = 'k', linewidth = 3)
    ax[i].text(x = 40.5, y = 200, s = "July " + str(days[i]), size = 60, color = 'k')
    ax[i].fill_between(bathy.lat, -depth_contour.values[:, 0], 210, color='lightgrey', alpha=0.7)
    lines5 = ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [0.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    lines1 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.3],  zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    lines2 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.15],  zorder = 2, colors = 'red', alpha = .4, linestyles = '-', linewidths = 5)
    lines3 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0],  zorder = 2, colors = 'red', alpha = .4, linestyles = '-', linewidths = 5)
    ax[i].contourf(xc, yc, binned_sigma.statistic.T, levels = [25.95, 26.4], colors='pink', alpha=0.4)
    lines5 = ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [34.75], zorder = 2, colors = 'k', linestyles = '--', linewidths = 5)
    lat = np.linspace(np.min(df.lat), np.max(df.lat), len(light_2d[i].ravel()))
   # ax[i].scatter(df.lat, df.depth)
    ax[i].plot(lat, light_2d[i], color = 'yellow', linewidth = 15, linestyle = 'dotted')
    labels = ['Target isopycnals:',  '26.0 $\sigma$', '26.15 $\sigma$','26.3 $\sigma$']#,  '25.75 $\sigma$']
    
    yt = [surface['depth min'][i], surface['depth max'][i], surface['depth max'][i], surface['depth min'][i], surface['depth min'][i]]
    xt = [surface['Lat max'][i], surface['Lat max'][i], surface['lat min'][i], surface['lat min'][i], surface['Lat max'][i]]
    ax[i].plot(xt, yt, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)

    yb = [bottom['depth min'][i], bottom['depth max'][i], bottom['depth max'][i], bottom['depth min'][i], bottom['depth min'][i]]
    xb = [bottom['Lat max'][i], bottom['Lat max'][i], bottom['lat min'][i], bottom['lat min'][i], bottom['Lat max'][i]]
    ax[i].plot(xb, yb, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)
    
    for j in range(len(labels)):
        ax[0].collections[j].set_label(labels[j])
        
    legend= fig.legend(loc='lower left', bbox_to_anchor=(0.02, -0.2), ncol = 4,
                      fontsize = 75, labelcolor = ['black',  'tomato', 'tomato', 'red'], facecolor='whitesmoke')

    ###PLOT SET UP## 
    ax[i].set_ylim(0, 210)
    
    ax[i].set_xlim(39.6, 40.6)
    
    ax[i].invert_yaxis()
    ax[i].invert_xaxis()
    
    ax[i].xaxis.set_tick_params(labelsize = 40)
    ax[i].yaxis.set_tick_params(labelsize = 40) 
    
    ax[i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')     
    
    if i == 0:
        ax[i].set_ylabel('Depth [m]', fontsize = 60) 
    
    ax[i].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
    set_cbar(fig, sm, 'Chlorophyll [mg/m$^3$]')
plt.subplots_adjust(hspace=0.5)      
# fig.suptitle('Position of the foot of 26.4 $\sigma$ vs 34.5 PSU (May 2019)', fontsize = 80, y = 0.95)
fig.tight_layout(pad = 4) 
fig.savefig(figures + 'figure_euphotic_july_ext.png', bbox_inches='tight')

#%%ONLY 16th JULY
transects = [tn1, tn2]
days = [6, 9]

fig, ax = plt.subplots(1, 2, dpi=100, figsize=([45,18]))

for i, df in enumerate(transects):
    x = np.arange(38, 42, 0.064) #0.064 get from diff
    y = np.arange(0, 300, 1)
    xx, yy = np.meshgrid(x, y)
    
    fluor = 1.0932*df.fluorecence -0.04250
    binned_chlor = ss.binned_statistic_2d(df.lat, df.depth, fluor, statistic='mean', bins=[x,y], expand_binnumbers=True)
    binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
    binned_hal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
    # to do a contour plot, you need to reference the center of the bins, not the edges
    # get the bin centers
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    
    # plot the data
    vmin = 0
    vmax = 5
    levels = np.linspace(vmin,vmax, 15)
    
    sm = ax[i].contourf(xc, yc, binned_chlor.statistic.T, levels = levels, vmin = vmin, vmax = vmax, cmap = 'cmo.algae',
                     extend ='both', zorder = -1)
    ax[i].plot(bathy.lat, -depth_contour.values, color = 'k', linewidth = 3)
    ax[i].text(x = 40.5, y = 200, s = "July " + str(days[i]), size = 90, color = 'k')
    ax[i].fill_between(bathy.lat, -depth_contour.values[:, 0], 210, color='lightgrey', alpha=0.7)
    lines5 = ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [0.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    lines1 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.3],  zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    lines2 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.15],  zorder = 2, colors = 'red', alpha = .4, linestyles = '-', linewidths = 5)
    lines3 = ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0],  zorder = 2, colors = 'red', alpha = .4, linestyles = '-', linewidths = 5)
    ax[i].contourf(xc, yc, binned_sigma.statistic.T, levels = [25.95, 26.4], colors='pink', alpha=0.4)
    lines5 = ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i].contour(xc, yc, binned_hal.statistic.T, levels = [34.75], zorder = 2, colors = 'k', linestyles = '--', linewidths = 5)
    lat = np.linspace(np.min(df.lat), np.max(df.lat), len(light_2d[i].ravel()))
   # ax[i].scatter(df.lat, df.depth)
    #ax[i].plot(lat, light_2d[i], color = 'gold', linewidth = 15, linestyle = '-')
    ax[i].plot(lat, light_2d[i], color = 'yellow', linewidth = 15, linestyle = 'dotted')
    labels = ['Target isopycnals:',  '26.0 $\sigma$', '26.15 $\sigma$','26.3 $\sigma$']#,  '25.75 $\sigma$']
    
    if i ==0:
        ax[i].yaxis.set_tick_params(labelsize = 70) 
        xb = [40.3352686674772, 40.3352686674772, 40.2369847537418, 40.2369847537418, 40.3352686674772]
        yb = [75, 100, 100, 75, 75]
        ax[i].plot(xb, yb, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)
        xt = [39.9442512553859, 39.9442512553859, 39.7501666666667, 39.7501666666667, 39.9442512553859]
        yt = [41.8749712460652, 65, 65, 41.8749712460652, 41.8749712460652]
        ax[i].plot(xt, yt, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)
    else:
       ax[i].yaxis.set_tick_params(labelsize = 0)
       xb = [40.3142075367576, 40.3142075367576, 40.1898267781432, 40.1898267781432, 40.3142075367576]
       yb = [70, 97, 97, 70, 70]
       ax[i].plot(xb, yb, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)
       xt = [40.0154774203023, 40.0154774203023, 39.9559026646974, 39.9559026646974, 40.0154774203023]
       yt = [45, 75, 75, 45, 45]
       ax[i].plot(xt, yt, c = 'b', linestyle = 'dotted', zorder = 2, linewidth = 15)
       
    for j in range(len(labels)):
        ax[0].collections[j].set_label(labels[j])
        
    legend= fig.legend(loc='lower left', bbox_to_anchor=(0.02, -0.2), ncol = 4,
                      fontsize = 90, labelcolor = ['red', 'tomato', 'tomato',  'red'], facecolor='whitesmoke')

    ###PLOT SET UP## 
    ax[i].set_ylim(0, 210)
    ax[i].set_xlim(39.6, 40.6)
    
    ax[i].xaxis.set_tick_params(labelsize = 70)
    
    ax[i].invert_yaxis()
    ax[i].invert_xaxis()
    
    
    ax[i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')     
    
    if i == 0:
        ax[i].set_ylabel('Depth [m]', fontsize = 90) 
    
    ax[i].set_xlabel('Latitude [$\degree$N]', fontsize = 90)
    set_cbar(fig, sm, 'Chlorophyll [mg/m$^3$]')
plt.subplots_adjust(hspace=0.5)      
# fig.suptitle('Position of the foot of 26.4 $\sigma$ vs 34.5 PSU (May 2019)', fontsize = 80, y = 0.95)
fig.tight_layout(pad = 4) 
fig.savefig(figures + 'figures_euphotic_july.png', bbox_inches='tight')
     
        