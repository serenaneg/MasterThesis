#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:30:22 2023

@author: serena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean 
import matplotlib.cm as cm
import scipy.stats as ss
import datetime
import xarray as xr
from scipy.interpolate import interp1d
from matplotlib.colors import ListedColormap

path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2018/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

path_pres = "/home/serena/Scrivania/Magistrale/thesis/figures_paper/"

file = path + 'bottle_ar29_hydro.csv'
bottle = pd.read_csv(file)

file2 = path + 'ctd_tn_withlocation_apr18.csv'
ctd_data = pd.read_csv(file2)
#some station are nan => delete
ctd_data = ctd_data.dropna(subset=['station'])
#%%
top = cm.get_cmap('YlGnBu_r', 128)  # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)  # combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))  # create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')
#%%
bathy = xr.open_dataset(bathy_path + "gebco_3d.nc")

lat_range = slice(39.6, 40.60)
lon_range = -70.83

bathy = bathy.sel(lat=lat_range)
bathy = bathy.sel(lon = lon_range, method = 'nearest')
depth = bathy.Z_3d_interpolated
#%%
bottle['day'] = pd.to_datetime(bottle['day'])
#remouce nan values from nitrate
bottle = bottle.dropna(subset=['no3'])

tn11 = bottle[(bottle['day'].dt.day == 17) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('13:15').time()) | (bottle['day'].dt.day == 18) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('20:43').time())]
tn22 = bottle[(bottle['day'].dt.day == 19) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('10:36').time()) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('22:44').time())]
tn33 = bottle[(bottle['day'].dt.day == 21) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('06:43').time()) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('18:51').time())]
tn55 = bottle[(bottle['day'].dt.day == 23) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('09:12').time()) | (bottle['day'].dt.day == 24) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('00:54').time())]
tn66 = bottle[(bottle['day'].dt.day == 25) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('15:33').time()) | (bottle['day'].dt.day == 26) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('10:27').time())]
tn77 = bottle[(bottle['day'].dt.day == 27) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('10:27').time()) | (bottle['day'].dt.day == 28) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('10:30').time())]

#tn44 latitude not soerted => sort

combined_tn_bott = pd.concat([tn11, tn22, tn33, tn55, tn66, tn77], ignore_index=True)
# plt.scatter(combined_tn['longitude'], combined_tn['latitude'])
#%%CTD DATA
ctd_data['day'] = pd.to_datetime(ctd_data['day'])

#salinity check
ctd_data = ctd_data[ctd_data['salinity'] >= 25]

tn1 = ctd_data[(ctd_data['day'].dt.day == 17) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('13:15:51').time()) | (ctd_data['day'].dt.day == 18) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('20:43:16').time())]
#remove station 15 not in bottle
tn1 = tn1[tn1.station != 15]
tn2 = ctd_data[(ctd_data['day'].dt.day == 19) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('10:36:30').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('22:44:02').time())]
tn3 = ctd_data[(ctd_data['day'].dt.day == 21) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('06:43:32').time()) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('18:51:18').time())]
tn5 = ctd_data[(ctd_data['day'].dt.day == 23) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('09:12:10').time()) | (ctd_data['day'].dt.day == 24) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('00:54:13').time())]
tn6 = ctd_data[(ctd_data['day'].dt.day == 25) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('15:33:00').time()) | (ctd_data['day'].dt.day == 26) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('10:27:27').time())]
tn7 = ctd_data[(ctd_data['day'].dt.day == 27) & (pd.to_datetime(ctd_data['day']).dt.time >= pd.to_datetime('10:27:54').time()) | (ctd_data['day'].dt.day == 28) & (pd.to_datetime(ctd_data['day']).dt.time <= pd.to_datetime('10:30:15').time())]

#%%
transects_ctd = [tn1, tn2, tn3, tn5, tn6, tn7]
days = [17, 19, 21,  23, 25, 27]
transects_bott = [tn11, tn22, tn33,  tn55, tn66, tn77]

#%%
transec = [None] * len(transects_bott)
for i in range(len(transects_bott)):
    df = transects_bott[i]
    lat_deg = df['latitude']
    lat_min = np.zeros_like(lat_deg)
    
    lon_deg = -df['longitude']
    lon_min = np.zeros_like(lat_deg)
    
    # Create a table for positions
    posits = pd.DataFrame({
        'station': df['station'],
        'lat degree': lat_deg,
        'lon degree': lon_deg,
        'depth': df['depth']
    })
    
    # Convert table to a dictionary
    posits = posits.to_dict('list')
    
    # Convert data to arrays
    # no3 = df['no3']
    # po = df['po4']
    # si = df['si']
    # pon = df['PON']
    # dens = df['density']
    # sal = df['salinity']
    
    # Get unique station values
    station_id = np.array(np.unique(df['station']), dtype = 'int')
    
    # Create an empty dictionary to store the filtered data
    mdict = {}
    
    # Loop through each station
    for j in range(len(station_id)):
        station = station_id[j]
        
        #interp each 1d profile on regular depth
        new_depth = np.arange(0, 220, 3)
        
        # Filter rows based on the station value
        sub_data = df[df['station'] == station][['no3', 'po4', 'si', 'nh4', 'PON', 'density', 'salinity', 'latitude', 'depth', 'fluorescence']]
        
        # Generate the field name dynamically
        field_name = f'rawprofile{station}'
        
        # Store the filtered data in the dictionary
        mdict[field_name] = sub_data
        
        # Interpolate each 1d profile on a regular depth grid (new_depth)
        interpolated_data = {}
        for var_name in ['no3', 'po4', 'si', 'nh4', 'PON', 'density', 'salinity', 'latitude', 'depth', 'fluorescence']:
            profile_depth = sub_data['depth'].values
            profile_data = sub_data[var_name].values
            
            # Use scipy.interpolate.interp1d to perform linear interpolation
            f = interp1d(profile_depth, profile_data, kind='linear', bounds_error=False, fill_value=np.nan)
            interpolated_data[var_name] = f(new_depth)
        
        # Store the interpolated data in the dictionary
        mdict[field_name] = pd.DataFrame(interpolated_data)
    
    # Store the posits dictionary in the mdict dictionary
    mdict['posits'] = posits
    
    transec[i] = mdict

#%%LOOP TO CLEAN PON    
transec_pon = [None] * len(transects_bott)
for i in range(len(transects_bott)):
    df = transects_bott[i]
    lat_deg = df['latitude']
    lat_min = np.zeros_like(lat_deg)
    
    lon_deg = -df['longitude']
    lon_min = np.zeros_like(lat_deg)
    
    # Create a table for positions
    posits = pd.DataFrame({
        'station': df['station'],
        'lat degree': lat_deg,
        'lon degree': lon_deg,
        'depth': df['depth']
    })
    
    # Convert table to a dictionary
    posits = posits.to_dict('list')
    
    # Convert data to arrays
    pon = df['PON']
    
    # Get unique station values
    station_id = np.array(np.unique(df['station']), dtype='int')
    
    # Create an empty dictionary to store the filtered data
    mdict = {}
    
    # Loop through each station
    for j in range(len(station_id)):
        station = station_id[j]
        
        # Interpolate each 1d profile on regular depth
        new_depth = np.linspace(0, 220, 50)
        
        # Filter rows based on the station value
        sub_data = df[df['station'] == station][['PON', 'latitude', 'depth']]
        
        # Drop rows with NaN values in 'PON' column
        cleaned_profile = sub_data.dropna(subset=['PON']).copy()
        
        # Generate the field name dynamically
        field_name = f'rawprofile{station}'
        
        # Store the filtered data in the dictionary
        mdict[field_name] = cleaned_profile
        
        # Interpolate each 1d profile on a regular depth grid (new_depth)
        interpolated_data = {}
        for var_name in ['PON', 'latitude', 'depth']:
            profile_depth = cleaned_profile['depth'].values
            profile_data = cleaned_profile[var_name].values
            
            # Use scipy.interpolate.interp1d to perform linear interpolation
            f = interp1d(profile_depth, profile_data, kind='linear', bounds_error=False, fill_value=np.nan)
            interpolated_data[var_name] = f(new_depth)
        
        # Store the interpolated data in the dictionary
        mdict[field_name] = pd.DataFrame(interpolated_data)
    
    # Store the posits dictionary in the mdict dictionary
    mdict['posits'] = posits
    
    transec_pon[i] = mdict

#%%
no3_2d = {}
po_2d = {}
pon_2d = {}
si_2d = {}
depth_2d ={}
lat_2d ={}
sal_2d ={}
dens_2d ={}
depth_pon_2d = {}
fluor_2d = {}
nh4_2d = {}
grid_size = 200

sorted_indices = []  # To store the sorted indices for depth
sorted_in = []
for k in range(len(transects_bott)):
    print(k)
    df = transec[k]
    df_pon = transec_pon[k]

    # Get all unique rawprofile field names
    rawprofile_fields = [field for field in df if field.startswith('rawprofile')]

    fluor_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    no3_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    nh4_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    po_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    pon_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    si_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    lat_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    dens_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    sal_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    depth_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    depth_grid_pon = np.full((grid_size, len(rawprofile_fields)), np.nan)
    # sorted_indices = [None] * len(rawprofile_fields)
    # sorted_in = [None] * len(rawprofile_fields)

    for i, profile_field in enumerate(rawprofile_fields):
        print(i)

        profile = df[profile_field]
        profile_pon_data = df_pon[profile_field]
    
        # Access other profile variables and reorder them based on sorted indices
        profile_no3 = profile['no3']
        profile_po = profile['po4']
        profile_nh4 = profile['nh4']
        profile_pon = profile_pon_data['PON']
        profile_si = profile['si']
        profile_lat = profile['latitude']
        profile_dens = profile['density']
        profile_sal = profile['salinity']
        profile_depth = profile['depth']
        profile_fluor = profile['fluorescence']
        profile_depth_pon = profile_pon_data['depth']


        # Store the sorted variables in the grid
        no3_grid[:len(profile_no3), i] = profile_no3
        nh4_grid[:len(profile_nh4), i] = profile_nh4
        pon_grid[:len(profile_pon), i] = profile_pon
        po_grid[:len(profile_po), i] = profile_po
        si_grid[:len(profile_si), i] = profile_si
        depth_grid[:len(profile_depth), i] = profile_depth
        lat_grid[:len(profile_lat), i] = profile_lat
        sal_grid[:len(profile_sal), i] = profile_sal
        dens_grid[:len(profile_dens), i] = profile_dens
        depth_grid_pon[:len(profile_depth_pon), i] = profile_depth_pon
        fluor_grid[:len(profile_no3), i] = profile_fluor


    fluor_2d[k] = fluor_grid
    no3_2d[k] = no3_grid
    nh4_2d[k] = nh4_grid
    po_2d[k] = po_grid
    pon_2d[k] = pon_grid
    si_2d[k] = si_grid
    depth_2d[k] = depth_grid
    lat_2d[k] = lat_grid
    sal_2d[k] = sal_grid
    dens_2d[k] = dens_grid
    depth_pon_2d[k] = depth_grid_pon
    
#%%
def set_cbar(fig, c, title, j, cax):

   # cax = fig.add_axes([0.038+ j*0.197, -0.05, 0.16, 0.04]) #small
    #cax = fig.add_axes([0.037 + j*0.246, -0.01, 0.2, 0.013]) #big  
    cbar = fig.colorbar(c, format='%.1f', spacing='proportional', cax=cax, shrink=0.9, pad = 6,
                        orientation = 'horizontal', location = "bottom")
    cbar.set_label(label = title, fontsize = 90, y = 0.5, labelpad = 30)
    cbar.ax.tick_params(which='minor', size=5, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=20, width=1, color='k', direction='in', labelsize = 70)
    return cbar
#%%
transects = [tn2, tn2]
days = [19, 19]

fig, ax = plt.subplots(1, 2, dpi=100, figsize=([45, 18]))

for i, df in enumerate(transects):
    
    x = np.arange(38, 42, 0.064) #0.064 get from diff
    y = np.arange(0, 220, 3)
    xx, yy = np.meshgrid(x, y)
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    
    cax = fig.add_axes([0.070+ i*0.49, -0.1, 0.42, 0.07]) #small

    
    if i == 0:
        binned_sal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
        binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
        #SALINITY    
        # plot the data
        vmin = 31.8
        vmax = 36
        # levels = np.arange(vmin,vmax, 0.05)
        levels = np.linspace(vmin,vmax, 15)
        
        sm2 = ax[i].contourf(xc, yc, binned_sal.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.haline')
        ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        #ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        # ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        ax[i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
        ax[i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
        ax[i].set_ylabel('Depth [m]', fontsize = 80)
        ax[i].set_title('Salinity', fontsize = 80)
        ax[i].yaxis.set_tick_params(length = 18, labelsize = 70) 
        ax[i].set_xlabel('Latitude [$\degree$N]', fontsize = 80)
        ax[i].xaxis.set_tick_params(length = 18, labelsize = 70)
        set_cbar(fig, sm2, '[PSU]', 0, cax) 


    
    elif i == 1:
        # temp
        binned_temp = ss.binned_statistic_2d(df.lat, df.depth, df.temperature, statistic='mean', bins=[x, y])
        binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
        binned_sal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
        #SALINITY    
        # plot the data
        vmin = 5
        vmax = 25
        # levels = np.arange(vmin,vmax, 0.05)
        levels = np.linspace(vmin,vmax, 15)
        
        sm0 = ax[i].contourf(xc, yc, binned_temp.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.thermal')
        ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        #ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        # ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        ax[i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
        ax [i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
        ax[i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 80)
        ax[i].set_title('Temperature', fontsize = 80)
        ax[i].xaxis.set_tick_params(length = 18, labelsize = 70)
        ax[i].yaxis.set_tick_params(length = 18,labelsize = 70, labelcolor = 'w') 
        set_cbar(fig, sm0, '[$^\circ$C]', 1, cax) 
        

    ax[i].set_ylim(0, 210)
    ax[i].set_xlim(39.6, 40.6)
    ax[i].invert_yaxis()
    ax[i].invert_xaxis()

    ax[i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')


fig.suptitle('CTD transect at 70.8$^\circ$W, April 19th 2018', fontsize = 90, y = 1)              
plt.subplots_adjust(hspace=0.5)      
fig.tight_layout(pad = 1) 
fig.savefig(path_pres + 'april_19_bott.png', bbox_inches='tight')

#%%
transects = [tn7, tn77]
days = [27, 27]

fig, ax = plt.subplots(1, 2, dpi=100, figsize=([45, 18]))

for i, df in enumerate(transects):
    
    x = np.arange(38, 42, 0.064) #0.064 get from diff
    y = np.arange(0, 220, 3)
    xx, yy = np.meshgrid(x, y)
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    
    if i == 0:
        binned_sal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
        binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
        #SALINITY    
        # plot the data
        vmin = 31.8
        vmax = 36
        # levels = np.arange(vmin,vmax, 0.05)
        levels = np.linspace(vmin,vmax, 15)
        
        sm2 = ax[i].contourf(xc, yc, binned_sal.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.haline')
        ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        #ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        # ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        ax[i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
        ax[i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
        ax[i].text(x = 40.4, y = 200, s = "April " + str(days[i]), size = 90, color = 'k')
        ax[i].set_ylabel('Depth [m]', fontsize = 90)
        ax[i].set_title('Salinity', fontsize = 90)
    
    elif i == 1:
        # nitrate
        vmin = 0.0
        vmax = 10
        levels = np.linspace(vmin,vmax, 12)
        
        sm = ax[i].contourf(lat_2d[5], depth_2d[5], no3_2d[5], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                         extend ='both')
        ax[i].contour(lat_2d[5], depth_2d[5], sal_2d[5], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        ax[i].contour(lat_2d[5], depth_2d[5], dens_2d[5], levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        ax[i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)

        # Fill the area below the line
        ax[i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
        
        ax[i].set_ylim(0, 210)
        ax[i].set_xlim(39.6, 40.5)
        ax[i].invert_yaxis()
        ax[i].invert_xaxis()
        ax[i].xaxis.set_tick_params(labelsize = 40)
        ax[i].yaxis.set_tick_params(labelsize = 40)   
        ax[i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
        
        ax[i].tick_params(axis = 'y', colors='white')
        ax[i].set_title('Nitrate (NO$^{-}_{3}$)', fontsize = 80)
        

    ax[i].set_ylim(0, 210)
    ax[i].set_xlim(39.6, 40.6)
    ax[i].invert_yaxis()
    ax[i].invert_xaxis()

    ax[i].xaxis.set_tick_params(labelsize = 0)
    ax[i].yaxis.set_tick_params(labelsize = 70) 

    ax[i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    # ax[i].set_xlabel('Latitude [$\degree$N]', fontsize = 90)
    # ax[i].xaxis.set_tick_params(labelsize = 70)

fig.suptitle('CTD transect at 70.8$^\circ$W', fontsize = 100, y = 0.96)              
plt.subplots_adjust(hspace=0.5)      
fig.tight_layout(pad = 6) 
fig.savefig(path_pres + 'april_27_bott.png', bbox_inches='tight')
#%%ALL VARIABLES
transects1 = [tn2, tn2, tn2, tn22, tn22]
days1 = [19, 19, 19, 19, 19]

transects2 = [tn7, tn7, tn7, tn77, tn77]
days2 = [27, 27, 27, 27, 27]

transect_tot = [transects1, transects2]
days_tot = [days1, days2]

fig, ax = plt.subplots(2, 5, dpi=100, figsize=([115, 36]))

for j in range(len(transect_tot)):
        
   transects = transect_tot[j]
   days = days_tot[j]
               
   for i in range(len(transects)):      
        df = transects[i]
        
        x = np.arange(38, 42, 0.064) #0.064 get from diff
        y = np.arange(0, 220, 3)
        xx, yy = np.meshgrid(x, y)
        xc = (x[:-1] + x[1:]) / 2
        yc = (y[:-1] + y[1:]) / 2
        
        if i == 0:
            binned_temp = ss.binned_statistic_2d(df.lat, df.depth, df.temperature, statistic='mean', bins=[x, y])
            binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
            binned_sal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
            #SALINITY    
            # plot the data
            vmin = 5
            vmax = 25
            # levels = np.arange(vmin,vmax, 0.05)
            levels = np.linspace(vmin,vmax, 15)
            
            sm0 = ax[j, i].contourf(xc, yc, binned_temp.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.thermal')
            ax[j, i].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
            #ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
            ax[j, i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
            # ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
            ax[j, i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
            ax[j, i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
            ax[j, i].text(x = 40.4, y = 200, s = "April " + str(days[i]), size = 90, color = 'k')
            ax[j, i].set_ylabel('Depth [m]', fontsize = 90)
            ax[1, i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 90)
            ax[0, i].set_title('Temperature', fontsize = 90)
            ax[j, i].yaxis.set_tick_params(length = 18, labelsize = 70)
            ax[0, i].xaxis.set_tick_params(labelsize = 0) 
            set_cbar(fig, sm0, '[$^\circ$C]', 0) 
    
        
        elif i == 1:
            binned_sal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
            binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
            #SALINITY    
            # plot the data
            vmin = 31.8
            vmax = 36
            # levels = np.arange(vmin,vmax, 0.05)
            levels = np.linspace(vmin,vmax, 15)
            
            sm1 = ax[j, i].contourf(xc, yc, binned_sal.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.haline')
            ax[j, i].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
            #ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
            ax[j, i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
            # ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
            ax[j, i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
            ax[j, i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
            ax[0, i].set_title('Salinity', fontsize = 90)
            ax[1, i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 90)
            ax[j, i].yaxis.set_tick_params(length = 18, labelsize = 70, labelcolor = 'w')
            ax[0, i].xaxis.set_tick_params(labelsize = 0) 
            set_cbar(fig, sm1, '[PSU]', 1) 
    
        elif i == 2:
            fluor = df.fluorecence * 0.99951089 + 0.031688137881046075
            binned_chl = ss.binned_statistic_2d(df.lat, df.depth, fluor, statistic='mean', bins=[x, y],
                                            expand_binnumbers=True) 
            binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
            #SALINITY    
            # plot the data
            vmin = 0
            vmax = 5
            # levels = np.arange(vmin,vmax, 0.05)
            levels = np.linspace(vmin,vmax, 15)
            
            sm2 = ax[j, i].contourf(xc, yc, binned_chl.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.algae')
            ax[j, i].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
            #ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
            ax[j, i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
            # ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
            ax[j, i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
            ax[j, i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
            ax[0, i].set_title('Chlorphyll', fontsize = 90)
            ax[1, i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 90)
            ax[j, i].yaxis.set_tick_params(length = 18, labelsize = 70, labelcolor = 'w')
            ax[0, i].xaxis.set_tick_params(labelsize = 0)
            set_cbar(fig, sm2, '[mg/m$^3$]', 2) 
    
        elif i == 3:
            # nitrate
            vmin = 0.0
            vmax = 10
            levels = np.linspace(vmin,vmax, 12)
            
            if j == 0:
                sm3 = ax[j, i].contourf(lat_2d[1], depth_2d[1], no3_2d[1], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                                 extend ='both')
                ax[j, i].contour(lat_2d[1], depth_2d[1], sal_2d[1], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
                ax[j, i].contour(lat_2d[1], depth_2d[1], dens_2d[1], levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
            else:
                sm4 = ax[j, i].contourf(lat_2d[5], depth_2d[5], no3_2d[5], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                                 extend ='both')
                ax[j, i].contour(lat_2d[5], depth_2d[5], sal_2d[5], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
                ax[j, i].contour(lat_2d[5], depth_2d[5], dens_2d[5], levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
              
            ax[j, i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
            ax[j, i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)           
            ax[j, i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
            ax[1, i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 90)
            ax[0, i].set_title('Nitrate (NO$^{-}_{3}$)', fontsize = 90)
            ax[j, i].yaxis.set_tick_params(length = 18, labelsize = 70, labelcolor = 'w') 
            ax[0, i].xaxis.set_tick_params(labelsize = 0) 
            set_cbar(fig, sm3, '[$\mu$mol/L]', 3) 
    
        elif i == 4:
             # pon
             vmin = 0.0
             vmax = 5
             levels = np.linspace(vmin,vmax, 12)
             
             if j == 0:
                 sm4 = ax[j, i].contourf(lat_2d[1], depth_pon_2d[1], pon_2d[1], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                                  extend ='both')
                 ax[j, i].contour(lat_2d[1], depth_2d[1], sal_2d[1], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
                 ax[j, i].contour(lat_2d[1], depth_2d[1], dens_2d[1], levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
                 
             else:
                 sm4 = ax[j, i].contourf(lat_2d[5], depth_pon_2d[5], pon_2d[5], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                                  extend ='both')
                 ax[j, i].contour(lat_2d[5], depth_2d[5], sal_2d[5], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
                 ax[j, i].contour(lat_2d[5], depth_2d[5], dens_2d[5], levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
                 
             ax[j, i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
             ax[j, i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)            
             ax[1, i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 90)
             ax[0, i].set_title('PON', fontsize = 90)
             ax[j, i].yaxis.set_tick_params(length = 18, labelsize = 70, labelcolor = 'w')
             ax[0, i].xaxis.set_tick_params(labelsize = 0) 
             set_cbar(fig, sm4, '[$\mu$mol/L]', 4) 
    
        ax[j, i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
        ax[j, i].set_ylim(0, 210)
        ax[j, i].set_xlim(39.6, 40.6)
        ax[j, i].invert_yaxis()
        ax[j, i].invert_xaxis()  
        ax[j, i].xaxis.set_tick_params(length = 18, labelsize = 70)

fig.suptitle('Transect at 70.8$^\circ$W. Not perturbed front (1st row) and perturbed front (2nd row)', fontsize=100, y = 1.01)     
plt.subplots_adjust(hspace=0.5)      
fig.tight_layout(pad = 3) 
fig.savefig(path_pres + 'april_ok.png', bbox_inches='tight')
#%%perturbed
transects = [tn7, tn7, tn7, tn77, tn77]
days = [27, 27, 27, 27, 27]

fig, ax = plt.subplots(1, 5, dpi=100, figsize=([115, 18]))

for i, df in enumerate(transects):
    
    x = np.arange(38, 42, 0.064) #0.064 get from diff
    y = np.arange(0, 220, 3)
    xx, yy = np.meshgrid(x, y)
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    
    if i == 0:
        binned_temp = ss.binned_statistic_2d(df.lat, df.depth, df.temperature, statistic='mean', bins=[x, y])
        binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
        binned_sal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
        #SALINITY    
        # plot the data
        vmin = 5
        vmax = 25
        # levels = np.arange(vmin,vmax, 0.05)
        levels = np.linspace(vmin,vmax, 15)
        
        sm0 = ax[i].contourf(xc, yc, binned_temp.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.thermal')
        ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        #ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        # ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        ax[i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
        ax[i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
        ax[i].text(x = 40.4, y = 200, s = "April " + str(days[i]), size = 90, color = 'k')
        ax[i].set_ylabel('Depth [m]', fontsize = 90)
        ax[i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 90)
        ax[i].set_title('Temperature', fontsize = 90)
        ax[i].yaxis.set_tick_params(labelsize = 70) 
        set_cbar(fig, sm0, '[$^\circ$C]', 0) 

    
    elif i == 1:
        binned_sal = ss.binned_statistic_2d(df.lat, df.depth, df.salinity, statistic='mean', bins=[x, y])
        binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
        #SALINITY    
        # plot the data
        vmin = 31.8
        vmax = 36
        # levels = np.arange(vmin,vmax, 0.05)
        levels = np.linspace(vmin,vmax, 15)
        
        sm1 = ax[i].contourf(xc, yc, binned_sal.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.haline')
        ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        #ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        # ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        ax[i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
        ax[i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
        ax[i].set_title('Salinity', fontsize = 90)
        ax[i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 90)
        ax[i].yaxis.set_tick_params(labelsize = 0) 
        set_cbar(fig, sm1, '[PSU]', 1) 

    elif i == 2:
        fluor = df.fluorecence * 0.99951089 + 0.031688137881046075
        binned_chl = ss.binned_statistic_2d(df.lat, df.depth, fluor, statistic='mean', bins=[x, y],
                                        expand_binnumbers=True) 
        binned_sigma = ss.binned_statistic_2d(df.lat, df.depth, df.density, statistic='mean', bins=[x, y])
        #SALINITY    
        # plot the data
        vmin = 0
        vmax = 5
        # levels = np.arange(vmin,vmax, 0.05)
        levels = np.linspace(vmin,vmax, 15)
        
        sm2 = ax[i].contourf(xc, yc, binned_chl.statistic.T, levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = 'cmo.algae')
        ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        #ax[i].contour(xc, yc, binned_sal.statistic.T, levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        # ax[i].contour(xc, yc, binned_sigma.statistic.T, levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        ax[i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
        ax[i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
        ax[i].set_title('Chlorphyll', fontsize = 90)
        ax[i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 90)
        ax[i].yaxis.set_tick_params(labelsize = 0)
        set_cbar(fig, sm2, '[mg/m$^3$]', 2) 

    elif i == 3:
        # nitrate
        vmin = 0.0
        vmax = 10
        levels = np.linspace(vmin,vmax, 12)
        
        sm3 = ax[i].contourf(lat_2d[5], depth_2d[5], no3_2d[5], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                         extend ='both')
        ax[i].contour(lat_2d[5], depth_2d[5], sal_2d[5], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
        ax[i].contour(lat_2d[5], depth_2d[5], dens_2d[5], levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
        ax[i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)

        # Fill the area below the line
        ax[i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
        
        ax[i].set_ylim(0, 210)
        ax[i].set_xlim(39.6, 40.5)
        ax[i].invert_yaxis()
        ax[i].invert_xaxis()
        ax[i].xaxis.set_tick_params(labelsize = 40)
        ax[i].yaxis.set_tick_params(labelsize = 40)   
        ax[i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
        
        ax[i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 90)
        ax[i].set_title('Nitrate (NO$^{-}_{3}$)', fontsize = 90)
        ax[i].yaxis.set_tick_params(labelsize = 0) 
        set_cbar(fig, sm3, '[$\mu$mol/L]', 3) 

    elif i == 4:
         # pon
         vmin = 0.0
         vmax = 5
         levels = np.linspace(vmin,vmax, 12)
         
         sm4 = ax[i].contourf(lat_2d[5], depth_2d[5], pon_2d[5], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                          extend ='both')
         ax[i].contour(lat_2d[5], depth_2d[5], sal_2d[5], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
         ax[i].contour(lat_2d[5], depth_2d[5], dens_2d[5], levels = [26.6], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
         ax[i].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)

         # Fill the area below the line
         ax[i].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
         
         ax[i].set_ylim(0, 210)
         ax[i].set_xlim(39.6, 40.5)
         ax[i].invert_yaxis()
         ax[i].invert_xaxis()
         ax[i].xaxis.set_tick_params(labelsize = 40)
         ax[i].yaxis.set_tick_params(labelsize = 40)   
         ax[i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
         
         ax[i].set_xlabel('Latitude [$^\circ$ N]', fontsize = 90)
         ax[i].set_title('PON', fontsize = 90)
         ax[i].yaxis.set_tick_params(labelsize = 0) 
         set_cbar(fig, sm4, '[$\mu$mol/L]', 4) 


    ax[i].set_ylim(0, 210)
    ax[i].set_xlim(39.6, 40.6)
    ax[i].invert_yaxis()
    ax[i].invert_xaxis()

    ax[i].xaxis.set_tick_params(labelsize = 70)

    ax[i].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    # ax[i].set_xlabel('Latitude [$\degree$N]', fontsize = 90)
    # ax[i].xaxis.set_tick_params(labelsize = 70)

fig.suptitle('CTD transect at 70.8$^\circ$W, Front perturbed', fontsize = 100, y = 0.96)              
plt.subplots_adjust(hspace=0.5)      
fig.tight_layout(pad = 6) 
fig.savefig(path_pres + 'april_Pert.png', bbox_inches='tight')

