#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:36:20 2023

@author: serena
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import xarray as xr
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

def set_cbar(fig, c, title, j):
    cax = fig.add_axes([0.037 + j*0.246, -0.01, 0.2, 0.013])    
    cbar = fig.colorbar(c, format='%.1f', spacing='proportional', cax=cax, shrink=0.8, pad = 6,
                        orientation = 'horizontal', location = "bottom")
    cbar.set_label(label = title, fontsize = 60, y = 0.5, labelpad = 30)
    cbar.ax.tick_params(which='minor', size=5, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=20, width=1, color='k', direction='in', labelsize = 50)
    return cbar


top = cm.get_cmap('YlGnBu_r', 128)  # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)  # combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))  # create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')
#%%
# Specify the path
path = '/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/'

# Specify the bbl path
plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2019/july/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

# Load bottle data for nutrient
file = path + 'bottle_tn368_hydro.csv'
bottle = pd.read_csv(file)
bottle['day'] = pd.to_datetime(bottle['day'])

bottle = bottle.dropna(subset=['no3'])
# bottle = bottle[(bottle['day'].dt.day == 6)]

#days 6, 7, 9, 14, 11-12, 16, 17-18
tn11 = bottle[(bottle['day'].dt.day == 6) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('00:07').time()) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('17:58').time())]
tn33 = bottle[(bottle['day'].dt.day == 9) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('09:49').time()) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('20:50').time())]
tn44 = bottle[(bottle['day'].dt.day == 11) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('18:21').time()) | (bottle['day'].dt.day == 12) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('13:51').time())]
tn55 = bottle[(bottle['day'].dt.day == 14) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('03:24').time()) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('20:00').time())]
tn61 = bottle[(bottle['day'].dt.day == 16) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('00:39:25').time()) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('13:18:41').time())]
tn62 = bottle[(bottle['day'].dt.day == 17) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('01:19:34').time()) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('04:11:33').time())]

tn66 = pd.concat([tn61, tn62])
tn77 = bottle[(bottle['day'].dt.day == 17) & (pd.to_datetime(bottle['day']).dt.time >= pd.to_datetime('12:39').time()) | (bottle['day'].dt.day == 18) & (pd.to_datetime(bottle['day']).dt.time <= pd.to_datetime('05:45').time())]

combined_tn_bot = pd.concat([tn11, tn33, tn44, tn55, tn66, tn77], ignore_index=True)

combined_tn_bot.to_csv(path + 'bottle_jul19_onlygood.csv', index=False)
#%%depth profile
bathy = xr.open_dataset(bathy_path + "gebco_3d.nc")

lat_range = slice(38.7, 40.60)
lon_range = -70.8

bathy = bathy.sel(lat=lat_range)
bathy = bathy.sel(lon = lon_range, method = 'nearest')
depth = bathy.Z_3d_interpolated
#%%
transects_bott = [tn11, tn33, tn44, tn55, tn66, tn77]

#create the dictionary with profiles and posits

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
    no3 = df['no3']
    po = df['po4']
    si = df['si']
    pon = df['PON']
    dens = df['density']
    sal = df['salinity']
    
    # Get unique station values
    station_id = np.array(np.unique(df['station']), dtype = 'int')
    
    # Create an empty dictionary to store the filtered data
    mdict = {}
    
    # Loop through each station
    for j in range(len(station_id)):
        station = station_id[j]
        
        #interp each 1d profile on regular depth
        new_depth = np.linspace(2, 200, 80)
        
        # Filter rows based on the station value
        sub_data = df[df['station'] == station][['no3', 'po4', 'si', 'PON', 'density', 'salinity', 'latitude', 'depth']]
        
        # Generate the field name dynamically
        field_name = f'rawprofile{station}'
        
        # Store the filtered data in the dictionary
        mdict[field_name] = sub_data
        
        # Interpolate each 1d profile on a regular depth grid (new_depth)
        interpolated_data = {}
        for var_name in ['no3', 'po4', 'si', 'PON', 'density', 'salinity', 'latitude', 'depth']:
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
        new_depth = np.linspace(2, 150, 30)
        
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
grid_size = 200

sorted_indices = []  # To store the sorted indices for depth
sorted_in = []
for k in range(len(transects_bott)):
    print(k)
    df = transec[k]
    df_pon = transec_pon[k]

    # Get all unique rawprofile field names
    rawprofile_fields = [field for field in df if field.startswith('rawprofile')]

    no3_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
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
        profile_pon = profile_pon_data['PON']
        profile_si = profile['si']
        profile_lat = profile['latitude']
        profile_dens = profile['density']
        profile_sal = profile['salinity']
        profile_depth = profile['depth']
        profile_depth_pon = profile_pon_data['depth']


        # Store the sorted variables in the grid
        no3_grid[:len(profile_no3), i] = profile_no3
        pon_grid[:len(profile_pon), i] = profile_pon
        po_grid[:len(profile_po), i] = profile_po
        si_grid[:len(profile_si), i] = profile_si
        depth_grid[:len(profile_depth), i] = profile_depth
        lat_grid[:len(profile_lat), i] = profile_lat
        sal_grid[:len(profile_sal), i] = profile_sal
        dens_grid[:len(profile_dens), i] = profile_dens
        depth_grid_pon[:len(profile_depth_pon), i] = profile_depth_pon


    no3_2d[k] = no3_grid
    po_2d[k] = po_grid
    pon_2d[k] = pon_grid
    si_2d[k] = si_grid
    depth_2d[k] = depth_grid
    lat_2d[k] = lat_grid
    sal_2d[k] = sal_grid
    dens_2d[k] = dens_grid
    depth_pon_2d[k] = depth_grid_pon

#%%
days = [6, 9, 11, 14, 16, 17]

fig, ax = plt.subplots(6, 4, dpi=50, figsize=([90, 84]))

for i, df in zip(range(len(transects_bott)), transects_bott):
    # nitrate
    vmin = 0.0
    vmax = 10
    levels = np.linspace(vmin,vmax, 12)
    
    sm = ax[i, 0].contourf(lat_2d[i], depth_2d[i], no3_2d[i], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                     extend ='both')
    ax[i, 0].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 0].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 0].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 0].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [25.8], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 0].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [26.15], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 0].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
    # ax[i, 0].plot(df.latitude, df.depth, 'o', linewidth = 3, markersize=25, markerfacecolor='w',
    #          markeredgewidth=1.5, markeredgecolor='k')


    # Fill the area below the line
    ax[i, 0].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
    ax[i, 0].text(x = 40.4, y = 200, s = "July " + str(days[i]), size = 60, color = 'k')
    
    # phosphate
    vmin = 0.0
    vmax = 1.5
    levels = np.linspace(vmin,vmax, 12)
    
    
    sm1 = ax[i, 2].contourf(lat_2d[i], depth_2d[i], po_2d[i], levels = levels, vmin = vmin, vmax = vmax, 
                      extend ='both', cmap = orange_blue)
    ax[i, 2].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 2].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 2].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
    ax[i, 2].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
        
    # PON
    vmin = 0.0
    vmax = 5.1
    levels = np.linspace(vmin,vmax, 12)
    
    sm2 = ax[i, 1].contourf(lat_2d[i], depth_pon_2d[i], pon_2d[i], levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = orange_blue)
    ax[i, 1].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 1].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 1].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 1].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [26.15], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 1].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [25.8], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
    ax[i, 1].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
    ax[i, 1].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
    
    #silicate
    vmin = 0.0
    vmax = 12
    levels = np.linspace(vmin,vmax, 12)
    
    sm3 = ax[i, 3].contourf(lat_2d[i], depth_2d[i], si_2d[i], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                     extend ='both')
    ax[i, 3].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
    ax[i, 3].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
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
    
    # if i == 5:
    ax[i, 2].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
    ax[i, 0].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
    ax[i, 1].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
    ax[i, 3].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
    
    set_cbar(fig, sm, '[$\mu$mol/m$^3$]', 0)
    set_cbar(fig, sm1, '[$\mu$mol/m$^3$]', 2)
    set_cbar(fig, sm2, '[$\mu$mol/m$^3$]', 1) 
    set_cbar(fig, sm3, '[$\mu$mol/m$^3$]', 3)
    
       
    if i == 0:
       ax[i, 0].set_title('Nitrate (NO$^{-}_{3}$)', fontsize = 70)
       ax[i, 1].set_title('PON', fontsize = 70)
       ax[i, 3].set_title('Silicon (Si(OH)$_4$)', fontsize = 70)
       ax[i, 2].set_title('Phosphate (PO$^{3^-}_4$)', fontsize = 70)
       
plt.subplots_adjust(hspace=0.5)      
fig.suptitle('Bottle transect at 70.8$^\circ$W', fontsize = 80, y = 0.99)
fig.tight_layout(pad = 6) 
fig.savefig(plot_path + 'bottle_july19.png', bbox_inches='tight')

#%%
fig, ax = plt.subplots(1, 4, dpi=50, figsize=([90, 15]))

# nitrate
i = 5
vmin = 0.0
vmax = 10
levels = np.linspace(vmin,vmax, 12)

sm = ax[0].contourf(lat_2d[i], depth_2d[i], no3_2d[i], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                 extend ='both')
ax[0].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[0].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[0].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[0].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [25.8], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[0].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [26.15], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[0].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
# ax[0].plot(df.latitude, df.depth, 'o', linewidth = 3, markersize=25, markerfacecolor='w',
#          markeredgewidth=1.5, markeredgecolor='k')


# Fill the area below the line
ax[0].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)

# phosphate
vmin = 0.0
vmax = 1.5
levels = np.linspace(vmin,vmax, 12)


sm1 = ax[2].contourf(lat_2d[i], depth_2d[i], po_2d[i], levels = levels, vmin = vmin, vmax = vmax, 
                  extend ='both', cmap = orange_blue)
ax[2].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[2].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[2].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)
ax[2].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
    
# PON
vmin = 0.0
vmax = 5.1
levels = np.linspace(vmin,vmax, 12)

sm2 = ax[1].contourf(lat_2d[i], depth_pon_2d[i], pon_2d[i], levels = levels, vmin = vmin, vmax = vmax, extend ='both', cmap = orange_blue)
ax[1].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.5],  zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[1].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[1].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [26.0], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[1].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [26.15], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[1].contour(lat_2d[i], depth_2d[i], dens_2d[i], levels = [25.8], zorder = 2, colors = 'red', linestyles = '-', linewidths = 5)
ax[1].plot(bathy.lat, -depth.values, color = 'k', linewidth = 3)
ax[1].fill_between(bathy.lat, -depth.values[:, 0], 210, color='lightgrey', alpha=0.7)

#silicate
vmin = 0.0
vmax = 12
levels = np.linspace(vmin,vmax, 12)

sm3 = ax[3].contourf(lat_2d[i], depth_2d[i], si_2d[i], levels = levels, vmin = vmin, vmax = vmax, cmap = orange_blue,
                 extend ='both')
ax[3].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.5], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
ax[3].contour(lat_2d[i], depth_2d[i], sal_2d[i], levels = [34.0], zorder = 2, colors = 'k', linestyles = '-', linewidths = 5)
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

ax[0].invert_yaxis()
ax[0].invert_xaxis()
ax[1].invert_yaxis()
ax[1].invert_xaxis()
ax[2].invert_yaxis()
ax[2].invert_xaxis()
ax[3].invert_yaxis()
ax[3].invert_xaxis()

ax[1].xaxis.set_tick_params(labelsize = 40)
ax[1].yaxis.set_tick_params(labelsize = 40) 
ax[0].xaxis.set_tick_params(labelsize = 40)
ax[0].yaxis.set_tick_params(labelsize = 40) 
ax[2].xaxis.set_tick_params(labelsize = 40)
ax[2].yaxis.set_tick_params(labelsize = 40) 
ax[3].xaxis.set_tick_params(labelsize = 40)
ax[3].yaxis.set_tick_params(labelsize = 40)

ax[0].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax[1].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax[2].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax[3].grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')

ax[0].set_ylabel('Depth [m]', fontsize = 60)

# if i == 5:
ax[0].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
ax[2].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
ax[1].set_xlabel('Latitude [$\degree$N]', fontsize = 60)
ax[3].set_xlabel('Latitude [$\degree$N]', fontsize = 60)

set_cbar(fig, sm, '[$\mu$mol/m$^3$]', 0)
set_cbar(fig, sm1, '[$\mu$mol/m$^3$]', 2)
set_cbar(fig, sm2, '[$\mu$mol/m$^3$]', 1) 
set_cbar(fig, sm3, '[$\mu$mol/m$^3$]', 3)

   
ax[0].set_title('Nitrate (NO$^{-}_{3}$)', fontsize = 70)
ax[1].set_title('PON', fontsize = 70)
ax[3].set_title('Silicon (Si(OH)$_4$)', fontsize = 70)
ax[2].set_title('Phosphate (PO$^{3^-}_4$)', fontsize = 70)
   
plt.subplots_adjust(hspace=0.5)      
fig.suptitle('Bottle transect at 70.8$^\circ$W, July 17th 2019', fontsize = 80, y = 0.99)
fig.tight_layout(pad = 6) 
fig.savefig(plot_path + 'bottle_17july.png', bbox_inches='tight')