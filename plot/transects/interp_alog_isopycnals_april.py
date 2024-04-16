#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:17:44 2024

@author: serena
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import xarray as xr
from scipy.interpolate import interp1d
import seaborn as sns
import cmocean
import scipy.stats
from scipy import interpolate

#colorbar
top = cm.get_cmap('YlGnBu_r', 128)  # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)  # combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))  # create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')
#%%
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2018/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2018/"

file = path + 'bottle_ar29_hydro.csv'
bottle = pd.read_csv(file)

file2 = path + 'ctd_tn_withlocation_apr18.csv'
ctd_data = pd.read_csv(file2)
#some station are nan => delete
ctd_data = ctd_data.dropna(subset=['station'])
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

transects_bott = [tn11, tn22, tn33, tn55, tn66, tn77]

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

transects_ctd = [tn1, tn2, tn3, tn5, tn6, tn7]
#%%depth profile
bathy = xr.open_dataset(bathy_path + "gebco_3d.nc")

lat_range = slice(38.7, 40.60)
lon_range = -70.8

bathy = bathy.sel(lat=lat_range)
bathy = bathy.sel(lon = lon_range, method = 'nearest')
depth = bathy.Z_3d_interpolated
#%%
transec = [None] * len(transects_bott)
n_rows =[]
for i in range(len(transects_bott)):
    bott = transects_bott[i]
    ctd = transects_ctd[i]
    
    # Get unique station values
    station_id = np.array(np.unique(bott['station']), dtype = 'int')
    
    # Create an empty dictionary to store the filtered data
    mdict_ctd = {}
    mdict_bott = {}
    
    # Loop through each station
    for j in range(len(station_id)):
        station = station_id[j]
        
        # Filter rows based on the statno3_2d[i]ion value
        sub_data_bott = bott[bott['station'] == station][['no3', 'PON', 'density', 'depth', 'latitude']]
        sub_data_ctd = ctd[ctd['station'] == station][['lat', 'density', 'depth', 'fluorecence']]
        
        # Generate the field name dynamically
        field_name = f'rawprofile{station}'
        
        # Store the filtered data in the dictionary
        mdict_bott[field_name] = sub_data_bott
        mdict_ctd[field_name] = sub_data_ctd

        len_max = len(sub_data_ctd['depth'])
        n_rows.append(len_max)
        #interp each 1d profile on regular depth
        new_depth = np.arange(0, np.max(n_rows), 5)
        
        # Interpolate each 1d profile on a regular depth grid (new_depth)
        interpolated_data_bott = {}
        for var_name in ['no3', 'PON', 'depth', 'latitude']:
                       
            profile_depth = sub_data_bott['depth'].values
            profile_data = sub_data_bott[var_name].values
            
            f = interp1d(profile_depth, profile_data, kind='linear', bounds_error=False, fill_value=np.nan)
            interpolated_data_bott[var_name] = f(new_depth)
                           
        # Store the interpolated data in the dictionary
        mdict_bott[field_name] = pd.DataFrame(interpolated_data_bott)
        
        interpolated_data_ctd = {}
        for var_name in ['density', 'fluorecence', 'depth', 'lat']:
            profile_depth = sub_data_ctd['depth'].values
            profile_data = sub_data_ctd[var_name].values
            
            # Use scipy.interpolate.interp1d to perform linear interpolation
            f = interp1d(profile_depth, profile_data, kind='linear', bounds_error=False, fill_value=np.nan)
            interpolated_data_ctd[var_name] = f(new_depth)
        
        mdict_ctd[field_name] = pd.DataFrame(interpolated_data_ctd)
    
    transec[i] = (mdict_ctd, mdict_bott)
    
#%%INTERPOLATE PON
transec_pon = [None] * len(transects_bott)
for i in range(len(transects_bott)):
    bott = transects_bott[i]
    
    # Get unique station values
    station_id = np.array(np.unique(bott['station']), dtype='int')
    
    # Create an empty dictionary to store the filtered data
    mdict = {}
    
    # Loop through each station
    for j in range(len(station_id)):
        station = station_id[j]
        
        # Interpolate each 1d profile on regular depth
        new_depth = np.linspace(0, 220, 50)
        
        # Filter rows based on the station value
        sub_data = bott[bott['station'] == station][['PON', 'latitude', 'depth', 'density']]
        
        # Drop rows with NaN values in 'PON' column
        cleaned_profile = sub_data.dropna(subset=['PON']).copy()
        
        # Generate the field name dynamically
        field_name = f'rawprofile{station}'
        
        # Store the filtered data in the dictionary
        mdict[field_name] = cleaned_profile
        
        # Interpolate each 1d profile on a regular depth grid (new_depth)
        interpolated_data = {}
        for var_name in ['PON', 'latitude', 'depth', 'density']:
            profile_depth = cleaned_profile['depth'].values
            profile_data = cleaned_profile[var_name].values
            
            # Use scipy.interpolate.interp1d to perform linear interpolation
            f = interp1d(profile_depth, profile_data, kind='linear', bounds_error=False, fill_value=np.nan)
            interpolated_data[var_name] = f(new_depth)
        
        # Store the interpolated data in the dictionary
        mdict[field_name] = pd.DataFrame(interpolated_data)
    
    transec_pon[i] = mdict

    
#%%
no3_2d = {}
pon_2d = {}
depth_2d ={}
lat_2d ={}
dens_2d ={}
fluor_2d ={}
grid_size = 200
depth_pon_2d = {}
density_pon_2d = {}
lat_2d_pon = {}

sorted_indices = []  # To store the sorted indices for depth
sorted_in = []
for k in range(len(transec)):
    print(k)
    
    df = transec[k]
    ctd = df[0]
    bott = df[1]
    
    pon = transec_pon[k]
    
    # Get all unique rawprofile field names
    rawprofile_fields = [field for field in bott if field.startswith('rawprofile')]
    
    fluor_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    no3_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    pon_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    depth_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    dens_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    lat_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)
    depth_grid_pon = np.full((grid_size, len(rawprofile_fields)), np.nan)
    density_grid_pon = np.full((grid_size, len(rawprofile_fields)), np.nan)
    lat_pon_grid = np.full((grid_size, len(rawprofile_fields)), np.nan)

    for i, profile_field in enumerate(rawprofile_fields):
        print(i)

        profile_bott = bott[profile_field]  
        profile_ctd = ctd[profile_field]  
        profile_PON = pon[profile_field]  
        
        # Access other profile variables and reorder them based on sorted indices
        profile_no3 = profile_bott['no3']
        profile_pon = profile_PON['PON']
        profile_lat = profile_ctd['lat']
        profile_lat_pon = profile_PON['latitude']
        profile_dens = profile_ctd['density']
        profile_depth = profile_ctd['depth']
        profile_fluor = profile_ctd['fluorecence']
        profile_depth_pon = profile_PON['depth']
        profile_density_pon = profile_PON['density']


        # Store the sorted variables in the grid
        no3_grid[:len(profile_no3), i] = profile_no3
        pon_grid[:len(profile_pon), i] = profile_pon
        depth_grid[:len(profile_depth), i] = profile_depth
        lat_grid[:len(profile_lat), i] = profile_lat
        dens_grid[:len(profile_dens), i] = profile_dens
        fluor_grid[:len(profile_dens), i] = profile_fluor
        depth_grid_pon[:len(profile_depth_pon), i] = profile_depth_pon
        density_grid_pon[:len(profile_density_pon), i] = profile_density_pon
        lat_pon_grid[:len(profile_lat_pon), i] = profile_lat_pon


    fluor_2d[k] = fluor_grid
    no3_2d[k] = no3_grid
    pon_2d[k] = pon_grid
    depth_2d[k] = depth_grid
    lat_2d[k] = lat_grid
    lat_2d_pon[k] = lat_pon_grid
    dens_2d[k] = dens_grid   
    depth_pon_2d[k] = depth_grid_pon
    density_pon_2d[k] = density_grid_pon

#%%DEFINE DELTA-RHO AND DELTA-Z = BOX IN WHICH CALCULATE DELTAS

def along_isopycnlas_interpolation(isopycnal, delta, transec, dens_2d, no3_2d, fluor_2d, lat_2d, pon_2d, depth_2d, depth_pon_2d, density_pon_2d, lat_2d_pon):
    
    results = []

    for k in range(len(transec)):
        
        density = dens_2d[k]
        density_pon = density_pon_2d[k]
        
        nit = no3_2d[k]
        chl = fluor_2d[k] *  0.9993071406551443 + 0.029374902209109433
        pon = pon_2d[k]
       
        lat = lat_2d[k]
        lat_pon = lat_2d_pon[k]
        z = depth_2d[k]
        z_pon = depth_pon_2d[k]
        
        target_density = isopycnal
        delta_rho =  delta
        # idxx = np.where(np.abs(density - target_density) <= delta_rho, 1 , np.nan) #2D matrix
        # nit_target = nit * idxx
        # lat_target = lat * idxx
        # sigma_target = density * idxx
        # # pon * idx
             
        #index from original matrix of the density between 26.3 +- 0.15
        idx = np.array(np.where(np.abs(density - target_density) <= delta_rho)).T
        idx_pon = np.array(np.where(np.abs(density_pon - target_density) <= delta_rho)).T
    
        #extract nitrogen correspondig to density
        nit_target = nit[idx[:, 0], idx[:, 1]]
        chl_target = chl[idx[:, 0], idx[:, 1]]
        pon_target = pon[idx_pon[:, 0], idx_pon[:, 1]] 
        
        lat_target = lat[idx[:, 0], idx[:, 1]]
        lat_target_pon = lat_pon[idx_pon[:, 0], idx_pon[:, 1]] 
        z_target = z[idx[:, 0], idx[:, 1]]
        z_target_pon = z_pon[idx_pon[:, 0], idx_pon[:, 1]]
        
        index = np.argsort(lat_target)
        coords_target = np.vstack((lat_target, z_target[index])).T
        
        index_pon = np.argsort(lat_target_pon)
        coords_target_pon = np.vstack((lat_target_pon, z_target_pon[index_pon])).T

        print('Original coords: ' + str(len(coords_target)))
        #extract from contour new coordinates for density contourf
        if k == 2:
            cs = plt.contourf(lat, z, density, levels = [target_density - delta_rho, target_density + delta_rho])

        else:
            cs = plt.contour(lat, z, density, levels = [target_density])
        
        index_isop  = np.argsort(cs.allsegs[0][0][:, 0])
        isop_coords = (np.sort(cs.allsegs[0][0][:, 0]), cs.allsegs[0][0][:, 1][index_isop]) #coords (Y, Z)
        print('Interpolated coords ' + str(len(cs.allsegs[0][0][:, 0])) + '\n')
        
        plt.plot(isop_coords[0], -isop_coords[1])
        plt.show()
        
        nit_linear = griddata(coords_target, nit_target, isop_coords, method = "linear")
        chl_linear = griddata(coords_target, chl_target, isop_coords, method = "linear")
        pon_linear = griddata(coords_target_pon, pon_target, isop_coords, method = "linear")
       # pon_linear = np.interp(isop_coords[0], lat_target_sorted, pon_target[index_pon])

        # Create a DataFrame for the current iteration's data and append it to the results list
        df = pd.DataFrame({
            'Latitude': np.flip(isop_coords[0]),
            'Nitrate': np.flip(nit_linear),
            'Chlorophyll': np.flip(chl_linear),
            'PON' : np.flip(pon_linear),
            'Depth': np.flip(isop_coords[1])
        })

        results.append(df)
        print(k)
        
    return results
#%%
path_deltas = '/home/serena/Scrivania/Magistrale/thesis/deltas/'
days = [17, 19, 21,  23, 25, 27]
#%%
results_266 = along_isopycnlas_interpolation(26.6, 0.05, transec, dens_2d, no3_2d, fluor_2d, lat_2d, pon_2d, depth_2d, depth_pon_2d, density_pon_2d, lat_2d_pon)

with pd.ExcelWriter(path_deltas + 'april_266_cor.xlsx') as writer:
    for i, x in enumerate(results_266):
        x.to_excel(writer, sheet_name='April%s' % days[i])

results_265 = along_isopycnlas_interpolation(26.5, 0.05, transec, dens_2d, no3_2d, fluor_2d, lat_2d, pon_2d, depth_2d, depth_pon_2d, density_pon_2d, lat_2d_pon)

with pd.ExcelWriter(path_deltas + 'april_265_cor.xlsx') as writer:
    for i, x in enumerate(results_265):
        x.to_excel(writer, sheet_name='April%s' % days[i])
        
results_264 = along_isopycnlas_interpolation(26.4, 0.1, transec, dens_2d, no3_2d, fluor_2d, lat_2d, pon_2d, depth_2d, depth_pon_2d, density_pon_2d, lat_2d_pon)

with pd.ExcelWriter(path_deltas + 'april_264_cor.xlsx') as writer:
    for i, x in enumerate(results_264):
        x.to_excel(writer, sheet_name='April%s' % days[i])       
