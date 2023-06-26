#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:55:56 2023

@author: serena
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import scipy.io as sio 
#%%
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"

#data from mat file
file = sio.loadmat(path + 'tn368_ctd_data.mat')
print(file)

data = file['data']
col_name = file['data_columns']

info = file['info_ctd_casts']
col_name_info = file['info_ctd_casts_columns']
cast_station = file['info_ctd_casts_stations'] #cast, station, stationId

#%%get info variables
lat = info[:,7]
lon = info[:,8]
cast_num = info[:,0] #141 casts

station = cast_station[:,1] #==cast
stationId = cast_station[:,2]

year = np.array(info[:,1], dtype = 'int')
month = np.array(info[:,2], dtype = 'int')
day = np.array(info[:,3], dtype = 'int')
hh = np.array(info[:,4], dtype = 'int')
mins = np.array(info[:,5], dtype = 'int')
sec = np.array(info[:,6], dtype = 'int')

#%%

#create date
date = []

# Iterate over the arrays and combine the values into a datetime object
for i in range(len(year)):
    dt = datetime.datetime(year[i], month[i], day[i], hh[i], mins[i], sec[i]).strftime('%Y-%m-%d %H:%M:%S')
    date.append(dt)
    
#%%
ctd_location_time = np.column_stack((date, cast_num, station, lat, lon))
#1 cast each time but more data for cast => cast is the link with data

#Add col names
column_names = ['Date', 'Cast_num', 'Station', 'latitude', 'longitude']

# Convert to DataFrame
ctd_location_time = pd.DataFrame(ctd_location_time, columns=column_names)

pd.DataFrame(ctd_location_time).to_csv( path +"ctd_location_time.csv")
    
#%%remouve bad flag -9.989999999999999992e-29
# bad_flag = -9.989999999999999992e-29

# # Find the rows where -9.99 is present
# rows_to_delete = np.where(np.any(data == bad_flag, axis=1))[0] 
# #axis = 1 delete row, axis = 0 delet colums
# #[0] extract the row index

# # Delete the rows from the matrix
# data_nobad = np.delete(data, rows_to_delete, axis=0)

#%%SALINTY CHECK
# salinity = data_nobad[:,20]
salinity = data[:,20]

#CHECK CONTROL ON SALINITY
sal = []
index = []
for i in range(len(salinity)):
    if salinity[i] <= 25:
        sal.append(salinity[i])
        index.append([i])
        
sal = np.array(sal)

data_nobad = np.delete(data, 253, axis=0)
# data_nobad = np.delete(data_nobad, 253, axis=0)


#%%SELECT ONLY INTRESTIN VARIABLES
#interesting column
indices_col = [0, 1, 2, 7, 19, 20, 22, 24]

# Creazione della nuova matrice selezionando solo le colonne desiderate
variables = data_nobad[:, indices_col]

#save as csv
#Add col names
column_names = ['Cast_num', 'pressure','temperature', 'fluorecence', 'depth', 'salinity', 'oxigen', 'density']

# Convert to DataFrame
ctd_tn368 = pd.DataFrame(variables, columns=column_names)

pd.DataFrame(ctd_tn368).to_csv( path +"ctd_tn368.csv")
#%%%CTD WITH LOCATION
infile = path + "ctd_location_time.csv"
loc = pd.read_csv(infile)

filedata = path + 'ctd_tn368.csv'
ctd = pd.read_csv(filedata)

lat = []
lon = []
day = []
station = []
for i in ctd.Cast_num:
    sub = loc[loc.Cast_num ==i]
    lati = sub.latitude.item()
    loni = sub.longitude.item()
    dayi = sub.Date.item()
    stati = sub.Station.item()
    lat.append(lati)
    lon.append(loni)
    day.append(dayi)
    station.append(stati)
ctd['lat'] = lat
ctd['lon'] = lon
ctd['day'] = day
ctd['station'] = station

ctd.to_csv(path + 'ctd_tn_withlocation.csv')

