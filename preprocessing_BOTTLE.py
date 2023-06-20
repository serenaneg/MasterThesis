#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:43:49 2023

@author: serena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import scipy.io as sio 

def set_cbar(fig, c, title, ax):  
    # if ax==ax2:
    #     cbar = plt.colorbar(c, format='%.0f', spacing='proportional', ax=ax, shrink=0.9, pad = 0.01,
    #                         orientation = 'vertical', location = "right")
    # elif ax==ax3:
    #     cbar = plt.colorbar(c, format='%.2f', spacing='proportional', ax=ax, shrink=0.9, pad = 0.01,
    #                         orientation = 'vertical', location = "right")
    # else:
    cbar = plt.colorbar(c, format='%.1f', spacing='proportional', ax=ax, shrink=0.9, pad = 0.01,
                        orientation = 'vertical', location = "right")

    cbar.set_label(label = title, fontsize = 35, y = 0.5, labelpad = 30)
    cbar.ax.tick_params(which='minor', size=15, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=20, width=1, color='k', direction='in', labelsize = 30)
    # fig.subplots_adjust(bottom=0.25)
        # cbar.ax.set_position([0.2, 0.08, 0.6, 0.08])
    return cbar
#%%
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"

#data from mat file
file = sio.loadmat(path + 'tn368_bottle_data_Jul_2022.mat')
print(file)

data = file['data']
nutrient = file['nuts_columns']
info_ctd = file['CTD_casts_info_columns']

#%%
cast_num = data[:,0] #141 casts
station = data[:,1] #==cast
stationId = data[:,2]

year = np.array(data[:,3], dtype = 'int')
month = np.array(data[:,4], dtype = 'int')
day = np.array(data[:,5], dtype = 'int')
hhmm = np.array(data[:,6], dtype = 'int')

lat = data[:,7]
lon = data[:,8]
depth = data[:,10]

#%%
date = []

# Iterate over the arrays and combine the values into a datetime object
for i in range(len(year)):
    dt = datetime.datetime(year[i], month[i], day[i]).strftime('%Y-%m-%d')
    date.append(dt)
    

#%%SELECT ONLY INTRESTIN VARIABLES
data_time = np.column_stack((data, date))

#interesting column
indices_col = [0, 1, 7, 8, 10, 44, 45, 46, 47, 52, 147]

# Creazione della nuova matrice selezionando solo le colonne desiderate
variables = data_time[:, indices_col]

#save as csv
#Add col names
column_names = ['Cast_num', 'station', 'latitude', 'longitude', 'depth', 'no3', 'nh4', 'po4', 'si', 'pon', 'day']

# Convert to DataFrame
bottle_tn368 = pd.DataFrame(variables, columns=column_names)

pd.DataFrame(bottle_tn368).to_csv(path +"bottle_tn368.csv")

#%%BOTTLE WITH SALINTY AND DENSITY 
#interesting column
indices_col = [0, 1, 7, 8, 10, 44, 45, 46, 47, 13, 20, 63, 31, 52]

# Creazione della nuova matrice selezionando solo le colonne desiderate
variables = data[:, indices_col]
data_time = np.column_stack((variables, date))

#save as csv
#Add col names
column_names = ['Cast_num', 'station', 'latitude', 'longitude', 
                'depth', 'no3', 'nh4', 'po4', 'si','density', 'salinity','chlor', 'fluor',
                'PON', 'day']

# Convert to DataFrame
bottle_tn368 = pd.DataFrame(data_time, columns=column_names)

pd.DataFrame(bottle_tn368).to_csv(path +"bottle_tn368_hydro.csv")


    