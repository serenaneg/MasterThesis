#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 11:44:23 2023

@author: serena
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset

bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"


# Carica i dati GEBCO
data = xr.open_dataset(bathy_path + "gebco_2023.nc")

lat = data.lat.to_numpy()
lon = data.lon.to_numpy()
depth = data.elevation.to_numpy()

# Definisci le coordinate della griglia 2D
x = np.arange(0, len(lon))  # Coordinate x (lon)
y = np.arange(0, len(lat))  # Coordinate y (lat)
X, Y = np.meshgrid(x, y)  # Griglia 2D delle coordinate

# Definisci i valori altimetrici della griglia 2D
Z = depth

# Definisci le coordinate della griglia 3D
z = np.linspace(0, 10000, num=100)  # Coordinate z (profondità)

# Effettua l'interpolazione bilineare per ottenere la griglia 3D
X_3d, Y_3d, Z_3d = np.meshgrid(x, y, z)  # Griglia 3D delle coordinate
points = np.column_stack((X.flatten(), Y.flatten()))  # Punti di input per l'interpolazione
values = Z.flatten()  # Valori altimetrici di input
points_3d = np.column_stack((X_3d.flatten(), Y_3d.flatten()))  # Punti di output per l'interpolazione
Z_3d_interpolated = griddata(points, values, points_3d, method='linear') # Interpolazione bilineare

Z_3d_reshaped = np.reshape(Z_3d_interpolated, X_3d.shape)

#%%
# Crea un DataArray con l'array Z_3d_interpolated
da = xr.DataArray(
    data=Z_3d_reshaped,
    dims=['lat', 'lon', 'depth'],
    coords={'lat': lat, 'lon': lon},
    attrs={'units': 'meters'}
)

# Crea un dataset con il DataArray
ds = xr.Dataset({'Z_3d_interpolated': da})

# Salva il dataset in un file NetCDF
ds.to_netcdf(bathy_path + "gebco_3d.nc")