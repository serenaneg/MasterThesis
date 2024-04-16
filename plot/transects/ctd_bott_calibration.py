#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:31:30 2023

@author: serena
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
from scipy import stats

#%%
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2018/"

file = path + 'bottle_ar29_hydro.csv'
bottle = pd.read_csv(file)

# file2 = path + 'ctd_tn_withlocation_apr18.csv'
# ctd_data = pd.read_csv(file2)
# #some station are nan => delete
# ctd_data = ctd_data.dropna(subset=['station'])


#%%APRIL
bottle['day'] = pd.to_datetime(bottle['day'])
#remouce nan values from nitrate
#bottle = bottle.dropna(subset=['no3'])

#%%
fluor = np.array(bottle.fluorescence)
chlor = np.array(bottle.chl)
phaeoa = np.array(bottle.phaeoa)

chlor = chlor + phaeoa
#remove nan
nan_indices = np.isnan(chlor)

clean_chlor = chlor[~nan_indices]
clean_fluor = fluor[~nan_indices]

plt.scatter(clean_chlor, clean_fluor)

# Calculate Z-scores for both 'x' and 'y' columns
zscore_chlor = zscore(clean_chlor)
zscore_fluor = zscore(clean_fluor)

# Define a threshold for the Z-score (e.g., 3) to identify outliers
threshold = 8

# Create a mask for outliers based on the Z-score threshold
outlier_mask = (np.abs(zscore_chlor) > threshold) | (np.abs(zscore_fluor) > threshold)

# Filter out the outliers from the data
filtered_chl = clean_chlor[~outlier_mask]
filtered_flor = clean_fluor[~outlier_mask]


# Plot the filtered data in a scatter plot
plt.scatter(filtered_chl, filtered_flor)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot without Outliers')
plt.show()

#%%
# filtered_chl = filtered_chl.reshape(-1, 1)
# filtered_flor = filtered_flor.reshape(-1, 1)

#X = fluorescence, y = clorophyll

# model = LinearRegression().fit(filtered_chl, filtered_flor)

# r_sq = model.score(filtered_chl, filtered_flor)

# print(f"intercept: {model.intercept_}")

# print(f"slope: {model.coef_}")
# # intercept: 0.031688137881046075
# # slope: [0.99951089]

stats = stats.linregress((filtered_chl, filtered_flor))
print(f'Slope = {stats.slope}')
print(f'Intercept = {stats.intercept}')
print(f'R_squared = {stats.rvalue}')


# Calculate RMSE
rmse = np.sqrt(np.mean((filtered_flor - filtered_chl) ** 2))
print(f"RMSE: {rmse}")

###########################################################################
#%%MAY
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"

file = path + 'bottle_rb1904_hydro.csv'
bottle = pd.read_csv(file)
#%%APRIL
bottle['day'] = pd.to_datetime(bottle['day'])

#%%
fluor = np.array(bottle.fluorescence)
chlor = np.array(bottle.chl)
phaeoa = np.array(bottle.phaeoa)

phytn = chlor + phaeoa
#remove nan
nan_indices = np.isnan(phytn)

clean_chlor = phytn[~nan_indices]
clean_fluor = fluor[~nan_indices]

plt.scatter(clean_chlor, clean_fluor)

# Calculate Z-scores for both 'x' and 'y' columns
zscore_chlor = zscore(clean_chlor)
zscore_fluor = zscore(clean_fluor)

# Define a threshold for the Z-score (e.g., 3) to identify outliers
threshold = 8
# Create a mask for outliers based on the Z-score threshold
outlier_mask = (np.abs(zscore_chlor) > threshold) | (np.abs(zscore_fluor) > threshold)

# Filter out the outliers from the data
filtered_chl = clean_chlor[~outlier_mask]
filtered_flor = clean_fluor[~outlier_mask]


# Plot the filtered data in a scatter plot
plt.scatter(filtered_chl, filtered_flor)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot without Outliers')

#%%
stats = stats.linregress((filtered_chl, filtered_flor))
print(f'Slope = {stats.slope}')
print(f'Intercept = {stats.intercept}')
print(f'R_squared = {stats.rvalue}')


# Calculate RMSE
rmse = np.sqrt(np.mean((filtered_flor - filtered_chl) ** 2))
print(f"RMSE: {rmse}")

#%%JULY
path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"

file = path + 'bottle_tn368_hydro.csv'
bottle = pd.read_csv(file)
#%%APRIL
bottle['day'] = pd.to_datetime(bottle['day'])

#%%
fluor = np.array(bottle.fluorescence)
chlor = np.array(bottle.chl)
phaeoa = np.array(bottle.phaeoa)

phytn = chlor + phaeoa
#remove nan
nan_indices = np.isnan(phytn)

clean_chlor = phytn[~nan_indices]
clean_fluor = fluor[~nan_indices]

plt.scatter(clean_chlor, clean_fluor)

# Calculate Z-scores for both 'x' and 'y' columns
zscore_chlor = zscore(clean_chlor)
zscore_fluor = zscore(clean_fluor)

# Define a threshold for the Z-score (e.g., 3) to identify outliers
threshold = 5
# Create a mask for outliers based on the Z-score threshold
outlier_mask = (np.abs(zscore_chlor) > threshold) | (np.abs(zscore_fluor) > threshold)

# Filter out the outliers from the data
filtered_chl = clean_chlor[~outlier_mask]
filtered_flor = clean_fluor[~outlier_mask]


# Plot the filtered data in a scatter plot
plt.scatter(filtered_chl, filtered_flor)
plt.xlabel('chl + phaoea')
plt.ylabel('fluorescence')
plt.title('Scatter Plot without Outliers')
plt.show()

#%%
stats = stats.linregress((filtered_chl, filtered_flor))
print(f'Slope = {stats.slope}')
print(f'Intercept = {stats.intercept}')
print(f'R_squared = {stats.rvalue}')


# Calculate RMSE
rmse = np.sqrt(np.mean((filtered_flor - filtered_chl) ** 2))
print(f"RMSE: {rmse}")
