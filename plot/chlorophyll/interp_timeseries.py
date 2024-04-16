
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:11:44 2023

@author: serena
"""

import xarray as xr
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from scipy import signal 
from scipy import interpolate

#%%FUNTIONS
def mean_values_list(variable):
            
    #create a new coodinates = day count
    # Calculate the start date
    variable['time'] = pd.to_datetime(variable['time'], format='%Y-%m-%d')
    start_date = variable['time'].min()
    
    # Calculate the time difference in days and store it in a new column
    variable['day_count'] = (variable['time'] - start_date).dt.days
    
    days = variable['day_count'].values.tolist()
    variable['time'] = variable['day_count']
       
    #daily mean of chl values => 1 value per day
    # variable_mean = variable.mean(dim=('longitude', 'latitude')) #for winds
    variable_mean = variable.mean(dim = ('lon', 'lat')) #for chl
 
    #remouve nan
    variable_mean = variable_mean.where(~np.isnan(variable_mean), 0)
    variable_list = variable_mean.values.tolist()
    
    return variable.day_count, variable_list

#%% LOAD DATA
path_chl = "/home/serena/Scrivania/Magistrale/thesis/data/MODIS_INTERPOLATED_DATA/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/chlor/"

df = xr.open_dataset(path_chl + "MODIS_chl_1D_nomissing.nc")

#%%
lat_range = slice(41, 38) #alto-basso
lon_range = slice(-72.6, -68)  #dx-sx
#chlorophyll
df = df.sel(lon=lon_range)
df = df.sel(lat=lat_range)

chl = df['chlor_a']
chl_days, chl_list = mean_values_list(chl)

ds_month = df.groupby('time.month').mean(skipna=True).load()
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

chl_month = ds_month['chlor_a']

#%%REMOVE ZERO VALUES FROM CHL
data = np.vstack([chl_days, chl_list])

#mask zeros
ma_data = np.ma.masked_equal(data,0)

#interp on complete timeserie
chl_ma = ma_data[1,:]
#%%
#delete masked array
chl_mask = chl_ma[~chl_ma.mask]

interp = interpolate.interp1d(np.arange(chl_mask.size), chl_mask , kind='nearest')
chl_stretch = interp(np.linspace(0, chl_mask.size-1, len(chl_list)))
 
chl = chl_stretch - np.mean(chl_stretch)

#%%TIME SERIES
year = np.array(range(0, 6574 + 1, 366))

fig2, bx = plt.subplots(1, 1, figsize = (42, 15))
bx.set_title("Time series of spatially avegraged Chl-a (01-01-2003 / 31-12-2020)", fontsize = 60, y = 1.05)
bx.plot(chl_days, chl, "-", label = "Interpolated", color = "seagreen") 
bx.plot(chl_days, chl_stretch, "-", label = "Interpolated", color = "seagreen") 
bx.plot(chl_days, chl_ma, 'o', color = 'navy', label = "Chlor data")
bx.set_xlim(0,6600)
bx.set_ylabel('Chl-a concentration  [mg/m$^{3}$]', fontsize = 50)
bx.set_xlabel('Years', fontsize= 50)
bx.set_yscale('log')
bx.yaxis.set_major_formatter(ScalarFormatter())
bx.set_xticks(year)
bx.set_xticklabels(['2003','2004', '2005','2006','2007','2008','2009','2010','2011', 
                    '2012','2013','2014','2015','2016','2017','2018','2019', '2020'])
bx.xaxis.set_tick_params(labelsize = 40)
bx.yaxis.set_tick_params(labelsize = 40)
bx.legend(loc='lower right', fontsize = 40)
bx.set_facecolor('whitesmoke')
bx.grid(linewidth = .75, linestyle='--',  color = 'gray')

fig2.tight_layout()
plt.show()
fig2.savefig(plot_path + "timeseries_chlor_interp.png", bbox_inches='tight')

#%%
april = np.array(range(120, 6574 + 1, 365))
dec = np.array(range(365, 6574 + 1, 365))


fig, bx = plt.subplots(1, 1, figsize = (42, 15))
bx.set_title("Time series of spatially averaged Chl-a (01-01-2003 / 31-12-2020)", fontsize = 60, y = 1.05)
bx.plot(chl_days, chl, "-", label = 'Chl(t) - Mean', color = "seagreen", linewidth = 2.5) 
bx.set_xlim(0,6600)
bx.set_ylabel('Chl-a concentration  [mg/$m^{3}$]', fontsize = 50)
bx.set_xlabel('Years', fontsize= 50)
bx.yaxis.set_major_formatter(ScalarFormatter())
bx.set_xticks(year)
bx.set_xticklabels(['2003','2004', '2005','2006','2007','2008','2009','2010','2011', 
                    '2012','2013','2014','2015','2016','2017','2018','2019', '2020'])
bx.xaxis.set_tick_params(labelsize = 40)
bx.yaxis.set_tick_params(labelsize = 40)
bx.set_facecolor('whitesmoke')
bx.grid(linewidth = .75, linestyle='--', color = 'gray')
bx.legend(loc='upper right', fontsize = 40)

# Iterate over each year
for yy in range(0, 18):
    march_start = yy * 365 + 59  
    april_end = yy * 365 + 120   
    bx.fill_between((march_start, april_end), 0, np.max(chl), color='red', alpha=0.2)
    half_oct = yy * 365 + 289 #half oct  
    nov_end = yy * 365 + 335
    bx.fill_between((half_oct, nov_end), 0, np.max(chl), color='blue', alpha=0.2)
for j in dec:  
    bx.vlines(x=j, ymin = 0, ymax = np.max(chl), colors='k', lw=3, linestyle = '--')

fig.tight_layout()
plt.show()
fig.savefig(plot_path + "chl_noMean.png", bbox_inches='tight')

#%%FILTERING
#running mean
chl_ds = pd.DataFrame({'Chlor': chl_stretch})

run_mean = chl_ds.rolling(20, center=True).mean()

#%%
minim = np.nanmin(run_mean.Chlor)
maxi = np.nanmax(run_mean.Chlor)

fig, bx = plt.subplots(1, 1, figsize = (42, 15))
bx.set_title("Filtered time series Chl-a (rolling mean 20 days window)", fontsize = 60, y = 1.05)
bx.plot(chl_days, run_mean.Chlor, color = 'seagreen', linewidth = 2.5)
bx.set_xlim(0,6600)
bx.set_ylim(bottom = minim-0.04, top = maxi+0.2)
bx.set_ylabel('Chl-a concentration  [mg/m$^{3}$]', fontsize = 50)
bx.set_xlabel('Years', fontsize= 50)
bx.set_yscale('log')
bx.yaxis.set_major_formatter(ScalarFormatter())
bx.set_xticks(year)
bx.set_xticklabels(['2003','2004', '2005','2006','2007','2008','2009','2010','2011', 
                    '2012','2013','2014','2015','2016','2017','2018','2019', '2020'])
bx.xaxis.set_tick_params(labelsize = 40)
bx.yaxis.set_tick_params(labelsize = 40)
bx.grid(linewidth = .75, linestyle='--', color = 'gray')
bx.set_facecolor('whitesmoke')
for i in april:
    bx.vlines(x=i, ymin = 0, ymax = maxi+0.2, colors='red', lw=3, linestyle = '--')
for j in dec:
    bx.vlines(x=j, ymin = 0, ymax = maxi+0.2, colors='k', lw=3, linestyle = '--')

fig.tight_layout()
plt.show()
fig.savefig(plot_path + "timeseries_filtered.png", bbox_inches='tight')